#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

std::string randomImagePath(const fs::path& root = "images") {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
            ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
            files.push_back(entry.path());
        }
    }
    if (files.empty()) throw std::runtime_error("No images found in " + root.string());

    static std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<std::size_t> dist(0, files.size() - 1);
    return files[dist(rng)].string();
}

// ---------------------------------------------------------------------
static Mat toGray(const Mat& src) {
    Mat g;
    if(src.channels()==3) cvtColor(src, g, COLOR_BGR2GRAY);
    else g = src.clone();
    return g;
}

static cv::Mat claheLabL(const cv::Mat& bgr, double clip=3.0, cv::Size tiles=cv::Size(8,8)){
    CV_Assert(bgr.channels()==3);
    cv::Mat lab; cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch; cv::split(lab, ch);           // L,a,b
    auto clahe = cv::createCLAHE(clip, tiles);
    clahe->apply(ch[0], ch[0]);                            // enhance L only
    cv::merge(ch, lab);
    cv::Mat out; cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
    return out;
}

static Mat chromaBinForeground(const Mat& bgr) {
    CV_Assert(bgr.channels()==3);
    Mat lab; cvtColor(bgr, lab, COLOR_BGR2Lab);
    vector<Mat> ch; split(lab, ch); // L,a,b

    // Model background chroma from image borders
    Mat border = Mat::zeros(bgr.size(), CV_8U);
    int m = std::max(5, std::min(bgr.rows, bgr.cols)/40);
    rectangle(border, Rect(0,0,bgr.cols,m),           255, FILLED);
    rectangle(border, Rect(0,bgr.rows-m,bgr.cols,m),  255, FILLED);
    rectangle(border, Rect(0,0,m,bgr.rows),           255, FILLED);
    rectangle(border, Rect(bgr.cols-m,0,m,bgr.rows),  255, FILLED);
    Scalar meanA = mean(ch[1], border);
    Scalar meanB = mean(ch[2], border);

    // Chroma distance map: sqrt((a-a_bg)^2 + (b-b_bg)^2)
    Mat a32, b32; ch[1].convertTo(a32, CV_32F); ch[2].convertTo(b32, CV_32F);
    Mat da = a32 - (float)meanA[0], db = b32 - (float)meanB[0];
    Mat dist; magnitude(da, db, dist);                    // CV_32F
    Mat dist8; normalize(dist, dist8, 0, 255, NORM_MINMAX); dist8.convertTo(dist8, CV_8U);

    // Robust global threshold on chroma distance
    Mat bw; threshold(dist8, bw, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // Small cleanup to fill tiny gaps
    morphologyEx(bw, bw, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
    return bw;
}

static int distanceAndComponents(const Mat& filled, Mat& dist32f, Mat& sureFG, Mat& markers) {
    Mat clean;
    morphologyEx(filled, clean, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));

    // Clean tiny noise first (optional area filter)
    Mat lbl, stats, centroids;
    int ncc = connectedComponentsWithStats(filled, lbl, stats, centroids, 8, CV_32S);
    Mat cleaned = Mat::zeros(filled.size(), CV_8U);
    for (int i = 1; i < ncc; ++i) {
        if (stats.at<int>(i, CC_STAT_AREA) >= 80)   // tune min area
            cleaned.setTo(255, (lbl == i));
    }
    
    // Seeds per component using local DT peak
    Mat seeds = Mat::zeros(filled.size(), CV_8U);
    for (int i = 1; i < ncc; ++i) {
        if (stats.at<int>(i, CC_STAT_AREA) < 80) continue;
        Mat compMask = (lbl == i);

        Mat dist;
        distanceTransform(compMask, dist, DIST_L2, 3);

        // --- NEW: normalize + smooth to reduce fragmented peaks ---
        normalize(dist, dist, 0, 1, NORM_MINMAX);
        GaussianBlur(dist, dist, Size(5,5), 0);

        double maxv = 0;
        minMaxLoc(dist, nullptr, &maxv, nullptr, nullptr, compMask);
        if (maxv < 0.05) continue;

        Mat s;
        double area = stats.at<int>(i, CC_STAT_AREA);
        Scalar meanDist = mean(dist, compMask);
        double ratio = meanDist[0] / maxv;
        double factor = std::clamp(0.45 + 0.3 * ratio + 200.0 / (area + 50.0), 0.40, 0.80);

        // --- threshold now uses smoothed, normalized distance map ---
        threshold(dist, s, factor * maxv, 255, THRESH_BINARY);

        s.convertTo(s, CV_8U);
        morphologyEx(s, s, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3,3))); // de-noise seeds
        seeds |= s;
    }


    // expose outputs in your function
    distanceTransform(cleaned, dist32f, DIST_L2, 3);  // keep DT if you still want to visualize it
    sureFG = seeds.clone();

    connectedComponents(sureFG, markers);
    markers += 1;
    Mat sureBG; dilate(cleaned, sureBG, getStructuringElement(MORPH_ELLIPSE, Size(5,5)), Point(-1,-1), 2);
    Mat unknown; subtract(sureBG, sureFG, unknown);
    markers.setTo(0, unknown > 0);

    // 🔹 Return number of distinct foreground components
    double minv, maxv;
    minMaxLoc(markers, &minv, &maxv);
    int numPills = static_cast<int>(maxv) - 1; // subtract background and shift
    return numPills;
}

static void drawMarkersOn(Mat& img, const Mat& markers) {
    double minv, maxv; minMaxLoc(markers, &minv, &maxv);
    for(int lbl=2; lbl<= (int)maxv; ++lbl){
        Mat mask = (markers == lbl);
        Moments m = moments(mask, true);
        if(m.m00 < 5.0) continue;
        Point2d c(m.m10/m.m00, m.m01/m.m00);
        circle(img, c, 6, Scalar(0,0,255), 2);
        putText(img, std::to_string(lbl-1), c + Point2d(8, -8),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 1);
    }
}

// Fill interior holes of foreground without contours
static Mat fillHoles(const Mat& bw) {
    CV_Assert(bw.type()==CV_8U);
    Mat ff = bw.clone();
    // Flood fill background from the outside border
    // Pad to avoid filling touching the edge ambiguity
    Mat pad(bw.rows+2, bw.cols+2, CV_8U, Scalar(0));
    bw.copyTo(pad(Rect(1,1,bw.cols,bw.rows)));
    floodFill(pad, Point(0,0), 255);
    Mat bg = pad(Rect(1,1,bw.cols,bw.rows));
    bitwise_not(bg, bg);             // holes-only mask
    Mat filled; bitwise_or(bw, bg, filled);
    return filled;
}

// Marker-controlled watershed without contours
static int splitByWatershed(const Mat& bgr, const Mat& fgMask, Mat& markersOut, Mat& sureFGOut) {
    CV_Assert(fgMask.type()==CV_8U && bgr.channels()==3);

    // Light cleanup and hole fill
    Mat cleaned;
    morphologyEx(fgMask, cleaned, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
    Mat filled = fillHoles(cleaned);

    // Distance transform → seeds
    Mat dist; distanceTransform(filled, dist, DIST_L2, 5);
    GaussianBlur(dist, dist, Size(3,3), 0.8);

    double maxv=0; minMaxLoc(dist, nullptr, &maxv);
    Mat sureFG; threshold(dist, sureFG, 0.35*maxv, 255, THRESH_BINARY); // tune 0.35 if under/over-splitting
    sureFG.convertTo(sureFG, CV_8U);
    morphologyEx(sureFG, sureFG, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(3,3)));

    // Connected components → initial markers: 0 unknown, 1 background, >=2 objects
    Mat markers; int ncc = connectedComponents(sureFG, markers, 8, CV_32S);
    markers += 1; // background becomes 1, objects 2..n

    // Sure background via dilation
    Mat sureBG;
    dilate(filled, sureBG, getStructuringElement(MORPH_ELLIPSE, Size(5,5)), Point(-1,-1), 2);

    // Unknown region → 0 in markers
    Mat unknown; subtract(sureBG, sureFG, unknown);
    markers.setTo(0, unknown > 0);

    // Watershed is driven by image gradients; keep it simple on grayscale
    Mat gray; cvtColor(bgr, gray, COLOR_BGR2GRAY);
    Mat gx, gy, grad;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);
    magnitude(gx, gy, grad);
    grad.convertTo(grad, CV_8U);

    // Run watershed
    watershed(bgr, markers);

    // Clean labels: zero outside foreground and on boundaries
    markers.setTo(0, filled == 0);   // outside FG
    markers.setTo(0, markers == -1); // boundaries

    // Count unique labels >=2
    double mn, mx; minMaxLoc(markers, &mn, &mx);
    int num = 0;
    for (int lbl = 2; lbl <= (int)mx; ++lbl) {
        if (countNonZero(markers == lbl) > 0) ++num;
    }

    markersOut = markers.clone();
    sureFGOut = sureFG.clone();
    return num;
}

// ---------------------------------------------------------------------
static int processClusteredAndOverlayMarkers(const Mat& img, const Mat& clus_img, Mat& visOut, Mat& sureFGOut) {
    
    Mat bw = chromaBinForeground(clus_img);

    Mat markers, sureFG;
    int numPills = splitByWatershed(clus_img, bw, markers, sureFG);

    visOut = img.clone();
    drawMarkersOn(visOut, markers);

    // Optional visualization
    imshow("SureFG", sureFG);
    imshow("Markers_on_Original", visOut);

    sureFGOut = sureFG;

    return numPills; // 🔹 return count
}

int main() {
    fs::path imgDir = "images";
    if (!fs::exists(imgDir) || !fs::is_directory(imgDir)) {
        cerr << "'images' directory not found!\n";
        return -1;
    }
    fs::create_directories("results"); // just to hold the log
    std::ofstream log("results/run_log.txt", std::ios::app);
    auto logln = [&](const std::string& s){ std::cout << s << '\n'; if (log) log << s << '\n'; };

    for (const auto& entry : fs::directory_iterator(imgDir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" &&
            ext != ".bmp" && ext != ".tif" && ext != ".tiff") continue;

        std::string path = entry.path().string();
        Mat img = imread(path);
        if (img.empty()) { logln("⚠️ Skipping unreadable file: " + path); continue; }

        // --- Preprocess & transform ---
        Mat clahe_lab = claheLabL(img);

        // --- Run pipeline (this also shows intermediate windows if enabled inside) ---
        Mat vis, sureFG;
        int numPills = processClusteredAndOverlayMarkers(img, clahe_lab, vis, sureFG);

        // --- Parse ground-truth from filename and log ---
        fs::path inPath(path);
        std::string baseName = inPath.stem().string(); // e.g., p19_44
        size_t us = baseName.find('_');
        int actual = 0;
        if (us != std::string::npos && us + 1 < baseName.size()) {
            try { actual = std::stoi(baseName.substr(us + 1)); } catch (...) {}
        }
        std::ostringstream oss;
        oss << "File: " << inPath.filename().string()
            << " | Detected: " << numPills;
        if (actual > 0) {
            double acc = 100.0 * (static_cast<double>(numPills) / actual);
            oss << " | Actual: " << actual << " | Accuracy: " << std::fixed << std::setprecision(2) << acc << "%";
            if (numPills > actual) oss << " | Note: possible false positives";
        } else {
            oss << " | Actual: N/A";
        }
        logln(oss.str());

        // --- Display images (no saving) ---
        imshow("CLAHE Transformed Image", clahe_lab);
        imshow("Sure Foreground", sureFG);
        imshow("Markers on Original", vis);

        // Controls: press ESC/Q to quit, any other key to proceed to next image
        int key = cv::waitKey(0);
        if (key == 27 || key == 'q' || key == 'Q') break;
        cv::destroyAllWindows();
    }

    logln("Finished displaying all images.");
    return 0;
}