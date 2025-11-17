#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Compute the median of a channel over a masked region.
// Used to estimate background chroma statistics robustly.
static double maskedMedian(const cv::Mat& channel, const cv::Mat& mask)
{
    CV_Assert(channel.type() == CV_32F);
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(channel.size() == mask.size());

    std::vector<float> vals;
    vals.reserve(channel.rows * channel.cols / 10);

    for (int y = 0; y < channel.rows; ++y) {
        const float* cptr = channel.ptr<float>(y);
        const uchar* mptr = mask.ptr<uchar>(y);
        for (int x = 0; x < channel.cols; ++x) {
            if (mptr[x]) vals.push_back(cptr[x]);
        }
    }

    if (vals.empty()) return 0.0;

    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    return vals[vals.size() / 2];
}

// Build "sure foreground" seeds per connected component,
// using local distance transforms instead of one global DT.
// static void buildSeedsPerComponent(const Mat& fgMask,
//                                    Mat& sure_fg,
//                                    Mat& dist_vis)
// {
//     CV_Assert(fgMask.type() == CV_8U);

//     Mat lbl, stats, centroids;
//     int ncc = connectedComponentsWithStats(fgMask, lbl, stats, centroids, 8, CV_32S);

//     Mat seeds = Mat::zeros(fgMask.size(), CV_8U);
//     Mat distAll = Mat::zeros(fgMask.size(), CV_32F);

//     Mat seOpen = getStructuringElement(MORPH_ELLIPSE, Size(3,3));

//     for (int i = 1; i < ncc; ++i) {
//         int area = stats.at<int>(i, CC_STAT_AREA);
//         if (area < 50) continue; // ignore junk

//         Mat compMask = (lbl == i);

//         Mat dist;
//         distanceTransform(compMask, dist, DIST_L2, 3);

//         double maxv = 0.0;
//         minMaxLoc(dist, nullptr, &maxv, nullptr, nullptr, compMask);
//         if (maxv < 2.0) continue;

//         // Normalize + smooth to stabilize peaks
//         normalize(dist, dist, 0.0, 1.0, NORM_MINMAX);
//         GaussianBlur(dist, dist, Size(5,5), 0);

//         // Local adaptive threshold: one or few seeds per object
//         // Factor can be tuned; 0.5 is a good starting point.
//         Mat s;
//         threshold(dist, s, 0.5, 1.0, THRESH_BINARY);
//         s.convertTo(s, CV_8U, 255.0);

//         morphologyEx(s, s, MORPH_OPEN, seOpen); // clean noise
//         seeds |= s;

//         // For visualization of DT, keep the max over all components
//         max(distAll, dist, distAll);
//     }

//     sure_fg = seeds;

//     normalize(distAll, dist_vis, 0, 255, NORM_MINMAX);
//     dist_vis.convertTo(dist_vis, CV_8U);
// }

static void buildSeedsPerComponent(const Mat& fgMask,
                                   Mat& sure_fg,
                                   Mat& dist_vis)
{
    CV_Assert(fgMask.type() == CV_8U);

    Mat lbl, stats, centroids;
    int ncc = connectedComponentsWithStats(fgMask, lbl, stats, centroids, 8, CV_32S);

    // --- 1) Collect areas to estimate "typical pill size" ---
    std::vector<int> areas;
    areas.reserve(ncc);
    for (int i = 1; i < ncc; ++i) {
        int a = stats.at<int>(i, CC_STAT_AREA);
        if (a >= 50) areas.push_back(a);
    }
    if (areas.empty()) {
        sure_fg = Mat::zeros(fgMask.size(), CV_8U);
        dist_vis = Mat::zeros(fgMask.size(), CV_8U);
        return;
    }
    std::nth_element(areas.begin(), areas.begin() + areas.size()/2, areas.end());
    double medArea = areas[areas.size()/2];

    Mat seeds   = Mat::zeros(fgMask.size(), CV_8U);
    Mat distAll = Mat::zeros(fgMask.size(), CV_32F);

    Mat seOpen  = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    Mat seClose = getStructuringElement(MORPH_ELLIPSE, Size(5,5));

    for (int i = 1; i < ncc; ++i) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area < 50) continue;

        Mat compMask = (lbl == i);

        // Distance transform for this component
        Mat dist;
        distanceTransform(compMask, dist, DIST_L2, 3);

        double maxv = 0.0;
        minMaxLoc(dist, nullptr, &maxv, nullptr, nullptr, compMask);
        if (maxv < 2.0) continue;

        Mat distNorm;
        normalize(dist, distNorm, 0.0, 1.0, NORM_MINMAX);

        Mat s = Mat::zeros(fgMask.size(), CV_8U);

        // ----------------------------------------------------
        // SMALL / ISOLATED COMPONENT: 1 SEED PER COMPONENT
        // ----------------------------------------------------
        if (area < 1.8 * medArea) {
            // take the global maximum of the DT and plant a small disc there
            Point maxLoc;
            minMaxLoc(distNorm, nullptr, nullptr, nullptr, &maxLoc, compMask);

            // radius ~ 30% of equivalent radius
            double eqR = std::sqrt(area / CV_PI);
            int r = std::max(2, (int)std::round(0.3 * eqR));

            circle(s, maxLoc, r, Scalar(255), FILLED);
        }
        // ----------------------------------------------------
        // LARGE / MERGED COMPONENT: multi-seed from DT
        // ----------------------------------------------------
        else {
            Mat sLocal;
            threshold(distNorm, sLocal, 0.65, 1.0, THRESH_BINARY);
            sLocal.convertTo(sLocal, CV_8U, 255.0);

            morphologyEx(sLocal, sLocal, MORPH_OPEN,  seOpen);
            morphologyEx(sLocal, sLocal, MORPH_CLOSE, seClose);

            // restrict seeds to this component
            bitwise_and(sLocal, compMask, s);
        }

        seeds |= s;
        max(distAll, distNorm, distAll);
    }

    sure_fg = seeds;

    normalize(distAll, dist_vis, 0, 255, NORM_MINMAX);
    dist_vis.convertTo(dist_vis, CV_8U);
}

// Generate a binary foreground mask based on chromatic contrast (Lab color space).
// Steps:
//   1. Convert BGR → Lab.
//   2. Estimate background chroma (a,b) using the border region.
//   3. Compute per-pixel chroma distance from that background median.
//   4. Normalize + Otsu threshold → binary mask.
//   5. Morphological closing to smooth edges and fill small gaps.
static cv::Mat chromaBinForeground(const cv::Mat& bgr)
{
    CV_Assert(bgr.channels() == 3);

    // --- Convert to Lab color space ---
    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch;
    cv::split(lab, ch); // L, a, b

    // --- Build border mask for background sampling ---
    cv::Mat border = cv::Mat::zeros(bgr.size(), CV_8U);
    int m = std::max(5, std::min(bgr.rows, bgr.cols) / 40);
    cv::rectangle(border, cv::Rect(0, 0, bgr.cols, m),               255, cv::FILLED);
    cv::rectangle(border, cv::Rect(0, bgr.rows - m, bgr.cols, m),    255, cv::FILLED);
    cv::rectangle(border, cv::Rect(0, 0, m, bgr.rows),               255, cv::FILLED);
    cv::rectangle(border, cv::Rect(bgr.cols - m, 0, m, bgr.rows),    255, cv::FILLED);

    // --- Convert channels to float for math ---
    cv::Mat L32, a32, b32;
    ch[0].convertTo(L32, CV_32F);
    ch[1].convertTo(a32, CV_32F);
    ch[2].convertTo(b32, CV_32F);

    // --- Compute robust background center (median chroma) ---
    double medA = maskedMedian(a32, border);
    double medB = maskedMedian(b32, border);

    // --- Compute chroma distance from background ---
    cv::Mat dA = a32 - static_cast<float>(medA);
    cv::Mat dB = b32 - static_cast<float>(medB);

    cv::Mat dist;
    cv::magnitude(dA, dB, dist); // Euclidean distance in chroma space

    // --- Normalize and threshold ---
    cv::Mat dist8;
    cv::normalize(dist, dist8, 0, 255, cv::NORM_MINMAX);
    dist8.convertTo(dist8, CV_8U);

    cv::Mat bw;
    cv::threshold(dist8, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // --- Force border to background ---
    bw.setTo(0, border);

    // --- Morphological cleanup (mild closing) ---
    cv::morphologyEx(bw, bw, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));

    return bw; // 255 = foreground, 0 = background
}

// ---------- Helper: count objects from watershed markers ----------
static int countObjectsFromMarkers(const Mat& markers) {
    CV_Assert(markers.type() == CV_32S);
    double minv, maxv;
    minMaxLoc(markers, &minv, &maxv);
    int maxLabel = static_cast<int>(maxv);
    int count = 0;
    for (int lbl = 2; lbl <= maxLabel; ++lbl) {
        if (countNonZero(markers == lbl) > 0)
            ++count;
    }
    return count;
}

int main() {
    fs::path imgDir = "images";
    if (!fs::exists(imgDir) || !fs::is_directory(imgDir)) {
        cerr << "'images' directory not found\n";
        return -1;
    }

    fs::create_directories("results");
    ofstream log("results/run_chroma_ws.txt", ios::app);
    auto logln = [&](const string& s) {
        cout << s << '\n';
        if (log) log << s << '\n';
    };

    double sum_acc = 0.0;
    int cnt_acc = 0;

    for (const auto& entry : fs::directory_iterator(imgDir)) {
        if (!entry.is_regular_file()) continue;

        string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" &&
            ext != ".bmp" && ext != ".tif" && ext != ".tiff")
            continue;

        string path = entry.path().string();
        Mat img = imread(path);
        if (img.empty()) {
            logln("Skipping unreadable file: " + path);
            continue;
        }

        // -------------------------------------------------
        // 1) Chromatic binary mask -> thresh
        // -------------------------------------------------
        Mat thresh = chromaBinForeground(img); // 255 = foreground blob

        imshow("Step 0 - Original", img);
        imshow("Step 1 - Chroma Mask", thresh);

        // -------------------------------------------------
        // 2) From chroma mask: opening, sure_bg, dist, sure_fg, unknown
        // -------------------------------------------------
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3));

        // Mat opening;
        // morphologyEx(thresh, opening, MORPH_OPEN, kernel, Point(-1,-1), 2);
        // imshow("Step 2 - Opening (Chroma Mask Cleaned)", opening);

        // Mat sure_bg;
        // dilate(opening, sure_bg, kernel, Point(-1,-1), 5);
        // Mat smoothed;
        // morphologyEx(opening, smoothed, MORPH_CLOSE,
        //             getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));
        // imshow("Step 2.5 - Smoothed Mask before DT", smoothed);
        // imshow("Step 3 - Sure Background (from Chroma)", sure_bg);

        Mat opening;
        morphologyEx(thresh, opening, MORPH_OPEN, kernel, Point(-1,-1), 2);
        imshow("Step 2 - Opening (Chroma Mask Cleaned)", opening);

        // Decide whether we actually need the strong closing.
        // Use median aspect ratio of connected components.
        Mat lblAR, statsAR, centAR;
        int nccAR = connectedComponentsWithStats(opening, lblAR, statsAR, centAR, 8, CV_32S);

        std::vector<double> aspect;
        aspect.reserve(std::max(0, nccAR - 1));
        for (int i = 1; i < nccAR; ++i) {
            int area  = statsAR.at<int>(i, CC_STAT_AREA);
            int w     = statsAR.at<int>(i, CC_STAT_WIDTH);
            int h     = statsAR.at<int>(i, CC_STAT_HEIGHT);
            if (area < 50) continue;
            double ar = (double)std::max(w, h) / std::max(1, std::min(w, h));
            aspect.push_back(ar);
        }

        double medAR = 1.0;
        if (!aspect.empty()) {
            std::nth_element(aspect.begin(),
                            aspect.begin() + aspect.size() / 2,
                            aspect.end());
            medAR = aspect[aspect.size() / 2];
        }

        // If pills are elongated (capsules), use strong closing.
        // If they are compact (squares / circles), skip it.
        Mat fgForDT = opening.clone();
        if (medAR > 1.5) {
            morphologyEx(fgForDT, fgForDT, MORPH_CLOSE,
                        getStructuringElement(MORPH_ELLIPSE, Size(9,9)));
        }
        imshow("Step 2.5 - Smoothed Mask before DT", fgForDT);

        // 2.4 Sure background from *smoothed* FG
        Mat sure_bg;
        dilate(fgForDT, sure_bg, kernel, Point(-1,-1), 3);
        imshow("Step 3 - Sure Background (from Chroma)", sure_bg);

        Mat dist_vis, sure_fg;
        buildSeedsPerComponent(fgForDT, sure_fg, dist_vis);
        imshow("Step 4 - Distance Transform (Chroma, per-component)", dist_vis);
        imshow("Step 5 - Sure Foreground (from Chroma, per-component)", sure_fg);

        Mat unknown;
        subtract(sure_bg, sure_fg, unknown);
        imshow("Step 6 - Unknown Region", unknown);

        // -------------------------------------------------
        // 3) Markers from sure_fg
        // -------------------------------------------------
        Mat markers;
        int nLabels = connectedComponents(sure_fg, markers, 8, CV_32S);
        (void)nLabels;

        markers += 1;                 // background => 1
        markers.setTo(0, unknown == 255); // unknown => 0

        // -------------------------------------------------
        // 4) Build Lab L-channel + CLAHE -> topography gradient
        // -------------------------------------------------
        Mat lab;
        cvtColor(img, lab, COLOR_BGR2Lab);
        vector<Mat> labCh;
        split(lab, labCh);            // labCh[0] = L
        Mat L = labCh[0];

        // CLAHE for lighting normalization
        Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8,8));
        Mat L_eq;
        clahe->apply(L, L_eq);
        imshow("Step 7 - L Channel CLAHE", L_eq);

        // Gaussian blur for noise suppression
        Mat L_blur;
        GaussianBlur(L_eq, L_blur, Size(9, 9), 3);

        // Sobel gradient magnitude
        Mat gx, gy, grad;
        Sobel(L_blur, gx, CV_32F, 1, 0, 3);
        Sobel(L_blur, gy, CV_32F, 0, 1, 3);
        magnitude(gx, gy, grad);

        Mat grad8u;
        normalize(grad, grad8u, 0, 255, NORM_MINMAX);
        grad8u.convertTo(grad8u, CV_8U);
        imshow("Step 8 - Gradient Magnitude (Topography)", grad8u);

        Mat wshedTopo;
        cvtColor(grad8u, wshedTopo, COLOR_GRAY2BGR);

        // -------------------------------------------------
        // 5) Run watershed on gradient topography with chroma-based markers
        // -------------------------------------------------
        watershed(wshedTopo, markers);

        // Visualize final boundaries on original image
        Mat finalSeg = img.clone();
        for (int y = 0; y < markers.rows; ++y) {
            const int* mp = markers.ptr<int>(y);
            Vec3b* cp = finalSeg.ptr<Vec3b>(y);
            for (int x = 0; x < markers.cols; ++x) {
                if (mp[x] == -1) {
                    cp[x] = Vec3b(0, 0, 255); // boundary in red
                }
            }
        }
        imshow("Step 9 - Final Segmentation (Chroma + Gradient)", finalSeg);

        int numDetected = countObjectsFromMarkers(markers);

        // -------------------------------------------------
        // 6) Accuracy from filename pXX_YY -> YY
        // -------------------------------------------------
        fs::path inPath(path);
        string baseName = inPath.stem().string(); // e.g. p18_79
        size_t us = baseName.find('_');
        int actual = 0;
        if (us != string::npos && us + 1 < baseName.size()) {
            try { actual = stoi(baseName.substr(us + 1)); }
            catch (...) { actual = 0; }
        }

        ostringstream oss;
        oss << "File: " << inPath.filename().string()
            << " | Detected: " << numDetected;
        if (actual > 0) {
            double acc = 100.0 * (static_cast<double>(numDetected) / actual);
            sum_acc += acc;
            cnt_acc += 1;
            oss << " | Actual: " << actual
                << " | Accuracy: " << fixed << setprecision(2) << acc << "%";
            if (numDetected > actual) oss << " | Note: possible over-count";
        } else {
            oss << " | Actual: N/A";
        }
        logln(oss.str());

        // Wait key per image
        int key = waitKey(0);
        if (key == 27 || key == 'q' || key == 'Q') {
            destroyAllWindows();
            break;
        }
        destroyAllWindows();
    }

    if (cnt_acc > 0) {
        double avg_acc = sum_acc / cnt_acc;
        ostringstream overall;
        overall << "Overall average accuracy across " << cnt_acc
                << " labeled image(s): " << fixed << setprecision(2)
                << avg_acc << "%";
        cout << overall.str() << '\n';
        if (log) log << overall.str() << '\n';
    } else {
        string msg = "Overall average accuracy: N/A (no labeled images found)";
        cout << msg << '\n';
        if (log) log << msg << '\n';
    }

    logln("Finished chroma+gradient watershed run.");
    return 0;
}