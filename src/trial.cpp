#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <random>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Helpers

static Mat toGray(const Mat& src) {
    Mat g;
    if (src.channels() == 3) cvtColor(src, g, COLOR_BGR2GRAY);
    else g = src.clone();
    return g;
}

// Grayscale-based foreground mask using CLAHE + Otsu + border polarity check
static Mat simpleGrayForeground(const Mat& bgr) {
    CV_Assert(bgr.channels() == 3);

    // 1) BGR -> Gray
    Mat gray = toGray(bgr);

    // 2) CLAHE on gray to normalize lighting
    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    Mat gray_eq;
    clahe->apply(gray, gray_eq);

    // 3) Otsu threshold in both polarities
    Mat bw1, bw2;
    threshold(gray_eq, bw1, 0, 255, THRESH_BINARY | THRESH_OTSU);
    threshold(gray_eq, bw2, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // 4) Build border mask (assumed background)
    Mat border = Mat::zeros(gray_eq.size(), CV_8U);
    int m = std::max(5, std::min(gray_eq.rows, gray_eq.cols) / 40);
    rectangle(border, Rect(0, 0, gray_eq.cols, m),                255, FILLED);
    rectangle(border, Rect(0, gray_eq.rows - m, gray_eq.cols, m), 255, FILLED);
    rectangle(border, Rect(0, 0, m, gray_eq.rows),                255, FILLED);
    rectangle(border, Rect(gray_eq.cols - m, 0, m, gray_eq.rows), 255, FILLED);

    // 5) Choose orientation where less of the border is foreground
    Mat tmp;
    bitwise_and(bw1, border, tmp);
    int fg1 = countNonZero(tmp);

    bitwise_and(bw2, border, tmp);
    int fg2 = countNonZero(tmp);

    Mat bw = (fg1 <= fg2) ? bw1 : bw2;

    // 6) Basic cleanup
    morphologyEx(bw, bw, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    morphologyEx(bw, bw, MORPH_CLOSE,
                 getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

    return bw; // 255 = foreground (pills), 0 = background
}

// Fill holes inside foreground blobs using flood fill from outside
static Mat fillHoles(const Mat& bw) {
    CV_Assert(bw.type() == CV_8U);
    // Pad image to avoid edge ambiguity
    Mat pad(bw.rows + 2, bw.cols + 2, CV_8U, Scalar(0));
    bw.copyTo(pad(Rect(1, 1, bw.cols, bw.rows)));

    // Flood fill from outside (true background)
    floodFill(pad, Point(0, 0), 255);

    // Crop back to original size
    Mat bg = pad(Rect(1, 1, bw.cols, bw.rows));

    // Invert: now "bg" has 255 where holes were; 0 elsewhere
    Mat holes;
    bitwise_not(bg, holes);

    // Combine original FG with holes to get solid blobs
    Mat filled;
    bitwise_or(bw, holes, filled);
    return filled;
}

// Watershed-based splitting with distance-transform seeds
static int splitByWatershed(const Mat& bgr,
                            const Mat& filledMask,
                            Mat& markersOut,
                            Mat& sureFGOut)
{
    CV_Assert(bgr.channels() == 3);
    CV_Assert(filledMask.type() == CV_8U);

    // 1) Clean mask and remove tiny blobs
    Mat cleaned;
    morphologyEx(filledMask, cleaned, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

    Mat lbl, stats, centroids;
    int ncc = connectedComponentsWithStats(cleaned, lbl, stats, centroids, 8, CV_32S);
    Mat big = Mat::zeros(cleaned.size(), CV_8U);
    const int MIN_AREA = 80; // tune if needed

    for (int i = 1; i < ncc; ++i) {
        if (stats.at<int>(i, CC_STAT_AREA) >= MIN_AREA)
            big.setTo(255, (lbl == i));
    }

    if (countNonZero(big) == 0) {
        markersOut = Mat::zeros(cleaned.size(), CV_32S);
        sureFGOut  = Mat::zeros(cleaned.size(), CV_8U);
        return 0;
    }

    // 2) Distance transform on cleaned mask
    Mat dist;
    distanceTransform(big, dist, DIST_L2, 5);
    GaussianBlur(dist, dist, Size(5, 5), 1.0);

    // 3) Seeds from high DT values
    double maxv = 0.0;
    minMaxLoc(dist, nullptr, &maxv);
    double alpha = 0.40; // fraction of max DT, deterministic

    Mat sureFG;
    threshold(dist, sureFG, alpha * maxv, 255, THRESH_BINARY);
    sureFG.convertTo(sureFG, CV_8U);
    morphologyEx(sureFG, sureFG, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

    // 4) Markers: label seeds, compute background, unknown
    Mat markers;
    int nSeeds = connectedComponents(sureFG, markers, 8, CV_32S);
    (void)nSeeds; // suppress unused warning
    markers += 1; // background will be 1

    Mat sureBG;
    dilate(big, sureBG,
           getStructuringElement(MORPH_ELLIPSE, Size(5, 5)),
           Point(-1, -1), 2);

    Mat unknown;
    subtract(sureBG, sureFG, unknown);
    markers.setTo(0, unknown > 0); // unknown region = 0

    // 5) Watershed
    Mat bgrCopy = bgr.clone();
    watershed(bgrCopy, markers);

    // 6) Cleanup markers:
    //   - remove background
    //   - remove watershed lines (-1)
    markers.setTo(0, big == 0);
    markers.setTo(0, markers == -1);

    // 7) Remove tiny fragments after watershed via area gate
    Mat pos = (markers > 1);
    Mat lbl2, stats2, cent2;
    int n2 = connectedComponentsWithStats(pos, lbl2, stats2, cent2, 8, CV_32S);

    vector<int> areas;
    areas.reserve(std::max(0, n2 - 1));
    for (int i = 1; i < n2; ++i)
        areas.push_back(stats2.at<int>(i, CC_STAT_AREA));

    int Amin = 20;
    if (!areas.empty()) {
        nth_element(areas.begin(), areas.begin() + areas.size() / 2, areas.end());
        int Amed = std::max(areas[areas.size() / 2], 1);
        Amin = std::max(Amin, (int)(0.30 * Amed));
    }

    for (int i = 1; i < n2; ++i) {
        if (stats2.at<int>(i, CC_STAT_AREA) < Amin)
            markers.setTo(0, lbl2 == i);
    }

    // 8) Count valid labels
    double mx = 0.0;
    minMaxLoc(markers, nullptr, &mx);
    int num = 0;
    for (int lblv = 2; lblv <= (int)mx; ++lblv) {
        if (countNonZero(markers == lblv) > 0) ++num;
    }

    markersOut = markers.clone();
    sureFGOut  = sureFG.clone();
    return num;
}

// Draw marker centroids and labels on image
static void drawMarkersOn(Mat& img, const Mat& markers) {
    double minv, maxv;
    minMaxLoc(markers, &minv, &maxv);
    for (int lbl = 2; lbl <= (int)maxv; ++lbl) {
        Mat mask = (markers == lbl);
        Moments m = moments(mask, true);
        if (m.m00 < 5.0) continue;
        Point2d c(m.m10 / m.m00, m.m01 / m.m00);
        circle(img, c, 6, Scalar(0, 0, 255), 2);
        putText(img, std::to_string(lbl - 1), c + Point2d(8, -8),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
    }
}

// Full pipeline for a single image: mask + fill + watershed + markers
static int processAndOverlayMarkers(const Mat& img,
                                    Mat& visOut,
                                    Mat& maskOut)
{
    // 1) Build foreground mask in gray space
    Mat bw = simpleGrayForeground(img);

    // 2) Fill internal holes so pills are solid blobs
    Mat filled = fillHoles(bw);

    // 3) Split touching pills via watershed
    Mat markers, sureFG;
    int numPills = splitByWatershed(img, filled, markers, sureFG);

    // 4) Visualization image
    visOut = img.clone();
    drawMarkersOn(visOut, markers);

    maskOut = filled.clone();

    return numPills;
}

// -----------------------------------------------------------------------------
// MAIN: iterate over all images, show windows, print per-image and average accuracy
int main() {
    fs::path imgDir = "images";
    if (!fs::exists(imgDir) || !fs::is_directory(imgDir)) {
        cerr << "'images' directory not found!\n";
        return -1;
    }

    double sum_acc = 0.0; // sum of per-image accuracies (%)
    int cnt_acc    = 0;   // number of images with valid GT

    for (const auto& entry : fs::directory_iterator(imgDir)) {
        if (!entry.is_regular_file()) continue;
        string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" &&
            ext != ".bmp" && ext != ".tif" && ext != ".tiff") continue;

        string path = entry.path().string();
        Mat img = imread(path);
        if (img.empty()) {
            cerr << "Failed to read image: " << path << endl;
            continue;
        }

        // Run pipeline
        Mat vis, mask;
        int numPills = processAndOverlayMarkers(img, vis, mask);

        // Extract filename parts
        fs::path inputPath(path);
        string baseName = inputPath.stem().string(); // e.g., "p19_44"

        // Parse actual pill count from filename
        int actualCount = 0;
        size_t pos = baseName.find('_');
        if (pos != string::npos && pos + 1 < baseName.size()) {
            try {
                actualCount = stoi(baseName.substr(pos + 1));
            } catch (...) {
                cerr << "Warning: Could not parse actual pill count from filename: "
                     << inputPath.filename().string() << "\n";
            }
        }

        // Print per-image stats
        cout << "File: " << inputPath.filename().string() << "\n";
        if (actualCount > 0) {
            double accuracy = (static_cast<double>(numPills) / actualCount) * 100.0;
            cout << "  Actual pills:   " << actualCount << "\n";
            cout << "  Detected pills: " << numPills    << "\n";
            if (numPills > actualCount)
                cout << "  Note: Detected more pills than actual (possible false positives).\n";
            cout << "  Accuracy:       " << fixed << setprecision(2) << accuracy << "%\n";

            sum_acc += accuracy;
            cnt_acc += 1;
        } else {
            cout << "  Actual pills:   N/A\n";
            cout << "  Detected pills: " << numPills << "\n";
            cout << "  Accuracy:       N/A (filename missing count)\n";
        }

        // Show requested windows for this image
        imshow("Original Image", img);
        imshow("Foreground Mask", mask);
        imshow("Markers on Original", vis);

        // Controls: ESC/Q to abort all, any other key → next image
        int key = waitKey(0);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
        destroyAllWindows();
    }

    // Overall average accuracy
    if (cnt_acc > 0) {
        double avg_acc = sum_acc / cnt_acc;
        cout << "\nOverall average accuracy across " << cnt_acc
             << " labeled image(s): " << fixed << setprecision(2)
             << avg_acc << "%\n";
    } else {
        cout << "\nOverall average accuracy: N/A (no labeled images found)\n";
    }

    return 0;
}