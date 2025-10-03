// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <set>

// using namespace cv;
// using std::cout;
// using std::endl;

// // Find regional maxima of a float distance map using dilation with a (2*minDist+1) kernel.
// // Returns an 8U mask with 255 at local maxima positions (restricted to 'mask' if provided).
// static cv::Mat peakLocalMaxLike(const cv::Mat& dist32f, int minDist, const cv::Mat& restrictMask /*8U or empty*/) {
//     CV_Assert(dist32f.type() == CV_32F);
//     CV_Assert(restrictMask.empty() || (restrictMask.type() == CV_8U && restrictMask.size() == dist32f.size()));

//     // Dilate with square kernel to enforce min peak spacing
//     int k = std::max(1, 2 * minDist + 1);
//     cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
//     cv::Mat dilated;
//     cv::dilate(dist32f, dilated, kernel);

//     // local maxima where dist == dilated (float compare with epsilon)
//     cv::Mat diff, eqMask;
//     cv::absdiff(dist32f, dilated, diff);
//     eqMask = diff <= 1e-6;  // CV_8U result due to implicit conversion

//     // Cast to 8U binary
//     cv::Mat peaks8u;
//     eqMask.convertTo(peaks8u, CV_8U, 255);

//     // Zero out peaks where distance is zero (outside foreground)
//     cv::Mat nonZeroDT = dist32f > 0.0f;
//     nonZeroDT.convertTo(nonZeroDT, CV_8U, 255);
//     cv::bitwise_and(peaks8u, nonZeroDT, peaks8u);

//     // Restrict to provided mask if any (e.g., foreground/thresh)
//     if (!restrictMask.empty())
//         cv::bitwise_and(peaks8u, restrictMask, peaks8u);

//     // Optional small opening to clean tiny clusters (keep it gentle)
//     cv::morphologyEx(peaks8u, peaks8u, cv::MORPH_OPEN,
//                      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));

//     return peaks8u; // 0/255
// }

// int main(int argc, char** argv) {
//     std::string path = (argc > 1) ? argv[1] : "images/p11_76.jpg";
//     int minDistance = 20;  // like peak_local_max(min_distance=20)

//     // 1) Load & mean-shift filter (same parameters as Python: 21, 51)
//     cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
//     if (image.empty()) { std::cerr << "Failed to read: " << path << "\n"; return 1; }

//     cv::Mat shifted;
//     cv::pyrMeanShiftFiltering(image, shifted, /*sp*/21, /*sr*/51);

//     cv::imshow("Input", image);

//     // 2) Grayscale + Otsu (same as Python)
//     cv::Mat gray;
//     cv::cvtColor(shifted, gray, cv::COLOR_BGR2GRAY);

//     cv::Mat thresh;
//     double otsuT = cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
//     (void)otsuT;
//     cv::imshow("Thresh", thresh);

//     // Ensure foreground is white (255) like your Python.
//     // If your objects are darker than background, use THRESH_BINARY_INV instead.

//     // 3) Distance transform (Euclidean, like scipy edt)
//     cv::Mat dist32f;
//     cv::distanceTransform(thresh, dist32f, cv::DIST_L2, 5); // CV_32F
//     // (Optional) visualize:
//     // cv::Mat distShow; cv::normalize(dist32f, distShow, 0, 1, cv::NORM_MINMAX); cv::imshow("DT", distShow);

//     // 4) Peak local maxima (min_distance ~ 20) within the foreground
//     cv::Mat peaks = peakLocalMaxLike(dist32f, minDistance, thresh); // 0/255

//     // 5) Connected components on local peaks -> markers
//     cv::Mat markers; // CV_32S
//     int nLabels = cv::connectedComponents(peaks, markers, 8, CV_32S); // background=0, peaks=1..n-1

//     // 6) Watershed: background must be 0; keep markers as-is; mask by thresh via the image content
//     // OpenCV watershed uses -1 for boundaries.
//     cv::Mat wshedInput = image.clone();
//     cv::watershed(wshedInput, markers);

//     // 7) Count unique segments excluding background (0) and boundaries (-1)
//     std::set<int> uniqueLabels;
//     uniqueLabels.clear();
//     for (int r = 0; r < markers.rows; ++r) {
//         const int* p = markers.ptr<int>(r);
//         for (int c = 0; c < markers.cols; ++c) {
//             int v = p[c];
//             if (v > 0) uniqueLabels.insert(v);
//         }
//     }
//     cout << "[INFO] " << uniqueLabels.size() << " unique segments found" << endl;

//     // 8) Draw circles & labels per segment (like Python)
//     cv::Mat output = image.clone();
//     int idx = 0; // to print #1..#N in display order (note: labels set is not sorted spatially)
//     for (int lbl : uniqueLabels) {
//         // Build a mask for this label
//         cv::Mat mask = cv::Mat::zeros(markers.size(), CV_8U);
//         mask.setTo(255, markers == lbl);

//         // Find contours (external)
//         std::vector<std::vector<cv::Point>> contours;
//         std::vector<cv::Vec4i> hierarchy;
//         cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//         if (contours.empty()) continue;

//         // Take largest contour
//         size_t best = 0;
//         double bestArea = 0.0;
//         for (size_t i = 0; i < contours.size(); ++i) {
//             double a = cv::contourArea(contours[i]);
//             if (a > bestArea) { bestArea = a; best = i; }
//         }

//         // Draw enclosing circle & label
//         cv::Point2f center; float radius = 0.f;
//         cv::minEnclosingCircle(contours[best], center, radius);
//         cv::circle(output, center, (int)radius, cv::Scalar(0,255,0), 2);

//         ++idx;
//         cv::putText(output, "#" + std::to_string(idx),
//                     cv::Point((int)center.x - 10, (int)center.y),
//                     cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
//     }

//     cv::imshow("Output", output);
//     cv::waitKey(0);
//     cv::destroyAllWindows();
//     return 0;
// }

// watershed_count.cpp
// g++ -std=c++17 watershed_count.cpp `pkg-config --cflags --libs opencv4` -O2 -o watershed_count

#include <opencv2/opencv.hpp>
#include <iostream>
#include <set>

using namespace cv;
using std::cout;
using std::endl;

// --- peak_local_max equivalent ----------------------------------------------
// Finds regional maxima of a CV_32F distance map using dilation with a
// (2*minDist+1) kernel; restricts to 'restrictMask' if provided.
static Mat peakLocalMaxLike(const Mat& dist32f, int minDist, const Mat& restrictMask /*8U or empty*/) {
    CV_Assert(dist32f.type() == CV_32F);
    CV_Assert(restrictMask.empty() || (restrictMask.type() == CV_8U && restrictMask.size() == dist32f.size()));

    int k = std::max(1, 2 * minDist + 1);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(k, k));
    Mat dilated;
    dilate(dist32f, dilated, kernel);

    // Regional maxima: dist == dilated (with epsilon tolerance)
    Mat diff, eqMask;
    absdiff(dist32f, dilated, diff);
    eqMask = diff <= 1e-6;            // CV_8U mask after implicit conversion

    Mat peaks8u;
    eqMask.convertTo(peaks8u, CV_8U, 255);

    // Only where distance > 0 (inside foreground)
    Mat fg;
    fg = dist32f > 0.0f;
    fg.convertTo(fg, CV_8U, 255);
    bitwise_and(peaks8u, fg, peaks8u);

    // Restrict to given mask (e.g., thresholded foreground)
    if (!restrictMask.empty())
        bitwise_and(peaks8u, restrictMask, peaks8u);

    // Gentle clean-up
    morphologyEx(peaks8u, peaks8u, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(3,3)));

    return peaks8u; // 0/255
}

// Quick visualization helper (optional)
static void showFloat(const std::string& win, const Mat& f32) {
    Mat s; normalize(f32, s, 0, 255, NORM_MINMAX);
    s.convertTo(s, CV_8U);
    imshow(win, s);
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "images/p11_76.jpg";
    int minDistance = (argc > 2) ? std::max(3, atoi(argv[2])) : 12; // like peak_local_max(min_distance)

    // 1) Load & mean-shift (like Python: 21, 51)
    Mat image = imread(path, IMREAD_COLOR);
    if (image.empty()) { std::cerr << "Failed to read: " << path << "\n"; return 1; }

    Mat shifted;
    pyrMeanShiftFiltering(image, shifted, /*sp*/21, /*sr*/51);
    imshow("Input", image);

    // 2) Grayscale
    Mat gray;
    cvtColor(shifted, gray, COLOR_BGR2GRAY);

    // 2a) Otsu both polarities; pick the one with larger DT max (automatic)
    Mat thrA, thrB;
    threshold(gray, thrA, 0, 255, THRESH_BINARY | THRESH_OTSU);
    threshold(gray, thrB, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    Mat dtA, dtB;
    distanceTransform(thrA, dtA, DIST_L2, 5);
    distanceTransform(thrB, dtB, DIST_L2, 5);

    double minA, maxA, minB, maxB;
    minMaxLoc(dtA, &minA, &maxA);
    minMaxLoc(dtB, &minB, &maxB);

    Mat thresh = (maxB > maxA) ? thrB : thrA; // choose polarity giving stronger interior distances
    bool usedInv = (maxB > maxA);

    // (Optional) small blur to stabilize DT ridges
    Mat dist32f;
    distanceTransform(thresh, dist32f, DIST_L2, 5);
    GaussianBlur(dist32f, dist32f, Size(3,3), 0);

    cout << "Foreground pixels = " << countNonZero(thresh)
         << " | DT max = " << (maxB > maxA ? maxB : maxA)
         << " | polarity = " << (usedInv ? "INV" : "NORMAL") << endl;

    imshow("Thresh", thresh);
    // showFloat("DT", dist32f); // uncomment if you want to inspect

    // 3) Peak local maxima (seed discovery)
    Mat peaks = peakLocalMaxLike(dist32f, minDistance, thresh);
    int peakCount = countNonZero(peaks);
    cout << "Peaks found = " << peakCount << endl;
    imshow("Peaks", peaks);

    // 3a) Fallback seeds if none (erode-based cores)
    Mat markers;
    if (peakCount == 0) {
        std::cerr << "[WARN] No peaks found; using eroded cores as seeds.\n";
        Mat seeds;
        erode(thresh, seeds, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
        int n = connectedComponents(seeds, markers, 8, CV_32S);
        (void)n;
    } else {
        int n = connectedComponents(peaks, markers, 8, CV_32S);
        (void)n;
    }

    // Restrict markers to the foreground (safety)
    markers.setTo(0, thresh == 0);

    // 4) Watershed
    Mat wshedInput = image.clone();
    watershed(wshedInput, markers);   // labels: -1 boundary, >=1 regions (because we ensured background=0)

    // 5) Build clean mask and count unique labels (>0, ignore -1)
    std::set<int> uniqueLabels;
    for (int r = 0; r < markers.rows; ++r) {
        const int* p = markers.ptr<int>(r);
        for (int c = 0; c < markers.cols; ++c) {
            int v = p[c];
            if (v > 0) uniqueLabels.insert(v);
        }
    }
    cout << "[INFO] " << uniqueLabels.size() << " unique segments found" << endl;

    // 6) Draw circles + labels (like Python)
    Mat output = image.clone();
    int idx = 0;
    for (int lbl : uniqueLabels) {
        Mat mask = Mat::zeros(markers.size(), CV_8U);
        mask.setTo(255, markers == lbl);

        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hier;
        findContours(mask, contours, hier, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (contours.empty()) continue;

        // largest contour
        size_t best = 0;
        double bestArea = 0.0;
        for (size_t i = 0; i < contours.size(); ++i) {
            double a = contourArea(contours[i]);
            if (a > bestArea) { bestArea = a; best = i; }
        }

        Point2f center; float radius = 0.f;
        minEnclosingCircle(contours[best], center, radius);
        circle(output, center, (int)radius, Scalar(0,255,0), 2);

        ++idx;
        putText(output, "#" + std::to_string(idx),
                Point((int)center.x - 10, (int)center.y),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,255), 2);
    }

    imshow("Output", output);
    waitKey(0);
    destroyAllWindows();
    return 0;
}