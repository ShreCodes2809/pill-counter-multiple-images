// Standup Update: Oct 7 -
// - I reduced the number of clusters from 3 to 2, performed adaptive thresholding using the gaussian method instead of the mean, and reduced the block size to '9'.
// - Additionally, I increased the value of constant 'C' to '7' since it biases towards filling less area in the contours.
// - Added code to calculate the accuracy for the number of pills detected in each image and got 100% accuracy on most of the images
// - Saved the images separately in a 'results' folder with the original image name along with the new number of pills detected

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <random>
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
Mat k_means(Mat input, int K) {
    Mat samples(input.rows * input.cols ,input.channels(), CV_32F);
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            for (int z = 0; z < input.channels(); z++) {
                if(input.channels() == 3)
                    samples.at<float>(y + x*input.rows, z) = input.at<Vec3b>(y, x)[z];
                else
                    samples.at<float>(y + x*input.rows, z) = input.at<uchar>(y, x);
            }
        }
    }

    Mat labels, centers;
    kmeans(samples, K, labels,
           TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10, 1.0),
           5, KMEANS_PP_CENTERS, centers);

    Mat new_img(input.size(), input.type());
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * input.rows, 0);
            if(input.channels() == 3) {
                for (int z = 0; z < input.channels(); z++)
                    new_img.at<Vec3b>(y, x)[z] = centers.at<float>(cluster_idx, z);
            } else {
                new_img.at<uchar>(y, x) = centers.at<float>(cluster_idx, 0);
            }
        }
    }
    return new_img;
}

// ---------------------------------------------------------------------
static Mat toGray(const Mat& src) {
    Mat g;
    if(src.channels()==3) cvtColor(src, g, COLOR_BGR2GRAY);
    else g = src.clone();
    return g;
}

static Mat adaptiveBinForeground(const Mat& gray, int block=9, double C=7) {
    Mat bin, bin_inv;
    adaptiveThreshold(gray, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, block, C);
    adaptiveThreshold(gray, bin_inv, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, block, C);
    return (countNonZero(bin) < countNonZero(bin_inv)) ? bin : bin_inv;
}

static void findAndFillContours(const Mat& bw, Mat& filled, vector<vector<Point>>& contours) {
    Mat src = bw.clone();
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    filled = Mat::zeros(bw.size(), CV_8U);
    drawContours(filled, contours, -1, Scalar(255), FILLED);
}

static int distanceAndComponents(const Mat& filled, Mat& dist32f, Mat& sureFG, Mat& markers) {
    Mat clean;
    morphologyEx(filled, clean, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));

    distanceTransform(clean, dist32f, DIST_L2, 3);
    Mat distNorm; normalize(dist32f, distNorm, 0, 1.0, NORM_MINMAX);
    threshold(distNorm, sureFG, 0.4, 255, THRESH_BINARY);
    sureFG.convertTo(sureFG, CV_8U);

    connectedComponents(sureFG, markers);
    markers += 1;

    Mat sureBG; dilate(clean, sureBG, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
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

// ---------------------------------------------------------------------
static int processClusteredAndOverlayMarkers(const Mat& img, const Mat& clus_img, Mat& visOut) {
    Mat gray = toGray(clus_img);
    Mat blurGray; medianBlur(gray, blurGray, 3);
    Mat bw = adaptiveBinForeground(blurGray);

    vector<vector<Point>> contours;
    Mat filled;
    findAndFillContours(bw, filled, contours);

    Mat dist32f, sureFG, markers;
    int numPills = distanceAndComponents(filled, dist32f, sureFG, markers);

    visOut = img.clone();
    drawMarkersOn(visOut, markers);

    // Optional visualization
    imshow("02_AdaptiveForeground", bw);
    imshow("03_FilledContours", filled);
    imshow("SureFG", sureFG);
    imshow("Markers_on_Original", visOut);

    return numPills; // 🔹 return count
}

// ---------------------------------------------------------------------
int main() {
    string path = randomImagePath();
    Mat img = imread(path);
    if (img.empty()) {
        cerr << "Failed to read image: " << path << endl;
        return -1;
    }

    int seg_clusters = 2;
    Mat clus_img = k_means(img, seg_clusters);

    imshow("Original Image", img);
    imshow("Clustered Image", clus_img);

    Mat vis;
    int numPills = processClusteredAndOverlayMarkers(img, clus_img, vis);

    // 🔹 Print number of pills detected
    cout << "Detected pills: " << numPills << endl;

    // 🔹 Extract filename parts
    fs::path inputPath(path);
    string baseName = inputPath.stem().string(); // e.g., "p19_44"
    string ext = inputPath.extension().string(); // e.g., ".png"

    // 🔹 Construct output filename: "<original_name>_<detected_count>.png"
    size_t pos = baseName.find('_');
    std::string prefix = (pos != std::string::npos) ? baseName.substr(0, pos) : baseName;
    string outName = prefix + "_" + to_string(numPills) + ext;
    fs::path outPath = fs::path("results") / outName;
    
    // 🔹 Extract actual number of pills from filename and compute accuracy
    int actualCount = 0;
    size_t pos_ = baseName.find('_');
    if (pos_ != std::string::npos && pos_ + 1 < baseName.size()) {
        try {
            actualCount = std::stoi(baseName.substr(pos_ + 1));
        } catch (...) {
            cerr << "Warning: Could not parse actual pill count from filename.\n";
        }
    }

    if (actualCount > 0) {
        double accuracy = (static_cast<double>(numPills) / actualCount) * 100.0;
        cout << "Actual pills: " << actualCount << endl;
        cout << "Detected pills: " << numPills << endl;
        if (numPills > actualCount)
            cout << "⚠️ Detected more pills than actual (possible false positives).\n";
        cout << "Accuracy: " << fixed << setprecision(2) << accuracy << "%\n";
    } else {
        cout << "⚠️ Could not determine actual pill count from filename.\n";
    }

    // 🔹 Save image
    imwrite(outPath.string(), vis);
    cout << "Saved result: " << outPath << endl;

    waitKey(0);
    return 0;
}