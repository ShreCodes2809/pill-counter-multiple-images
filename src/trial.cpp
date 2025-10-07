#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <random>
#include <vector>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

namespace fs = std::filesystem;

std::string randomImagePath(const fs::path& root = "images") {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        // normalize extension to lowercase for the comparison
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png"
            || ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
            files.push_back(entry.path());
        }
    }
    if (files.empty()) throw std::runtime_error("No images found in " + root.string());

    static std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<std::size_t> dist(0, files.size() - 1);
    return files[dist(rng)].string();
}

Mat k_means(Mat input, int K){
    Mat samples(input.rows * input.cols ,input.channels(), CV_32F);
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            for (int z = 0; z < input.channels(); z++) {
                if(input.channels() == 3) {
                    samples.at<float>(y + x*input.rows, z) = input.at<Vec3b>(y, x)[z];
                }
                else {
                    samples.at<float>(y + x*input.rows, z) = input.at<uchar>(y, x);
                }
            }
        }
    }

    Mat labels;
    int iters = 5;
    Mat centers;
    kmeans(samples, K, labels, TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 10, 1.0), iters, KMEANS_PP_CENTERS, centers);

    Mat new_img(input.size(), input.type());
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * input.rows, 0);
            if(input.channels() == 3) {
                for (int z = 0; z < input.channels(); z++){
                    new_img.at<Vec3b>(y, x)[z] = centers.at<float>(cluster_idx, z);
                }
            }
            else {
                new_img.at<uchar>(y, x) = centers.at<float>(cluster_idx, 0);
            }
        }
    }

    return new_img;
}

// --- helpers ---
static Mat toGray(const Mat& src){
    Mat g;
    if(src.channels()==3) cvtColor(src, g, COLOR_BGR2GRAY);
    else g = src.clone();
    return g;
}

static Mat adaptiveBinForeground(const Mat& gray, int block=31, double C=5){
    // Make pills (foreground) white regardless of lighter/darker background
    Mat bin, bin_inv;
    adaptiveThreshold(gray, bin,     255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,      block, C);
    adaptiveThreshold(gray, bin_inv, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV,  block, C);
    // Heuristic: pills occupy less area than background → pick smaller white area
    return (countNonZero(bin) < countNonZero(bin_inv)) ? bin : bin_inv;
}

static void findAndFillContours(const Mat& bw, Mat& filled, vector<vector<Point>>& contours){
    Mat src = bw.clone();
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    filled = Mat::zeros(bw.size(), CV_8U);
    drawContours(filled, contours, -1, Scalar(255), FILLED);
}

static void distanceAndComponents(const Mat& filled, Mat& dist32f, Mat& sureFG, Mat& markers){
    // Clean small noise first
    Mat clean;
    morphologyEx(filled, clean, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3,3)), Point(-1,-1), 1);

    // Distance transform
    distanceTransform(clean, dist32f, DIST_L2, 3);
    Mat distNorm; normalize(dist32f, distNorm, 0, 1.0, NORM_MINMAX);
    threshold(distNorm, sureFG, 0.4, 255, THRESH_BINARY); // keep confident peaks
    sureFG.convertTo(sureFG, CV_8U);

    // Connected components on sure foreground → markers (labels start at 1)
    connectedComponents(sureFG, markers);
    // Make background 0, shift labels by +1 so that 0 can be reserved (if watershed later)
    markers += 1;

    // Optional: mark unknown region (sureBG - sureFG) as 0 (not strictly required here)
    Mat sureBG; dilate(clean, sureBG, getStructuringElement(MORPH_ELLIPSE, Size(5,5)), Point(-1,-1), 2);
    Mat unknown; subtract(sureBG, sureFG, unknown);
    markers.setTo(0, unknown > 0);
}

static void drawMarkersOn(Mat& img, const Mat& markers){
    // Draw a small circle & id at each component centroid (id >= 2 after shift)
    double minv, maxv; minMaxLoc(markers, &minv, &maxv);
    for(int lbl=2; lbl<= (int)maxv; ++lbl){
        Mat mask = (markers == lbl);
        Moments m = moments(mask, true);
        if(m.m00 < 5.0) continue; // skip tiny blobs
        Point2d c(m.m10/m.m00, m.m01/m.m00);
        circle(img, c, 6, Scalar(0,0,255), 2);
        putText(img, std::to_string(lbl-1), c + Point2d(8, -8), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 1);
    }
}

// --- pipeline ---
// Given: Mat img (original BGR), Mat clus_img (your 3-cluster image)
static void processClusteredAndOverlayMarkers(const Mat& img, const Mat& clus_img){
    // 1) threshold
    Mat gray = toGray(clus_img);
    Mat bw   = adaptiveBinForeground(gray);

    // 2) find & fill contours
    vector<vector<Point>> contours;
    Mat filled;
    findAndFillContours(bw, filled, contours);

    // 3) distance transform + connected components → markers
    Mat dist32f, sureFG, markers;
    distanceAndComponents(filled, dist32f, sureFG, markers);

    // 4) overlay markers on original image
    Mat vis = img.clone();
    drawMarkersOn(vis, markers);

    // (Optional) quick visualization windows
    imshow("01_Gray", gray);
    imshow("02_AdaptiveForeground", bw);
    imshow("03_FilledContours", filled);
    Mat distShow; normalize(dist32f, distShow, 0, 255, NORM_MINMAX); distShow.convertTo(distShow, CV_8U);
    imshow("04_Distance", distShow);
    imshow("05_SureFG", sureFG);
    imshow("06_Markers_on_Original", vis);
    waitKey(0);
}

int main() {
    string path = randomImagePath();
    Mat img = cv::imread(path);
    // resize(img, img, Size(400,400), INTER_AREA);

    //total number of clusters in which the input will be segmented:
    int seg_clusters = 3; 
    
    Mat clus_img = k_means(img, seg_clusters);

    imshow("Clustered Image", clus_img);
    processClusteredAndOverlayMarkers(img, clus_img);
    
    waitKey(0);
    return 0;
}