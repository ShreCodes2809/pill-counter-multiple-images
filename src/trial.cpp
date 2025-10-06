#include <opencv2/opencv.hpp>
#include <opencv2/xphoto/white_balance.hpp>
#include <iostream>
#include <set>
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

// I implemented the basic watershed algorithm on one of the images where both the pills and background
// have highly saturated colors even though they are clearly contrasting colors. The flow of the algorithm
// included the following steps: threshold → open → dilate → distance transform → foreground/background/unknown
// → connected components → watershed → draw boundaries

// int main() {
//     // Read image
//     // string path = randomImagePath();
//     Mat img = cv::imread("images/p11_76.jpg");

//     // Convert to grayscale
//     Mat gray;
//     cvtColor(img, gray, COLOR_BGR2GRAY);

//     // Otsu threshold (inverse binary)
//     Mat thresh;
//     threshold(gray, thresh, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

//     // Noise removal: morphological opening (3x3 kernel, 2 iterations)
//     Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
//     Mat opening;
//     morphologyEx(thresh, opening, MORPH_OPEN, kernel, Point(-1, -1), 2);

//     // Sure background area: dilate (3 iterations)
//     Mat sure_bg;
//     dilate(opening, sure_bg, kernel, Point(-1, -1), 3);

//     // Distance transform on opening
//     Mat dist_transform;
//     distanceTransform(opening, dist_transform, DIST_L2, 5); // CV_32F output

//     // Threshold distance map to get sure foreground (0.7 * max)
//     double maxVal;
//     minMaxLoc(dist_transform, nullptr, &maxVal);
//     Mat sure_fg;
//     threshold(dist_transform, sure_fg, 0.7 * maxVal, 255, THRESH_BINARY);

//     // sure_fg must be 8U for later steps
//     sure_fg.convertTo(sure_fg, CV_8U);

//     // Unknown region = sure_bg - sure_fg
//     Mat unknown;
//     subtract(sure_bg, sure_fg, unknown);

//     // Marker labelling on sure foreground
//     Mat markers;
//     int nLabels = connectedComponents(sure_fg, markers); // markers: CV_32S

//     // Add one so background is 1 instead of 0
//     markers = markers + 1;

//     // Mark unknown region with 0
//     markers.setTo(0, unknown == 255);

//     // Watershed (modifies markers in place)
//     Mat img_for_ws = img.clone();
//     watershed(img_for_ws, markers);

//     // Draw boundaries (pixels with label -1). Keep same BGR as your Python: [255, 0, 0]
//     Mat boundaryMask = (markers == -1);
//     img.setTo(Scalar(0, 255, 255), boundaryMask);

//     // Show result
//     imshow("Original Image with Watershed Boundaries", img);
//     waitKey(0);

//     return 0;
// }

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

int main() {
    string path = randomImagePath();
    Mat img = cv::imread(path);
    Mat blur; GaussianBlur(img, blur, Size(5, 5), 0, 0);
    Mat lab; cvtColor(blur, lab, COLOR_BGR2Lab);
    vector<Mat> ch; split(lab, ch);
    Mat L = ch[0];
    imshow("Luminance", L);

    Mat thresh; adaptiveThreshold(L, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 15, 5);
    imshow("Thresholded Image", thresh);

    //total number of clusters in which the input will be segmented:
    int seg_clusters = 3; 
    
    Mat clustered_image = k_means(img, seg_clusters);

    imshow("Clustered Image", clustered_image);
    waitKey(0);

    return 0;
}