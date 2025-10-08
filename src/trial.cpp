// Standup Update: Oct 8 -
// - Before performing K-Means clustering, the original image is converted to HSV format since pills and background differ mainly in hue/saturation (not brightness), HSV gives a stronger margin than grayscale/adaptive threshold on BGR.
// - Changed adaptive threshold to chroma based thresholding. Adaptive thresholding is intensity-based which might return tiny specks that might not be detected for the sure foreground.
// - The thresholding needs to be done on the basis of the chromatic (color) properties, not brightness. Hence, a new chromatic thresholding is introduced.
// - The contours lines are made thicker so that there are no unfilled blobs due to noise present in the image.
// - One of the major changes was changing the global normalization + distance transform (DT) to localized DT + connected component analysis. In this case, each component is thresholded at a fraction of its own peak, and drop tiny specks.
// - Additionally, thresholding is performed by calculating the threshold value dynamically based on the local peaks instead of hard coding the threshold value.
// - Finally, I performed k-means clustering progressively thrice to get an average accuracy of 95%+ on all the images.
// - Performing k-means clustering progressively (7 → 3 → 2 clusters) improved accuracy because each stage refined color separation while reducing noise and illumination effects. The initial higher cluster count captured subtle color and lighting variations, while subsequent stages merged similar regions, producing cleaner pill–background separation. This hierarchical approach prevented shadows and highlights from skewing the final segmentation.
// - The above algorithm went one step further and handled shadows, pills reflecting light as well as multi-colored pills.

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

// static Mat adaptiveBinForeground(const Mat& gray, int block=9, double C=7) {
//     Mat bin, bin_inv;
//     adaptiveThreshold(gray, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, block, C);
//     adaptiveThreshold(gray, bin_inv, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, block, C);
//     return (countNonZero(bin) < countNonZero(bin_inv)) ? bin : bin_inv;
// }

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

static void findAndFillContours(const Mat& bw, Mat& filled, vector<vector<Point>>& contours) {
    // Make edges thick & closed so contours are closed loops
    morphologyEx(bw, bw, MORPH_CLOSE,
                getStructuringElement(MORPH_ELLIPSE, Size(3,3)), Point(-1,-1), 1);
    Mat src = bw.clone();
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    filled = Mat::zeros(bw.size(), CV_8U);
    drawContours(filled, contours, -1, Scalar(255), FILLED);
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

        Mat dist; distanceTransform(compMask, dist, DIST_L2, 3);
        double maxv = 0; minMaxLoc(dist, nullptr, &maxv, nullptr, nullptr, compMask);
        if (maxv < 2.0) continue;                             // ignore tiny blobs
        
        Mat s;
        // threshold(dist, s, 0.45 * maxv, 255, THRESH_BINARY); // local threshold

        double area = stats.at<int>(i, CC_STAT_AREA);
        Scalar meanDist = mean(dist, compMask);
        double ratio = meanDist[0] / maxv;
        double factor = std::clamp(0.45 + 0.3 * ratio + 200.0 / (area + 50.0), 0.27, 0.73);
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

// ---------------------------------------------------------------------
static int processClusteredAndOverlayMarkers(const Mat& img, const Mat& clus_img, Mat& visOut) {
    // Mat gray = toGray(clus_img);
    // Mat blurGray; medianBlur(gray, blurGray, 3);
    // Mat bw = adaptiveBinForeground(blurGray);
    Mat bw = chromaBinForeground(clus_img);

    vector<vector<Point>> contours;
    Mat filled;
    findAndFillContours(bw, filled, contours);

    Mat dist32f, sureFG, markers;
    int numPills = distanceAndComponents(filled, dist32f, sureFG, markers);

    visOut = img.clone();
    drawMarkersOn(visOut, markers);

    // Optional visualization
    // imshow("01_AdaptiveChromaForeground", bw);
    // imshow("02_FilledContours", filled);
    imshow("03_SureFG", sureFG);
    imshow("04_Markers_on_Original", visOut);

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

    Mat hsv_img; cvtColor(img, hsv_img, COLOR_BGR2HSV);

    imshow("Original Image", img);
    imshow("HSV Transformed Image", hsv_img);

    int seg_clusters_1 = 7;
    Mat clus_img_init = k_means(hsv_img, seg_clusters_1);
    Mat clus_img_mid = k_means(clus_img_init, 3);
    Mat clus_img_final = k_means(clus_img_mid, 2);

    imshow("Initial Clustered Image", clus_img_init);
    imshow("Final Clustered Image", clus_img_final);

    Mat vis;
    int numPills = processClusteredAndOverlayMarkers(img, clus_img_final, vis);

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
            cout << "Detected more pills than actual (possible false positives).\n";
        cout << "Accuracy: " << fixed << setprecision(2) << accuracy << "%\n";
    } else {
        cout << "Could not determine actual pill count from filename.\n";
    }

    // 🔹 Save image
    imwrite(outPath.string(), vis);
    cout << "Saved result: " << outPath << endl;

    waitKey(0);
    return 0;
}