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

std::string randomImagePath(const fs::path& root = "images") 
/**
 * @brief Selects a random image file path from a given directory.
 *
 * This function scans the specified directory for valid image files 
 * (extensions: .jpg, .jpeg, .png, .bmp, .tif, .tiff) and randomly 
 * selects one file path to return.
 *
 * @param root The root directory to search for image files. 
 *             Defaults to "images".
 * @return std::string The full path of a randomly selected image file.
 *
 * @throws std::runtime_error If no image files are found in the directory.
 */
{
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

static cv::Mat claheLabL(const cv::Mat& bgr, double clip=3.0, cv::Size tiles=cv::Size(8,8))
/**
 * @brief Enhances image contrast by applying CLAHE to the L-channel in Lab color space.
 *
 * This function converts a BGR image to Lab color space, applies 
 * Contrast Limited Adaptive Histogram Equalization (CLAHE) on the 
 * luminance (L) channel to improve contrast, and then converts the 
 * image back to BGR format. The a and b channels (color information) 
 * remain unchanged.
 *
 * @param bgr Input 3-channel BGR image.
 * @param clip Clip limit for CLAHE contrast enhancement. Default is 3.0.
 * @param tiles Size of the grid for histogram equalization. Default is 8x8.
 *
 * @return cv::Mat Contrast-enhanced BGR image.
 *
 * @note This operation helps normalize lighting and improve visual clarity
 *       without altering the color balance.
 */
{
    CV_Assert(bgr.channels()==3);
    cv::Mat lab; cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch; cv::split(lab, ch);           // L,a,b
    auto clahe = cv::createCLAHE(clip, tiles);
    clahe->apply(ch[0], ch[0]);                            // enhance L only
    cv::merge(ch, lab);
    cv::Mat out; cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
    return out;
}

static double maskedMedian(const cv::Mat& channel, const cv::Mat& mask)
{
    CV_Assert(channel.type() == CV_8U || channel.type() == CV_16U || channel.type() == CV_32F);

    std::vector<float> vals;
    vals.reserve(channel.rows * channel.cols / 10); // small guess
    for (int y = 0; y < channel.rows; ++y) {
        const uchar* m = mask.ptr<uchar>(y);
        const float*  c = channel.ptr<float>(y);
        for (int x = 0; x < channel.cols; ++x)
            if (m[x]) vals.push_back(c[x]);
    }
    if (vals.empty()) return 0.0;
    std::nth_element(vals.begin(), vals.begin() + vals.size()/2, vals.end());
    return vals[vals.size()/2];
}

static Mat chromaBinForeground(const Mat& bgr)
/**
 * @brief Generates a binary foreground mask based on chromatic contrast from the background.
 *
 * This function identifies foreground regions (e.g., pills) by analyzing color
 * differences in the Lab color space. It models the background chroma using the 
 * image borders and computes the Euclidean distance of each pixel’s (a,b) chroma 
 * values from the background mean. The distance map is then globally thresholded 
 * using Otsu’s method to produce a binary mask.
 *
 * Steps:
 *  1. Convert input BGR image to Lab color space.
 *  2. Estimate background chroma (mean a,b) from the image borders.
 *  3. Compute per-pixel chroma distance from background.
 *  4. Normalize and apply Otsu thresholding to extract the foreground.
 *  5. Apply morphological closing to fill small gaps and smooth edges.
 *
 * @param bgr Input 3-channel BGR image.
 * @return cv::Mat Binary mask (255 = foreground, 0 = background).
 *
 * @note This method works well when the foreground objects have distinct chromatic
 *       properties compared to the background, even under variable lighting.
 */
{
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
    // Scalar meanA = mean(ch[1], border);
    // Scalar meanB = mean(ch[2], border);

    Mat L32, a32, b32;
    ch[0].convertTo(L32, CV_32F);
    ch[1].convertTo(a32, CV_32F);
    ch[2].convertTo(b32, CV_32F);

    // robust background center using medians
    double medL = maskedMedian(L32, border);
    double medA = maskedMedian(a32, border);
    double medB = maskedMedian(b32, border);

    // optional: median absolute deviation for scale (robust std)
    auto mad = [&](const Mat& ch, double med) -> double {
        std::vector<float> vals;
        for (int y = 0; y < ch.rows; ++y) {
            const uchar* m = border.ptr<uchar>(y);
            const float* c = ch.ptr<float>(y);
            for (int x = 0; x < ch.cols; ++x)
                if (m[x]) vals.push_back(std::abs(c[x] - med));
        }
        if (vals.empty()) return 1.0;
        std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
        return std::max(static_cast<double>(vals[vals.size() / 2]), 1e-3);
    };


    double sL = mad(L32, medL);
    double sA = mad(a32, medA);
    double sB = mad(b32, medB);

    // compute weighted Lab distance
    Mat dL = (L32 - medL)/sL;
    Mat dA = (a32 - medA)/sA;
    Mat dB = (b32 - medB)/sB;

    Mat dist;
    sqrt(0.0*dL.mul(dL) + dA.mul(dA) + dB.mul(dB), dist);
    normalize(dist, dist, 0, 255, NORM_MINMAX);
    dist.convertTo(dist, CV_8U);
    Mat bw;
    threshold(dist, bw, 0, 255, THRESH_BINARY | THRESH_OTSU);
    morphologyEx(bw, bw, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
    return bw;
}

static int distanceAndComponents(const Mat& filled, Mat& dist32f, Mat& sureFG, Mat& markers)
/**
 * @brief Generates marker regions and seed points for watershed segmentation.
 *
 * This function refines a binary mask (`filled`) and computes both the 
 * distance transform and connected components to create markers for 
 * subsequent watershed-based separation of touching objects.
 *
 * Steps:
 *  1. Apply morphological opening to remove small noise.
 *  2. Identify connected components and filter out tiny blobs 
 *     (area < 80 pixels).
 *  3. For each valid component:
 *      - Compute its distance transform.
 *      - Normalize and smooth the distance map to avoid fragmented peaks.
 *      - Derive a per-component threshold based on its area and mean 
 *        distance-to-max ratio.
 *      - Threshold to produce localized seed regions for each blob.
 *  4. Combine all seed masks to form the final `sureFG` (sure foreground).
 *  5. Create watershed markers:
 *      - `connectedComponents()` assigns unique labels to each seed.
 *      - Dilated `cleaned` mask defines the sure background.
 *      - Unknown areas (BG − FG) are set to 0 in the markers.
 *
 * @param filled Input binary mask (typically from chroma-based segmentation).
 * @param dist32f Output 32-bit float distance transform for visualization.
 * @param sureFG Output binary mask representing the sure foreground regions.
 * @param markers Output integer label image used as markers for watershed.
 * @return int Number of detected foreground components (objects/pills).
 *
 * @note This method provides a deterministic and adaptive approach to 
 *       seed generation, balancing local distance peaks and object size 
 *       for robust separation in later watershed segmentation.
 */
{
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

static void drawMarkersOn(Mat& img, const Mat& markers)
/**
 * @brief Overlays numerical markers and centroids on an image.
 *
 * This function visualizes watershed or connected-component markers by 
 * computing each labeled region’s centroid and drawing both a red circle 
 * and an identifying label number at that location on the input image.
 *
 * Steps:
 *  1. Iterate over all labels in `markers` (excluding background labels 0 and 1).
 *  2. For each label:
 *      - Extract its binary mask.
 *      - Compute spatial moments to find its centroid.
 *      - Skip regions with negligible area (m00 < 5).
 *      - Draw a red circle at the centroid and place the label number next to it.
 *
 * @param img Input/output BGR image on which markers and labels are drawn.
 * @param markers Integer label image where each connected region has a unique ID.
 *
 * @note Background (label 0/1) is ignored. This visualization is used for 
 *       verifying segmentation accuracy and ensuring correct object counting.
 */
{
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
static Mat fillHoles(const Mat& bw)
/**
 * @brief Fills internal holes in a binary mask without using contours.
 *
 * This function performs a flood fill from the outer border of the image
 * to identify the background, then inverts it to locate internal holes
 * within the foreground. The resulting holes are merged back with the 
 * original mask to produce a solid filled version.
 *
 * Steps:
 *  1. Pad the binary image by 1 pixel on all sides to prevent edge ambiguity.
 *  2. Perform flood fill from the top-left corner (assumed background).
 *  3. Invert the flood-filled result to isolate holes.
 *  4. Merge the holes with the original binary mask using bitwise OR.
 *
 * @param bw Input binary mask (8-bit, 0 = background, 255 = foreground).
 * @return cv::Mat Binary mask with all interior holes filled.
 *
 * @note This method ensures continuous foreground regions and improves
 *       robustness for later watershed or connected-component analysis.
 */
{
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
static int splitByWatershed(const Mat& bgr, const Mat& fgMask, Mat& markersOut, Mat& sureFGOut)
/**
 * @brief Splits touching foreground objects using marker-controlled watershed.
 *
 * Given a binary foreground mask and the original BGR image, this routine:
 *  1) Cleans the mask (small opening) and fills interior holes.
 *  2) Computes the distance transform (DT) on the filled mask and blurs it.
 *  3) Derives seed regions (sure foreground) by thresholding DT at a fixed
 *     fraction of its global maximum (0.35 * max(DT)), then cleans seeds.
 *  4) Builds watershed markers: connectedComponents(seeds) → labels (2..N),
 *     sets background to 1, and marks unknown = (dilated FG − seeds) as 0.
 *  5) Runs OpenCV watershed on the input color image using these markers.
 *  6) Zeros labels outside the original foreground and along watershed ridges.
 *  7) Returns the count of labeled objects (labels ≥ 2).
 *
 * @param bgr         Input 3-channel BGR image (used by cv::watershed).
 * @param fgMask      Binary foreground mask (CV_8U; 255 = FG, 0 = BG).
 * @param markersOut  Output label image (CV_32S): 0=unknown/invalid, 1=BG, 2..K=objects.
 * @param sureFGOut   Output binary mask (CV_8U) of "sure foreground" seed regions.
 * @return int        Number of detected objects (labels ≥ 2).
 *
 * @note The DT seed threshold (0.35 * max(DT)) controls splitting: lower → more seeds/splits,
 *       higher → fewer. Watershed is invoked as watershed(bgr, markers); the locally computed
 *       Sobel magnitude (grad) is not passed to watershed (OpenCV uses the provided image).
 *
 * @warning If fgMask under-segments (misses parts of objects), seeds will be sparse and the
 *          watershed cannot recover missing regions. Ensure fgMask quality before this step.
 */
{
    CV_Assert(fgMask.type()==CV_8U && bgr.channels()==3);

    // --- 1) Clean + fill holes → "chroma-based input filled binary mask"
    Mat cleaned;
    morphologyEx(fgMask, cleaned, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
    Mat filled = fillHoles(cleaned);

    // Display filled chroma-based FG mask
    imshow("Step 1 - Filled FG Mask (Chroma)", filled);

    // --- 2) Distance transform → "distance transform visualization"
    Mat dist;
    distanceTransform(filled, dist, DIST_L2, 5);
    GaussianBlur(dist, dist, Size(3,3), 0.8);

    // Visualize DT as 8-bit image
    Mat dist_vis;
    normalize(dist, dist_vis, 0, 255, NORM_MINMAX);
    dist_vis.convertTo(dist_vis, CV_8U);
    imshow("Step 2 - Distance Transform", dist_vis);

    // --- 3) DT threshold → seeds → "output sure foreground"
    double maxv = 0;
    minMaxLoc(dist, nullptr, &maxv);
    Mat sureFG;
    threshold(dist, sureFG, 0.35 * maxv, 255, THRESH_BINARY); // tune 0.35 if needed
    sureFG.convertTo(sureFG, CV_8U);
    morphologyEx(sureFG, sureFG, MORPH_OPEN,
                 getStructuringElement(MORPH_ELLIPSE, Size(3,3)));

    imshow("Step 3 - Sure Foreground (Seeds)", sureFG);

    // --- 4) Build markers
    Mat markers;
    int ncc = connectedComponents(sureFG, markers, 8, CV_32S);
    (void)ncc; // silence unused warning
    markers += 1; // background becomes 1, objects 2..n

    Mat sureBG;
    dilate(filled, sureBG,
           getStructuringElement(MORPH_ELLIPSE, Size(5,5)),
           Point(-1,-1), 2);

    Mat unknown;
    subtract(sureBG, sureFG, unknown);
    markers.setTo(0, unknown > 0);

    // --- 5) Watershed
    watershed(bgr, markers);

    // Clean labels: zero outside foreground and on boundaries
    markers.setTo(0, filled == 0);   // outside FG
    markers.setTo(0, markers == -1); // boundaries

    // --- 6) Visualize markers as a label image ("markers")
    double minv, mx;
    minMaxLoc(markers, &minv, &mx);

    Mat markersColor(markers.size(), CV_8UC3, Scalar(0,0,0));
    for (int y = 0; y < markers.rows; ++y) {
        const int* mptr = markers.ptr<int>(y);
        Vec3b* cptr = markersColor.ptr<Vec3b>(y);
        for (int x = 0; x < markers.cols; ++x) {
            int lbl = mptr[x];
            if (lbl <= 1) {
                cptr[x] = Vec3b(0,0,0); // background / unknown
            } else {
                // deterministic pseudo-color per label
                uchar r = (lbl * 50) % 256;
                uchar g = (lbl * 80) % 256;
                uchar b = (lbl * 110) % 256;
                cptr[x] = Vec3b(b,g,r);
            }
        }
    }
    imshow("Step 4 - Markers (Label Image)", markersColor);

    // --- 7) Count labels
    int num = 0;
    for (int lbl = 2; lbl <= (int)mx; ++lbl) {
        if (countNonZero(markers == lbl) > 0) ++num;
    }

    markersOut = markers.clone();
    sureFGOut  = sureFG.clone();
    return num;
}

// ---------------------------------------------------------------------
static int processClusteredAndOverlayMarkers(const Mat& img, const Mat& clus_img, Mat& visOut, Mat& sureFGOut)
/**
 * @brief Full per-image pipeline: build FG mask, split with watershed, and overlay markers.
 *
 * Orchestrates the segmentation workflow for one image:
 *  1) Computes a chroma-based foreground mask from the contrast-normalized image (`clus_img`)
 *     via chromaBinForeground().
 *  2) Splits touching objects using splitByWatershed(), producing labeled markers and a
 *     "sure foreground" seed mask.
 *  3) Overlays marker centroids/IDs on a copy of the original image (`img`) for visualization.
 *  4) Displays optional debug windows: "SureFG" and "Markers_on_Original".
 *
 * @param img         Original BGR image (used only for clean visualization output).
 * @param clus_img    Preprocessed BGR image (e.g., CLAHE on L) used to derive the FG mask.
 * @param visOut      Output BGR image with marker circles and label IDs drawn.
 * @param sureFGOut   Output binary mask (CV_8U) of "sure foreground" seed regions.
 * @return int        Estimated object count (number of labels ≥ 2 from watershed).
 *
 * @note The quality of `chromaBinForeground()` strongly influences splitting success.
 *       `visOut` uses the original `img` to avoid visual artifacts from preprocessing.
 */
{
    
    // Step A: chroma-based foreground on preprocessed image
    Mat bw = chromaBinForeground(clus_img);

    // Step B: split touching objects using watershed
    Mat markers, sureFG;
    int numPills = splitByWatershed(clus_img, bw, markers, sureFG);

    // Step C: overlay markers on original image
    visOut = img.clone();
    drawMarkersOn(visOut, markers);

    // Optional: also show the raw bw if you want
    imshow("Step 0 - Raw Chroma Foreground", bw);

    sureFGOut = sureFG;

    return numPills;
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

    double sum_acc = 0.0;   // sum of per-image accuracies (%)
    int cnt_acc = 0;     // number of images with ground-truth

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
            sum_acc += acc;
            cnt_acc += 1;
            oss << " | Actual: " << actual << " | Accuracy: " << std::fixed << std::setprecision(2) << acc << "%";
            if (numPills > actual) oss << " | Note: possible false positives";
        } else {
            oss << " | Actual: N/A";
        }
        logln(oss.str());

        // --- Display final views for this image ---
        imshow("Original Image", img);
        imshow("Markers on Original", vis);

        // Controls: press ESC/Q to quit, any other key to proceed to next image
        int key = cv::waitKey(0);
        if (key == 27 || key == 'q' || key == 'Q') break;
        cv::destroyAllWindows();
    }

    if (cnt_acc > 0) {
        double avg_acc = sum_acc / cnt_acc;
        std::ostringstream overall;
        overall << "Overall average accuracy across " << cnt_acc
                << " labeled image(s): " << std::fixed << std::setprecision(2)
                << avg_acc << "%";
        std::cout << overall.str() << '\n';
        if (log) log << overall.str() << '\n';
    } else {
        std::string msg = "Overall average accuracy: N/A (no labeled images found)";
        std::cout << msg << '\n';
        if (log) log << msg << '\n';
    }


    logln("Finished displaying all images.");
    return 0;
}