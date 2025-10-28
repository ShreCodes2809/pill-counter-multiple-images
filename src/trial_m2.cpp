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

// Local maxima by dilation equality
static Mat localMaxima(const Mat& src32f, int nms_win){
    Mat dil, eq;
    dilate(src32f, dil, getStructuringElement(MORPH_RECT, Size(nms_win, nms_win)));
    compare(src32f, dil, eq, CMP_EQ);
    Mat nz; threshold(src32f, nz, 1e-6, 255, THRESH_BINARY); nz.convertTo(nz, CV_8U);
    Mat lm; bitwise_and(eq, nz, lm); lm.convertTo(lm, CV_8U, 255);
    return lm;
}

// Median of vector<double>
static double median(std::vector<double> v){
    if(v.empty()) return 0.0;
    size_t n=v.size()/2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    double m=v[n];
    if(v.size()%2==0){
        std::nth_element(v.begin(), v.begin()+n-1, v.end());
        m = 0.5*(m+v[n-1]);
    }
    return m;
}

// Input: BGR image src
// Output: uint8 mask fg where pills are 255
static Mat makeForegroundMask(const Mat& src){
    Mat lab; cvtColor(src, lab, COLOR_BGR2Lab);
    vector<Mat> ch; split(lab, ch);
    // Chromatic distance from background mode
    Mat ab; merge(std::vector<Mat>{ch[1], ch[2]}, ab);
    // K-means to 2 classes in ab
    Mat samples; ab.convertTo(samples, CV_32F); samples = samples.reshape(1, src.rows*src.cols);
    Mat labels, centers;
    kmeans(samples, 2, labels, TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER, 50, 1e-3), 3, KMEANS_PP_CENTERS, centers);
    labels = labels.reshape(1, src.rows);
    // Choose pill class by larger mean L contrast
    Scalar m0 = mean(ch[0], labels==0), m1 = mean(ch[0], labels==1);
    int pillClass = (m0[0] < m1[0]) ? 0 : 1; // pills usually darker in L after shading
    Mat fg = (labels==pillClass);
    fg.convertTo(fg, CV_8U, 255);
    morphologyEx(fg, fg, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
    morphologyEx(fg, fg, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
    // Remove specks
    Mat lbl, stats, cents;
    int n = connectedComponentsWithStats(fg, lbl, stats, cents, 8, CV_32S);
    Mat out = Mat::zeros(fg.size(), CV_8U);
    for(int i=1;i<n;i++){
        int area = stats.at<int>(i, CC_STAT_AREA);
        if(area >= 80) out.setTo(255, lbl==i);
    }
    return out;
}

struct UnitAreaInfo { double unit_area=0; double min_area=0; double max_area_single=0; int n_single=0; };

static UnitAreaInfo estimateUnitArea(const Mat& fg){
    Mat lbl, stats, cents;
    int n = connectedComponentsWithStats(fg, lbl, stats, cents, 8, CV_32S);
    vector<double> singles;
    for(int i=1;i<n;i++){
        Rect r(stats.at<int>(i, CC_STAT_LEFT),
               stats.at<int>(i, CC_STAT_TOP),
               stats.at<int>(i, CC_STAT_WIDTH),
               stats.at<int>(i, CC_STAT_HEIGHT));
        Mat mask = (lbl(r)==i);
        Mat dt; distanceTransform(mask, dt, DIST_L2, 3);
        // NMS window scales with sqrt(area)
        int A = stats.at<int>(i, CC_STAT_AREA);
        int win = std::max(3, int(0.15*std::sqrt((double)A))*2+1);
        Mat peaks = localMaxima(dt, win);
        int k = countNonZero(peaks);
        if(k==1) singles.push_back((double)A);
    }
    UnitAreaInfo ui;
    ui.unit_area = median(singles);
    ui.min_area = 0.25*ui.unit_area;
    ui.max_area_single = 2.0*ui.unit_area;
    ui.n_single = (int)singles.size();
    return ui;
}

static Mat seedsFromDT(const Mat& dt, double unit_area){
    int nms = std::max(3, int(0.15*std::sqrt(unit_area))*2+1);
    Mat peaks = localMaxima(dt, nms);
    // Height threshold
    double h = 0.33*std::sqrt(unit_area);
    Mat high; threshold(dt, high, h, 255, THRESH_BINARY);
    high.convertTo(high, CV_8U);
    Mat seeds; bitwise_and(peaks, high, seeds);
    // Clean tiny seed dust
    morphologyEx(seeds, seeds, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
    return seeds;
}

struct SplitResult { Mat labels; int count=0; };

static SplitResult splitComponentWatershed(const Mat& src, const Mat& fg, const UnitAreaInfo& ui){
    // Precompute gradient for watershed
    Mat gray; cvtColor(src, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5,5), 0);
    Mat gx, gy; Sobel(gray, gx, CV_32F, 1, 0, 3); Sobel(gray, gy, CV_32F, 0, 1, 3);
    Mat grad; magnitude(gx, gy, grad); normalize(grad, grad, 0, 255, NORM_MINMAX); grad.convertTo(grad, CV_8U);
    Mat lbl, stats, cents;
    int n = connectedComponentsWithStats(fg, lbl, stats, cents, 8, CV_32S);

    Mat finalLabels = Mat::zeros(fg.size(), CV_32S);
    int globalId = 1;

    for(int i=1;i<n;i++){
        Rect r(stats.at<int>(i, CC_STAT_LEFT),
               stats.at<int>(i, CC_STAT_TOP),
               stats.at<int>(i, CC_STAT_WIDTH),
               stats.at<int>(i, CC_STAT_HEIGHT));
        Mat compMask = (lbl(r)==i);
        if(countNonZero(compMask)==0) continue;

        // Distance transform in ROI
        Mat dt; distanceTransform(compMask, dt, DIST_L2, 3);

        // Seeds
        Mat seeds = seedsFromDT(dt, ui.unit_area);
        int seedCount = countNonZero(seeds);
        if(seedCount==0 && stats.at<int>(i, CC_STAT_AREA) > 1.5*ui.unit_area){
            // relax h slightly
            Mat peaks = localMaxima(dt, std::max(3, int(0.12*std::sqrt(ui.unit_area))*2+1));
            Mat high; threshold(dt, high, 0.28*std::sqrt(ui.unit_area), 255, THRESH_BINARY);
            high.convertTo(high, CV_8U);
            bitwise_and(peaks, high, seeds);
            morphologyEx(seeds, seeds, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
        }

        Mat markers = Mat::zeros(compMask.size(), CV_32S);
        if(countNonZero(seeds)>0){
            Mat seedLbl; connectedComponents(seeds, seedLbl, 8, CV_32S);
            seedLbl.copyTo(markers, seeds);
        }else{
            // fallback one seed at max DT
            Point mx; minMaxLoc(dt, nullptr, nullptr, nullptr, &mx);
            markers.at<int>(mx) = 1;
        }

        // Watershed input is 3-channel
        Mat gradROI = grad(r);
        Mat colorGrad; cvtColor(gradROI, colorGrad, COLOR_GRAY2BGR);

        // Mask background to zero
        markers.setTo(0, compMask==0);

        watershed(colorGrad, markers); // modifies markers in place

        // markers: -1 border, 0 background, 1..N regions
        // Keep only regions inside compMask
        // Reindex to global ids, drop too-small regions
        int maxId = 0; minMaxLoc(markers, nullptr, nullptr, nullptr, nullptr, compMask);
        maxId = *std::max_element((int*)markers.datastart, (int*)markers.dataend);
        // Build area per id
        std::map<int,int> area;
        for(int y=0;y<markers.rows;y++){
            const int* mp = markers.ptr<int>(y);
            const uchar* cp = compMask.ptr<uchar>(y);
            for(int x=0;x<markers.cols;x++){
                if(cp[x] && mp[x]>0) area[mp[x]]++;
            }
        }
        // Assign
        for(int y=0;y<markers.rows;y++){
            const int* mp = markers.ptr<int>(y);
            const uchar* cp = compMask.ptr<uchar>(y);
            int* fp = finalLabels.ptr<int>(y + r.y);
            for(int x=0;x<markers.cols;x++){
                if(!cp[x]) continue;
                int id = mp[x];
                if(id<=0) continue;
                if(area[id] < ui.min_area) continue; // drop tiny fragments
                // map local id to global
                int gid = globalId + id - 1;
                fp[x + r.x] = gid;
            }
        }
        // Advance global id by kept regions count
        int kept = 0; for(auto& kv: area) if(kv.second >= ui.min_area) kept++;
        globalId += std::max(kept, 1);
    }
    SplitResult sr; sr.labels = finalLabels; sr.count = globalId-1;
    return sr;
}

struct CountResult { int count=0; double unit_area=0; Mat labels; };

static CountResult countPills(const Mat& src){
    Mat fg = makeForegroundMask(src);
    UnitAreaInfo ui = estimateUnitArea(fg);
    // First pass
    SplitResult sr = splitComponentWatershed(src, fg, ui);

    // Sanity: per parent component area conservation
    Mat parentLbl, stats, cents;
    int n = connectedComponentsWithStats(fg, parentLbl, stats, cents, 8, CV_32S);
    bool redo=false;
    for(int i=1;i<n;i++){
        Rect r(stats.at<int>(i, CC_STAT_LEFT),
               stats.at<int>(i, CC_STAT_TOP),
               stats.at<int>(i, CC_STAT_WIDTH),
               stats.at<int>(i, CC_STAT_HEIGHT));
        int A = stats.at<int>(i, CC_STAT_AREA);
        Mat roiLab = sr.labels(r);
        Mat roiPar = (parentLbl(r)==i);
        // Sum child area inside this parent
        int sumChild = 0;
        for(int y=0;y<roiLab.rows;y++){
            const int* L = roiLab.ptr<int>(y);
            const uchar* P = roiPar.ptr<uchar>(y);
            for(int x=0;x<roiLab.cols;x++){
                if(P[x] && L[x]>0) sumChild++;
            }
        }
        double gap = double(std::abs(A - sumChild))/std::max(1,A);
        if(gap > 0.12){ redo=true; break; }
    }
    if(redo){
        // Slightly relax seeds and rerun
        ui.min_area = 0.20*ui.unit_area;
        sr = splitComponentWatershed(src, fg, ui);
    }

    CountResult cr; cr.count = sr.count; cr.unit_area = ui.unit_area; cr.labels = sr.labels;
    return cr;
}

int main(){
    string path = randomImagePath();
    Mat src = imread(path);
    if(src.empty()) return 0;
    CountResult cr = countPills(src);
    std::cout << "Count: " << cr.count << "\n";
    std::cout << "Unit area: " << cr.unit_area << "\n";

    // Visualize
    Mat vis = src.clone();
    RNG rng(1234);
    std::map<int, Vec3b> lut;
    for(int y=0;y<cr.labels.rows;y++){
        const int* L = cr.labels.ptr<int>(y);
        Vec3b* V = vis.ptr<Vec3b>(y);
        for(int x=0;x<cr.labels.cols;x++){
            int id = L[x];
            if(id>0){
                if(!lut.count(id)) lut[id]=Vec3b((uchar)rng.uniform(0,255),(uchar)rng.uniform(0,255),(uchar)rng.uniform(0,255));
                V[x] = 0.6*V[x] + 0.4*lut[id];
            }
        }
    }
    imwrite("pill_segments.png", vis);
    return 0;
}