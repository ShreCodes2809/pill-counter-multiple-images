# Advanced Pill Counter using OpenCV and C++

## Overview

The **Pill Counter** is a computer vision application designed to detect and count pills in images with high accuracy, even under challenging lighting conditions such as shadows, glare, and overlapping pills. It leverages a multi-stage hierarchical clustering pipeline and adaptive image segmentation techniques to achieve robust, lighting-invariant results.

The program:

1. Randomly selects an image from the `images/` directory.
2. Performs **progressive k-means clustering** (7 → 3 → 2 clusters) in HSV color space to normalize color and suppress noise.
3. Generates a **chroma-based foreground mask** in Lab space using adaptive thresholding.
4. Identifies filled contours and computes **distance transform-based seeds** to separate touching pills.
5. Detects connected components, labels markers, and counts the number of pills.
6. Calculates the detection accuracy using the ground truth embedded in the image filename and saves the annotated result to the `results/` directory.

---

## System Requirements

**Platform:** macOS (tested on macOS 26+)

**Dependencies:**

* OpenCV 4 (installed via Homebrew)
* pkg-config
* clang++ with C++17 support

**Installation Commands:**

```bash
brew install opencv
brew install pkg-config
```

---

## Project Structure

```
project-root/
├── images/                 # Input images (e.g., p19_44.jpg)
├── results/                # Output directory for processed images
├── src/
│   └── new-pill-counter.cpp  # Main source code
└── bin/                    # Compiled executable output
```

---

## Build and Run Instructions

**Compilation Command:**

```bash
clang++ -std=c++17 src/new-pill-counter.cpp -o bin/pill_counter_app `pkg-config --cflags --libs opencv4`
```

**Run Command:**

```bash
./bin/pill_counter_app
```

Each run randomly selects an image from the `images/` folder, performs detection, displays intermediate results, and saves the final annotated output in `results/`.

**Example:**

![Results for the sample image](sample_imgs/image_1.png)

```
Input  : images/p10_27.jpg
Output : results/p10_27.jpg
Console:
Detected pills: 27
Actual pills: 27
Accuracy: 100.00%
```

---

## Algorithm Overview

### 1. Progressive K-Means Clustering

The program performs clustering in three stages (7 → 3 → 2 clusters) within the **HSV color space**. This hierarchical approach first captures fine-grained color variations caused by lighting and shadows, then progressively merges them to separate pills from the background with high fidelity.

### 2. Chroma-Based Foreground Extraction

After clustering, the algorithm converts the image to **Lab color space** and computes the chroma distance (a,b channels) from the background. Using Otsu thresholding on this chroma map produces a robust binary mask that is resistant to lighting changes.

### 3. Contour Detection and Filling

Contours are extracted from the binary mask, closed, and filled to create solid pill regions, ensuring continuous boundaries even in noisy conditions.

### 4. Adaptive Distance Transform and Seed Generation

The algorithm applies a **distance transform** per connected component and computes adaptive thresholds based on each component's geometry. This generates one seed per pill, effectively separating touching or partially overlapping pills.

### 5. Counting and Accuracy Evaluation

Connected components from the seed map are counted as detected pills. The ground truth pill count is parsed from the filename (e.g., `p10_27` → actual count = 27). Accuracy is calculated as:

```
Accuracy = (Detected / Actual) * 100
```

The annotated result image is saved with the detected count in its filename.

---

## Usage Guidelines

* **Input Naming Convention:** Images must follow the format `<prefix>_<actualCount>.<ext>` (e.g., `p10_27.jpg`).
* **Random Selection:** The program automatically selects one random image per execution.
* **Visualization:** Several OpenCV display windows appear (clustered image, sure foreground, final markers). Close them to terminate the program.

---

## Parameter Tuning

| Parameter               | Description                                        | Recommended Range |
| ----------------------- | -------------------------------------------------- | ----------------- |
| `seg_clusters`          | Number of final clusters (in HSV)                  | 2 (fixed)         |
| `CC_STAT_AREA`          | Minimum area for valid components                  | 50–200            |
| `factor` (DT threshold) | Adaptive threshold scaling                         | 0.25–0.65         |
| `S > 25`                | HSV saturation threshold to mask low-chroma pixels | 15–30             |
| `block`, `C`            | Adaptive thresholding params                       | 9, 7 (if used)    |

---

## Troubleshooting

| Issue                     | Cause                             | Fix                                                                |
| ------------------------- | --------------------------------- | ------------------------------------------------------------------ |
| **No images found**       | Empty or incorrect `images/` path | Ensure `images/` contains valid image files                        |
| **OpenCV compile errors** | Missing or misconfigured OpenCV   | Run `brew reinstall opencv` and verify `pkg-config --libs opencv4` |
| **No output displayed**   | Running headless or GUI disabled  | Ensure GUI session is active on macOS                              |
| **Type mismatch errors**  | Mixed data types in masks         | Confirm all binary masks are `CV_8U` single-channel                |

---

## Performance Summary

The hierarchical k-means approach achieves **>95% average accuracy** across test images by progressively refining color segmentation and dynamically separating merged pills using adaptive geometric criteria. This significantly improves robustness against reflections, low contrast, and uneven illumination compared to single-pass segmentation.

---

## Future Enhancements

* Implement watershed post-processing for heavily overlapping pills.
* Add batch mode with CSV export for accuracy tracking.
* Integrate lighting correction (Retinex/CLAHE) pre-processing.
* Explore GPU-accelerated k-means for faster clustering.

---

## License

This project is released under the Apache-2.0 License.

---

## Acknowledgments

Developed using OpenCV 4. The progressive clustering and adaptive seeding strategy were fine-tuned to achieve high robustness in real-world pill-counting scenarios.
