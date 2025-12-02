# Advanced Pill Counter using OpenCV (Chroma-Based + Gradient Watershed)

## Overview

This system detects and counts pills using a chromatic segmentation pipeline combined with marker-controlled watershed driven by gradient topography. It is designed to handle capsules, round pills, shadows, lighting variation, and partial overlaps.

The updated algorithm replaces earlier HSV/k-means and contour-based approaches with:

* Chroma-based foreground extraction (Lab a*, b* median background estimation)
* Adaptive separation logic based on shape (round vs capsule)
* Distance Transform–derived markers
* CLAHE-enhanced L-channel gradient magnitude as watershed topography
* Final watershed labeling and object counting

All outputs and logs are stored in the `results/` directory.

## System Requirements

**Platform:** macOS (tested on macOS 26+)

**Dependencies:**

* OpenCV 4
* pkg-config
* clang++ with C++17 support

Install:

```bash
brew install opencv
brew install pkg-config
```

## Project Structure

```
project-root/
├── images/                        # Input pill images
├── results/                       # Logs and output segmentation
├── trial.cpp                      # Main source code
└── bin/
```

## Build and Run Instructions

### Compile

```bash
clang++ -std=c++17 trial.cpp -o bin/pill_counter `pkg-config --cflags --libs opencv4`
```

### Run

```bash
./bin/pill_counter
```

The program iterates over all images, displays intermediate stages, and logs accuracy in `results/run_chroma_ws.txt`.

## Algorithm Overview

### 1. Chroma-Based Foreground Extraction (Lab)

* Convert BGR → Lab
* Build border mask and compute median background chroma (a*, b*)
* Compute chroma distance map
* Normalize + Otsu threshold → binary mask
* Morphological closing for smoothing

### 2. Shape-Adaptive Pill Mode Selection

Aspect ratios of connected components determine:

* **Round mode** if median AR < 1.10
* **Capsule mode** otherwise

### 3. Distance Transform–Based Sure Foreground and Background

#### Round pills:

* Distance transform on mask
* Threshold at alpha ≈ 0.65
* Morphological open for cleanup
* Background via dilation

#### Capsules:

* Light cleaning
* DT threshold at alpha_caps ≈ 0.35
* Remove narrow bridges
* Background via light dilation

Produces:

* `sure_fg`
* `sure_bg`
* `unknown = sure_bg - sure_fg`

### 4. Gradient Topography for Watershed

* Extract L channel from Lab
* Apply CLAHE
* Gaussian blur
* Compute Sobel gradients gx, gy
* Gradient magnitude → watershed topography

### 5. Marker-Controlled Watershed

* Connected components on `sure_fg` → markers
* Unknown region set to 0
* Watershed applied on gradient topography
* Boundaries drawn in red

### 6. Counting and Accuracy

* Filename format: `pXX_YY.jpg` → actual count = YY
* Detected count = number of watershed labels ≥ 2
* Accuracy logged per image

## Usage Guidelines

* Images must follow `<prefix>_<actualCount>.<ext>` naming
* Close windows to proceed
* Logs saved automatically

## Parameter Tuning

| Parameter        | Meaning                      | Typical Values  |
| ---------------- | ---------------------------- | --------------- |
| `alpha`          | DT shrink factor (round)     | 0.60–0.70       |
| `alpha_caps`     | DT core threshold (capsules) | 0.30–0.45       |
| AR threshold     | Shape mode decision          | 1.05–1.15       |
| Border thickness | Background sampling          | rows/40–rows/30 |

## Troubleshooting

| Issue           | Cause                 | Fix                                      |
| --------------- | --------------------- | ---------------------------------------- |
| Empty mask      | Low contrast          | Add slight blur before chroma extraction |
| Over-splitting  | DT threshold too high | Reduce alpha/alpha_caps                  |
| Under-splitting | DT threshold too low  | Increase alpha/alpha_caps                |
| Wrong mode      | AR threshold mismatch | Adjust round/capsule cutoff              |

## Performance Summary

The updated pipeline provides robust segmentation under:

* Lighting variation
* Shadows and glare
* Capsules and elongated shapes
* Slight overlaps

Accuracy is consistently high due to chroma modeling and gradient-topography watershed.
