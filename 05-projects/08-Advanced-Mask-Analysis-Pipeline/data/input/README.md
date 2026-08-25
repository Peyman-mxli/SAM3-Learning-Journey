# Input Data

This directory contains the original input images used by **Project 08 — Advanced Mask Analysis Pipeline**.

The images stored here are processed by the Project 08 pipeline but are not modified directly.

---

## Purpose

The input images provide the source data for:

```text
YOLOv8 Detection
        ↓
Detection Filtering
        ↓
SAM 3 Segmentation
        ↓
Mask Analysis
        ↓
Visualization
        ↓
Structured Results
```

---

## Initial Input Images

The initial validation dataset will contain:

```text
input/
├── README.md
├── bus.jpg
└── zidane.jpg
```

These images were previously used during the segmentation exercises and provide two different object configurations for validating the reusable Project 08 pipeline.

---

## bus.jpg

`bus.jpg` contains multiple detectable object classes.

It is useful for validating:

- Multi-object detection
- Multiple SAM 3 prompts
- Multiple segmentation masks
- Class identification
- Detection confidence
- Mask-area calculations
- Bounding-box-area calculations
- Occupancy ratios
- Multi-object structured results

---

## zidane.jpg

`zidane.jpg` provides a second image with a different scene and object configuration.

It is used to verify that the same Project 08 pipeline can process more than one image without changing the implementation.

This validates the reusable design:

```text
bus.jpg ───────┐
               ↓
         Same Pipeline
               ↓
          Result Set A


zidane.jpg ────┐
               ↓
         Same Pipeline
               ↓
          Result Set B
```

---

## Supported Formats

The current pipeline discovers the following image formats:

```text
.jpg
.jpeg
.png
```

Only supported image files inside this directory are processed.

---

## Input Rules

Files in this directory should be treated as original source data.

The pipeline should:

```text
READ input images
        ↓
PROCESS them
        ↓
SAVE new results elsewhere
```

It should not overwrite or modify the original images.

Generated visualizations belong in:

```text
data/output/
```

Structured analytical results belong in:

```text
results/json/
results/csv/
```

---

## Expected Final Structure

```text
data/input/
├── README.md
├── bus.jpg
└── zidane.jpg
```

---
