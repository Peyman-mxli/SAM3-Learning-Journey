# Source Code

This directory contains the Python source code for **Project 08 — Advanced Mask Analysis Pipeline**.

The source code implements the main computer-vision pipeline used to detect objects, generate SAM 3 segmentation masks, analyze those masks, visualize the results, and produce structured analytical data.

---

## Purpose

The `src/` directory separates the project implementation from:

- Input data
- Generated outputs
- Analytical results
- Documentation

The main processing workflow is:

```text
Input Image
     ↓
YOLOv8 Detection
     ↓
sv.Detections
     ↓
Optional Detection Filtering
     ↓
Bounding-Box Prompts
     ↓
SAM 3 Segmentation
     ↓
Segmentation Masks
     ↓
Mask Analysis
     ↓
Visualization
     ↓
Structured Results
```

---

## Planned Files

```text
src/
├── README.md
├── mask_analysis_pipeline.py
└── utils.py
```

### `mask_analysis_pipeline.py`

This will contain the main Project 08 processing pipeline.

Responsibilities include:

- Loading input images
- Running YOLOv8 detection
- Converting results to `sv.Detections`
- Filtering detections when required
- Sending bounding-box prompts to SAM 3
- Generating segmentation masks
- Calculating mask metrics
- Creating visual annotations
- Producing structured results
- Saving generated outputs

### `utils.py`

This file will contain reusable helper functions used by the main pipeline.

Possible responsibilities include:

- Image validation
- Directory validation
- Mask-area calculations
- Bounding-box calculations
- Occupancy-ratio calculations
- JSON serialization helpers
- CSV export helpers

---

## Mask Analysis

For each segmented object, the source code will calculate:

```text
Object
├── Class ID
├── Class Name
├── Confidence
├── Bounding Box
├── Bounding-Box Area
├── Mask Area
└── Mask / Box Occupancy Ratio
```

The occupancy ratio is calculated as:

```text
Occupancy Ratio = Mask Area / Bounding-Box Area
```

This provides a quantitative comparison between the rectangular YOLO detection and the pixel-level SAM 3 segmentation mask.

---

## Reusable Design

The implementation should avoid duplicating processing logic.

The main pipeline will follow a reusable design such as:

```python
analyze_image(
    image,
    yolo_model,
    sam_model
)
```

Conceptually:

```text
Image A ─────┐
             ↓
       analyze_image()
             ↓
         Result A


Image B ─────┐
             ↓
       analyze_image()
             ↓
         Result B
```

The models should be loaded once and reused when processing multiple images.

---

## Technologies

The source code will use:

- Python
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- OpenCV
- NumPy
- JSON
- CSV

Important Supervision components include:

```python
sv.Detections
sv.MaskAnnotator
sv.BoxAnnotator
```

---

## Development Order

The source implementation will be developed incrementally:

```text
01
Input Validation
      ↓
02
YOLOv8 Detection
      ↓
03
Detection Filtering
      ↓
04
SAM 3 Segmentation
      ↓
05
Mask Metrics
      ↓
06
Visualization
      ↓
07
Structured Results
      ↓
08
Multi-Image Processing
```

Each stage will be validated before moving to the next one.

---

## Current Status

```text
src/ documentation:       CREATED ✅
Main pipeline:             PENDING
Utility functions:         PENDING
Detection implementation:  PENDING
SAM 3 implementation:      PENDING
Mask analysis:             PENDING
Result export:             PENDING

Status: IN PROGRESS
```
