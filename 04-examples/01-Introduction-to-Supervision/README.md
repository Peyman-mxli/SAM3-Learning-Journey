# Introduction to Supervision — Code Examples

This folder contains runnable Python examples based on the **Introduction to Supervision** session of the SAM3 course.

The examples demonstrate how **Ultralytics YOLO**, **OpenCV**, and **Supervision** can be combined to build a basic computer vision pipeline.

---

## Examples

### 01 — Image Loading

**File:** `01-image-loading.py`

Load an image with OpenCV, inspect its dimensions, convert it from BGR to RGB, and display it.

### 02 — YOLO Object Detection

**File:** `02-yolo-detection.py`

Load a pretrained YOLO model and run object detection on an image.

### 03 — Supervision Detections

**File:** `03-supervision-detections.py`

Convert Ultralytics YOLO predictions into the standardized `sv.Detections` format.

### 04 — Confidence Threshold

**File:** `04-confidence-threshold.py`

Filter detections according to their confidence scores.

### 05 — Image Annotation

**File:** `05-image-annotation.py`

Use `BoxAnnotator` and `LabelAnnotator` to visualize detected objects.

### 06 — YOLO Model Comparison

**File:** `06-yolo-model-comparison.py`

Compare predictions produced by YOLOv8 Nano and YOLOv8 Small.

### 07 — Custom Image Detection

**File:** `07-custom-image.py`

Apply the complete YOLO + Supervision pipeline to another image.

### 08 — Save Predictions as JSON

**File:** `08-save-predictions-json.py`

Convert detection results into structured data and save the predictions as JSON.

---

## Computer Vision Pipeline

```text
Image
  ↓
OpenCV
  ↓
YOLO
  ↓
Ultralytics Results
  ↓
sv.Detections
  ↓
Filtering / Analysis
  ↓
Supervision Annotators
  ↓
Annotated Image
```

---

## Technologies

- Python
- OpenCV
- NumPy
- Matplotlib
- Ultralytics YOLO
- Supervision

---

## Purpose

These files are simplified, runnable examples extracted from the concepts and practical exercises studied during the session.

The detailed explanations are documented separately in:

```text
08-course-notes/
└── 01-Introduction-to-Supervision/
```

The original course notebook is preserved in:

```text
03-notebooks/
└── 01_a_introduccion_supervision.ipynb
```
