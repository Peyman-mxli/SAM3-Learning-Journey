# Practical Exercises — Introduction to Supervision

This directory contains the practical exercises completed during the **Introduction to Supervision** session of the SAM3 Computer Vision course.

The objective of these exercises is to move from theoretical concepts to a complete working computer vision pipeline using **OpenCV, Ultralytics YOLO, and Supervision**.

---

## Practical Workflow

Throughout these exercises, we build the following pipeline:

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
Analysis / Filtering
  ↓
Supervision Annotators
  ↓
Annotated Image
```

---

## Exercises

### 1. Image Loading

Download an example image, load it with OpenCV, inspect its dimensions, and understand the difference between BGR and RGB.

[`01-image-loading.md`](01-image-loading.md)

### 2. YOLO Object Detection

Load a pretrained YOLO model and perform object detection on an image.

[`02-yolo-detection.md`](02-yolo-detection.md)

### 3. Supervision Detections

Convert YOLO predictions into `sv.Detections` and inspect:

- Bounding boxes
- Confidence scores
- Class IDs
- Class names

[`03-supervision-detections.md`](03-supervision-detections.md)

### 4. Confidence Experiment

Change the YOLO confidence threshold and analyze how stricter confidence requirements affect the number of detections.

```text
Default Confidence
        ↓
More Predictions

Higher Confidence
        ↓
Fewer Predictions
```

[`04-confidence-experiment.md`](04-confidence-experiment.md)

### 5. Image Annotation

Use Supervision to draw:

- Bounding boxes
- Class labels
- Confidence scores

and generate a final annotated image.

[`05-image-annotation.md`](05-image-annotation.md)

### 6. YOLO Model Comparison

Compare:

```text
yolov8n.pt
     vs.
yolov8s.pt
```

and analyze differences in detection results.

[`06-yolo-model-comparison.md`](06-yolo-model-comparison.md)

### 7. Custom Image Challenge

Apply the complete pipeline to a different image and analyze:

- Correct detections
- Missed objects
- Incorrect detections
- Confidence scores

[`07-custom-image-challenge.md`](07-custom-image-challenge.md)

---

## Main Libraries

```python
import supervision as sv
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
```

---

## Core Supervision Conversion

One of the most important operations in these exercises is:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

This converts the YOLO result into Supervision's standardized detection structure.

---

## Detection Data

After conversion, we can access:

```python
detections.xyxy
detections.confidence
detections.class_id
```

These answer three fundamental questions:

```text
WHERE?     → xyxy
HOW SURE?  → confidence
WHAT?      → class_id
```

---

## Final Practical Architecture

```text
                  INPUT IMAGE
                       │
                       ▼
                     OpenCV
                       │
                       ▼
                      YOLO
                       │
                       ▼
              Model Predictions
                       │
                       ▼
                sv.Detections
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        xyxy       confidence    class_id
          │            │            │
          └────────────┼────────────┘
                       ▼
                Analyze / Filter
                       │
                       ▼
                  Annotators
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Bounding Boxes          Labels
             │                   │
             └─────────┬─────────┘
                       ▼
                Annotated Image
```

---

## Learning Goal

By completing these exercises, we should be able to independently build the basic workflow:

```text
Load → Detect → Convert → Inspect → Filter → Annotate → Analyze
```

This practical workflow provides the foundation for more advanced computer vision tasks later in the SAM3 course.
