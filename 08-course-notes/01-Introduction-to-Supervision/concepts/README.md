# Supervision Concepts

This directory contains the theoretical and technical concepts covered during the **Introduction to Supervision** session of the SAM3 Computer Vision course.

The goal is to document each concept separately so the material can be used as a structured reference throughout the course.

## Concepts

### 1. Supervision
Introduction to the Supervision library and its role as a common interface for computer vision applications.

[`supervision.md`](supervision.md)

### 2. YOLO Object Detection
Understanding YOLO, object detection, model inference, classes, bounding boxes, and confidence scores.

[`yolo.md`](yolo.md)

### 3. Bounding Boxes
Understanding the `xyxy` coordinate format, image coordinates, bounding-box dimensions, and object localization.

[`bounding-boxes.md`](bounding-boxes.md)

### 4. `sv.Detections`
The central Supervision data structure used to represent bounding boxes, confidence scores, and class IDs.

[`detections.md`](detections.md)

### 5. Confidence Scores
Understanding model confidence, confidence thresholds, and how stricter thresholds affect predictions.

[`confidence-scores.md`](confidence-scores.md)

### 6. Image Annotations
Using Supervision annotators to transform raw detections into human-readable visual results.

[`annotations.md`](annotations.md)

### 7. COCO Dataset
Understanding the dataset and object categories used by the pretrained YOLO model in this lesson.

[`coco-dataset.md`](coco-dataset.md)

### 8. Detection Filtering
Filtering predictions according to confidence scores and object classes.

[`detection-filtering.md`](detection-filtering.md)

### 9. Computer Vision Pipeline
Understanding the complete architecture:

```text
Image → YOLO → sv.Detections → Processing → Annotation → Result
```

[`computer-vision-pipeline.md`](computer-vision-pipeline.md)

### 10. YOLO Model Comparison
Comparing YOLOv8 Nano and Small models and understanding the trade-off between model size, speed, and detection capability.

[`yolo-model-comparison.md`](yolo-model-comparison.md)

### 11. OpenCV Image Processing
Loading images, understanding image arrays, BGR vs. RGB, and connecting OpenCV with the detection pipeline.

[`opencv-image-processing.md`](opencv-image-processing.md)

### 12. Saving Predictions to JSON
Understanding how structured prediction data can be preserved for later analysis.

[`json-predictions.md`](json-predictions.md)

### 13. Extension Challenge
Applying the complete YOLO + Supervision workflow to a new image and analyzing model behavior.

[`extension-challenge.md`](extension-challenge.md)

---

## Core Architecture

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
                   Ultralytics Results
                            │
                            ▼
              sv.Detections.from_ultralytics()
                            │
                            ▼
                      sv.Detections
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
           xyxy         confidence      class_id
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                    Filter / Analyze
                            │
                            ▼
                       Annotators
                            │
                            ▼
                    Annotated Image
```

## Technologies Covered

- Python
- Google Colab
- OpenCV
- NumPy
- Matplotlib
- Ultralytics
- YOLOv8
- Supervision
- COCO
- JSON

## Main Takeaway

The fundamental concept behind this lesson is:

```text
Model Predictions
       ↓
sv.Detections
       ↓
Standardized Processing
       ↓
Analysis + Visualization
```

Supervision provides the common layer that allows computer vision predictions to be processed, filtered, analyzed, and visualized in a structured way.
