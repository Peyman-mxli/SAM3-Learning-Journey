# YOLO + Supervision Object Detector

A practical computer vision project developed as part of my **SAM3 Learning Journey**.

This project combines the concepts learned during the **Introduction to Supervision** session into a complete object-detection pipeline.

---

## Project Objective

Build a reusable computer vision application capable of:

1. Loading an image
2. Running object detection with YOLO
3. Converting YOLO predictions to `sv.Detections`
4. Filtering predictions by confidence
5. Creating human-readable labels
6. Drawing bounding boxes
7. Annotating detected objects
8. Saving the annotated image
9. Exporting detection information to JSON

---

## Pipeline

```text
Input Image
     ↓
OpenCV
     ↓
YOLOv8
     ↓
Ultralytics Results
     ↓
sv.Detections
     ↓
Confidence Filtering
     ↓
Bounding Boxes + Labels
     ↓
Annotated Image
     ↓
┌─────────────────┬─────────────────┐
↓                                   ↓
Output Image                    JSON Results
```

---

## Technologies

- Python
- OpenCV
- Ultralytics YOLO
- Supervision
- NumPy
- JSON

---

## Project Structure

```text
01-YOLO-Supervision-Object-Detector/
├── README.md
├── object_detector.py
├── requirements.txt
├── input/
└── output/
```

---

## Features

- Pretrained YOLO object detection
- Supervision detection management
- Configurable confidence threshold
- Bounding-box visualization
- Class-name labels
- Confidence-score labels
- Annotated image export
- JSON prediction export
- Reusable Python structure

---

## Learning Goals

This project brings together the individual concepts studied during the course into one complete workflow.

Instead of testing each concept separately, the goal is to understand how the components work together as a small computer vision application.

---

## Related Material

Course documentation:

```text
08-course-notes/01-Introduction-to-Supervision/
```

Runnable examples:

```text
04-examples/01-Introduction-to-Supervision/
```

Original notebook:

```text
03-notebooks/01_a_introduccion_supervision.ipynb
```

---

## Status

🚧 Project under development
