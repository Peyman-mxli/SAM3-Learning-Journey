# Annotation and Visualization — Concepts

This directory contains detailed concept notes from the **Annotation and Visualization** lesson of my SAM3 Computer Vision Learning Journey.

The lesson focuses on transforming raw object-detection results into clear and useful visual representations using the **Supervision** library.

---

## Topics Covered

### 1. Supervision Annotators

Supervision provides specialized Annotator classes for visualizing computer vision detections.

Examples studied in this lesson include:

- `BoxAnnotator`
- `LabelAnnotator`
- `EllipseAnnotator`
- `DotAnnotator`

Each Annotator adds a different visual representation to the detected objects.

---

### 2. Bounding Box Visualization

`BoxAnnotator` draws rectangular bounding boxes around detected objects.

Bounding boxes make it possible to visually identify the location and size of each detection.

---

### 3. Detection Labels

`LabelAnnotator` displays information associated with detections.

Labels can contain information such as:

- Object class
- Confidence score
- Detection information

Example:

```text
person 91%
car 84%
bus 92%
