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
---

4. Annotation Customization

Annotators can be customized to improve visualization.

Common customization options include:

Colors
Color palettes
Line thickness
Text scale
Label appearance

This makes visualization adaptable to different computer vision applications.

5. Alternative Annotators

Bounding boxes are not the only way to visualize detections.

The lesson also explores:

sv.EllipseAnnotator()

and:

sv.DotAnnotator()

These provide alternative ways to represent detected objects.

6. Annotation Layers

Multiple Annotators can be applied sequentially to the same image.

Example:

Original Image
      ↓
BoxAnnotator
      ↓
EllipseAnnotator
      ↓
DotAnnotator
      ↓
LabelAnnotator
      ↓
Final Visualization

Each Annotator receives the image produced by the previous annotation step.

This creates a layered visualization pipeline.

7. YOLO and Supervision

YOLO and Supervision perform different responsibilities.

YOLO
  ↓
Object Detection
  ↓
Detection Data
  ↓
Supervision
  ↓
Visualization

YOLO determines what and where objects are.

Supervision determines how those detections are visualized.

Related Material
Course Notebook
../01_b_anotacion_visualizacion.ipynb
Code Examples
../../../04-examples/02-Annotation-and-Visualization/
Practical Project
../../../05-projects/02-Multi-Annotator-Visualization-Pipeline/
Learning Objective

The objective of this lesson is to understand how raw detection data can be transformed into customizable and layered visual representations.

These concepts provide the foundation for building more advanced computer vision visualization pipelines later in the SAM3 Learning Journey.


Commit message:

```text
Expand Annotation and Visualization concept notes
