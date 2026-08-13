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

Example:

```python
box_annotator = sv.BoxAnnotator()
```

The Annotator can then be applied to detections:

```python
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

---

### 3. Detection Labels

`LabelAnnotator` displays information associated with detections.

Labels can contain information such as:

- Object class
- Confidence score
- Detection information

Example labels:

```text
person 91%
car 84%
bus 92%
```

Example:

```python
label_annotator = sv.LabelAnnotator()
```

---

### 4. Annotation Customization

Annotators can be customized to improve the appearance and readability of computer vision results.

Common customization options include:

- Colors
- Color palettes
- Line thickness
- Text scale
- Label appearance

For example:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)
```

A label Annotator can also be customized:

```python
label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)
```

Customization makes visualizations easier to understand and allows them to be adapted to different computer vision applications.

---

### 5. Alternative Annotators

Bounding boxes are not the only way to visualize detections.

Supervision provides several alternative Annotators.

#### EllipseAnnotator

```python
ellipse_annotator = sv.EllipseAnnotator()
```

`EllipseAnnotator` draws elliptical shapes around detected objects.

#### DotAnnotator

```python
dot_annotator = sv.DotAnnotator()
```

`DotAnnotator` adds detection points to detected objects.

These Annotators can be used independently or combined with other visualization methods.

---

### 6. Annotation Layers

Multiple Annotators can be applied sequentially to the same image.

For example:

```text
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
```

Each Annotator receives the image produced by the previous annotation step.

Example:

```python
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

annotated_image = ellipse_annotator.annotate(
    scene=annotated_image,
    detections=detections
)

annotated_image = dot_annotator.annotate(
    scene=annotated_image,
    detections=detections
)

annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)
```

This creates a layered visualization pipeline.

---

### 7. YOLO and Supervision

YOLO and Supervision perform different responsibilities in the computer vision pipeline.

```text
Input Image
     ↓
YOLO
     ↓
Object Detection
     ↓
Detection Data
     ↓
sv.Detections
     ↓
Supervision
     ↓
Visualization
```

### YOLO

YOLO determines:

- What objects are present
- Where the objects are located
- How confident the model is

### Supervision

Supervision helps determine:

- How detections are represented
- Which Annotators are used
- How labels appear
- How visualization layers are combined

This separation makes the computer vision pipeline easier to customize and maintain.

---

## Related Material

### Course Notebook

```text
../01_b_anotacion_visualizacion.ipynb
```

### Code Examples

```text
../../../04-examples/02-Annotation-and-Visualization/
```

### Practical Project

```text
../../../05-projects/02-Multi-Annotator-Visualization-Pipeline/
```

---

## Learning Objective

The objective of this lesson is to understand how raw detection data can be transformed into customizable and layered visual representations.

The main workflow is:

```text
Detection
    ↓
Structured Data
    ↓
Annotation
    ↓
Customization
    ↓
Layer Composition
    ↓
Final Visualization
```

These concepts provide the foundation for building more advanced computer vision visualization pipelines later in the **SAM3 Learning Journey**.

---

## Key Takeaway

Object detection and visualization are separate stages of a computer vision application.

**YOLO detects the objects.**

**Supervision provides tools for processing and visualizing those detections.**

By combining multiple Supervision Annotators, we can create reusable and highly customizable visualization pipelines.
