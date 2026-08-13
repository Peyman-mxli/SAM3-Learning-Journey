# Supervision Annotators

## Overview

Supervision Annotators are visualization tools used to transform computer vision detection data into visual elements that can be drawn on images.

An object-detection model such as YOLO produces detection information, while Supervision Annotators determine how that information is displayed.

---

## Detection and Visualization

The general workflow is:

```text
Image
  ↓
YOLO
  ↓
Detection Results
  ↓
sv.Detections
  ↓
Supervision Annotators
  ↓
Visualized Image
```

YOLO is responsible for detecting objects.

Supervision is responsible for processing and visualizing those detections.

---

## `sv.Detections`

YOLO results can be converted into a Supervision `Detections` object:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

The resulting object provides structured detection information such as:

- Bounding box coordinates
- Class IDs
- Confidence scores

This detection data can then be passed to different Annotators.

---

## BoxAnnotator

`BoxAnnotator` draws rectangular bounding boxes around detected objects.

```python
box_annotator = sv.BoxAnnotator()
```

It can be applied to an image:

```python
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

Conceptually:

```text
Detection
    ↓
Bounding Box
    ↓
Object Location Becomes Visible
```

---

## LabelAnnotator

`LabelAnnotator` adds textual information to detections.

```python
label_annotator = sv.LabelAnnotator()
```

Labels can contain information such as:

```text
person 91%
car 84%
bus 92%
```

Example:

```python
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)
```

---

## EllipseAnnotator

`EllipseAnnotator` provides another way of representing detections.

```python
ellipse_annotator = sv.EllipseAnnotator()
```

Instead of relying only on rectangular bounding boxes, ellipses provide an alternative visual representation around detected objects.

---

## DotAnnotator

`DotAnnotator` adds detection points to the visualization.

```python
dot_annotator = sv.DotAnnotator()
```

This provides another visual layer that can be combined with other Annotators.

---

## Combining Annotators

Annotators can be applied sequentially:

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
Final Image
```

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

Each Annotator adds another visualization layer to the existing image.

---

## Why Annotators Are Useful

Annotators separate detection logic from visualization logic.

```text
AI Model
   ↓
Detection
   ↓
Structured Detection Data
   ↓
Visualization
```

This means the same detections can be visualized in different ways without rerunning the object-detection model.

---

## Key Takeaway

Supervision Annotators provide reusable components for visualizing computer vision detections.

Different Annotators can be combined to create customized visualization pipelines while keeping the underlying detection data unchanged.
