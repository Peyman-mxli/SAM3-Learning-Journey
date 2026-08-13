# Annotation and Visualization — Practical Exercises

This directory contains hands-on exercises from the **Annotation and Visualization** lesson of my SAM3 Computer Vision Learning Journey.

The exercises progress from basic bounding-box visualization to a complete multi-annotator visualization workflow using **YOLOv8**, **Ultralytics**, **Supervision**, and **OpenCV**.

---

## Exercises

### 01 — Basic Box Annotation

[`01-basic-box-annotation.md`](./01-basic-box-annotation.md)

Introduction to the basic detection and annotation workflow.

Topics:

- Loading an image with OpenCV
- Running YOLOv8 object detection
- Using confidence thresholds
- Converting results to `sv.Detections`
- Using `BoxAnnotator`
- Changing bounding-box thickness
- Saving annotated images

---

### 02 — Label Annotation

[`02-label-annotation.md`](./02-label-annotation.md)

Extends bounding-box visualization by adding detection information.

Topics:

- `LabelAnnotator`
- Class IDs
- Class names
- Confidence scores
- Creating custom labels
- Confidence percentage formatting
- Text scale
- Box + Label composition
- Annotation order

---

### 03 — Annotation Customization

[`03-annotation-customization.md`](./03-annotation-customization.md)

Explores how annotation appearance can be customized independently from YOLO detection results.

Topics:

- `ColorPalette.DEFAULT`
- Bounding-box thickness
- Label text scale
- Custom label formatting
- Visualization readability
- Confidence thresholds
- Detection vs. visualization configuration

---

### 04 — Multi-Annotator Challenge

[`04-multi-annotator-challenge.md`](./04-multi-annotator-challenge.md)

Combines multiple Supervision Annotators into a layered visualization pipeline.

Topics:

- `BoxAnnotator`
- `EllipseAnnotator`
- `DotAnnotator`
- `LabelAnnotator`
- Layer composition
- Layer ordering
- Reusing `sv.Detections`
- Building custom visualization pipelines
- Alternative Annotators
- Multi-layer visualization

---

## Learning Progression

The exercises are designed to build on each other:

```text
Exercise 01
Basic Bounding Boxes
        ↓
Exercise 02
Boxes + Labels
        ↓
Exercise 03
Visualization Customization
        ↓
Exercise 04
Multi-Annotator Pipeline
        ↓
Practical Project
```

Each exercise introduces additional functionality while reusing the same fundamental computer vision workflow.

---

## Core Workflow

The exercises follow this general architecture:

```text
Input Image
     ↓
OpenCV
     ↓
YOLOv8
     ↓
Detection Results
     ↓
sv.Detections
     ↓
Supervision Annotators
     ↓
Visualization
     ↓
Output Image
```

---

## Detection vs. Visualization

One of the most important concepts practiced in these exercises is the separation between detection and visualization.

### YOLO

YOLO determines:

```text
What object was detected?
Where is the object?
How confident is the prediction?
```

### Supervision

Supervision helps determine:

```text
How should the detection be visualized?
```

Therefore:

```text
YOLO
  ↓
Detection Data
  ↓
sv.Detections
  ↓
Supervision
  ↓
Visual Representation
```

Changing annotation colors, thickness, text size, or visualization layers does not modify the underlying YOLO prediction.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| YOLOv8 | Object detection |
| Ultralytics | YOLO model interface |
| Supervision | Detection processing and visualization |
| OpenCV | Image loading and saving |

---

## Connection to Course Concepts

The theoretical explanations for these exercises are available in:

```text
../concepts/
```

The concepts section covers:

```text
01-supervision-annotators.md
02-annotation-customization.md
03-annotation-layers.md
```

The recommended learning flow is:

```text
Read Concept
     ↓
Study Example
     ↓
Complete Exercise
     ↓
Experiment
     ↓
Build Project
```

---

## Connection to Project 02

The exercises lead directly into:

```text
05-projects/
└── 02-Multi-Annotator-Visualization-Pipeline/
```

The project transforms the concepts practiced here into a complete reusable Python application.

The progression is:

```text
Course Lesson
     ↓
Concept Notes
     ↓
Practical Exercises
     ↓
Code Examples
     ↓
Multi-Annotator Visualization Pipeline
```

---

## Exercise Status

| # | Exercise | Status |
|---|---|---|
| 01 | Basic Box Annotation | Completed |
| 02 | Label Annotation | Completed |
| 03 | Annotation Customization | Completed |
| 04 | Multi-Annotator Challenge | Completed |

**Total exercises: 4**

---

## Key Takeaway

These exercises demonstrate how raw object-detection results can progressively become rich visual representations.

```text
Detection
    ↓
Bounding Boxes
    ↓
Labels
    ↓
Customization
    ↓
Multiple Annotation Layers
    ↓
Complete Visualization Pipeline
```

The same detection data can support many different visualization styles, making the pipeline flexible and reusable.

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
