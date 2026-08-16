# SAM3 — Code Examples

This directory contains clean, runnable Python examples based on the practical concepts covered throughout my **SAM3 Computer Vision Learning Journey**.

Unlike the detailed course notes, these files focus on small, reusable code examples that can be executed and studied independently.

---

## Available Examples

### 00 — Agentic AI Programming

[`00-Agentic-AI-Programming/`](./00-Agentic-AI-Programming/)

Examples covering:

- AI-assisted code generation
- Prompt engineering for code
- OpenCV image processing
- AI-assisted debugging
- Error analysis and refinement

---

### 01 — Introduction to Supervision

[`01-Introduction-to-Supervision/`](./01-Introduction-to-Supervision/)

Examples covering:

- OpenCV image loading
- YOLOv8 object detection
- `sv.Detections`
- Bounding boxes
- Confidence scores
- Confidence thresholds
- Supervision annotations
- YOLO model comparison
- Custom-image detection
- JSON prediction export

---

### 02 — Annotation and Visualization

[`02-Annotation-and-Visualization/`](./02-Annotation-and-Visualization/)

Examples covering:

- `BoxAnnotator`
- `RoundBoxAnnotator`
- `HaloAnnotator`
- `BlurAnnotator`
- `BoxCornerAnnotator`
- `LabelAnnotator`
- Bounding box and label composition
- Annotation colors and color palettes
- Bounding box thickness
- Label text scale
- Annotation layer order
- Custom multi-annotator visualization pipelines
- `DotAnnotator`
- `EllipseAnnotator`

The examples demonstrate how the same YOLO detection results can be presented using different visualization techniques and how multiple Supervision Annotators can be composed as visual layers.

---

### 03 — Filtering and Manipulating Detections

[`03-Filtering-and-Manipulating-Detections/`](./03-Filtering-and-Manipulating-Detections/)

Examples covering:

- Confidence filtering
- Boolean masks
- Class filtering
- Multiple filtering conditions
- Class exclusion
- Bounding-box area
- Size filtering
- Detection merging
- Non-Maximum Suppression
- Confidence sorting
- Top-N detection selection
- Bounding-box center calculations
- Spatial filtering

These examples demonstrate how raw YOLO predictions can be transformed into application-specific detections using **Supervision** and **NumPy**.

The filtering workflow can be represented as:

```text
Raw Detections
      ↓
Confidence Filtering
      ↓
Class Filtering
      ↓
Size Filtering
      ↓
NMS
      ↓
Top-N Selection
      ↓
Spatial Filtering
      ↓
Application-Specific Detections
```

---

## Repository Organization

The SAM3 Learning Journey separates different types of material:

```text
03-notebooks/
    Original Google Colab / Jupyter notebooks

04-examples/
    Small runnable Python examples

05-projects/
    Larger practical projects

08-course-notes/
    Detailed class notes and explanations

09-assets/
    Images, banners, and supporting resources
```

---

## Example Workflow

Many of the computer vision examples follow this architecture:

```text
Input
  ↓
Python
  ↓
OpenCV
  ↓
AI Model
  ↓
Predictions
  ↓
Supervision
  ↓
Processing / Analysis
  ↓
Visualization / Output
```

As the course progresses, this workflow is extended with additional visualization and processing layers.

For example:

```text
Input Image
    ↓
YOLO
    ↓
Detection Results
    ↓
sv.Detections
    ↓
Filtering / Manipulation
    ↓
Supervision Annotators
    ↓
Visualization / Output
```

---

## Technologies

Examples in this directory may use:

- Python
- OpenCV
- NumPy
- Matplotlib
- Ultralytics YOLO
- Supervision
- JSON
- Google Colab

---

## Example Directory Structure

```text
04-examples/
│
├── 00-Agentic-AI-Programming/
│
├── 01-Introduction-to-Supervision/
│
├── 02-Annotation-and-Visualization/
│   ├── README.md
│   ├── 01_box_and_label.py
│   ├── 02_compare_annotators.py
│   ├── 03_customize_visualization.py
│   ├── 04_layer_order.py
│   └── 05_custom_composition.py
│
├── 03-Filtering-and-Manipulating-Detections/
│   ├── README.md
│   ├── 01_confidence_filtering.py
│   ├── 02_class_and_boolean_filtering.py
│   ├── 03_size_filtering.py
│   ├── 04_nms_and_top_n.py
│   └── 05_spatial_filtering.py
│
└── README.md
```

---

## 02 — Annotation and Visualization Examples

These examples focus on transforming raw object detections into clear visual results.

### Basic Box and Label

```text
01_box_and_label.py
```

Demonstrates the standard visualization pipeline:

```text
Image
  ↓
YOLO
  ↓
Detections
  ↓
BoxAnnotator
  ↓
LabelAnnotator
  ↓
Annotated Image
```

### Annotator Comparison

```text
02_compare_annotators.py
```

Compares:

```python
sv.BoxAnnotator()
sv.RoundBoxAnnotator()
sv.HaloAnnotator()
sv.BlurAnnotator()
sv.BoxCornerAnnotator()
```

### Visualization Customization

```text
03_customize_visualization.py
```

Experiments with:

```python
color=
thickness=
text_scale=
```

and:

```python
sv.Color.RED
sv.Color.GREEN
sv.ColorPalette.DEFAULT
```

### Layer Order

```text
04_layer_order.py
```

Demonstrates why annotation order matters:

```text
Box → Label
```

versus:

```text
Label → Box
```

The Annotator applied last appears visually on top of the previous layers.

### Custom Composition

```text
05_custom_composition.py
```

Combines multiple Annotators:

```text
Original Image
      ↓
EllipseAnnotator
      ↓
DotAnnotator
      ↓
LabelAnnotator
      ↓
Custom Visualization
```

---

## 03 — Filtering and Manipulating Detections Examples

These examples focus on selecting, removing, ranking, and manipulating object detections after model inference.

### Confidence Filtering

```text
01_confidence_filtering.py
```

Demonstrates how a Boolean confidence mask can remove low-confidence predictions:

```python
high_confidence = detections[
    detections.confidence > 0.50
]
```

The basic concept is:

```text
All Detections
      ↓
Confidence > Threshold
      ↓
High-Confidence Detections
```

---

### Class and Boolean Filtering

```text
02_class_and_boolean_filtering.py
```

Demonstrates how detections can be selected according to class:

```python
detections.class_id == 0
```

Multiple conditions can also be combined:

```python
(
    detections.class_id == 0
)
&
(
    detections.confidence > 0.60
)
```

This example also demonstrates class exclusion using:

```python
detections.class_id != 0
```

---

### Size Filtering

```text
03_size_filtering.py
```

Demonstrates how bounding-box area can be used to remove small detections:

```python
large_detections = detections[
    detections.area > 5000
]
```

The example also inspects:

```text
Minimum Area
Maximum Area
Average Area
```

before applying the size filter.

---

### NMS and Top-N Selection

```text
04_nms_and_top_n.py
```

Demonstrates how multiple detection collections can be merged:

```python
merged = sv.Detections.merge([
    detections_low,
    detections_high
])
```

Non-Maximum Suppression is then applied:

```python
after_nms = merged.with_nms(
    threshold=0.50
)
```

Finally, NumPy sorts the detections by confidence:

```python
indices_top = np.argsort(
    after_nms.confidence
)[::-1][:TOP_N]
```

This produces a workflow such as:

```text
Detection Set A
        +
Detection Set B
        ↓
      Merge
        ↓
       NMS
        ↓
Confidence Sorting
        ↓
      Top-N
```

---

### Spatial Filtering

```text
05_spatial_filtering.py
```

Demonstrates how bounding-box coordinates can be used to filter detections according to their location.

The horizontal center of each bounding box is calculated with:

```python
centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2
```

The image midpoint is calculated with:

```python
image_midpoint = image.shape[1] / 2
```

The final filter keeps detections located in the right half:

```python
right_side_detections = detections[
    centers_x > image_midpoint
]
```

This introduces the idea of **region-based detection filtering**.

---

## From Examples to Project

The five Lesson 03 examples isolate individual concepts:

```text
01_confidence_filtering.py
        ↓
02_class_and_boolean_filtering.py
        ↓
03_size_filtering.py
        ↓
04_nms_and_top_n.py
        ↓
05_spatial_filtering.py
```

These concepts are then combined into:

[Project 03 — Detection Filtering and NMS Pipeline](../05-projects/03-Detection-Filtering-and-NMS-Pipeline/)

The progression is:

```text
Small Examples
      ↓
Understand Individual Techniques
      ↓
Combine Techniques
      ↓
Complete Filtering Pipeline
      ↓
Test in Google Colab
      ↓
Generate Real Output
```

---

## Purpose

The purpose of this directory is to transform the concepts studied during the course into practical, reusable code.

Each example is designed to isolate an important concept so it can be studied independently before being incorporated into larger computer vision projects.

The repository therefore follows a progression from:

```text
Course Concept
      ↓
Notebook Experiment
      ↓
Reusable Example
      ↓
Practical Project
```

As new SAM3 sessions are completed, additional example directories will be added here.

---

## Current Progress

| # | Topic | Examples |
|---|---|---:|
| 00 | Agentic AI Programming | 2 |
| 01 | Introduction to Supervision | 8 |
| 02 | Annotation and Visualization | 5 |
| 03 | Filtering and Manipulating Detections | 5 |

**Total runnable examples: 20**

---

## Related Repository Sections

### Course Notebooks

```text
03-notebooks/
```

Contains the original Jupyter / Google Colab notebooks used during the lessons.

### Practical Projects

```text
05-projects/
```

Contains larger projects that combine multiple concepts into reusable computer vision applications.

### Course Notes

```text
08-course-notes/
```

Contains detailed explanations and documentation for each lesson.

---

## Author

**Peyman Miyandashti**

SAM3 Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
