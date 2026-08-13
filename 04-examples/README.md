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
Supervision Annotators
    ↓
Boxes / Labels / Effects
    ↓
Final Visualization
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
└── README.md
```

---

## 02 — Annotation and Visualization Examples

The newest examples focus on transforming raw object detections into clear visual results.

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

**Total runnable examples: 15**

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
