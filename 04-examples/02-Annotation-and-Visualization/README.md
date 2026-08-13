# Annotation and Visualization — Examples

This folder contains practical Python examples from the **Annotation and Visualization with Supervision** lesson.

The examples demonstrate how YOLO detections can be visualized and customized using different Supervision Annotators.

---

## Examples

| # | File | Description |
|---|---|---|
| 01 | `01_box_and_label.py` | Basic bounding boxes and detection labels |
| 02 | `02_compare_annotators.py` | Compare different Supervision Annotators |
| 03 | `03_customize_visualization.py` | Customize colors, thickness, and text scale |
| 04 | `04_layer_order.py` | Compare Box → Label and Label → Box layer order |
| 05 | `05_custom_composition.py` | Combine multiple Annotators into a custom visualization |

---

## 1. Box and Label

File:

```text
01_box_and_label.py
```

Introduces the standard object detection visualization pipeline:

```text
Input Image
     ↓
YOLOv8
     ↓
Detections
     ↓
BoxAnnotator
     ↓
LabelAnnotator
     ↓
Annotated Image
```

The example demonstrates:

- Loading YOLOv8
- Running object detection
- Converting YOLO results to `sv.Detections`
- Creating labels from class names and confidence scores
- Drawing bounding boxes
- Adding labels
- Saving the annotated image

---

## 2. Comparing Annotators

File:

```text
02_compare_annotators.py
```

Compares several visualization methods provided by Supervision:

```python
sv.BoxAnnotator()
sv.RoundBoxAnnotator()
sv.HaloAnnotator()
sv.BlurAnnotator()
sv.BoxCornerAnnotator()
```

The purpose of this example is to demonstrate that the same detections can be represented using different visual styles.

---

## 3. Customizing Visualization

File:

```text
03_customize_visualization.py
```

Explores customization options such as:

```python
color=
thickness=
text_scale=
```

It also demonstrates predefined Supervision colors:

```python
sv.Color.RED
sv.Color.GREEN
```

and the default color palette:

```python
sv.ColorPalette.DEFAULT
```

This makes it possible to adapt visualization styles to different images and applications.

---

## 4. Annotation Layer Order

File:

```text
04_layer_order.py
```

Demonstrates why Annotator order matters.

The example compares:

```text
Box → Label
```

with:

```text
Label → Box
```

The recommended order is:

```text
Original Image
      ↓
BoxAnnotator
      ↓
LabelAnnotator
      ↓
Final Visualization
```

Because the label is applied last, it remains visually on top of the bounding box and is generally easier to read.

---

## 5. Custom Annotator Composition

File:

```text
05_custom_composition.py
```

Demonstrates how several Annotators can be combined into a single visualization pipeline.

The example uses:

```python
sv.EllipseAnnotator()
sv.DotAnnotator()
sv.LabelAnnotator()
```

Pipeline:

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

This demonstrates one of the central concepts of the lesson:

> Annotators can be composed as visual layers.

---

## Requirements

Install the required libraries with:

```bash
pip install supervision ultralytics matplotlib opencv-python
```

Main technologies:

- Python
- YOLOv8
- Ultralytics
- Supervision
- OpenCV
- Matplotlib

---

## Running the Examples

Place an image named:

```text
image.jpg
```

in the working directory.

Then run an example:

```bash
python 01_box_and_label.py
```

or:

```bash
python 02_compare_annotators.py
```

The examples that use Matplotlib will display their results in a visualization window or notebook environment.

---

## Related Course Material

Course notes:

```text
08-course-notes/02-Annotation-and-Visualization/
```

Lesson notebook:

```text
03-notebooks/01_b_anotacion_visualizacion.ipynb
```

These examples provide reusable Python implementations of the concepts explored in the lesson notebook.

---

## Key Concept

```text
YOLO
  ↓
Object Detection
  ↓
sv.Detections
  ↓
Supervision Annotators
  ↓
Custom Visualization
```

YOLO determines **what and where objects are**.

Supervision Annotators determine **how those detections are visually presented**.

---

## Author

**Peyman Miyandashti**

SAM3 Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
