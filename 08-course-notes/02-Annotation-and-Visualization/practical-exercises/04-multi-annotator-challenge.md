# Exercise 04 — Multi-Annotator Challenge

## Objective

The objective of this final exercise is to combine multiple **Supervision Annotators** into a single layered computer vision visualization pipeline.

This exercise brings together the concepts practiced throughout the Annotation and Visualization lesson.

You will combine:

- YOLOv8 object detection
- `sv.Detections`
- Confidence scores
- Custom labels
- `BoxAnnotator`
- `EllipseAnnotator`
- `DotAnnotator`
- `LabelAnnotator`
- Annotation customization
- Annotation layer ordering

---

## Final Challenge

Build a visualization pipeline with at least four annotation layers:

```text
Input Image
     ↓
YOLOv8
     ↓
Detection Results
     ↓
sv.Detections
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

The goal is to understand how multiple Annotators can work with the same detection data.

---

## Step 1 — Import the Libraries

```python
import cv2
import supervision as sv

from ultralytics import YOLO
```

---

## Step 2 — Configure the Pipeline

```python
MODEL_NAME = "yolov8n.pt"

IMAGE_PATH = "image.jpg"
OUTPUT_PATH = "multi_annotator_result.jpg"

CONFIDENCE_THRESHOLD = 0.50
```

These values control:

- YOLO model
- Input image
- Output image
- Detection confidence threshold

---

## Step 3 — Load the Image

```python
image = cv2.imread(IMAGE_PATH)
```

Verify that the image was loaded:

```python
if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )
```

---

## Step 4 — Load YOLOv8

```python
model = YOLO(MODEL_NAME)
```

YOLO will perform the object-detection stage of the pipeline.

---

## Step 5 — Run Object Detection

```python
results = model(
    image,
    conf=CONFIDENCE_THRESHOLD
)[0]
```

The model produces the raw detection results.

---

## Step 6 — Convert Results to Supervision

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

Now the YOLO predictions can be used by Supervision Annotators.

The same `detections` object will be reused throughout the visualization pipeline.

---

## Step 7 — Create Detection Labels

Create labels containing the class name and confidence score:

```python
labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]
```

Example:

```text
person 92%
car 87%
bus 81%
dog 94%
```

---

# Building the Visualization Layers

## Layer 1 — BoxAnnotator

Create the bounding-box Annotator:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)
```

Apply it:

```python
scene = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

The first visualization layer is now complete.

---

## Layer 2 — EllipseAnnotator

Create:

```python
ellipse_annotator = sv.EllipseAnnotator()
```

Apply it to the result of the previous layer:

```python
scene = ellipse_annotator.annotate(
    scene=scene,
    detections=detections
)
```

The image now contains:

```text
Bounding Boxes
+
Ellipses
```

---

## Layer 3 — DotAnnotator

Create:

```python
dot_annotator = sv.DotAnnotator()
```

Apply it:

```python
scene = dot_annotator.annotate(
    scene=scene,
    detections=detections
)
```

The visualization now contains:

```text
Bounding Boxes
+
Ellipses
+
Detection Points
```

---

## Layer 4 — LabelAnnotator

Create:

```python
label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)
```

Apply the labels last:

```python
scene = label_annotator.annotate(
    scene=scene,
    detections=detections,
    labels=labels
)
```

The final visualization now contains:

```text
Bounding Boxes
+
Ellipses
+
Detection Points
+
Class Labels
+
Confidence Scores
```

---

# Why Labels Are Applied Last

The order of annotation layers matters.

For example:

```text
Box
 ↓
Ellipse
 ↓
Dot
 ↓
Label
```

places the textual information on top of the previous visualization layers.

This generally makes labels easier to read.

If another visual Annotator is applied after the labels, its graphics may overlap some of the text.

---

# Complete Challenge Code

```python
import cv2
import supervision as sv

from ultralytics import YOLO


MODEL_NAME = "yolov8n.pt"

IMAGE_PATH = "image.jpg"
OUTPUT_PATH = "multi_annotator_result.jpg"

CONFIDENCE_THRESHOLD = 0.50


# Load image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


# Load YOLO model
model = YOLO(MODEL_NAME)


# Run object detection
results = model(
    image,
    conf=CONFIDENCE_THRESHOLD
)[0]


# Convert YOLO results to Supervision
detections = sv.Detections.from_ultralytics(
    results
)


# Create labels
labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]


# Create Annotators
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)

ellipse_annotator = sv.EllipseAnnotator()

dot_annotator = sv.DotAnnotator()

label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)


# Layer 1 — Boxes
scene = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)


# Layer 2 — Ellipses
scene = ellipse_annotator.annotate(
    scene=scene,
    detections=detections
)


# Layer 3 — Dots
scene = dot_annotator.annotate(
    scene=scene,
    detections=detections
)


# Layer 4 — Labels
scene = label_annotator.annotate(
    scene=scene,
    detections=detections,
    labels=labels
)


# Save result
success = cv2.imwrite(
    OUTPUT_PATH,
    scene
)

if not success:
    raise RuntimeError(
        f"Could not save image: {OUTPUT_PATH}"
    )


print(
    f"Detected objects: {len(detections)}"
)

print(
    f"Final visualization saved to: {OUTPUT_PATH}"
)
```

---

# Understanding the Pipeline

The most important part of this exercise is the reuse of:

```python
detections
```

The same detection information is passed to every Annotator.

```text
                    ┌── BoxAnnotator
                    │
                    ├── EllipseAnnotator
                    │
sv.Detections ──────┼── DotAnnotator
                    │
                    └── LabelAnnotator
```

YOLO does not need to run again for each visualization style.

---

# Understanding `scene`

The variable:

```python
scene
```

contains the current version of the annotated image.

The process works like this:

```text
image.copy()
     ↓
scene
     ↓
Add Box
     ↓
scene
     ↓
Add Ellipse
     ↓
scene
     ↓
Add Dot
     ↓
scene
     ↓
Add Label
     ↓
Final scene
```

Each step builds on the previous step.

---

# Experiment 1 — Change the Layer Order

Try this order:

```text
Label
 ↓
Box
 ↓
Ellipse
 ↓
Dot
```

Compare it with:

```text
Box
 ↓
Ellipse
 ↓
Dot
 ↓
Label
```

Observe how the final visualization changes.

### Question

Which order produces the clearest result?

---

# Experiment 2 — Remove One Layer

Remove:

```python
ellipse_annotator
```

Run the pipeline again.

Then restore it and remove:

```python
dot_annotator
```

Compare the results.

This demonstrates that visualization layers can be added or removed independently.

---

# Experiment 3 — Change Box Thickness

Try:

```python
thickness=1
```

Then:

```python
thickness=5
```

Observe how the bounding-box appearance changes while the detections remain the same.

---

# Experiment 4 — Change Label Size

Try:

```python
text_scale=0.3
```

Then:

```python
text_scale=1.0
```

Compare the readability of the final visualization.

---

# Experiment 5 — Use a Different Annotator

Supervision provides additional Annotators.

Try replacing one visualization layer with:

```python
sv.RoundBoxAnnotator()
```

or:

```python
sv.BoxCornerAnnotator()
```

The pipeline could become:

```text
YOLO
 ↓
sv.Detections
 ↓
RoundBoxAnnotator
 ↓
DotAnnotator
 ↓
LabelAnnotator
 ↓
Final Visualization
```

This demonstrates how the visualization pipeline can be customized without changing the detection model.

---

# Extension Challenge

Create your own visualization combination using at least **three different Annotators**.

For example:

```python
scene = image.copy()

scene = sv.RoundBoxAnnotator().annotate(
    scene=scene,
    detections=detections
)

scene = sv.DotAnnotator().annotate(
    scene=scene,
    detections=detections
)

scene = sv.LabelAnnotator().annotate(
    scene=scene,
    detections=detections,
    labels=labels
)
```

Your goal is to create a visualization that is:

- Clear
- Readable
- Informative
- Visually organized

---

# Expected Result

After successful execution, the program should generate:

```text
multi_annotator_result.jpg
```

The output should contain multiple visualization layers applied to the same YOLO detections.

---

# From Exercise to Project

This exercise demonstrates the core idea behind the larger project:

```text
05-projects/
└── 02-Multi-Annotator-Visualization-Pipeline/
```

The learning progression is:

```text
Concept
   ↓
Basic Exercise
   ↓
Label Exercise
   ↓
Customization Exercise
   ↓
Multi-Annotator Challenge
   ↓
Reusable Project
```

The project takes these individual lesson concepts and organizes them into a complete Python application.

---

## Questions for Review

1. Why can multiple Annotators use the same `sv.Detections` object?
2. Why does annotation order matter?
3. What is the purpose of the `scene` variable?
4. Why is `image.copy()` used at the beginning?
5. What does `BoxAnnotator` add?
6. What does `EllipseAnnotator` add?
7. What does `DotAnnotator` add?
8. What does `LabelAnnotator` add?
9. Does adding more Annotators change YOLO's predictions?
10. Why is it useful to separate object detection from visualization?

---

# Key Takeaway

A single set of object detections can support many different visualization styles.

```text
YOLO
  ↓
One Detection Result
  ↓
sv.Detections
  ↓
Multiple Annotators
  ↓
Multiple Visual Layers
  ↓
Final Annotated Image
```

The detection model answers:

```text
What is in the image?
Where is it?
How confident is the prediction?
```

The visualization pipeline answers:

```text
How should those detections be displayed?
```

This separation makes computer vision applications more flexible, reusable, and easier to customize.
