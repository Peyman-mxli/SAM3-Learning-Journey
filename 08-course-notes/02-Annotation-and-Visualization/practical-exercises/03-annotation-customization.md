# Exercise 03 — Annotation Customization

## Objective

The objective of this exercise is to practice customizing the appearance of object-detection visualizations using **Supervision Annotators**.

The previous exercises introduced:

- `BoxAnnotator`
- `LabelAnnotator`
- Class labels
- Confidence scores

This exercise focuses on changing how those annotations appear without changing the underlying YOLO detections.

---

## Concepts Practiced

This exercise focuses on:

- `BoxAnnotator`
- `LabelAnnotator`
- `ColorPalette`
- Bounding-box thickness
- Label text scale
- Custom detection labels
- Visualization readability
- Separating detection from visualization

---

## Important Concept

Changing an Annotator does **not** change the YOLO prediction.

For example:

```text
YOLO Detection
      ↓
sv.Detections
      ↓
Same Detection Data
      ↓
Different Visualization Settings
      ↓
Different Appearance
```

The following information remains unchanged:

- Bounding-box coordinates
- Class IDs
- Confidence scores
- Number of accepted detections

Only the visualization changes.

---

## Step 1 — Import the Libraries

```python
import cv2
import supervision as sv

from ultralytics import YOLO
```

---

## Step 2 — Configure the Exercise

Define the main configuration:

```python
MODEL_NAME = "yolov8n.pt"

IMAGE_PATH = "image.jpg"
OUTPUT_PATH = "customized_image.jpg"

CONFIDENCE_THRESHOLD = 0.50
```

Keeping configuration values near the beginning of the program makes the experiment easier to modify.

---

## Step 3 — Load the Image

Load the image using OpenCV:

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

Create the model:

```python
model = YOLO(MODEL_NAME)
```

---

## Step 5 — Run Object Detection

Run inference:

```python
results = model(
    image,
    conf=CONFIDENCE_THRESHOLD
)[0]
```

The confidence threshold determines which predictions are accepted.

---

## Step 6 — Convert to Supervision

Convert the YOLO results into `sv.Detections`:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

The detection information is now ready for visualization.

---

## Step 7 — Create Labels

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
```

---

# Customizing BoxAnnotator

## Basic BoxAnnotator

A basic Annotator can be created with:

```python
box_annotator = sv.BoxAnnotator()
```

Supervision will use its default visualization settings.

---

## Custom Box Thickness

The bounding-box line thickness can be changed:

```python
box_annotator = sv.BoxAnnotator(
    thickness=3
)
```

Conceptually:

```text
thickness=1
     ↓
Thin Boxes

thickness=3
     ↓
Medium Boxes

thickness=5
     ↓
Thicker Boxes
```

The best value depends on the image resolution and intended visualization.

---

## Using a Color Palette

A color palette can be provided to the Annotator:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)
```

A palette can help make detections visually distinguishable.

Conceptually:

```text
Detection Data
      ↓
Color Palette
      ↓
Visual Color Assignment
      ↓
Annotated Image
```

---

# Customizing LabelAnnotator

## Basic LabelAnnotator

Create a basic label Annotator:

```python
label_annotator = sv.LabelAnnotator()
```

---

## Custom Text Scale

The label size can be changed using:

```python
label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)
```

Try different values:

```python
text_scale=0.3
```

```python
text_scale=0.6
```

```python
text_scale=1.0
```

Larger values create larger text.

---

## Choosing a Good Text Size

A good label size should be:

- Large enough to read
- Small enough not to hide the object
- Appropriate for the image resolution
- Consistent across the visualization

The goal is readability.

---

# Combining Customized Annotators

Create both Annotators:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)

label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)
```

Apply the bounding boxes first:

```python
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

Then add the labels:

```python
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)
```

The pipeline becomes:

```text
Original Image
      ↓
YOLO Detection
      ↓
sv.Detections
      ↓
Custom BoxAnnotator
      ↓
Custom LabelAnnotator
      ↓
Final Visualization
```

---

# Complete Exercise Code

```python
import cv2
import supervision as sv

from ultralytics import YOLO


MODEL_NAME = "yolov8n.pt"

IMAGE_PATH = "image.jpg"
OUTPUT_PATH = "customized_image.jpg"

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


# Create customized Annotators
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)

label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)


# Apply bounding boxes
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)


# Apply labels
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)


# Save result
success = cv2.imwrite(
    OUTPUT_PATH,
    annotated_image
)

if not success:
    raise RuntimeError(
        f"Could not save image: {OUTPUT_PATH}"
    )


print(
    f"Detected objects: {len(detections)}"
)

print(
    f"Customized image saved to: {OUTPUT_PATH}"
)
```

---

# Experiment 1 — Compare Box Thickness

Run the program three times.

### Version A

```python
thickness=1
```

### Version B

```python
thickness=3
```

### Version C

```python
thickness=5
```

Compare the results.

Ask:

- Which is easiest to see?
- Does a thick box hide image details?
- Does a thin box become difficult to see?

---

# Experiment 2 — Compare Text Scale

Try:

```python
text_scale=0.3
```

Then:

```python
text_scale=0.6
```

Then:

```python
text_scale=1.0
```

Compare the readability of the labels.

---

# Experiment 3 — Change Confidence Threshold

Try:

```python
CONFIDENCE_THRESHOLD = 0.25
```

Then:

```python
CONFIDENCE_THRESHOLD = 0.50
```

Then:

```python
CONFIDENCE_THRESHOLD = 0.75
```

This experiment is different from changing annotation appearance.

Changing:

```python
thickness
```

or:

```python
text_scale
```

changes only the visualization.

Changing:

```python
CONFIDENCE_THRESHOLD
```

can change which detections are accepted.

---

## Important Difference

```text
Thickness
Text Scale
Color Palette
      ↓
Visualization Settings
      ↓
Detection Data Does Not Change
```

But:

```text
Confidence Threshold
      ↓
Detection Filtering
      ↓
Accepted Detections May Change
```

This distinction is important when designing computer vision pipelines.

---

# Experiment 4 — Class Name Only

Remove the confidence score from the labels.

Use:

```python
labels = [
    results.names[class_id]
    for class_id in detections.class_id
]
```

Instead of:

```text
person 92%
```

the result becomes:

```text
person
```

Compare both visualization styles.

---

# Challenge

Create two different visualization styles using the **same YOLO detections**.

## Style A

Use:

```python
thickness=1
text_scale=0.4
```

## Style B

Use:

```python
thickness=5
text_scale=1.0
```

Do not run YOLO twice.

Use the same:

```python
detections
```

object to create both visualizations.

Conceptually:

```text
                    ┌─ Style A
                    │
YOLO → Detections ──┤
                    │
                    └─ Style B
```

This demonstrates that detection data and visualization configuration are independent.

---

## Expected Result

The exercise should generate:

```text
customized_image.jpg
```

The image should contain:

- Detected objects
- Customized bounding boxes
- Class names
- Confidence scores
- Customized label text

---

## Questions for Review

1. What does `BoxAnnotator` control?
2. What does `LabelAnnotator` control?
3. What does `thickness` change?
4. What does `text_scale` change?
5. What is the purpose of `ColorPalette`?
6. Does changing box thickness change the YOLO prediction?
7. Does changing text scale change confidence scores?
8. Why can the same `detections` object be used with different Annotators?
9. How is changing the confidence threshold different from changing annotation appearance?
10. Why is visualization readability important?

---

## Key Takeaway

Annotation customization changes **how detections are displayed**, not what YOLO detected.

```text
YOLO
  ↓
Detection Data
  ↓
sv.Detections
  ↓
Visualization Configuration
  ↓
Customized Annotators
  ↓
Final Image
```

This separation allows developers to experiment with different visualization styles without rerunning or modifying the underlying object-detection model.
