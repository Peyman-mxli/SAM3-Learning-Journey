# Exercise 02 — Label Annotation

## Objective

The objective of this exercise is to extend basic object detection by adding **class names and confidence scores** to detected objects using Supervision's `LabelAnnotator`.

In the previous exercise, detections were visualized using bounding boxes.

Now we will combine:

```text
BoxAnnotator
+
LabelAnnotator
```

to create a more informative visualization.

---

## Concepts Practiced

This exercise focuses on:

- YOLOv8 object detection
- `sv.Detections`
- Detection class IDs
- Confidence scores
- Creating custom labels
- `BoxAnnotator`
- `LabelAnnotator`
- Text scale customization
- Combining annotation layers
- Annotation order

---

## Expected Workflow

```text
Input Image
     ↓
YOLOv8
     ↓
Detection Results
     ↓
sv.Detections
     ↓
Create Labels
     ↓
BoxAnnotator
     ↓
LabelAnnotator
     ↓
Final Visualization
```

---

## Step 1 — Import the Libraries

```python
import cv2
import supervision as sv

from ultralytics import YOLO
```

---

## Step 2 — Load the Image

Define the input image:

```python
IMAGE_PATH = "image.jpg"
```

Load it with OpenCV:

```python
image = cv2.imread(IMAGE_PATH)
```

Verify that the image exists:

```python
if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )
```

---

## Step 3 — Load YOLOv8

Load the YOLOv8 Nano model:

```python
model = YOLO("yolov8n.pt")
```

---

## Step 4 — Run Object Detection

Run YOLO with a confidence threshold:

```python
results = model(
    image,
    conf=0.50
)[0]
```

YOLO returns information about the detected objects.

This includes:

- Bounding boxes
- Class IDs
- Confidence scores

---

## Step 5 — Convert Results to `sv.Detections`

Convert the Ultralytics results:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

Now the detection information can be easily used with Supervision Annotators.

---

## Step 6 — Inspect Class IDs

Each detected object has a class ID.

For example:

```python
print(detections.class_id)
```

A possible result could look like:

```text
[0 0 5 2]
```

Each number represents a YOLO object class.

YOLO provides the class names through:

```python
results.names
```

For example:

```text
0 → person
2 → car
5 → bus
```

---

## Step 7 — Inspect Confidence Scores

Confidence scores can be accessed using:

```python
print(detections.confidence)
```

A possible result could look like:

```text
[0.91 0.87 0.82 0.76]
```

These values represent how confident the model is about each prediction.

---

## Step 8 — Create Labels

Now combine the class name and confidence score.

```python
labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]
```

The result may look like:

```text
person 91%
person 87%
bus 82%
car 76%
```

---

## Understanding `zip()`

The code:

```python
zip(
    detections.class_id,
    detections.confidence
)
```

pairs each class ID with its corresponding confidence score.

Conceptually:

```text
class_id     confidence
   0    +       0.91
   0    +       0.87
   5    +       0.82
   2    +       0.76
```

Then:

```python
results.names[class_id]
```

converts the class ID into a readable class name.

---

## Understanding Confidence Formatting

This expression:

```python
{confidence:.0%}
```

converts a decimal confidence value into a percentage.

For example:

```text
0.91 → 91%
0.87 → 87%
0.76 → 76%
```

This makes the labels easier to read.

---

## Step 9 — Create BoxAnnotator

Create the bounding-box Annotator:

```python
box_annotator = sv.BoxAnnotator(
    thickness=3
)
```

---

## Step 10 — Create LabelAnnotator

Create the label Annotator:

```python
label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)
```

The `text_scale` parameter controls the size of the label text.

---

## Step 11 — Apply Bounding Boxes

First apply the bounding boxes:

```python
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

The visualization now contains bounding boxes.

---

## Step 12 — Apply Labels

Next, apply the labels to the image already containing the boxes:

```python
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)
```

The visualization now contains:

```text
Bounding Boxes
+
Class Names
+
Confidence Scores
```

---

## Why Annotation Order Matters

The recommended order is:

```text
BoxAnnotator
     ↓
LabelAnnotator
```

This means the labels are drawn after the bounding boxes.

Conceptually:

```text
Original Image
      ↓
Draw Boxes
      ↓
Draw Labels on Top
      ↓
Final Image
```

If another Annotator is applied after the labels, its graphics may overlap the text.

---

## Step 13 — Save the Result

Define the output path:

```python
OUTPUT_PATH = "labeled_image.jpg"
```

Save the result:

```python
success = cv2.imwrite(
    OUTPUT_PATH,
    annotated_image
)
```

Verify the result:

```python
if not success:
    raise RuntimeError(
        f"Could not save image: {OUTPUT_PATH}"
    )
```

---

## Complete Exercise Code

```python
import cv2
import supervision as sv

from ultralytics import YOLO


IMAGE_PATH = "image.jpg"
OUTPUT_PATH = "labeled_image.jpg"

CONFIDENCE_THRESHOLD = 0.50


# Load image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


# Load YOLO
model = YOLO("yolov8n.pt")


# Run detection
results = model(
    image,
    conf=CONFIDENCE_THRESHOLD
)[0]


# Convert results to Supervision
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
    f"Labeled image saved to: {OUTPUT_PATH}"
)
```

---

## Expected Result

The final visualization should contain bounding boxes with labels similar to:

```text
┌──────────────────────┐
│ person 91%           │
│                      │
│    Detected Object   │
│                      │
└──────────────────────┘
```

The output file will be:

```text
labeled_image.jpg
```

---

## Experiment 1 — Change Text Scale

Start with:

```python
text_scale=0.3
```

Then try:

```python
text_scale=0.6
```

Finally:

```python
text_scale=1.0
```

Compare the results.

### Question

Which text size is easiest to read without covering too much of the detected object?

The answer may depend on:

- Image resolution
- Object size
- Number of detections
- Intended display size

---

## Experiment 2 — Remove Confidence Scores

Change:

```python
labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]
```

to:

```python
labels = [
    results.names[class_id]
    for class_id in detections.class_id
]
```

Now compare:

```text
person 91%
```

with:

```text
person
```

Consider which format is more useful for debugging and model evaluation.

---

## Experiment 3 — Change the Annotation Order

Try applying `LabelAnnotator` first:

```python
annotated_image = label_annotator.annotate(
    scene=image.copy(),
    detections=detections,
    labels=labels
)
```

Then apply `BoxAnnotator`:

```python
annotated_image = box_annotator.annotate(
    scene=annotated_image,
    detections=detections
)
```

Compare this with:

```text
Box → Label
```

Observe whether the visualization becomes less readable.

---

## Challenge

Modify the labels so they include the class ID.

Instead of:

```text
person 91%
```

create:

```text
ID:0 | person | 91%
```

A possible solution structure is:

```python
labels = [
    f"ID:{class_id} | {results.names[class_id]} | {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]
```

Run the pipeline and inspect the result.

---

## Questions for Review

1. What is the purpose of `LabelAnnotator`?
2. What does `detections.class_id` contain?
3. What does `detections.confidence` contain?
4. What does `results.names[class_id]` return?
5. Why is `zip()` used when creating labels?
6. What does `:.0%` do?
7. Why should labels usually be applied after bounding boxes?
8. Does changing `text_scale` modify the YOLO detections?

---

## Key Takeaway

Bounding boxes show **where** objects are located, while labels explain **what** was detected and how confident the model is.

```text
YOLO Detection
      ↓
sv.Detections
      ↓
Bounding Box
      +
Class Name
      +
Confidence Score
      ↓
Clear Visualization
```

By combining `BoxAnnotator` and `LabelAnnotator`, raw detection results become much easier to interpret.
