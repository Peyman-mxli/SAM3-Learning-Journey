# Exercise 01 — Basic Box Annotation

## Objective

The objective of this exercise is to practice the basic object-detection and annotation workflow using **YOLOv8**, **OpenCV**, and **Supervision**.

In this exercise, an image will be processed by YOLOv8 and the detected objects will be visualized using Supervision's `BoxAnnotator`.

---

## Concepts Practiced

This exercise reinforces the following concepts:

- Loading images with OpenCV
- Loading a YOLOv8 model
- Running object detection
- Using a confidence threshold
- Converting YOLO results to `sv.Detections`
- Creating a `BoxAnnotator`
- Drawing bounding boxes
- Saving an annotated image

---

## Expected Workflow

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
BoxAnnotator
     ↓
Annotated Image
     ↓
Saved Output
```

---

## Step 1 — Install Dependencies

Install the required libraries:

```bash
pip install ultralytics supervision opencv-python
```

The main libraries used are:

| Library | Purpose |
|---|---|
| Ultralytics | Load and run YOLOv8 |
| Supervision | Manage detections and annotations |
| OpenCV | Load and save images |

---

## Step 2 — Import the Libraries

```python
import cv2
import supervision as sv

from ultralytics import YOLO
```

---

## Step 3 — Load the Image

Define the path to the input image:

```python
IMAGE_PATH = "image.jpg"
```

Load the image using OpenCV:

```python
image = cv2.imread(IMAGE_PATH)
```

Verify that the image was loaded correctly:

```python
if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )
```

---

## Step 4 — Load YOLOv8

Load the YOLOv8 Nano model:

```python
model = YOLO("yolov8n.pt")
```

`yolov8n.pt` is the Nano version of YOLOv8.

It is useful for learning and experimentation because it is lightweight and fast.

---

## Step 5 — Run Object Detection

Run the model on the image:

```python
results = model(
    image,
    conf=0.50
)[0]
```

The value:

```python
conf=0.50
```

defines the confidence threshold.

Detections below this threshold are filtered out.

---

## Step 6 — Convert Results to Supervision

Convert the YOLO results into a Supervision `Detections` object:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

The `detections` object provides structured information about the detected objects.

It may contain:

- Bounding box coordinates
- Class IDs
- Confidence scores

---

## Step 7 — Create the BoxAnnotator

Create a Supervision `BoxAnnotator`:

```python
box_annotator = sv.BoxAnnotator(
    thickness=3
)
```

The Annotator is responsible for drawing bounding boxes around detected objects.

---

## Step 8 — Annotate the Image

Apply the bounding boxes:

```python
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

The original image is copied before annotations are added:

```python
image.copy()
```

This keeps the original image unchanged.

---

## Step 9 — Save the Result

Save the annotated image:

```python
OUTPUT_PATH = "annotated_image.jpg"

success = cv2.imwrite(
    OUTPUT_PATH,
    annotated_image
)
```

Verify that the image was saved correctly:

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
OUTPUT_PATH = "annotated_image.jpg"
CONFIDENCE_THRESHOLD = 0.50


# Load image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


# Load YOLO model
model = YOLO("yolov8n.pt")


# Run object detection
results = model(
    image,
    conf=CONFIDENCE_THRESHOLD
)[0]


# Convert YOLO results to Supervision
detections = sv.Detections.from_ultralytics(
    results
)


# Create annotator
box_annotator = sv.BoxAnnotator(
    thickness=3
)


# Annotate image
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
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
    f"Annotated image saved to: {OUTPUT_PATH}"
)
```

---

## Expected Result

After running the code, the directory should contain:

```text
image.jpg
annotated_image.jpg
```

The output image should contain bounding boxes around the objects detected by YOLOv8.

Example:

```text
Original Image
      ↓
YOLO Detection
      ↓
┌──────────────────┐
│ Detected Object  │
└──────────────────┘
      ↓
annotated_image.jpg
```

---

## Experiment 1 — Change the Confidence Threshold

Change:

```python
CONFIDENCE_THRESHOLD = 0.50
```

to:

```python
CONFIDENCE_THRESHOLD = 0.25
```

Run the program again.

Observe whether more objects are detected.

Then try:

```python
CONFIDENCE_THRESHOLD = 0.75
```

Compare the results.

### Question

What happens when the confidence threshold increases?

Think about the relationship:

```text
Lower Threshold
      ↓
More detections
      ↓
Potentially more false positives

Higher Threshold
      ↓
Fewer detections
      ↓
Only higher-confidence predictions
```

---

## Experiment 2 — Change Box Thickness

Try:

```python
box_annotator = sv.BoxAnnotator(
    thickness=1
)
```

Then:

```python
box_annotator = sv.BoxAnnotator(
    thickness=5
)
```

Compare the visualization.

This demonstrates that annotation appearance can be customized without changing the underlying detection results.

---

## Challenge

Modify the exercise so that labels are also displayed.

Create:

```python
label_annotator = sv.LabelAnnotator()
```

Then generate labels containing:

```text
class name + confidence
```

Example:

```text
person 92%
car 87%
dog 81%
```

Apply the `LabelAnnotator` after the `BoxAnnotator`.

The final pipeline should become:

```text
Input Image
     ↓
YOLOv8
     ↓
sv.Detections
     ↓
BoxAnnotator
     ↓
LabelAnnotator
     ↓
Final Image
```

---

## Questions for Review

1. What is the purpose of YOLOv8 in this exercise?
2. What information is stored inside `sv.Detections`?
3. What does `BoxAnnotator` do?
4. Why is `image.copy()` used?
5. What does the confidence threshold control?
6. What happens when the confidence threshold is reduced?
7. Does changing the box thickness change the YOLO predictions?
8. Why is detection separated from visualization?

---

## Key Takeaway

YOLO and Supervision perform different responsibilities.

```text
YOLO
 ↓
Detect Objects
 ↓
Detection Data
 ↓
Supervision
 ↓
Visualize Detections
```

YOLO determines **what and where objects are**, while Supervision determines **how those detections are visualized**.

This separation makes computer vision pipelines easier to customize and reuse.
