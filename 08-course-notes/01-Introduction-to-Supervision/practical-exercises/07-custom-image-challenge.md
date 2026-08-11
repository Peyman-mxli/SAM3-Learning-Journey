# Exercise 07 — Custom Image Challenge

## Objective

The objective of this final practical exercise is to apply the complete **YOLO + Supervision pipeline** to a new image.

Until now, we have worked mainly with the provided `bus.jpg` example.

Now we will independently:

- Choose another image.
- Load it with OpenCV.
- Run YOLO inference.
- Convert predictions to `sv.Detections`.
- Inspect the detections.
- Create labels.
- Annotate the image.
- Analyze the model's performance.

---

## 1. Choose a New Image

You can use your own image or download one from the internet.

The course provides the following example:

```python
import urllib.request

urllib.request.urlretrieve(
    "https://ultralytics.com/images/zidane.jpg",
    "assets/zidane.jpg"
)
```

The image will be saved as:

```text
assets/zidane.jpg
```

---

## 2. Load the Image

Load the image using OpenCV:

```python
image_custom = cv2.imread(
    "assets/zidane.jpg"
)
```

Check its dimensions:

```python
print(
    "Image shape:",
    image_custom.shape
)
```

---

## 3. Display the Original Image

Convert BGR to RGB before displaying it with Matplotlib:

```python
plt.figure(figsize=(12, 7))

plt.imshow(
    cv2.cvtColor(
        image_custom,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")
plt.title("Custom Test Image")
plt.show()
```

---

## 4. Run YOLO

Use the pretrained YOLO model:

```python
results_custom = model(
    image_custom
)[0]
```

Our pipeline is now:

```text
Custom Image
     ↓
OpenCV
     ↓
YOLO
     ↓
Predictions
```

---

## 5. Convert to `sv.Detections`

Convert the Ultralytics results:

```python
detections_custom = (
    sv.Detections.from_ultralytics(
        results_custom
    )
)
```

Now we can use the standard Supervision interface.

---

## 6. Count the Objects

Check how many objects were detected:

```python
print(
    "Objects detected:",
    len(detections_custom)
)
```

---

## 7. Inspect the Predictions

Bounding boxes:

```python
print(
    detections_custom.xyxy
)
```

Confidence scores:

```python
print(
    detections_custom.confidence
)
```

Class IDs:

```python
print(
    detections_custom.class_id
)
```

---

## 8. Inspect the Detected Classes

Translate the numerical class IDs into human-readable names:

```python
for class_id in sorted(
    set(detections_custom.class_id)
):
    print(
        f"Class {class_id}: "
        f"{results_custom.names[class_id]}"
    )
```

---

## 9. Create Labels

Create labels containing the class name and confidence:

```python
labels_custom = [
    f"{results_custom.names[class_id]} {conf:.0%}"
    for class_id, conf in zip(
        detections_custom.class_id,
        detections_custom.confidence
    )
]
```

Example:

```text
person 96%
person 92%
tie 84%
```

---

## 10. Annotate the Image

Draw the bounding boxes:

```python
annotated_custom = (
    box_annotator.annotate(
        scene=image_custom.copy(),
        detections=detections_custom
    )
)
```

Then add the labels:

```python
annotated_custom = (
    label_annotator.annotate(
        scene=annotated_custom,
        detections=detections_custom,
        labels=labels_custom
    )
)
```

---

## 11. Display the Final Result

```python
plt.figure(figsize=(12, 7))

plt.imshow(
    cv2.cvtColor(
        annotated_custom,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")

plt.title(
    "Custom Image — YOLO + Supervision"
)

plt.show()
```

---

## 12. Complete Challenge Code

```python
import urllib.request
import cv2
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLO

# Download image
urllib.request.urlretrieve(
    "https://ultralytics.com/images/zidane.jpg",
    "assets/zidane.jpg"
)

# Load image
image_custom = cv2.imread(
    "assets/zidane.jpg"
)

# Load YOLO
model = YOLO("yolov8n.pt")

# Run inference
results_custom = model(
    image_custom
)[0]

# Convert to Supervision
detections_custom = (
    sv.Detections.from_ultralytics(
        results_custom
    )
)

# Create annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Create labels
labels_custom = [
    f"{results_custom.names[class_id]} {conf:.0%}"
    for class_id, conf in zip(
        detections_custom.class_id,
        detections_custom.confidence
    )
]

# Draw boxes
annotated_custom = (
    box_annotator.annotate(
        scene=image_custom.copy(),
        detections=detections_custom
    )
)

# Add labels
annotated_custom = (
    label_annotator.annotate(
        scene=annotated_custom,
        detections=detections_custom,
        labels=labels_custom
    )
)

# Display result
plt.figure(figsize=(12, 7))

plt.imshow(
    cv2.cvtColor(
        annotated_custom,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")
plt.title(
    "Custom Image — YOLO + Supervision"
)
plt.show()
```

---

# Analysis

After running the experiment, do not only look at the final image.

Analyze the model's behavior.

## Questions

### 1. How many objects were detected?

Use:

```python
len(detections_custom)
```

### 2. Which objects were detected?

Inspect:

```python
detections_custom.class_id
```

and:

```python
results_custom.names
```

### 3. Which detection had the highest confidence?

Inspect:

```python
detections_custom.confidence
```

### 4. Did YOLO miss any visible objects?

If an object exists but YOLO does not detect it, this is a:

```text
False Negative
```

### 5. Did YOLO detect something incorrectly?

If YOLO predicts an object that is not actually present, this is a:

```text
False Positive
```

---

# Final Challenge Pipeline

```text
                    CUSTOM IMAGE
                         │
                         ▼
                       OpenCV
                         │
                         ▼
                        YOLO
                         │
                         ▼
                Ultralytics Results
                         │
                         ▼
                   sv.Detections
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        xyxy         confidence      class_id
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                      Analyze
                         │
                         ▼
                  Create Labels
                         │
                         ▼
                    Annotators
                         │
                         ▼
                  Annotated Image
                         │
                         ▼
                  Evaluate Result
```

---

## Key Takeaway

The most important result of this challenge is understanding that the same pipeline can be reused with different images:

```text
ANY IMAGE
    ↓
YOLO
    ↓
sv.Detections
    ↓
Analyze
    ↓
Annotate
    ↓
Result
```

We are no longer simply running the provided example.

We can now independently apply the basic **YOLO + Supervision computer vision workflow** to a new image and evaluate the model's predictions.
