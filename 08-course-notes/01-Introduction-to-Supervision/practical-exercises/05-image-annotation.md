# Exercise 05 — Image Annotation with Supervision

## Objective

The objective of this exercise is to transform raw YOLO detections into a visual result using **Supervision annotators**.

We will learn how to:

- Create a `BoxAnnotator`.
- Create a `LabelAnnotator`.
- Generate labels from class names and confidence scores.
- Draw bounding boxes.
- Add labels.
- Preserve the original image.
- Display the final result with Matplotlib.

---

## 1. Starting Point

From the previous exercises, we already have:

```python
image
results
detections
```

Our pipeline currently looks like:

```text
Image
  ↓
YOLO
  ↓
Ultralytics Results
  ↓
sv.Detections
```

Now we will add visualization.

---

## 2. Create the Box Annotator

Supervision provides:

```python
box_annotator = sv.BoxAnnotator()
```

The `BoxAnnotator` uses the coordinates stored in:

```python
detections.xyxy
```

to draw rectangles around detected objects.

---

## 3. Create the Label Annotator

Create another annotator:

```python
label_annotator = sv.LabelAnnotator()
```

The `LabelAnnotator` allows us to display information such as:

```text
person 95%
bus 91%
car 87%
```

---

## 4. Create the Labels

We combine the class name and confidence score:

```python
labels = [
    f"{results.names[class_id]} {conf:.0%}"
    for class_id, conf in zip(
        detections.class_id,
        detections.confidence
    )
]
```

This combines:

```text
class_id
   ↓
results.names
   ↓
Class Name
```

with:

```text
confidence
   ↓
Percentage
```

to produce:

```text
Class Name + Confidence
```

---

## 5. Understanding `zip()`

The code uses:

```python
zip(
    detections.class_id,
    detections.confidence
)
```

This allows us to process the class and confidence belonging to the same detection together.

Conceptually:

```text
class_id     confidence

   0     +      0.95
   5     +      0.91
   0     +      0.87

          ↓

person 95%
bus 91%
person 87%
```

---

## 6. Draw Bounding Boxes

Now annotate the image:

```python
annotated = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

Supervision reads the bounding-box coordinates from `detections` and draws them on the image.

---

## 7. Why Use `image.copy()`?

Notice that we use:

```python
image.copy()
```

instead of:

```python
image
```

This protects the original image.

```text
Original Image
      │
      ├────────────→ Remains Original
      │
      ▼
  image.copy()
      │
      ▼
 Box Annotation
      │
      ▼
Annotated Copy
```

This is useful when running multiple experiments on the same image.

---

## 8. Add the Labels

Now apply the label annotator:

```python
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels
)
```

The same image now contains:

```text
Bounding Boxes
      +
Class Names
      +
Confidence Scores
```

---

## 9. Annotation Sequence

The complete annotation sequence is:

```text
Original Image
      ↓
image.copy()
      ↓
BoxAnnotator
      ↓
Image + Bounding Boxes
      ↓
LabelAnnotator
      ↓
Image + Boxes + Labels
      ↓
Final Annotated Image
```

---

## 10. Display the Annotated Image

Use Matplotlib:

```python
plt.figure(figsize=(12, 7))

plt.imshow(
    cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")

plt.title(
    "YOLO + Supervision Detection"
)

plt.show()
```

---

## 11. Why Convert BGR to RGB?

Our annotated image is still an OpenCV image.

OpenCV uses:

```text
BGR
```

Matplotlib expects:

```text
RGB
```

Therefore:

```python
cv2.cvtColor(
    annotated,
    cv2.COLOR_BGR2RGB
)
```

ensures the colors are displayed correctly.

---

## 12. Complete Exercise

```python
import supervision as sv
import cv2
import matplotlib.pyplot as plt

# Create annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Create labels
labels = [
    f"{results.names[class_id]} {conf:.0%}"
    for class_id, conf in zip(
        detections.class_id,
        detections.confidence
    )
]

# Draw bounding boxes
annotated = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

# Add labels
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels
)

# Display result
plt.figure(figsize=(12, 7))

plt.imshow(
    cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")
plt.title(
    "Pipeline: YOLO + Supervision"
)
plt.show()
```

---

## 13. Complete Pipeline So Far

```text
                    IMAGE
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
          ┌───────────┴───────────┐
          ▼                       ▼
 detections.xyxy          detections.class_id
                                  +
                         detections.confidence
          │                       │
          ▼                       ▼
   BoxAnnotator             Create Labels
          │                       │
          └───────────┬───────────┘
                      ▼
               LabelAnnotator
                      │
                      ▼
               Annotated Image
                      │
                      ▼
                  Matplotlib
```

---

## Key Takeaways

The main Supervision annotation tools used in this exercise are:

```python
sv.BoxAnnotator()
sv.LabelAnnotator()
```

The complete annotation process is:

```text
Detections
    ↓
Create Labels
    ↓
Draw Bounding Boxes
    ↓
Add Labels
    ↓
Display Result
```

The final image converts numerical model predictions into a result that humans can easily inspect and understand.
