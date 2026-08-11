# Image Annotations with Supervision

## What is an Annotation?

An **annotation** is visual information added to an image to show the results produced by a computer vision model.

Annotations can include:

- Bounding boxes
- Class names
- Confidence scores
- Labels
- Tracking information
- Segmentation masks

They make model predictions easier for humans to understand.

---

## Annotation Pipeline

After YOLO detects objects, Supervision can visualize those detections.

```text
Original Image
      ↓
YOLO
      ↓
sv.Detections
      ↓
Supervision Annotators
      ↓
Annotated Image
```

---

## Box Annotator

Supervision provides a box annotator for drawing bounding boxes around detected objects.

In the course notebook:

```python
box_annotator = sv.BoxAnnotator()
```

The annotator is then applied to the image:

```python
annotated = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

The result is an image containing rectangles around the detected objects.

---

## Why Use `image.copy()`?

The course uses:

```python
image.copy()
```

instead of modifying the original image directly.

This is important because it preserves the original image.

```text
Original Image
      │
      ├── Remains unchanged
      │
      └── Copy
           ↓
       Annotation
           ↓
     Annotated Image
```

This allows us to perform additional experiments without starting from an image that already contains annotations.

---

## Label Annotator

Bounding boxes show **where** an object is located.

Labels tell us **what** was detected.

Supervision provides:

```python
label_annotator = sv.LabelAnnotator()
```

We can then apply it:

```python
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels
)
```

---

## Creating Labels

The course combines the class name with the confidence score:

```python
labels = [
    f"{results.names[class_id]} {conf:.0%}"
    for class_id, conf in zip(
        detections.class_id,
        detections.confidence
    )
]
```

This can produce labels such as:

```text
person 95%
bus 92%
person 87%
```

Each label gives us two important pieces of information:

```text
Class Name + Confidence Score
```

---

## Combining Box and Label Annotators

The complete process is:

```python
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{results.names[class_id]} {conf:.0%}"
    for class_id, conf in zip(
        detections.class_id,
        detections.confidence
    )
]

annotated = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels
)
```

---

## Visualization Pipeline

```text
                    ORIGINAL IMAGE
                          │
                          ▼
                     image.copy()
                          │
                          ▼
                    BoxAnnotator
                          │
                          ▼
                   Bounding Boxes
                          │
                          ▼
                   LabelAnnotator
                          │
                          ▼
              Class + Confidence Labels
                          │
                          ▼
                   ANNOTATED IMAGE
```

---

## Displaying the Result

The notebook uses Matplotlib to display the final image:

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
    "Pipeline completo: detección + anotación con Supervision"
)

plt.show()
```

OpenCV uses **BGR**, while Matplotlib expects **RGB**, so the image is converted with:

```python
cv2.cvtColor(
    annotated,
    cv2.COLOR_BGR2RGB
)
```

---

## Why Annotations Matter

Raw detection data may look like:

```text
class_id = 0
confidence = 0.94
xyxy = [100, 50, 400, 300]
```

This information is useful for a computer, but an annotated image is much easier for a person to understand.

```text
Raw Predictions
      ↓
Supervision
      ↓
Visual Annotations
      ↓
Human-Readable Result
```

---

## Key Concept

The model performs the detection:

```text
YOLO → Predictions
```

Supervision converts and visualizes those predictions:

```text
Predictions
     ↓
sv.Detections
     ↓
BoxAnnotator + LabelAnnotator
     ↓
Annotated Image
```

Annotations transform numerical model outputs into visual results that can be inspected and understood.
