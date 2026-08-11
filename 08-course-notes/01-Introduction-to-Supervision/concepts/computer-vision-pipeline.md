# Computer Vision Pipeline

## What is a Computer Vision Pipeline?

A **computer vision pipeline** is a sequence of steps that transforms an input image into useful information.

In this lesson, the central pipeline is:

```text
Image
  ↓
YOLO
  ↓
sv.Detections
  ↓
Annotated Image
```

Each component has a specific responsibility.

---

## Step 1 — Input Image

The process begins with an image.

In the course notebook, the example image is downloaded and loaded with OpenCV:

```python
image = cv2.imread("assets/bus.jpg")
```

At this point, the image is represented as a NumPy array.

---

## Step 2 — YOLO Model

The image is passed to YOLO:

```python
results = model(image)[0]
```

YOLO performs **inference** and attempts to identify objects in the image.

The model produces information such as:

```text
Bounding Boxes
Class IDs
Confidence Scores
```

---

## Step 3 — Convert to `sv.Detections`

The raw Ultralytics result is converted into Supervision's standard representation:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

Now the predictions can be accessed through:

```python
detections.xyxy
detections.confidence
detections.class_id
```

---

## Step 4 — Inspect the Predictions

We can determine how many objects were detected:

```python
print(len(detections))
```

We can inspect their positions:

```python
print(detections.xyxy)
```

Their confidence scores:

```python
print(detections.confidence)
```

And their classes:

```python
print(detections.class_id)
```

---

## Step 5 — Create Human-Readable Labels

YOLO provides class names through:

```python
results.names
```

We can combine the class name and confidence:

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
bus 91%
person 87%
```

---

## Step 6 — Annotate the Image

Supervision can draw the detections:

```python
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
```

First, draw the bounding boxes:

```python
annotated = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

Then add the labels:

```python
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels
)
```

---

## Step 7 — Display the Result

The final annotated image can be displayed with Matplotlib:

```python
plt.imshow(
    cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")
plt.show()
```

The BGR-to-RGB conversion is necessary because OpenCV and Matplotlib use different default channel orders.

---

## Complete Architecture

The complete process can be represented as:

```text
                 INPUT IMAGE
                      │
                      ▼
                    OpenCV
                      │
                      ▼
                     YOLO
                      │
                      ▼
             Ultralytics Result
                      │
                      ▼
    sv.Detections.from_ultralytics()
                      │
                      ▼
                sv.Detections
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
      xyxy       confidence      class_id
        │             │             │
        └─────────────┼─────────────┘
                      ▼
               Process / Filter
                      │
                      ▼
                  Annotators
                      │
             ┌────────┴────────┐
             ▼                 ▼
       Bounding Boxes        Labels
             │                 │
             └────────┬────────┘
                      ▼
               Annotated Image
                      │
                      ▼
                 Matplotlib
```

---

## Separation of Responsibilities

One important software-design concept in this pipeline is that each component has its own responsibility.

```text
OpenCV
→ Image loading and processing

YOLO
→ Object detection

sv.Detections
→ Standardized detection representation

Supervision Annotators
→ Visualization

Matplotlib
→ Displaying the final result
```

This separation makes the pipeline easier to understand, debug, modify, and extend.

---

## Changing the Model

Because Supervision provides a standardized detection representation, the model can change while much of the rest of the pipeline remains similar.

Conceptually:

```text
YOLO ─────────────┐
                  │
Other Detector ───┼──→ sv.Detections
                  │
Future Model ─────┘
                         ↓
                    Processing
                         ↓
                    Annotation
```

This is one of the main reasons Supervision is useful in computer vision projects.

---

## Key Concept

The fundamental architecture to remember from this lesson is:

```text
Image → Model → sv.Detections → Processing → Annotation → Result
```

For our practical example:

```text
Image → YOLO → sv.Detections → Annotated Image
```

This pipeline forms the foundation for more advanced computer vision workflows involving tracking, segmentation, zones, and video analysis.
