# Extension Challenge

## Objective

The goal of this challenge is to apply the complete YOLO + Supervision pipeline to a new image.

Instead of only following the provided example, we test whether we can reproduce the workflow independently.

---

## Challenge Workflow

```text
Choose an Image
      ↓
Download Image
      ↓
Load with OpenCV
      ↓
Run YOLO
      ↓
Convert to sv.Detections
      ↓
Inspect Predictions
      ↓
Annotate Image
      ↓
Analyze Results
```

---

## Step 1 — Choose an Image

Select an image from the internet that contains recognizable objects.

The course also provides an example image:

```python
import urllib.request

urllib.request.urlretrieve(
    "https://ultralytics.com/images/zidane.jpg",
    "assets/zidane.jpg"
)
```

---

## Step 2 — Load the Image

Use OpenCV:

```python
image = cv2.imread(
    "assets/zidane.jpg"
)
```

Verify the image:

```python
print(image.shape)
```

---

## Step 3 — Run YOLO

Use the pretrained model:

```python
results = model(image)[0]
```

YOLO will analyze the image and return its predictions.

---

## Step 4 — Convert to Supervision

Convert the Ultralytics result:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

Now we can work with:

```python
detections.xyxy
detections.confidence
detections.class_id
```

---

## Step 5 — Inspect the Results

Check the number of detected objects:

```python
print(
    "Objects detected:",
    len(detections)
)
```

Inspect confidence scores:

```python
print(detections.confidence)
```

Inspect classes:

```python
print(detections.class_id)
```

---

## Step 6 — Create Labels

Create human-readable labels:

```python
labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]
```

---

## Step 7 — Annotate

Create the annotated image:

```python
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

## Step 8 — Analyze the Model

Running the code is only part of the challenge.

We should also evaluate what the model produced.

Ask:

1. What objects were detected?
2. How many objects were detected?
3. Which classes were identified?
4. Which prediction had the highest confidence?
5. Which prediction had the lowest confidence?
6. Did YOLO miss any visible objects?
7. Did YOLO detect anything incorrectly?

---

## False Positives

A **false positive** occurs when the model predicts an object that is not actually present.

```text
Model says:
"Object detected"
       ↓
But the object is not actually there
       ↓
False Positive
```

---

## False Negatives

A **false negative** occurs when an object exists in the image but the model fails to detect it.

```text
Real Object Exists
       ↓
Model does not detect it
       ↓
False Negative
```

---

## Experiment with Confidence

Try a stricter threshold:

```python
results_strict = model(
    image,
    conf=0.8
)[0]
```

Then compare the number of predictions.

```text
Default Threshold
       ↓
More Predictions

Higher Threshold
       ↓
Fewer Predictions
```

---

## Experiment with Another YOLO Model

Compare:

```python
YOLO("yolov8n.pt")
```

with:

```python
YOLO("yolov8s.pt")
```

Observe whether the larger model changes:

- Detection count
- Confidence
- Small-object detection
- Occluded-object detection

---

## Key Concept

The important lesson is that the input can change while the architecture remains the same:

```text
ANY IMAGE
    ↓
YOLO
    ↓
sv.Detections
    ↓
Analysis
    ↓
Annotation
    ↓
Result
```

The challenge demonstrates that the YOLO + Supervision pipeline can be reused with new images instead of being limited to the original course example.
