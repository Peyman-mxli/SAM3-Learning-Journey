# Confidence Scores

## What is a Confidence Score?

A **confidence score** represents how certain an object detection model is about a prediction.

The value normally ranges from:

```text
0.0 → Very uncertain
1.0 → Very confident
```

For example:

```text
0.95 → 95% confidence
0.82 → 82% confidence
0.51 → 51% confidence
```

A higher value means the model is more confident that the detected object belongs to the predicted class.

---

## Confidence in `sv.Detections`

Supervision stores confidence scores in:

```python
detections.confidence
```

We can inspect them with:

```python
print(detections.confidence)
```

Example output:

```text
[0.94, 0.89, 0.76]
```

Each value corresponds to one detected object.

Conceptually:

```text
Detection 1 → Person → 94%
Detection 2 → Bus    → 89%
Detection 3 → Person → 76%
```

---

## Confidence Threshold

A **confidence threshold** determines the minimum confidence required for a prediction to be accepted.

For example:

```python
results = model(image, conf=0.8)[0]
```

Here:

```text
conf = 0.8
```

means we are asking YOLO to use a stricter confidence threshold.

---

## Lower vs Higher Confidence Threshold

A lower threshold generally allows more predictions:

```text
Lower Threshold
      ↓
More Detections
      ↓
More Uncertain Predictions May Remain
```

A higher threshold is more restrictive:

```text
Higher Threshold
      ↓
Fewer Detections
      ↓
More Confident Predictions Remain
```

---

## Course Experiment

In the course notebook, a stricter detection is created with:

```python
results_estricto = model(
    image,
    conf=0.8
)[0]
```

The results are then converted to Supervision:

```python
detections_estricto = (
    sv.Detections.from_ultralytics(
        results_estricto
    )
)
```

We can compare the number of detections:

```python
print(
    f"Default: {len(detections)} objects"
)

print(
    f"Strict: {len(detections_estricto)} objects"
)
```

---

## Why Does a Higher Threshold Detect Fewer Objects?

The model usually has greater confidence when objects are:

- Large
- Clearly visible
- Well represented in the training data
- Easy to distinguish from the background

Increasing the confidence threshold removes predictions where the model is less certain.

---

## Confidence and Individual Detections

We can inspect the confidence of a specific detection:

```python
first = detections[0]

print(first.confidence[0])
```

Or display it as a percentage:

```python
print(
    f"Confidence: {first.confidence[0]:.1%}"
)
```

For example:

```text
Confidence: 94.2%
```

---

## Confidence in the Detection Pipeline

```text
Image
  ↓
YOLO
  ↓
Predictions
  ↓
Confidence Scores
  ↓
Confidence Threshold
  ↓
Accepted Detections
  ↓
sv.Detections
  ↓
Annotation
```

---

## Why Confidence Matters

Confidence thresholds can affect the behavior of a computer vision application.

A very low threshold may produce more detections but can include uncertain predictions.

A very high threshold may produce cleaner results but can also remove real objects that the model detected with lower confidence.

The appropriate threshold depends on the application.

---

## Key Concept

Remember:

```text
LOWER threshold
      ↓
MORE predictions

HIGHER threshold
      ↓
FEWER, more confident predictions
```

In Supervision, the confidence values can be accessed with:

```python
detections.confidence
```

And in YOLO, a confidence threshold can be specified with:

```python
model(image, conf=0.8)
```
