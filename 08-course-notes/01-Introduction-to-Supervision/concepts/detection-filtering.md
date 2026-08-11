# Detection Filtering

## What is Detection Filtering?

**Detection filtering** allows us to keep only the predictions that satisfy specific conditions.

After YOLO detects objects and the results are converted into `sv.Detections`, we may not want to use every prediction.

For example, we may want to keep only:

- High-confidence detections
- Specific object classes
- Objects located in a particular area
- Objects with a certain bounding-box size

---

## Filtering by Confidence

Every detection contains a confidence score:

```python
detections.confidence
```

For example:

```text
[0.95, 0.83, 0.62, 0.41]
```

We can keep only detections above a specific confidence:

```python
filtered_detections = detections[
    detections.confidence > 0.80
]
```

Now only predictions with confidence greater than `0.80` remain.

---

## Confidence Filtering Concept

```text
All Detections
      ↓
Confidence Check
      ↓
Is confidence > 0.80?
      │
   ┌──┴──┐
   │     │
  YES    NO
   │     │
 Keep   Remove
   │
   ▼
Filtered Detections
```

---

## YOLO Confidence Threshold

We can also apply a confidence threshold directly when running YOLO:

```python
results_strict = model(
    image,
    conf=0.8
)[0]
```

Then convert the results:

```python
detections_strict = (
    sv.Detections.from_ultralytics(
        results_strict
    )
)
```

This tells YOLO to return only predictions that satisfy the stricter confidence requirement.

---

## Comparing Detection Counts

We can compare the number of detections:

```python
print(
    f"Original detections: {len(detections)}"
)

print(
    f"Strict detections: {len(detections_strict)}"
)
```

A higher confidence threshold will normally result in fewer detections.

---

## Filtering by Class

We can also filter detections according to their class ID.

For example, COCO uses:

```text
0 → person
2 → car
5 → bus
7 → truck
```

To keep only people:

```python
people = detections[
    detections.class_id == 0
]
```

To keep only cars:

```python
cars = detections[
    detections.class_id == 2
]
```

---

## Combining Filters

Multiple conditions can also be combined.

For example, keep only people with confidence greater than `0.80`:

```python
filtered = detections[
    (detections.class_id == 0)
    &
    (detections.confidence > 0.80)
]
```

Conceptually:

```text
All Detections
      ↓
Class = Person?
      ↓
Confidence > 0.80?
      ↓
Keep Detection
```

---

## Why Filtering Matters

Real computer vision models may generate many predictions.

Not all predictions are equally useful.

Filtering allows us to control which detections continue through the application:

```text
YOLO
  ↓
All Predictions
  ↓
sv.Detections
  ↓
Filtering
  ↓
Relevant Predictions
  ↓
Annotation / Analysis
```

---

## Key Concept

Detection filtering helps transform raw model predictions into results that are useful for a specific application.

Remember:

```python
detections.confidence
```

controls filtering based on model certainty.

And:

```python
detections.class_id
```

allows filtering based on object category.

The general workflow is:

```text
Detect
  ↓
Convert
  ↓
Filter
  ↓
Analyze
  ↓
Annotate
```
