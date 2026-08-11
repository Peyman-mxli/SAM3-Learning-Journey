# Exercise 04 — Confidence Threshold Experiment

## Objective

The objective of this exercise is to understand how the **confidence threshold** affects YOLO object detection results.

We will compare the original detections with a stricter confidence threshold.

---

## 1. What is Confidence?

Every YOLO prediction includes a confidence score.

For example:

```text
Person → 95%
Bus    → 91%
Person → 64%
```

The confidence score represents how certain the model is about its prediction.

In Supervision, these values are available through:

```python
detections.confidence
```

---

## 2. Run YOLO with a Stricter Threshold

The course experiment uses:

```python
results_strict = model(
    image,
    conf=0.8
)[0]
```

The important parameter is:

```python
conf=0.8
```

This makes the detection process more restrictive.

---

## 3. Convert the New Results

Convert the stricter YOLO results to Supervision:

```python
detections_strict = (
    sv.Detections.from_ultralytics(
        results_strict
    )
)
```

Now we have two detection sets:

```text
detections
→ Original detections

detections_strict
→ Detections using conf=0.8
```

---

## 4. Compare Detection Counts

We can compare the number of objects detected:

```python
print(
    f"Original: "
    f"{len(detections)} objects"
)

print(
    f"Confidence 0.8: "
    f"{len(detections_strict)} objects"
)
```

---

## 5. Expected Behavior

Increasing the confidence threshold usually produces fewer detections.

```text
Lower Threshold
      ↓
More Predictions
      ↓
Some Predictions May Be Less Certain
```

Compared with:

```text
Higher Threshold
      ↓
Fewer Predictions
      ↓
More Confident Predictions Remain
```

---

## 6. Inspect the Confidence Scores

Original detections:

```python
print(
    detections.confidence
)
```

Strict detections:

```python
print(
    detections_strict.confidence
)
```

This allows us to compare which predictions remain after increasing the threshold.

---

## 7. Analyze the Difference

Ask:

1. How many objects were detected originally?
2. How many remain with `conf=0.8`?
3. Which objects disappeared?
4. What were their confidence scores?
5. Were the removed predictions actually incorrect?
6. Did the stricter threshold remove any real objects?

---

## 8. Why Can Objects Disappear?

The model may have lower confidence when an object is:

- Small
- Partially hidden
- Far from the camera
- Difficult to distinguish
- Surrounded by a complex background

Therefore, increasing the threshold can remove these detections.

---

## 9. Complete Exercise

```python
# Original detection
results = model(image)[0]

detections = (
    sv.Detections.from_ultralytics(
        results
    )
)

# Stricter detection
results_strict = model(
    image,
    conf=0.8
)[0]

detections_strict = (
    sv.Detections.from_ultralytics(
        results_strict
    )
)

# Compare results
print(
    f"Original: "
    f"{len(detections)} objects"
)

print(
    f"Confidence 0.8: "
    f"{len(detections_strict)} objects"
)

print("\nOriginal confidence:")
print(detections.confidence)

print("\nStrict confidence:")
print(detections_strict.confidence)
```

---

## 10. Experiment with Different Thresholds

We can also experiment with other values:

```python
model(image, conf=0.3)
model(image, conf=0.5)
model(image, conf=0.7)
model(image, conf=0.8)
model(image, conf=0.9)
```

Then compare:

```text
0.3 → More detections
0.5 → Moderate filtering
0.7 → Stricter
0.8 → Very strict
0.9 → Only very confident detections
```

---

## 11. Detection Pipeline

```text
                 IMAGE
                   │
                   ▼
                  YOLO
                   │
                   ▼
          Confidence Threshold
                   │
            ┌──────┴──────┐
            ▼             ▼
         Accepted       Rejected
       Predictions     Predictions
            │
            ▼
      sv.Detections
            │
            ▼
          Analysis
```

---

## Key Takeaways

The main parameter in this exercise is:

```python
conf=0.8
```

The general relationship is:

```text
Lower Confidence Threshold
          ↓
More Detections

Higher Confidence Threshold
          ↓
Fewer Detections
```

A higher threshold does not automatically mean a better model.

The correct confidence threshold depends on the requirements of the computer vision application.
