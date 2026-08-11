# Exercise 06 — YOLO Model Comparison

## Objective

The objective of this exercise is to compare two pretrained YOLOv8 models:

```text
YOLOv8 Nano  → yolov8n.pt
YOLOv8 Small → yolov8s.pt
```

We will analyze whether changing the model size affects the detection results.

---

## 1. Models to Compare

The first model used in the lesson is:

```python
model_n = YOLO("yolov8n.pt")
```

This is the **Nano** version.

Now load the Small model:

```python
model_s = YOLO("yolov8s.pt")
```

---

## 2. General Model Comparison

The course presents the following general relationship:

| Model | Approximate Size | Speed | Accuracy |
|---|---:|:---:|:---:|
| Nano | ~6 MB | +++ | + |
| Small | ~22 MB | ++ | ++ |
| Medium | ~52 MB | + | +++ |

The basic trade-off is:

```text
Smaller Model
      ↓
Faster
      ↓
Lower Computational Cost
```

versus:

```text
Larger Model
      ↓
More Computation
      ↓
Potentially Better Detection
```

---

## 3. Run YOLOv8 Nano

Run inference with the Nano model:

```python
results_n = model_n(image)[0]
```

Convert the predictions:

```python
detections_n = (
    sv.Detections.from_ultralytics(
        results_n
    )
)
```

---

## 4. Run YOLOv8 Small

Now run the Small model:

```python
results_s = model_s(image)[0]
```

Convert its predictions:

```python
detections_s = (
    sv.Detections.from_ultralytics(
        results_s
    )
)
```

---

## 5. Compare Detection Counts

Compare how many objects each model detected:

```python
print(
    f"YOLOv8 Nano: "
    f"{len(detections_n)} objects"
)

print(
    f"YOLOv8 Small: "
    f"{len(detections_s)} objects"
)
```

This gives us a simple first comparison.

---

## 6. Compare Confidence Scores

Inspect the confidence values from Nano:

```python
print(
    "Nano confidence:"
)

print(
    detections_n.confidence
)
```

Then inspect Small:

```python
print(
    "Small confidence:"
)

print(
    detections_s.confidence
)
```

The confidence scores may change even when both models detect the same object.

---

## 7. Compare Detected Classes

Nano:

```python
for class_id in sorted(
    set(detections_n.class_id)
):
    print(
        results_n.names[class_id]
    )
```

Small:

```python
for class_id in sorted(
    set(detections_s.class_id)
):
    print(
        results_s.names[class_id]
    )
```

Now we can determine whether one model detects classes that the other model misses.

---

## 8. What Should We Observe?

When comparing the models, ask:

1. Did both models detect the same number of objects?
2. Did they detect the same classes?
3. Which model produced higher confidence scores?
4. Did the Small model detect additional objects?
5. Did either model miss an object?
6. Were small objects detected differently?
7. Were partially hidden objects detected differently?

---

## 9. Occluded Objects

An object is **occluded** when part of it is hidden.

For example:

```text
Person
  ↓
Partially Hidden
  ↓
More Difficult to Detect
```

The course encourages us to observe whether the larger model handles small or partially occluded objects differently.

---

## 10. Supervision Makes Comparison Easier

Both models produce different Ultralytics results:

```text
YOLOv8 Nano ─────┐
                 │
YOLOv8 Small ────┤
                 ▼
        Ultralytics Results
                 │
                 ▼
     from_ultralytics()
                 │
                 ▼
          sv.Detections
```

Because both are converted to `sv.Detections`, we can analyze them using the same code.

---

## 11. Complete Exercise

```python
from ultralytics import YOLO
import supervision as sv

# Load models
model_n = YOLO("yolov8n.pt")
model_s = YOLO("yolov8s.pt")

# Nano inference
results_n = model_n(image)[0]

detections_n = (
    sv.Detections.from_ultralytics(
        results_n
    )
)

# Small inference
results_s = model_s(image)[0]

detections_s = (
    sv.Detections.from_ultralytics(
        results_s
    )
)

# Compare detection counts
print(
    f"YOLOv8 Nano: "
    f"{len(detections_n)} objects"
)

print(
    f"YOLOv8 Small: "
    f"{len(detections_s)} objects"
)

# Compare confidence
print("\nNano confidence:")
print(detections_n.confidence)

print("\nSmall confidence:")
print(detections_s.confidence)
```

---

## 12. Experiment Architecture

```text
                    IMAGE
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       YOLOv8 Nano       YOLOv8 Small
             │                 │
             ▼                 ▼
          Results           Results
             │                 │
             ▼                 ▼
      sv.Detections     sv.Detections
             │                 │
             └────────┬────────┘
                      ▼
                    Compare
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
      Count       Confidence      Classes
        │             │             │
        └─────────────┼─────────────┘
                      ▼
                   Analysis
```

---

## Key Takeaways

Changing the model affects the **inference stage**, but our Supervision workflow remains consistent:

```text
Different YOLO Model
        ↓
Different Predictions
        ↓
sv.Detections
        ↓
Same Analysis Tools
```

The goal is not automatically to choose the largest model.

The best model depends on the balance required between:

```text
Speed
  ↕
Model Size
  ↕
Computational Cost
  ↕
Detection Performance
```
