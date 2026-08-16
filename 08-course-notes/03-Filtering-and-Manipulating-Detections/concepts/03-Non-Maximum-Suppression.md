# Non-Maximum Suppression (NMS)

## Overview

Object detection models can sometimes produce **multiple bounding boxes for the same object**.

For example, YOLO may detect one person several times with slightly different bounding boxes.

This creates duplicate detections.

**Non-Maximum Suppression (NMS)** is a post-processing technique used to remove these redundant bounding boxes while keeping the strongest prediction.

---

# 1. The Duplicate Detection Problem

Imagine that YOLO detects the same person three times:

```text
Person A → Confidence 0.94
Person A → Confidence 0.87
Person A → Confidence 0.72
```

The bounding boxes may look like:

```text
┌─────────────────────┐
│  ┌────────────────┐ │
│  │ ┌────────────┐ │ │
│  │ │   PERSON   │ │ │
│  │ └────────────┘ │ │
│  └────────────────┘ │
└─────────────────────┘
```

Although there is only one person, the model produced several overlapping predictions.

Without post-processing, an application could incorrectly count these as multiple objects.

---

# 2. What Is NMS?

NMS stands for:

**Non-Maximum Suppression**

The goal of NMS is to:

1. Find overlapping bounding boxes.
2. Compare their confidence scores.
3. Keep the strongest prediction.
4. Suppress weaker duplicate predictions.

Conceptually:

```text
Multiple Overlapping Boxes
          │
          ▼
Compare Confidence Scores
          │
          ▼
Keep Highest Confidence
          │
          ▼
Remove Redundant Boxes
          │
          ▼
Clean Detections
```

---

# 3. Confidence and NMS

Suppose three bounding boxes detect the same object:

```text
Box A → 94%
Box B → 86%
Box C → 71%
```

If their overlap is high enough, NMS will generally keep:

```text
Box A → 94%
```

and suppress:

```text
Box B → 86%
Box C → 71%
```

The highest-confidence detection becomes the surviving prediction.

---

# 4. Understanding Bounding-Box Overlap

NMS needs a way to determine whether two bounding boxes represent the same object.

This is commonly measured using **Intersection over Union (IoU)**.

IoU compares:

```text
Area where two boxes overlap
```

with:

```text
Total combined area of both boxes
```

The basic formula is:

```text
IoU = Intersection Area / Union Area
```

The result ranges from:

```text
0.0 → No overlap

1.0 → Perfect overlap
```

---

# 5. Visualizing IoU

Consider two bounding boxes:

```text
Box A
┌───────────────┐
│               │
│      ┌────────┼───────┐
│      │////////│       │
│      │////////│       │
└──────┼────────┘       │
       │                │
       └────────────────┘
             Box B
```

The `/` region represents the intersection.

IoU measures how large this intersection is compared with the total area covered by both boxes.

---

# 6. NMS in Supervision

Supervision provides NMS directly through `sv.Detections`.

Example:

```python
clean_detections = detections.with_nms(
    threshold=0.5
)
```

The `threshold` controls how much overlap is allowed before a lower-confidence detection is suppressed.

---

# 7. Understanding the Threshold

In the lesson, different NMS thresholds are explored:

```python
0.3
0.5
0.8
```

The threshold changes how aggressively overlapping boxes are removed.

### Threshold = 0.3

```text
Lower overlap required
        ↓
More aggressive suppression
        ↓
More boxes may be removed
```

### Threshold = 0.5

```text
Moderate overlap requirement
        ↓
Balanced suppression
```

### Threshold = 0.8

```text
Large overlap required
        ↓
Less aggressive suppression
        ↓
More boxes may remain
```

---

# 8. Lower vs Higher NMS Threshold

A useful way to remember this is:

```text
LOW NMS threshold
        ↓
STRICT about overlap
        ↓
More suppression
        ↓
Fewer boxes
```

Compared with:

```text
HIGH NMS threshold
        ↓
Allows more overlap
        ↓
Less suppression
        ↓
More boxes
```

---

# 9. Creating Duplicate Detections for the Experiment

The lesson demonstrates NMS by running the same model with two different confidence thresholds.

First:

```python
results_low = model(
    image,
    conf=0.3
)[0]
```

This allows more low-confidence detections.

Then:

```python
results_high = model(
    image,
    conf=0.7
)[0]
```

This produces fewer but more confident detections.

---

# 10. Converting the Results

The YOLO results are converted into `sv.Detections`:

```python
det_low = sv.Detections.from_ultralytics(
    results_low
)

det_high = sv.Detections.from_ultralytics(
    results_high
)
```

Now both detection collections can be manipulated using Supervision.

---

# 11. Merging Detections

Supervision provides:

```python
sv.Detections.merge()
```

to combine multiple detection collections.

Example:

```python
merged = sv.Detections.merge([
    det_low,
    det_high
])
```

Because both detection sets came from the same image and model, the same objects may appear in both collections.

This intentionally creates duplicate predictions for the NMS experiment.

---

# 12. Before NMS

After merging:

```python
print(
    f"After merge: {len(merged)}"
)
```

The collection contains detections from both inference runs.

Conceptually:

```text
Low-Confidence Run
        │
        ├──────────┐
        │          │
        ▼          ▼
   Detection A   Detection B

High-Confidence Run
        │
        ├──────────┐
        │          │
        ▼          ▼
   Detection A   Detection B

        ↓ MERGE ↓

A + B + A + B

Duplicates Included
```

---

# 13. Applying NMS After Merge

NMS can now clean the merged detections:

```python
without_duplicates = merged.with_nms(
    threshold=0.5
)
```

The workflow becomes:

```text
det_low
   │
   ├────────────┐
   │            │
det_high        │
   │            │
   └────────────┘
        │
        ▼
sv.Detections.merge()
        │
        ▼
Merged Detections
        │
        ▼
with_nms(threshold=0.5)
        │
        ▼
Clean Detections
```

---

# 14. Comparing Before and After NMS

Before NMS:

```text
Object A
├── Box 1
└── Box 2

Object B
├── Box 3
└── Box 4
```

After NMS:

```text
Object A
└── Best Box

Object B
└── Best Box
```

This produces cleaner detection results.

---

# 15. Experimenting with Different Thresholds

The notebook tests:

```python
for thresh in [0.3, 0.5, 0.8]:

    filtered = sv.Detections.merge(
        [det_low, det_high]
    ).with_nms(
        threshold=thresh
    )
```

This allows us to visually compare how different NMS thresholds affect the final detections.

---

# 16. Why Threshold Selection Matters

There is no universal NMS threshold that works perfectly for every problem.

Imagine a crowded image containing several people standing very close together.

Their bounding boxes may naturally overlap.

If the NMS threshold is too aggressive:

```text
Person A
+
Person B
+
Large overlap
        ↓
NMS incorrectly assumes duplicate
        ↓
One valid person may disappear
```

Therefore, NMS parameters must be selected according to the application.

---

# 17. Example: Crowd Detection

For crowded environments:

```text
Many real objects
+
Bounding boxes naturally overlap
```

A very aggressive NMS configuration may remove valid detections.

The system needs to distinguish between:

```text
Duplicate boxes for ONE object
```

and:

```text
Overlapping boxes for DIFFERENT objects
```

This is one reason why NMS tuning is important.

---

# 18. Example: Vehicle Detection

Consider traffic monitoring.

A vehicle detector may initially produce:

```text
Car 1 → Box A
Car 1 → Box B
Car 2 → Box C
Bus   → Box D
```

Without NMS, the system might count:

```text
4 vehicles
```

even though there are only:

```text
3 vehicles
```

After NMS:

```text
Car 1 → Best Box
Car 2 → Box C
Bus   → Box D
```

Now the result is more accurate.

---

# 19. NMS as Post-Processing

NMS happens **after object detection**.

A simplified pipeline is:

```text
Input Image
     │
     ▼
YOLO Model
     │
     ▼
Raw Predictions
     │
     ▼
Confidence Filtering
     │
     ▼
NMS
     │
     ▼
Final Bounding Boxes
```

This makes NMS a **post-processing technique**.

---

# 20. NMS and Detection Quality

NMS does not retrain or modify the neural network.

Instead, it improves how the model's predictions are interpreted.

The model produces candidate detections.

NMS determines which overlapping candidates should survive.

```text
MODEL
  ↓
Produces Predictions
  ↓
NMS
  ↓
Cleans Predictions
```

---

# Important Supervision Methods

### Merge Detections

```python
sv.Detections.merge([
    detection_set_1,
    detection_set_2
])
```

### Apply NMS

```python
detections.with_nms(
    threshold=0.5
)
```

### Count Detections

```python
len(detections)
```

These operations make it easy to inspect how NMS changes the number of detections.

---

# Key Takeaways

- Object detectors can produce duplicate bounding boxes.
- NMS stands for **Non-Maximum Suppression**.
- NMS removes redundant overlapping detections.
- Higher-confidence boxes are generally preserved.
- IoU measures bounding-box overlap.
- IoU ranges from `0` to `1`.
- `sv.Detections.merge()` combines multiple detection collections.
- `.with_nms()` applies NMS in Supervision.
- Lower NMS thresholds generally produce more aggressive suppression.
- Higher thresholds allow more overlap between detections.
- NMS is an important object-detection post-processing technique.
- Threshold selection depends on the application and scene.

---

## Related Documentation

[Boolean Filtering](01-Boolean-Filtering.md)

[Confidence and Class Filtering](02-Confidence-and-Class-Filtering.md)

[Back to Concepts](README.md)

[Lesson 03 — Filtering and Manipulating Detections](../README.md)

[Main SAM3 Learning Journey Repository](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
