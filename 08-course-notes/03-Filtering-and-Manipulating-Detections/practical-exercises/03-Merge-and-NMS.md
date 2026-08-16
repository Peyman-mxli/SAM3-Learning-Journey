# Exercise 03 — Merge and Non-Maximum Suppression (NMS)

## Objective

The goal of this exercise is to understand how to:

- Generate detections using different confidence thresholds
- Combine multiple `sv.Detections` objects
- Create duplicate detections for experimentation
- Apply Non-Maximum Suppression (NMS)
- Compare different NMS thresholds
- Understand how NMS removes redundant bounding boxes

---

## Why Do We Need NMS?

Object detection models can sometimes generate multiple bounding boxes around the same object.

For example:

```text
Same Bus
   │
   ├── Detection A → 92%
   ├── Detection B → 85%
   └── Detection C → 73%
```

If these boxes strongly overlap, they probably represent the same object.

Non-Maximum Suppression helps remove the duplicates.

---

## Step 1 — Generate Two Detection Sets

In this lesson, the same YOLO model is executed using two different confidence thresholds.

### Lower Confidence Threshold

```python
results_low = model(
    image,
    conf=0.3
)[0]
```

A lower threshold allows more predictions to remain.

### Higher Confidence Threshold

```python
results_high = model(
    image,
    conf=0.7
)[0]
```

A higher threshold keeps fewer, more confident predictions.

---

## Step 2 — Convert YOLO Results to Supervision

Convert both YOLO results into `sv.Detections`:

```python
det_low = sv.Detections.from_ultralytics(
    results_low
)

det_high = sv.Detections.from_ultralytics(
    results_high
)
```

Now both collections can be manipulated using Supervision.

---

## Step 3 — Compare Detection Counts

We can inspect how many detections each inference produced:

```python
print(
    f"Low confidence: {len(det_low)}"
)

print(
    f"High confidence: {len(det_high)}"
)
```

Conceptually:

```text
YOLO conf=0.3
      │
      ▼
More Detections

YOLO conf=0.7
      │
      ▼
Fewer Detections
```

Some objects appear in both collections.

---

## Step 4 — Merge the Detections

Supervision provides:

```python
sv.Detections.merge()
```

to combine multiple detection collections.

```python
merged = sv.Detections.merge([
    det_low,
    det_high
])
```

Now `merged` contains predictions from both inference runs.

---

## Why Does This Create Duplicates?

Both detection collections were generated using:

- The same image
- The same YOLO model

Only the confidence threshold changed.

Therefore, an object detected in both inference runs may appear twice after merging.

For example:

```text
det_low
├── Person A
├── Person B
└── Bus

det_high
├── Person A
└── Bus

          ↓ MERGE ↓

merged
├── Person A
├── Person B
├── Bus
├── Person A   ← duplicate
└── Bus        ← duplicate
```

This gives us a useful example for testing NMS.

---

## Step 5 — Inspect the Merged Result

```python
print(
    f"Individual detections: "
    f"low_conf={len(det_low)}, "
    f"high_conf={len(det_high)}"
)

print(
    f"After merge "
    f"(duplicates included): {len(merged)}"
)
```

The merged collection should contain more detections because duplicate objects are included.

---

# Step 6 — Apply NMS

Now apply Non-Maximum Suppression:

```python
without_duplicates = merged.with_nms(
    threshold=0.5
)
```

This evaluates overlapping bounding boxes and suppresses redundant detections.

---

## Understanding `threshold=0.5`

The threshold controls how much overlap is allowed.

```python
threshold=0.5
```

means that strongly overlapping detections can be treated as duplicates.

When this happens, NMS prefers the stronger prediction and suppresses the weaker one.

Conceptually:

```text
Detection A → 92%
      │
      │ strong overlap
      ▼
Detection B → 74%

        ↓ NMS ↓

Detection A → KEEP
Detection B → REMOVE
```

---

## Step 7 — Compare Before and After NMS

We can compare the number of detections:

```python
print(
    f"Before NMS: {len(merged)}"
)

print(
    f"After NMS: {len(without_duplicates)}"
)
```

The expected workflow is:

```text
Merged Detections
       │
       ▼
Duplicates Included
       │
       ▼
NMS
       │
       ▼
Redundant Boxes Removed
       │
       ▼
Cleaner Detections
```

---

## Step 8 — Visualize the Difference

Before NMS:

```python
mostrar(
    merged,
    "Before NMS (with duplicates)"
)
```

After NMS:

```python
mostrar(
    without_duplicates,
    "After NMS (threshold=0.5)"
)
```

This makes it easier to visually understand what NMS is doing.

---

# Complete Merge + NMS Exercise

```python
# Run the model with two confidence thresholds

results_low = model(
    image,
    conf=0.3
)[0]

results_high = model(
    image,
    conf=0.7
)[0]


# Convert YOLO results to Supervision

det_low = sv.Detections.from_ultralytics(
    results_low
)

det_high = sv.Detections.from_ultralytics(
    results_high
)


# Merge detections

merged = sv.Detections.merge([
    det_low,
    det_high
])


# Display detection counts

print(
    f"Individual detections: "
    f"low_conf={len(det_low)}, "
    f"high_conf={len(det_high)}"
)

print(
    f"After merge "
    f"(duplicates included): {len(merged)}"
)


# Apply NMS

without_duplicates = merged.with_nms(
    threshold=0.5
)


print(
    f"After NMS "
    f"(duplicates removed): "
    f"{len(without_duplicates)}"
)


# Visualize

mostrar(
    merged,
    "Before NMS (with duplicates)"
)

mostrar(
    without_duplicates,
    "After NMS (threshold=0.5)"
)
```

---

# Experiment — Different NMS Thresholds

The lesson also compares three NMS thresholds:

```text
0.3
0.5
0.8
```

The experiment:

```python
fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

for ax, thresh in zip(
    axes,
    [0.3, 0.5, 0.8]
):

    filtered = sv.Detections.merge(
        [det_low, det_high]
    ).with_nms(
        threshold=thresh
    )

    labels = [
        results.names[c]
        for c in filtered.class_id
    ]

    scene = box_annotator.annotate(
        scene=image.copy(),
        detections=filtered
    )

    scene = label_annotator.annotate(
        scene=scene,
        detections=filtered,
        labels=labels
    )

    ax.imshow(
        cv2.cvtColor(
            scene,
            cv2.COLOR_BGR2RGB
        )
    )

    ax.set_title(
        f"NMS threshold={thresh}\n"
        f"({len(filtered)} objects)"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()
```

---

# Comparing the Thresholds

### NMS = 0.3

```text
Less overlap required
        ↓
More aggressive suppression
        ↓
More boxes removed
```

### NMS = 0.5

```text
Moderate overlap requirement
        ↓
Balanced suppression
```

### NMS = 0.8

```text
More overlap allowed
        ↓
Less aggressive suppression
        ↓
More boxes remain
```

---

## Important Observation

A lower NMS threshold generally means:

```text
Lower Threshold
      ↓
Stricter Overlap Rule
      ↓
More Suppression
      ↓
Fewer Detections
```

A higher threshold generally means:

```text
Higher Threshold
      ↓
More Overlap Allowed
      ↓
Less Suppression
      ↓
More Detections
```

---

# NMS and IoU

NMS typically relies on **Intersection over Union (IoU)** to measure the overlap between bounding boxes.

The basic idea is:

```text
IoU = Intersection / Union
```

Where:

```text
Intersection
=
Area shared by both boxes
```

and:

```text
Union
=
Total area covered by both boxes
```

IoU ranges from:

```text
0.0 → No overlap

1.0 → Complete overlap
```

---

# Why Threshold Selection Matters

Consider two people standing very close together.

Their bounding boxes may overlap naturally.

```text
Person A      Person B
┌──────────┐
│       ┌──┼───────┐
│       │  │       │
│       │  │       │
└───────┼──┘       │
        └──────────┘
```

These are two real objects.

If NMS is too aggressive, one valid detection could potentially be removed.

Therefore, the NMS threshold should be selected according to the application and scene.

---

# Practical Example — Traffic Monitoring

Imagine a traffic detector produces:

```text
Car A → Box 1
Car A → Box 2
Car B → Box 3
Bus   → Box 4
```

Without NMS:

```text
Detected boxes = 4
```

But there are actually only:

```text
3 vehicles
```

After NMS:

```text
Car A → Best Box
Car B → Box 3
Bus   → Box 4
```

Now the result better represents the real scene.

---

# Merge vs NMS

These operations have different purposes.

## Merge

```python
sv.Detections.merge()
```

means:

> Combine multiple detection collections.

Merge can increase the number of detections.

---

## NMS

```python
detections.with_nms()
```

means:

> Remove redundant overlapping detections.

NMS can reduce the number of detections.

---

## Combined Workflow

```text
Detection Set A
       │
       ├─────────────┐
       │             │
Detection Set B      │
       │             │
       └─────────────┘
             │
             ▼
           MERGE
             │
             ▼
     Combined Detections
             │
             ▼
            NMS
             │
             ▼
      Clean Detections
```

---

# What I Practiced

In this exercise, I practiced:

- Running inference with different confidence thresholds
- Creating multiple `sv.Detections` objects
- Combining detections with `sv.Detections.merge()`
- Understanding why duplicate detections occur
- Applying `.with_nms()`
- Comparing detection counts
- Experimenting with NMS thresholds
- Understanding the relationship between NMS and IoU
- Visualizing detections before and after NMS

---

# Key Takeaways

- Multiple detection sources can be combined using `sv.Detections.merge()`.
- Merging predictions can create duplicate detections.
- NMS removes redundant overlapping bounding boxes.
- Higher-confidence detections are generally preserved.
- `.with_nms(threshold=...)` applies NMS in Supervision.
- Lower NMS thresholds generally produce stronger suppression.
- Higher NMS thresholds allow more overlapping detections.
- NMS commonly uses IoU to determine bounding-box overlap.
- NMS is an important post-processing step in object detection pipelines.
- Threshold selection depends on the application and scene.

---

## Related Exercises

[Previous: Class Filtering](02-Class-Filtering.md)

[Back to Practical Exercises](README.md)

[Next: Top Confidence Detections](04-Top-Confidence-Detections.md)

---

## Related Concepts

[Non-Maximum Suppression](../concepts/03-Non-Maximum-Suppression.md)

[Boolean Filtering](../concepts/01-Boolean-Filtering.md)

---

## Main Lesson

[03 — Filtering and Manipulating Detections](../README.md)

---

## Repository

[SAM3 Learning Journey](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
