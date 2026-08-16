# Exercise 01 — Confidence Filtering

## Objective

The goal of this exercise is to filter object detections according to their **confidence score**.

Object detection models such as YOLO assign a confidence value to every prediction.

Using Supervision, we can easily remove detections that do not meet a minimum confidence requirement.

---

## Starting Point

After running YOLO and converting the results to `sv.Detections`, we have:

```python
results = model(image)[0]

detections = sv.Detections.from_ultralytics(
    results
)
```

The `detections` object contains all predictions produced by the model.

---

## Step 1 — Access Confidence Scores

Confidence values are available through:

```python
detections.confidence
```

For example:

```python
print(detections.confidence)
```

A possible output could look like:

```text
[0.91 0.87 0.82 0.66 0.44]
```

Each value belongs to one detected object.

---

## Step 2 — Create a Boolean Mask

We want to keep only detections with confidence greater than:

```text
50%
```

Create the condition:

```python
mask = detections.confidence > 0.5
```

This produces a Boolean array similar to:

```text
[True, True, True, True, False]
```

Each value answers:

> Is this detection's confidence greater than 0.5?

---

## Step 3 — Apply the Mask

Apply the Boolean mask to the detections:

```python
high_confidence = detections[mask]
```

Now `high_confidence` contains only predictions above the selected threshold.

The same operation can also be written directly:

```python
high_confidence = detections[
    detections.confidence > 0.5
]
```

---

## Step 4 — Compare the Results

We can compare the number of detections before and after filtering:

```python
print(
    f"Total: {len(detections)} | "
    f"Confidence > 0.5: {len(high_confidence)}"
)
```

Conceptually:

```text
All Detections
      │
      ▼
Confidence > 0.5?
      │
   ┌──┴──┐
  Yes    No
   │      │
 Keep   Remove
   │
   ▼
High-Confidence Detections
```

---

## Complete Exercise

```python
# Create Boolean mask
mask = detections.confidence > 0.5

# Apply the mask
high_confidence = detections[mask]

# Compare results
print(
    f"Total: {len(detections)} | "
    f"Confidence > 0.5: {len(high_confidence)}"
)

# Visualize
mostrar(
    high_confidence,
    "Only confidence > 0.5"
)
```

---

## Experimenting with Thresholds

We can change the confidence threshold to observe how the results change.

### 30% Confidence

```python
filtered = detections[
    detections.confidence > 0.3
]
```

This is less restrictive and normally keeps more detections.

### 50% Confidence

```python
filtered = detections[
    detections.confidence > 0.5
]
```

This provides moderate filtering.

### 70% Confidence

```python
filtered = detections[
    detections.confidence > 0.7
]
```

This is more restrictive.

### 90% Confidence

```python
filtered = detections[
    detections.confidence > 0.9
]
```

Only extremely confident predictions remain.

---

## Threshold Comparison

| Threshold | Behavior |
|---:|---|
| `0.3` | More detections, including uncertain predictions |
| `0.5` | Moderate filtering |
| `0.7` | More selective |
| `0.9` | Only very high-confidence predictions |

---

## Important Observation

Increasing the confidence threshold does **not** make the model itself more accurate.

Instead, it changes which predictions we accept.

The model has already generated its detections.

The filter decides:

```text
Which predictions should remain?
```

This is a **post-processing operation**.

---

## Practical Example

Imagine a system detects:

```text
Person → 94%
Bus    → 88%
Person → 72%
Car    → 47%
Person → 31%
```

With:

```python
detections.confidence > 0.5
```

the result becomes:

```text
Person → 94%   KEEP
Bus    → 88%   KEEP
Person → 72%   KEEP
Car    → 47%   REMOVE
Person → 31%   REMOVE
```

The model predictions remain unchanged, but the application only uses detections that satisfy its confidence requirement.

---

## Real-World Applications

Confidence filtering is useful in:

- Security systems
- Traffic monitoring
- Object counting
- Retail analytics
- Industrial inspection
- Robotics
- Autonomous systems
- Wildlife monitoring

Different applications may require different confidence thresholds.

---

## What I Practiced

In this exercise, I practiced:

- Accessing `detections.confidence`
- Creating Boolean masks
- Applying masks to `sv.Detections`
- Selecting high-confidence predictions
- Comparing detection counts
- Understanding confidence thresholds
- Understanding detection post-processing

---

## Key Takeaways

- YOLO assigns a confidence score to every detection.
- Supervision stores these values in `detections.confidence`.
- Boolean conditions can select detections above a threshold.
- `detections[mask]` applies the filter.
- Higher thresholds are more restrictive.
- Lower thresholds keep more predictions.
- Confidence filtering is a post-processing operation.
- The best threshold depends on the application.

---

## Related Exercises

[Back to Practical Exercises](README.md)

[Next: Class Filtering](02-Class-Filtering.md)

---

## Related Concepts

[Boolean Filtering](../concepts/01-Boolean-Filtering.md)

[Confidence and Class Filtering](../concepts/02-Confidence-and-Class-Filtering.md)

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
