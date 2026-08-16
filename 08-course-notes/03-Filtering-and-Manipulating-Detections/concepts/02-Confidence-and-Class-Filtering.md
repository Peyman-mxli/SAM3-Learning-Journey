# Confidence and Class Filtering

## Overview

Object detection models such as YOLO can detect many objects in a single image.

However, not every prediction is useful.

Two of the most common ways to control which detections we keep are:

- **Confidence filtering**
- **Class filtering**

These techniques allow us to select predictions based on how certain the model is and what type of object was detected.

---

# 1. Understanding Confidence Scores

Every detection produced by YOLO includes a **confidence score**.

The confidence score represents how certain the model is about its prediction.

For example:

```text
Person → 0.94
Bus    → 0.87
Person → 0.63
Car    → 0.41
```

These values can also be interpreted as percentages:

```text
0.94 → 94%
0.87 → 87%
0.63 → 63%
0.41 → 41%
```

A higher confidence score means the model is more certain about the prediction.

---

# 2. Accessing Confidence Scores

When using Supervision, confidence scores are available through:

```python
detections.confidence
```

For example:

```python
print(detections.confidence)
```

A possible result could be:

```text
[0.94 0.87 0.63 0.41]
```

Each value corresponds to one detected object.

---

# 3. Filtering by Confidence

We can create a condition that keeps only predictions above a certain confidence threshold.

For example:

```python
mask = detections.confidence > 0.5
```

Then apply the mask:

```python
high_confidence = detections[mask]
```

Or write everything in one line:

```python
high_confidence = detections[
    detections.confidence > 0.5
]
```

This means:

> Keep only detections with confidence greater than 50%.

---

# 4. Understanding the Confidence Threshold

Changing the confidence threshold changes how strict the filter is.

For example:

```text
Threshold = 0.3
↓
More detections
↓
Higher possibility of uncertain predictions
```

```text
Threshold = 0.5
↓
Balanced filtering
```

```text
Threshold = 0.8
↓
Fewer detections
↓
Only highly confident predictions
```

There is no single confidence threshold that is correct for every application.

The appropriate threshold depends on the problem being solved.

---

# 5. Why Confidence Filtering Is Important

Low-confidence detections may represent:

- False positives
- Difficult objects
- Small objects
- Partially visible objects
- Background patterns
- Ambiguous predictions

Confidence filtering helps remove predictions that do not meet the certainty requirements of the application.

For example, a system could require:

```python
detections.confidence > 0.8
```

when high certainty is more important than detecting every possible object.

---

# 6. Understanding Class IDs

Object detection models also assign a **class ID** to every prediction.

The class ID identifies what type of object the model believes it detected.

For example, in the COCO dataset:

```text
0 → person
1 → bicycle
2 → car
3 → motorcycle
5 → bus
7 → truck
```

The class IDs are stored in:

```python
detections.class_id
```

---

# 7. Inspecting Detected Classes

We can inspect the classes detected in an image.

```python
for class_id in sorted(set(detections.class_id)):
    n = (detections.class_id == class_id).sum()

    print(
        f"Class {class_id} "
        f"({results.names[class_id]}): "
        f"{n} detections"
    )
```

This tells us:

- Which classes were detected
- Their class IDs
- Their class names
- How many objects belong to each class

---

# 8. Filtering a Specific Class

Suppose we only want to detect people.

For COCO:

```text
person = class_id 0
```

We can filter the detections using:

```python
persons = detections[
    detections.class_id == 0
]
```

Now the `persons` variable contains only detections classified as people.

---

# 9. Excluding a Class

Sometimes we want the opposite.

Instead of selecting one class, we may want to remove it.

For example, COCO class `5` represents a bus.

```python
without_buses = detections[
    detections.class_id != 5
]
```

This means:

> Keep every detection except buses.

---

# 10. Combining Class and Confidence Filtering

Class filtering becomes more powerful when combined with confidence filtering.

Suppose we want:

```text
Person
AND
Confidence > 60%
```

We can write:

```python
persons_high_confidence = detections[
    (detections.class_id == 0)
    & (detections.confidence > 0.6)
]
```

The detection must satisfy **both conditions**.

---

# 11. How the Combined Filter Works

Imagine YOLO produced:

| Object | Class ID | Confidence |
|---|---:|---:|
| Person | 0 | 0.92 |
| Person | 0 | 0.55 |
| Bus | 5 | 0.95 |
| Person | 0 | 0.81 |

Our condition is:

```python
(detections.class_id == 0) & (detections.confidence > 0.6)
```

The evaluation becomes:

| Object | Person? | Confidence > 0.6? | Keep? |
|---|---|---|---|
| Person 0.92 | True | True | Yes |
| Person 0.55 | True | False | No |
| Bus 0.95 | False | True | No |
| Person 0.81 | True | True | Yes |

Only detections satisfying both requirements remain.

---

# 12. Filtering Multiple Classes

Boolean logic can also be used when more than one class is relevant.

For example, suppose an application wants both cars and buses.

```python
vehicles = detections[
    (detections.class_id == 2)
    | (detections.class_id == 5)
]
```

Here:

```text
| = OR
```

The detection is kept if either condition is true.

---

# 13. Practical Example: Traffic Monitoring

Imagine a traffic monitoring system.

The system may only need:

- Cars
- Buses
- Trucks
- Confidence greater than 70%

Conceptually:

```text
Raw YOLO Detections
        │
        ▼
Select Vehicle Classes
        │
        ▼
Confidence > 70%
        │
        ▼
Relevant Traffic Detections
```

This reduces unnecessary information before later processing such as:

- Tracking
- Counting
- Speed estimation
- Traffic analysis

---

# 14. Practical Example: Person Detection

A security system may only care about people.

Its filtering pipeline could be:

```text
YOLO
  │
  ▼
All Detections
  │
  ▼
class_id == 0
  │
  ▼
confidence > 0.6
  │
  ▼
Reliable Person Detections
```

In code:

```python
persons = detections[
    (detections.class_id == 0)
    & (detections.confidence > 0.6)
]
```

---

# 15. Confidence vs Class Filtering

These two filters answer different questions.

### Confidence Filtering

Asks:

> How certain is the model?

Example:

```python
detections.confidence > 0.6
```

### Class Filtering

Asks:

> What type of object was detected?

Example:

```python
detections.class_id == 0
```

### Combined Filtering

Asks:

> Is this the object type I need AND is the model confident enough?

Example:

```python
(detections.class_id == 0) & (detections.confidence > 0.6)
```

---

# Why These Filters Matter

Raw object detection output may contain more information than an application needs.

Filtering allows us to transform:

```text
Everything the model detected
```

into:

```text
Only the detections relevant to the task
```

This can make later stages of the computer vision pipeline:

- Cleaner
- Faster
- Easier to analyze
- More application-specific
- Less affected by uncertain predictions

---

# Key Takeaways

- Every detection contains a confidence score.
- `detections.confidence` provides access to confidence values.
- Confidence thresholds remove uncertain predictions.
- Every detection also contains a class ID.
- `detections.class_id` provides access to detected classes.
- `==` can select a specific class.
- `!=` can exclude a specific class.
- `&` combines class and confidence requirements.
- `|` can select detections belonging to multiple classes.
- Confidence and class filtering are fundamental detection post-processing techniques.

---

## Related Documentation

[Boolean Filtering](01-Boolean-Filtering.md)

[Back to Concepts](README.md)

[Lesson 03 — Filtering and Manipulating Detections](../README.md)

[Main SAM3 Learning Journey Repository](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
