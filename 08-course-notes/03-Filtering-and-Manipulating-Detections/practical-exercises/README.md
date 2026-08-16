# Practical Exercises — Filtering and Manipulating Detections

## Overview

This folder contains the practical exercises from **Lesson 03 — Filtering and Manipulating Detections**.

The exercises demonstrate how to take raw object detections produced by YOLO and manipulate them using the **Supervision** library.

The main goal is to learn how to select exactly the detections needed for a computer vision application.

---

## Exercises

### 1. Confidence Filtering

Filter detections according to their confidence score.

```python
high_confidence = detections[
    detections.confidence > 0.5
]
```

This keeps only predictions with confidence greater than 50%.

[View Exercise](01-Confidence-Filtering.md)

---

### 2. Class Filtering

Select detections belonging to a specific object class.

```python
persons = detections[
    detections.class_id == 0
]
```

For the COCO dataset, class `0` represents a person.

The exercise also demonstrates combining class and confidence filters.

[View Exercise](02-Class-Filtering.md)

---

### 3. Merge and Non-Maximum Suppression

Combine detections from multiple inference results:

```python
merged = sv.Detections.merge([
    det_low,
    det_high
])
```

Then remove duplicate detections:

```python
without_duplicates = merged.with_nms(
    threshold=0.5
)
```

This exercise demonstrates how NMS cleans overlapping predictions.

[View Exercise](03-Merge-and-NMS.md)

---

### 4. Top Confidence Detections

Sort detections according to confidence and select the strongest predictions.

```python
indices_top3 = np.argsort(
    detections.confidence
)[::-1][:3]

top3 = detections[indices_top3]
```

This returns the three detections with the highest confidence scores.

[View Exercise](04-Top-Confidence-Detections.md)

---

### 5. Right-Half Detection Challenge

Calculate the horizontal center of every bounding box:

```python
centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2
```

Then compare it with the image midpoint:

```python
image_midpoint = image.shape[1] / 2
```

Finally:

```python
right_detections = detections[
    centers_x > image_midpoint
]
```

This keeps only objects whose center is located in the right half of the image.

[View Exercise](05-Right-Half-Challenge.md)

---

## Practical Workflow

The exercises follow this general workflow:

```text
Input Image
     │
     ▼
YOLO Inference
     │
     ▼
sv.Detections
     │
     ├───────────────┐
     │               │
     ▼               ▼
Confidence        Class
Filtering         Filtering
     │               │
     └───────┬───────┘
             │
             ▼
        Size Filtering
             │
             ▼
        Merge + NMS
             │
             ▼
       Sort by Confidence
             │
             ▼
       Spatial Filtering
             │
             ▼
       Final Detections
```

---

## Main Supervision Operations

| Operation | Example |
|---|---|
| Confidence filter | `detections[detections.confidence > 0.5]` |
| Class filter | `detections[detections.class_id == 0]` |
| Exclude class | `detections[detections.class_id != 5]` |
| Area filter | `detections[detections.area > 5000]` |
| Merge | `sv.Detections.merge([...])` |
| NMS | `detections.with_nms(threshold=0.5)` |
| Bounding boxes | `detections.xyxy` |
| Confidence values | `detections.confidence` |
| Class IDs | `detections.class_id` |

---

## Skills Practiced

These exercises reinforce:

- NumPy-style Boolean filtering
- Detection confidence thresholds
- Object class selection
- Combining multiple conditions
- Bounding-box manipulation
- Detection area analysis
- Non-Maximum Suppression
- Detection merging
- Confidence sorting
- Bounding-box center calculation
- Spatial filtering
- Detection post-processing

---

## Why These Exercises Matter

Running an object detection model is only the beginning of a computer vision pipeline.

Real applications need rules that determine which predictions are useful.

For example:

```text
Detect people
      ↓
Confidence > 60%
      ↓
Remove small detections
      ↓
Remove duplicates
      ↓
Keep only a specific region
      ↓
Use final detections
```

These exercises provide the foundation for building those rules.

---

## Exercise Files

1. [Confidence Filtering](01-Confidence-Filtering.md)
2. [Class Filtering](02-Class-Filtering.md)
3. [Merge and NMS](03-Merge-and-NMS.md)
4. [Top Confidence Detections](04-Top-Confidence-Detections.md)
5. [Right-Half Detection Challenge](05-Right-Half-Challenge.md)

---

## Related Documentation

[Lesson Concepts](../concepts/README.md)

[Lesson 03 — Filtering and Manipulating Detections](../README.md)

[Main SAM3 Learning Journey Repository](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
