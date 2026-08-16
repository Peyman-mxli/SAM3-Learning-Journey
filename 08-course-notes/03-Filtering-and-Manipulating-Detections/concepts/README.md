# Concepts — Filtering and Manipulating Detections

This folder contains the main theoretical concepts covered in **Lesson 03 — Filtering and Manipulating Detections**.

The purpose of these notes is to understand how detections produced by an object detection model can be selected, filtered, combined, sorted, and cleaned before being used in a real computer vision application.

---

## Concepts Covered

### 1. `sv.Detections` as a Filterable Data Structure

Supervision represents detected objects using the `sv.Detections` class.

A detection can contain information such as:

- Bounding-box coordinates
- Confidence score
- Class ID
- Masks
- Tracker IDs
- Additional metadata

Because many of these values are stored as NumPy arrays, detections can be filtered using Boolean conditions.

Example:

```python
persons = detections[detections.class_id == 0]
```

---

### 2. Boolean Filtering

A Boolean condition produces an array containing `True` and `False` values.

Example:

```python
mask = detections.confidence > 0.5
```

The mask can then be applied:

```python
high_confidence = detections[mask]
```

Only detections corresponding to `True` values remain.

---

### 3. Confidence Filtering

Every prediction produced by an object detection model has a confidence score.

Example:

```python
high_confidence = detections[
    detections.confidence > 0.5
]
```

This removes predictions with confidence scores of 50% or lower.

Confidence filtering helps reduce uncertain predictions and detection noise.

---

### 4. Class Filtering

Detections can be selected according to their class ID.

Example:

```python
persons = detections[
    detections.class_id == 0
]
```

For the COCO dataset:

```text
class_id 0 = person
```

This allows a computer vision application to focus only on relevant object categories.

---

### 5. Combining Conditions

Multiple conditions can be applied simultaneously.

Example:

```python
persons_safe = detections[
    (detections.class_id == 0)
    & (detections.confidence > 0.6)
]
```

This keeps only:

```text
Person
AND
Confidence > 60%
```

When working with NumPy arrays, `&` performs an element-wise logical AND operation.

---

### 6. Excluding Classes

The `!=` operator can be used to remove a specific class.

Example:

```python
without_buses = detections[
    detections.class_id != 5
]
```

This keeps every detected object except buses.

---

### 7. Non-Maximum Suppression

**Non-Maximum Suppression (NMS)** is a post-processing technique used to remove duplicate bounding boxes.

Sometimes multiple boxes detect the same object.

NMS compares overlapping detections and keeps the strongest prediction while suppressing redundant ones.

With Supervision:

```python
clean_detections = detections.with_nms(
    threshold=0.5
)
```

---

### 8. NMS Threshold

The NMS threshold controls how much overlap is allowed between bounding boxes.

Examples:

```text
0.3 → More aggressive suppression

0.5 → Moderate suppression

0.8 → More overlap allowed
```

A lower threshold generally removes overlapping boxes more aggressively.

The correct value depends on the computer vision application.

---

### 9. Merging Detections

Supervision can combine multiple `Detections` objects.

Example:

```python
merged = sv.Detections.merge([
    detections_a,
    detections_b
])
```

This can be useful when predictions come from:

- Multiple inference runs
- Different models
- Different confidence thresholds
- Different detection sources

After merging detections, NMS can help remove duplicates.

---

### 10. Bounding-Box Area

Supervision provides the area of detected bounding boxes through:

```python
detections.area
```

For example:

```python
large_objects = detections[
    detections.area > 5000
]
```

This keeps only detections larger than `5000 px²`.

Size filtering can help remove very small or irrelevant predictions.

---

### 11. Sorting by Confidence

NumPy's `argsort()` can be used to rank detections according to confidence.

Example:

```python
indices = np.argsort(
    detections.confidence
)[::-1]
```

The `[::-1]` reverses the order so the highest-confidence detections appear first.

To select only the Top 3:

```python
indices_top3 = np.argsort(
    detections.confidence
)[::-1][:3]

top3 = detections[indices_top3]
```

---

### 12. Spatial Filtering

Bounding-box coordinates can be used to determine where an object appears inside an image.

Bounding boxes follow the format:

```text
[x1, y1, x2, y2]
```

The horizontal center is:

```python
center_x = (x1 + x2) / 2
```

For all detections:

```python
centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2
```

This information can be used to filter objects according to their location.

---

### 13. Filtering the Right Half of an Image

The image width is available through:

```python
image.shape[1]
```

The midpoint is:

```python
image_midpoint = image.shape[1] / 2
```

Then:

```python
mask = centers_x > image_midpoint
right_detections = detections[mask]
```

This keeps detections whose center is located on the right side of the image.

---

## Detection Post-Processing Pipeline

These concepts can be combined:

```text
Object Detection Model
        │
        ▼
   Raw Detections
        │
        ▼
Confidence Filtering
        │
        ▼
   Class Filtering
        │
        ▼
    Size Filtering
        │
        ▼
        NMS
        │
        ▼
Spatial Filtering
        │
        ▼
 Final Detections
```

---

## Why These Concepts Matter

Real computer vision systems rarely use every raw prediction produced by a model.

Post-processing allows us to transform raw model output into information that is useful for a specific application.

For example, a system could request:

> Detect only people with more than 60% confidence, remove duplicate detections, ignore very small objects, and keep only people located inside a specific region.

The techniques from this lesson provide the foundation for implementing this type of logic.

---

## Related Lesson

[03 — Filtering and Manipulating Detections](../README.md)

---

## Main Repository

[SAM3 Learning Journey](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
