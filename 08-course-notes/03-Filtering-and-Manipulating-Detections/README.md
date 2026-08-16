# 03 — Filtering and Manipulating Detections

## Overview

This lesson focuses on one of the most important parts of a computer vision pipeline: **selecting and manipulating detections after a model has produced its predictions**.

Object detection models such as YOLO can return many detected objects. However, in real-world applications, we usually do not need every detection.

We may want to:

- Keep only specific object classes
- Remove low-confidence predictions
- Filter objects according to their size
- Combine multiple filtering conditions
- Remove duplicate bounding boxes
- Select the most confident detections
- Filter objects according to their position in an image

In this lesson, I learned how to perform these operations using the `sv.Detections` class from the **Supervision** library.

---

## Learning Objectives

By the end of this lesson, I learned how to:

- Understand `sv.Detections` as a filterable data structure
- Create Boolean masks for detections
- Filter detections by confidence score
- Filter detections by class
- Combine multiple filtering conditions
- Use element-wise Boolean operators such as `&`
- Merge detections from multiple sources
- Apply Non-Maximum Suppression (NMS)
- Understand the effect of different NMS thresholds
- Filter detections according to bounding-box area
- Sort detections by confidence
- Select the Top-N most confident detections
- Calculate the center of bounding boxes
- Filter objects according to their position in an image

---

## Technologies Used

The practical examples in this lesson use:

- Python
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Supervision](https://github.com/roboflow/supervision)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

Main imports:

```python
import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

# 1. Understanding `sv.Detections`

The `sv.Detections` object can be understood similarly to a table.

Each detected object represents one row containing information such as:

| Bounding Box | Confidence | Class ID |
|---|---:|---:|
| `[10, 20, 50, 80]` | 0.92 | 0 |
| `[100, 30, 200, 120]` | 0.71 | 0 |
| `[300, 10, 600, 400]` | 0.85 | 5 |

This makes it possible to filter detections in a way similar to filtering rows in a spreadsheet or values in a NumPy array.

For example:

```python
detections[detections.class_id == 0]
```

This means:

> Keep only the detections whose `class_id` is equal to `0`.

In the COCO dataset, class `0` represents a **person**.

---

# 2. Boolean Masks

Filtering in Supervision relies heavily on **Boolean masks**.

A Boolean mask is an array containing values such as:

```text
True
False
True
False
```

Each value corresponds to one detection.

For example:

```python
mask = detections.confidence > 0.5
```

For every detected object, Python checks whether its confidence is greater than `0.5`.

A possible result could look like:

```python
[True, True, False, True]
```

We can then apply the mask:

```python
high_confidence = detections[mask]
```

Only detections corresponding to `True` values are kept.

---

# 3. Filtering by Confidence

Object detection models assign a **confidence score** to each prediction.

The confidence score represents how certain the model is about a detected object.

```python
mask = detections.confidence > 0.5
high_confidence = detections[mask]
```

This keeps only detections with confidence greater than **50%**.

Filtering low-confidence predictions is useful because they may represent:

- Incorrect detections
- Background objects
- Ambiguous objects
- Detection noise

Increasing the confidence threshold generally produces fewer detections but requires greater certainty from the model.

---

# 4. Filtering by Class

We can select detections according to their object class.

For example:

```python
persons = detections[detections.class_id == 0]
```

Because COCO class `0` represents `person`, this operation removes all other detected object types.

We can inspect the detected classes using:

```python
for class_id in sorted(set(detections.class_id)):
    n = (detections.class_id == class_id).sum()
    print(
        f"Class {class_id} "
        f"({results.names[class_id]}): "
        f"{n} detections"
    )
```

This is useful when an application only needs certain types of objects.

Examples include:

- Detecting only people in a security system
- Detecting only vehicles in traffic analysis
- Detecting only animals in wildlife monitoring
- Detecting only products in a retail application

---

# 5. Combining Multiple Conditions

Multiple filtering conditions can be combined.

For example, we may want:

**Only people AND confidence greater than 60%.**

```python
safe_persons = detections[
    (detections.class_id == 0)
    & (detections.confidence > 0.6)
]
```

The `&` operator performs an element-by-element logical AND operation.

Both conditions must be true for a detection to remain.

### Important: `&` vs `and`

When working with NumPy arrays or `sv.Detections`, use:

```python
&
```

instead of:

```python
and
```

Python's normal `and` operator is designed for individual Boolean values.

Detection filters operate on arrays containing many Boolean values, so they require element-wise operators.

Correct:

```python
(class_condition) & (confidence_condition)
```

Incorrect:

```python
(class_condition) and (confidence_condition)
```

---

# 6. Excluding a Class

Filtering can also be reversed.

Instead of selecting a class, we can remove it.

```python
without_buses = detections[
    detections.class_id != 5
]
```

In the COCO dataset:

```text
class_id 5 = bus
```

The `!=` operator means:

> Keep everything whose class ID is NOT equal to 5.

---

# 7. Non-Maximum Suppression (NMS)

Object detection models can sometimes generate multiple bounding boxes around the same object.

These boxes may overlap heavily and represent duplicate detections.

**Non-Maximum Suppression (NMS)** is used to remove these duplicates.

The basic idea is:

1. Compare overlapping bounding boxes.
2. Measure how much they overlap.
3. Keep the detection with the highest confidence.
4. Remove lower-confidence boxes when their overlap exceeds the selected threshold.

With Supervision:

```python
filtered = detections.with_nms(
    threshold=0.5
)
```

---

# 8. Creating and Merging Detections

The lesson demonstrates NMS by generating detections using two different confidence thresholds:

```python
results_low = model(image, conf=0.3)[0]
results_high = model(image, conf=0.7)[0]
```

Convert them into Supervision detections:

```python
det_low = sv.Detections.from_ultralytics(results_low)
det_high = sv.Detections.from_ultralytics(results_high)
```

Then combine them:

```python
merged = sv.Detections.merge([
    det_low,
    det_high
])
```

Because both detection sets were generated from the same image, some objects can appear more than once.

---

# 9. Applying NMS

After merging the detections:

```python
without_duplicates = merged.with_nms(
    threshold=0.5
)
```

The process can be represented as:

```text
Before NMS
    ↓
Multiple overlapping detections
    ↓
Apply NMS
    ↓
Keep strongest predictions
    ↓
Cleaner detection results
```

---

# 10. Understanding the NMS Threshold

The lesson experiments with:

```text
0.3
0.5
0.8
```

A lower threshold is generally **more aggressive** because less overlap is required before one of the boxes is suppressed.

```text
Threshold 0.3
More aggressive suppression

Threshold 0.5
Moderate suppression

Threshold 0.8
More overlap allowed
```

The correct threshold depends on the application.

---

# 11. Filtering by Object Size

Supervision can calculate the area of each detected bounding box:

```python
areas = detections.area
```

We can inspect statistics:

```python
print(areas.min())
print(areas.max())
print(areas.mean())
```

We can also filter according to area:

```python
large_objects = detections[
    detections.area > 5000
]
```

This keeps only objects whose bounding boxes contain more than `5000 px²`.

### Why Filter by Size?

Very small detections can sometimes represent:

- Noise
- Distant objects
- False positives
- Objects that are not useful for the application

---

# 12. Selecting the Most Confident Detections

NumPy can sort detections according to confidence.

```python
top3_indices = np.argsort(
    detections.confidence
)[::-1][:3]
```

Then:

```python
top3 = detections[top3_indices]
```

The operation:

```python
np.argsort(detections.confidence)
```

sorts the indices from lowest to highest.

Then:

```python
[::-1]
```

reverses the order.

Finally:

```python
[:3]
```

keeps the first three.

The result is the **Top 3 most confident detections**.

---

# 13. Spatial Filtering

Detections can also be filtered according to where they appear inside an image.

This is useful for:

- Regions of interest
- Traffic lanes
- Restricted areas
- Entry and exit zones
- Surveillance systems
- Industrial inspection

The extension challenge asks us to keep only objects located in the **right half of the image**.

---

# 14. Calculating the Center of a Bounding Box

A bounding box uses the format:

```text
[x1, y1, x2, y2]
```

Where:

- `x1` = left coordinate
- `y1` = top coordinate
- `x2` = right coordinate
- `y2` = bottom coordinate

The horizontal center is calculated using:

```python
centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2
```

Formula:

```text
center_x = (x1 + x2) / 2
```

---

# 15. Right-Half Image Challenge

The width of the image is:

```python
image.shape[1]
```

The horizontal midpoint is:

```python
image_midpoint = image.shape[1] / 2
```

Create the Boolean mask:

```python
mask = centers_x > image_midpoint
```

Then filter:

```python
right_detections = detections[mask]
```

### Complete Solution

```python
centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2

image_midpoint = image.shape[1] / 2

mask = centers_x > image_midpoint

right_detections = detections[mask]
```

---

# 16. Complete Detection Filtering Pipeline

The concepts from this lesson can be combined into a complete post-processing pipeline:

```text
Image
  ↓
YOLO
  ↓
Raw Detections
  ↓
Confidence Filtering
  ↓
Class Filtering
  ↓
Size Filtering
  ↓
NMS
  ↓
Spatial Filtering
  ↓
Final Detections
```

An important lesson from this workflow is:

> Model inference is only one part of a computer vision pipeline. Post-processing determines which predictions are actually useful for the application.

---

# Key Concepts

| Concept | Purpose |
|---|---|
| `detections.confidence` | Access confidence scores |
| `detections.class_id` | Access detected classes |
| `detections.xyxy` | Access bounding-box coordinates |
| `detections.area` | Calculate bounding-box areas |
| Boolean masks | Select specific detections |
| `&` | Combine filtering conditions |
| `!=` | Exclude detections |
| `sv.Detections.merge()` | Combine detection sets |
| `.with_nms()` | Remove duplicate detections |
| `np.argsort()` | Sort detections |
| `image.shape[1]` | Obtain image width |

---

# What I Learned

This lesson helped me understand that object detection does not end when the model returns bounding boxes.

The predictions can be manipulated and filtered according to the requirements of the application.

I learned how to use Supervision detections similarly to NumPy arrays, allowing me to create Boolean conditions for selecting exactly the objects I need.

I also learned how Non-Maximum Suppression helps remove duplicate predictions and how bounding-box coordinates can be used to perform spatial filtering.

These techniques are important building blocks for creating more advanced computer vision systems.

---

# Key Takeaways

- `sv.Detections` can be filtered using NumPy-style Boolean masks.
- Confidence filtering removes uncertain predictions.
- Class filtering allows applications to focus on specific objects.
- Multiple conditions can be combined using `&`.
- NMS removes overlapping duplicate detections.
- Bounding-box area can be used to filter objects by size.
- NumPy can sort detections according to confidence.
- Bounding-box coordinates allow detections to be filtered spatially.
- Detection post-processing is essential for practical computer vision applications.

---

## Lesson Status

**Lesson:** 03 — Filtering and Manipulating Detections  
**Notebook:** `02_a_filtrado_detecciones.ipynb`  
**Topic:** Detection Filtering and Post-Processing  
**Status:** Documented

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)

---

[Back to SAM3 Learning Journey](https://github.com/Peyman-mxli/SAM3-Learning-Journey)
