# Filtering and Manipulating Detections — Code Examples

## Overview

This folder contains small, focused, runnable Python examples based on **Lesson 03 — Filtering and Manipulating Detections**.

The goal of these examples is to isolate the most important post-processing techniques used with `sv.Detections` so each concept can be studied and tested independently.

These examples are smaller than the complete Project 03 and are designed to demonstrate one specific idea at a time.

---

## Examples

### 01 — Confidence Filtering

File:

```text
01_confidence_filtering.py
```

Demonstrates how to:

- Access `detections.confidence`
- Create a Boolean mask
- Remove low-confidence detections
- Keep only predictions above a selected threshold

---

### 02 — Class and Boolean Filtering

File:

```text
02_class_and_boolean_filtering.py
```

Demonstrates how to:

- Access `detections.class_id`
- Select a specific class
- Exclude a class
- Combine multiple Boolean conditions
- Use `&`, `|`, and `!=`

---

### 03 — Size Filtering

File:

```text
03_size_filtering.py
```

Demonstrates how to:

- Access `detections.area`
- Inspect bounding-box areas
- Remove very small detections
- Keep only detections above a minimum size

---

### 04 — NMS and Top-N Selection

File:

```text
04_nms_and_top_n.py
```

Demonstrates how to:

- Merge multiple detection collections
- Apply Non-Maximum Suppression
- Compare detection counts
- Sort detections by confidence
- Select the Top-N strongest predictions

---

### 05 — Spatial Filtering

File:

```text
05_spatial_filtering.py
```

Demonstrates how to:

- Access `detections.xyxy`
- Calculate bounding-box centers
- Read image dimensions
- Filter detections by position
- Keep only detections located in the right half of an image

---

## Folder Structure

```text
03-Filtering-and-Manipulating-Detections/
│
├── README.md
├── 01_confidence_filtering.py
├── 02_class_and_boolean_filtering.py
├── 03_size_filtering.py
├── 04_nms_and_top_n.py
└── 05_spatial_filtering.py
```

---

## Learning Goal

These examples follow the progression:

```text
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
Top-N Selection
      ↓
Spatial Filtering
```

The examples help bridge the gap between the theoretical lesson and the complete Project 03 implementation.

---

## Related Course Lesson

[Lesson 03 — Filtering and Manipulating Detections](../../08-course-notes/03-Filtering-and-Manipulating-Detections/)

---

## Related Project

[Project 03 — Detection Filtering and NMS Pipeline](../../05-projects/03-Detection-Filtering-and-NMS-Pipeline/)

---

## Main Repository

[SAM3 Learning Journey](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
