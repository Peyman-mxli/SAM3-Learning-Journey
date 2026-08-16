# Detection Filtering and NMS Pipeline

## Overview

This project applies the concepts learned in **Lesson 03 — Filtering and Manipulating Detections** and combines them into a complete object-detection post-processing pipeline.

The project uses:

- YOLOv8
- Supervision
- OpenCV
- NumPy

The goal is to take raw object detections and keep only the predictions that are useful for the application.

The pipeline includes:

- Confidence filtering
- Class filtering
- Size filtering
- Non-Maximum Suppression
- Top-N confidence selection
- Spatial filtering
- Final annotated visualization

---

## Project Workflow

```text
Input Image
     ↓
YOLOv8
     ↓
Raw Predictions
     ↓
sv.Detections
     ↓
Confidence Filtering
     ↓
Class Filtering
     ↓
Size Filtering
     ↓
Merge / NMS
     ↓
Top-N Selection
     ↓
Spatial Filtering
     ↓
Final Detections
     ↓
Annotated Output
```

---

## Project Structure

```text
03-Detection-Filtering-and-NMS-Pipeline/
│
├── assets/
│   ├── input/
│   │   └── image.jpg
│   │
│   ├── output/
│   │   └── filtered_detections.jpg
│   │
│   └── screenshots/
│
├── detection_filter_pipeline.py
├── requirements.txt
└── README.md
```

---

## Main Features

### Confidence Filtering

Removes detections below a selected confidence threshold.

Example:

```python
detections = detections[
    detections.confidence > 0.5
]
```

---

### Class Filtering

Keeps only detections belonging to selected object classes.

Example:

```python
detections = detections[
    detections.class_id == 0
]
```

For the COCO dataset:

```text
class_id 0 = person
```

---

### Size Filtering

Removes detections whose bounding boxes are too small.

Example:

```python
detections = detections[
    detections.area > 5000
]
```

This helps eliminate small or irrelevant detections.

---

### Non-Maximum Suppression

NMS removes redundant overlapping bounding boxes.

Example:

```python
detections = detections.with_nms(
    threshold=0.5
)
```

This keeps the strongest predictions while removing duplicate detections.

---

### Top-N Detection Selection

The detections can be ranked according to confidence.

Example:

```python
indices = np.argsort(
    detections.confidence
)[::-1][:3]

detections = detections[
    indices
]
```

This keeps only the three most confident predictions.

---

### Spatial Filtering

The project can also filter objects based on their position inside the image.

For example, keep only objects located in the right half:

```python
centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2

image_midpoint = image.shape[1] / 2

detections = detections[
    centers_x > image_midpoint
]
```

---

## Technologies Used

- [Python](https://www.python.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Supervision](https://github.com/roboflow/supervision)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Place an image inside:

```text
assets/input/
```

For example:

```text
assets/input/image.jpg
```

Then run:

```bash
python detection_filter_pipeline.py
```

---

## Expected Output

The project will:

1. Load the input image
2. Run YOLOv8 object detection
3. Convert predictions into `sv.Detections`
4. Apply detection filtering
5. Remove duplicate detections with NMS
6. Select relevant predictions
7. Annotate the final detections
8. Save the result

The output image will be saved to:

```text
assets/output/filtered_detections.jpg
```

---

## Example Processing Logic

```text
Raw Detections
      ↓
Confidence > 50%
      ↓
Selected Classes
      ↓
Area > Minimum Size
      ↓
NMS
      ↓
Top-N
      ↓
Spatial Condition
      ↓
Final Output
```

This demonstrates how raw model predictions can be transformed into application-specific results.

---

## Why This Project Matters

Object detection models often return more information than an application needs.

A real computer vision system may require rules such as:

```text
Only people
+
Confidence > 60%
+
Large enough bounding box
+
No duplicate detections
+
Located in a specific region
```

This project demonstrates how Supervision and NumPy can implement these rules efficiently.

---

## Concepts Applied

This project applies:

- `sv.Detections`
- Boolean masks
- Confidence filtering
- Class filtering
- Bounding-box area
- Detection merging
- Non-Maximum Suppression
- Intersection over Union
- NumPy sorting
- Top-N detection selection
- Bounding-box center calculations
- Spatial filtering
- Detection visualization

---

## Related Course Lesson

[Lesson 03 — Filtering and Manipulating Detections](../../08-course-notes/03-Filtering-and-Manipulating-Detections/)

---

## Related Projects

[Project 01 — YOLO Supervision Object Detector](../01-YOLO-Supervision-Object-Detector/)

[Project 02 — Multi-Annotator Visualization Pipeline](../02-Multi-Annotator-Visualization-Pipeline/)

---

## Repository

[SAM3 Learning Journey](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
