# Practical — Segmentation with SAM 3

This folder contains the practical implementation for **Session 06 — Segmentation with SAM**.

The practical exercise demonstrates how to combine **YOLO**, **SAM 3**, and **Supervision** to move from object detection with bounding boxes to precise pixel-level segmentation masks.

---

## Practical Objective

The objective of this practical is to build and understand the following computer vision pipeline:

YOLO Detection  
↓  
Bounding Boxes  
↓  
SAM 3 Prompts  
↓  
Segmentation Masks  
↓  
Supervision Detections  
↓  
Mask Analysis and Visualization

The practical focuses on understanding how segmentation masks are generated, represented, analyzed, and stored.

---

## Main Steps

The practical workflow includes:

1. Load an input image.
2. Run YOLO object detection.
3. Convert YOLO results into `sv.Detections`.
4. Extract detected bounding boxes.
5. Use those bounding boxes as prompts for SAM 3.
6. Generate segmentation masks.
7. Convert SAM results into `sv.Detections`.
8. Inspect individual segmentation masks.
9. Calculate mask areas.
10. Extract objects using pixel-level masks.
11. Compare mask area with bounding-box area.
12. Encode segmentation masks for JSON storage.

---

## Technologies Used

- Python
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- NumPy
- OpenCV
- Matplotlib
- JSON
- Base64

---

## YOLO + SAM 3 Pipeline

YOLO is responsible for detecting objects and generating bounding boxes.

SAM 3 receives those bounding boxes as prompts and generates precise segmentation masks.

Conceptually:

Input Image  
↓  
YOLOv8  
↓  
Object Detections  
↓  
Bounding Box Prompts  
↓  
SAM 3  
↓  
Pixel-Level Masks

This allows the strengths of both models to be combined in a single workflow.

---

## Segmentation Masks

The generated masks are represented as boolean NumPy arrays.

Each pixel contains:

- `True` if the pixel belongs to the object
- `False` if the pixel belongs to the background

This makes it possible to perform operations such as:

- Measuring object area
- Removing image backgrounds
- Extracting individual objects
- Comparing object area with bounding-box area
- Saving segmentation results for later analysis

---

## Expected Outputs

The practical should produce or demonstrate:

- YOLO object detections
- Bounding boxes used as SAM prompts
- SAM 3 segmentation masks
- Raw mask visualization
- Pixel-level object extraction
- Mask area measurements
- Bounding-box vs. mask-area comparison
- Serialized segmentation data

---

## Practical Structure

The practical resources for this session are organized inside this directory.

```text
practical/
├── README.md
├── assets/
│   ├── input/
│   └── output/
└── segmentation_with_sam.py

The input directory stores source media used by the practical.

The output directory stores generated segmentation results.

The Python implementation contains the reproducible version of the workflow demonstrated during the session.

Learning Outcome

After completing this practical, I understand how to connect an object detector with a segmentation model and work directly with pixel-level segmentation data.

The key concept is that YOLO provides the approximate location of an object through a bounding box, while SAM 3 converts that spatial prompt into a much more precise representation of the object's actual shape.


Create **only this file for now**.

When it is uploaded to GitHub, send me **`done`** or a screenshot, and we continue to **Step 3**.
