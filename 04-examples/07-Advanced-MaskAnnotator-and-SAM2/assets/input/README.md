# Input Assets — Advanced MaskAnnotator and SAM2

This directory contains the input images used by the examples in **07 — Advanced MaskAnnotator and SAM2**.

---

## Input Images

```text
bus.jpg
zidane.jpg
```

### `bus.jpg`

Used by:

- `01_basic_mask_annotator.py`
- `02_mask_opacity.py`
- `03_mask_and_box_annotator.py`
- `04_person_only_segmentation.py`
- `05_reusable_segmentation_function.py`

The image is used to demonstrate:

- YOLOv8 object detection
- SAM 3 segmentation
- `MaskAnnotator`
- Mask opacity
- Mask + bounding-box visualization
- Person-only segmentation
- Reusable segmentation pipelines

---

### `zidane.jpg`

Used by:

- `05_reusable_segmentation_function.py`

This second image demonstrates that the same reusable YOLO + SAM 3 segmentation pipeline can process different input images.

---

## Purpose

These images provide reproducible inputs for the Session 07 examples.

The general workflow is:

```text
Input Image
     ↓
YOLOv8
     ↓
Detections
     ↓
Bounding-Box Prompts
     ↓
SAM 3
     ↓
Segmentation Masks
     ↓
Visualization / Analysis
```

---

## Structure

```text
input/
├── README.md
├── bus.jpg
└── zidane.jpg
```

---

## Commit Message

```text
Add README for Session 07 input assets
```
