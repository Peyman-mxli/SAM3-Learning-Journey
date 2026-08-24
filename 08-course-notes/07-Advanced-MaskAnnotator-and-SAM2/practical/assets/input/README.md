# Input Assets — Session 07 Advanced MaskAnnotator and SAM2

This directory contains the original input images used by the practical implementation of **Session 07 — Advanced MaskAnnotator and SAM2**.

---

## Purpose

The input images are used to demonstrate:

- YOLOv8 object detection
- SAM 3 segmentation
- Bounding-box prompting
- Mask visualization
- Mask opacity experiments
- Class filtering before SAM
- Person-only segmentation
- Reuse of the same pipeline on a different image

---

## Input Files

The practical uses two images:

```text
bus.jpg
zidane.jpg
```

---

## bus.jpg

The first image is used for the main segmentation experiments.

It is used to demonstrate:

```text
YOLO Detection
      ↓
Bounding Boxes
      ↓
SAM 3
      ↓
Segmentation Masks
      ↓
MaskAnnotator
      ↓
Visualization
```

It is also used for:

- Bounding-box vs. mask comparison
- Mask opacity comparison
- Person-only filtering
- Person-only SAM segmentation

---

## zidane.jpg

The second image is used to demonstrate that the same segmentation pipeline can be reused without rewriting the processing logic.

The workflow remains:

```text
New Image
   ↓
YOLOv8
   ↓
sv.Detections
   ↓
Bounding Boxes
   ↓
SAM 3
   ↓
MaskAnnotator
```

This demonstrates that the pipeline is reusable across different input images.

---

## File Organization

After the two images are added, the directory should look like:

```text
input/
├── README.md
├── bus.jpg
└── zidane.jpg
```

---

## Source Media

The practical uses the Ultralytics sample images:

```text
bus.jpg
zidane.jpg
```

These are the same images used in the original Session 07 class notebook.

---

## Important

The original source images should remain unchanged.

Generated visualizations belong inside:

```text
../output/
```

The SAM 3 checkpoint should not be stored in this directory.

The validated model path is:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```
