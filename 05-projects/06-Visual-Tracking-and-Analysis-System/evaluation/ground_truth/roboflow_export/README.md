# Roboflow Ground-Truth Export

This directory contains the manually annotated ground-truth dataset exported from Roboflow for **Project 06 — Visual Tracking and Analysis System**.

---

## Annotation Project

The ground-truth dataset was created using a Roboflow **Instance Segmentation** project.

The evaluation class is:

```text
person
```

All annotations were created manually using segmentation polygons.

---

## Dataset

The export contains:

- 20 evaluation images
- 10 frames from Session 001
- 10 frames from Session 002
- Manual instance-segmentation annotations
- COCO-compatible annotation data

---

## Export Format

The dataset was exported using:

```text
COCO Segmentation
```

The exported dataset contains a `train/` directory with the evaluation images and:

```text
_annotations.coco.json
```

The JSON file contains the image metadata, categories, and polygon annotations required for the evaluation pipeline.

---

## File Integrity

The exported image filenames should not be renamed.

The COCO annotation file references the exported filenames directly, so changing them could break the relationship between images and annotations.

---

## Purpose

This export provides the reference annotations required to evaluate the existing Project 06 predictions using:

- Intersection over Union (IoU)
- Dice coefficient
- Precision
- Recall
- False positives
- Omissions

---

## Structure

```text
roboflow_export/
├── README.md
└── train/
    ├── _annotations.coco.json
    └── [20 evaluation images]
```

---

## Project

This dataset belongs to:

[Visual Tracking and Analysis System](../../../README.md)

---

## Author

**Peyman Miyandashti**

GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)

LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)
