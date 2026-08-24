# Input Assets — Project 07 SAM 3 Segmentation Pipeline

This directory contains the original input images used by **Project 07 — SAM 3 Segmentation Pipeline**.

---

## Purpose

Images stored in this directory are used as source media for the complete segmentation pipeline.

The workflow begins here:

    Input Image
         ↓
    YOLOv8 Detection
         ↓
    Confidence Filtering
         ↓
    Bounding Boxes
         ↓
    SAM 3 Segmentation
         ↓
    Pixel-Level Masks

The original input images should remain unchanged during processing.

---

## Initial Validation Image

The initial project validation uses:

    bus.jpg

This image is useful because it contains multiple objects that can be detected by YOLOv8 and subsequently segmented using SAM 3.

The image allows the project to test:

- Multi-object detection
- Confidence filtering
- Bounding-box extraction
- SAM 3 bounding-box prompts
- Multiple segmentation masks
- Object extraction
- Mask-area analysis
- Final visualization

---

## Input Requirements

Supported images should be readable by OpenCV.

Common formats include:

    .jpg
    .jpeg
    .png

The Python pipeline will verify that the selected input image exists and can be loaded successfully before performing inference.

---

## File Organization

The initial structure is:

    input/
    ├── README.md
    └── bus.jpg

Additional images can be added later for further testing and validation.

---

## Reproducibility

Input files are kept separate from generated results.

The project should never overwrite the original source image.

Generated files belong inside:

    ../output/

This separation allows the same input image to be processed repeatedly while keeping the original media intact.

---

## Important

The SAM 3 model checkpoint does **not** belong in this directory.

The validated model is stored externally in Google Drive:

    /content/drive/MyDrive/SAM3-Models/sam3.pt

Only source media required by the project should be stored inside `input/`.
