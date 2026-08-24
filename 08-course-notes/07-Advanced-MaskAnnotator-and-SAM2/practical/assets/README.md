# Assets — Session 07 Advanced MaskAnnotator and SAM2

This directory contains the input media and generated output files used by the practical implementation of **Session 07 — Advanced MaskAnnotator and SAM2**.

The assets are separated into input and output directories so that original source images remain separate from generated visualizations.

---

## Directory Structure

```text
assets/
├── README.md
├── input/
└── output/
```

---

## Input

The `input/` directory stores the original images used by the practical.

The initial practical uses:

```text
bus.jpg
zidane.jpg
```

These images are used to demonstrate:

- YOLOv8 detection
- SAM 3 segmentation
- Bounding-box prompts
- Mask visualization
- Mask opacity experiments
- Person-only segmentation
- Reuse of the same pipeline on a second image

---

## Output

The `output/` directory stores generated practical results.

Expected outputs include:

```text
bounding_boxes.png
segmentation_masks.png
bbox_vs_mask.png
opacity_comparison.png
person_only_segmentation.png
second_image_segmentation.png
```

These files provide visual evidence of the different experiments performed during the practical.

---

## Workflow

```text
Input Images
     ↓
   input/
     ↓
YOLOv8 Detection
     ↓
SAM 3 Segmentation
     ↓
MaskAnnotator
     ↓
Visualization Experiments
     ↓
   output/
```

---

## Reproducibility

Keeping input and generated output files separate makes the practical easier to:

- Reproduce
- Debug
- Compare
- Validate
- Document

The original input images should remain unchanged during processing.

---

## SAM 3 Model

The SAM 3 checkpoint is not stored in this directory.

The validated Google Colab environment uses:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

The model remains external because of its large file size.
