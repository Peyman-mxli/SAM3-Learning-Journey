# Output Assets — Session 07 Advanced MaskAnnotator and SAM2

This directory contains the generated visual outputs from the practical implementation of **Session 07 — Advanced MaskAnnotator and SAM2**.

The outputs demonstrate how YOLOv8 detections and SAM 3 segmentation masks can be visualized, customized, filtered, and reused using Supervision.

---

## Expected Outputs

After successfully running the practical, this directory should contain:

```text
output/
├── README.md
├── bounding_boxes.png
├── segmentation_masks.png
├── bbox_vs_mask.png
├── opacity_comparison.png
├── person_only_segmentation.png
└── second_image_segmentation.png
```

---

## bounding_boxes.png

This image contains the YOLOv8 detections visualized using:

```python
sv.BoxAnnotator()
```

It represents the object-detection stage before pixel-level segmentation.

Conceptually:

```text
Input Image
     ↓
YOLOv8
     ↓
Bounding Boxes
     ↓
bounding_boxes.png
```

---

## segmentation_masks.png

This image contains the SAM 3 segmentation results visualized using:

```python
sv.MaskAnnotator()
```

The visualization demonstrates the transition from rectangular object localization to pixel-level segmentation.

```text
YOLO Bounding Boxes
        ↓
SAM 3 Prompts
        ↓
Segmentation Masks
        ↓
segmentation_masks.png
```

---

## bbox_vs_mask.png

This output combines segmentation masks and bounding boxes in the same visualization.

The image demonstrates the difference between:

```text
Bounding Box
     ↓
Approximate rectangular region
```

and:

```text
Segmentation Mask
     ↓
Pixel-level object region
```

The visualization is created by applying:

```text
MaskAnnotator
      ↓
BoxAnnotator
```

to the same image.

---

## opacity_comparison.png

This output compares multiple `MaskAnnotator` opacity values.

The practical evaluates:

```text
0.2
0.5
0.9
```

Conceptually:

```text
Opacity 0.2
    ↓
Highly transparent mask

Opacity 0.5
    ↓
Balanced visualization

Opacity 0.9
    ↓
Strong mask visualization
```

This experiment demonstrates how mask opacity changes the balance between segmentation visibility and visibility of the original image.

---

## person_only_segmentation.png

This output demonstrates filtering detections **before** SAM segmentation.

The workflow is:

```text
Input Image
     ↓
YOLOv8
     ↓
All Detections
     ↓
Person Class Filter
     ↓
Person Bounding Boxes
     ↓
SAM 3
     ↓
Person Masks
     ↓
person_only_segmentation.png
```

Only detections with:

```text
COCO Class ID = 0
```

are passed to SAM.

This demonstrates how filtering can reduce unnecessary segmentation inference.

---

## second_image_segmentation.png

This output demonstrates reuse of the same segmentation pipeline on:

```text
zidane.jpg
```

The processing logic remains unchanged:

```text
Second Input Image
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
        ↓
second_image_segmentation.png
```

This validates that the workflow is reusable and not tied to a single source image.

---

## Practical Output Flow

The complete output-generation process is:

```text
bus.jpg
   ↓
YOLOv8
   ↓
SAM 3
   ↓
   ├─────────────────────────────┐
   │                             │
   ↓                             ↓
Bounding Boxes              Segmentation Masks
   │                             │
   └──────────────┬──────────────┘
                  ↓
          Combined Visualization
                  ↓
          Opacity Experiment
                  ↓
          Person-Only Filtering

zidane.jpg
   ↓
Same YOLO + SAM Pipeline
   ↓
Second Image Visualization
```

---

## Purpose

These generated files provide visual evidence that the Session 07 practical successfully demonstrates:

- YOLOv8 object detection
- SAM 3 segmentation
- `sv.MaskAnnotator`
- `sv.BoxAnnotator`
- Mask opacity customization
- Bounding-box and mask composition
- Detection filtering before segmentation
- Person-only segmentation
- Pipeline reuse across multiple images

---

## Important

These files are **generated outputs**.

The original source images belong in:

```text
../input/
```

The generated files should not replace or modify the original images.

The SAM 3 checkpoint is also not stored here because of its large size.

Validated external model path:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

---

## Validation Status

The output filenames documented here correspond to the Session 07 practical script:

```text
advanced_mask_annotator.py
```

The final validation status and actual generated results should be documented after the practical has been executed successfully in Google Colab.
