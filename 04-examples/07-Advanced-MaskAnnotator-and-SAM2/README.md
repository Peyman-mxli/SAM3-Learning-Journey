# 07 — Advanced MaskAnnotator and SAM2

This directory contains focused Python examples based on **Session 07 — Advanced MaskAnnotator and SAM2** from my SAM3 Computer Vision Learning Journey.

These examples continue from the segmentation concepts introduced in Session 06 and focus on how segmentation masks can be **visualized, customized, filtered, combined with detections, and reused**.

The session also introduces the conceptual transition from static-image segmentation toward temporal segmentation with SAM2.

All six examples in this directory were executed and validated successfully in Google Colab.

---

## Topics Covered

The examples in this folder cover:

- `sv.MaskAnnotator`
- `sv.BoxAnnotator`
- YOLOv8 object detection
- SAM 3 segmentation
- Bounding-box prompts
- Segmentation-mask visualization
- Mask opacity
- Combining masks and bounding boxes
- Detection filtering before SAM
- Person-only segmentation
- Reusable segmentation functions
- Processing different input images with the same pipeline
- SAM2 temporal segmentation concepts
- Temporal memory
- Mask propagation across video frames

---

## Directory Structure

```text
07-Advanced-MaskAnnotator-and-SAM2/
├── README.md
├── 01_basic_mask_annotator.py
├── 02_mask_opacity.py
├── 03_mask_and_box_annotator.py
├── 04_person_only_segmentation.py
├── 05_reusable_segmentation_function.py
├── 06_sam2_temporal_concept.py
└── assets/
    ├── README.md
    └── output/
        ├── README.md
        ├── 01_basic_mask_annotator_output.png
        ├── 02_mask_opacity_output.png
        ├── 03_mask_and_box_annotator_output.png
        ├── 04_person_only_segmentation_output.png
        ├── 05_reusable_bus_output.png
        └── 05_reusable_zidane_output.png
```

The first five examples generate visual outputs.

Example 05 processes two different images and therefore generates two output files.

Example 06 is intentionally conceptual and does not generate an image.

---

# 01 — Basic MaskAnnotator

Script:

[`01_basic_mask_annotator.py`](./01_basic_mask_annotator.py)

Output:

[`01_basic_mask_annotator_output.png`](./assets/output/01_basic_mask_annotator_output.png)

This example introduces:

```python
sv.MaskAnnotator()
```

The workflow is:

```text
Input Image
     ↓
YOLOv8
     ↓
Object Detections
     ↓
Bounding Boxes
     ↓
SAM 3
     ↓
Segmentation Masks
     ↓
MaskAnnotator
     ↓
Annotated Image
```

YOLOv8 first detects the objects.

The resulting bounding boxes are used as prompts for SAM 3.

SAM 3 generates pixel-level segmentation masks, and `MaskAnnotator` overlays those masks on the original image.

## Validated Results

Input:

```text
bus.jpg
```

Image shape:

```text
(1080, 810, 3)
```

YOLO detected:

```text
4 persons
1 bus
1 stop sign
```

Results:

```text
YOLO detections:     6
SAM masks generated: 6
```

Status:

```text
PASSED ✅
```

---

# 02 — Mask Opacity

Script:

[`02_mask_opacity.py`](./02_mask_opacity.py)

Output:

[`02_mask_opacity_output.png`](./assets/output/02_mask_opacity_output.png)

This example demonstrates how the `opacity` parameter changes the appearance of segmentation masks.

The following values were tested:

```text
0.2
0.5
0.9
```

Example:

```python
mask_annotator = sv.MaskAnnotator(
    opacity=0.5
)
```

Conceptually:

```text
Opacity 0.2
    ↓
Original Image More Visible
```

```text
Opacity 0.5
    ↓
Balanced Visualization
```

```text
Opacity 0.9
    ↓
Segmentation Mask More Visible
```

## Validated Results

```text
YOLO detections:     6
SAM masks generated: 6

Opacity 0.2: ✅
Opacity 0.5: ✅
Opacity 0.9: ✅
```

The comparison image was generated successfully.

Status:

```text
PASSED ✅
```

---

# 03 — Mask + Box Annotator

Script:

[`03_mask_and_box_annotator.py`](./03_mask_and_box_annotator.py)

Output:

[`03_mask_and_box_annotator_output.png`](./assets/output/03_mask_and_box_annotator_output.png)

This example combines:

```python
sv.MaskAnnotator()
```

with:

```python
sv.BoxAnnotator()
```

The workflow becomes:

```text
Original Image
      ↓
YOLOv8
      ↓
Bounding Boxes
      ↓
SAM 3
      ↓
Segmentation Masks
      ↓
MaskAnnotator
      ↓
BoxAnnotator
      ↓
Combined Visualization
```

The final image contains three visualization layers:

```text
1. Original image
2. SAM 3 segmentation masks
3. YOLO bounding boxes
```

This makes it possible to compare:

```text
Bounding Box
     ↓
Rectangular Object Localization
```

with:

```text
Segmentation Mask
     ↓
Pixel-Level Object Representation
```

## Validated Results

```text
YOLO detections:     6
SAM masks generated: 6
```

Status:

```text
PASSED ✅
```

---

# 04 — Person-Only Segmentation

Script:

[`04_person_only_segmentation.py`](./04_person_only_segmentation.py)

Output:

[`04_person_only_segmentation_output.png`](./assets/output/04_person_only_segmentation_output.png)

This example combines filtering concepts from earlier sessions with SAM segmentation.

Instead of sending every YOLO detection to SAM, the detections are filtered first.

For the COCO `person` class:

```python
person_detections = detections[
    detections.class_id == 0
]
```

The workflow becomes:

```text
Input Image
     ↓
YOLOv8
     ↓
All Detections
     ↓
Person Filter
     ↓
Person Bounding Boxes
     ↓
SAM 3
     ↓
Person Segmentation Masks
     ↓
MaskAnnotator + BoxAnnotator
```

The important concept is:

```text
Filter BEFORE SAM
```

instead of:

```text
Segment Everything
      ↓
Discard Unwanted Results
```

## Validated Results

```text
Total YOLO detections:       6
Person detections:           4
Objects removed before SAM:  2
Person masks generated:      4
```

Therefore:

```text
6 YOLO Detections
        ↓
Person Filter
        ↓
4 Person Detections
        ↓
SAM 3
        ↓
4 Person Masks
```

Status:

```text
PASSED ✅
```

---

# 05 — Reusable Segmentation Function

Script:

[`05_reusable_segmentation_function.py`](./05_reusable_segmentation_function.py)

Outputs:

- [`05_reusable_bus_output.png`](./assets/output/05_reusable_bus_output.png)
- [`05_reusable_zidane_output.png`](./assets/output/05_reusable_zidane_output.png)

This example restructures the YOLO + SAM workflow into a reusable:

```python
segment_image()
```

function.

The reusable function performs:

```text
Image
  ↓
YOLOv8
  ↓
sv.Detections
  ↓
Bounding Boxes
  ↓
SAM 3
  ↓
Segmentation Masks
  ↓
MaskAnnotator
  ↓
BoxAnnotator
  ↓
Result
```

The models are loaded once and the same function is then used to process different images.

## Validated Result — bus.jpg

```text
YOLO detections: 6
SAM masks:       6
```

Output:

```text
05_reusable_bus_output.png
```

## Validated Result — zidane.jpg

YOLO detected:

```text
2 persons
1 tie
```

Results:

```text
YOLO detections: 3
SAM masks:       3
```

Output:

```text
05_reusable_zidane_output.png
```

The same function successfully processed both images:

```text
bus.jpg ─────────┐
                 ↓
           segment_image()
                 ↓
             Result A


zidane.jpg ──────┐
                 ↓
           segment_image()
                 ↓
             Result B
```

This demonstrates:

```text
Reusable Pipeline
instead of
Repeated Code
```

Status:

```text
PASSED ✅
```

---

# 06 — SAM2 Temporal Concept

Script:

[`06_sam2_temporal_concept.py`](./06_sam2_temporal_concept.py)

This example is intentionally conceptual.

It does **not** load a SAM2 model and does **not** generate an image.

The purpose is to demonstrate the temporal-segmentation concepts introduced in Session 07.

Static segmentation follows:

```text
Image
  ↓
Prompt
  ↓
Segmentation
  ↓
Mask
```

Each image is processed independently.

Temporal segmentation introduces information across frames:

```text
Initial Frame
      ↓
Object Prompt
      ↓
Initial Mask
      ↓
Temporal Memory
      ↓
Next Frame
      ↓
Updated Mask
      ↓
Update Temporal Memory
      ↓
Future Frames
```

The conceptual difference is:

```text
Static Segmentation
        ↓
Spatial Information
```

versus:

```text
Temporal Segmentation
        ↓
Spatial Information
        +
Temporal Information
```

The example also demonstrates the concept of mask propagation:

```text
Initial Frame
     ↓
Object Prompt
     ↓
Initial Mask
     ↓
Temporal Memory
     ↓
Frame 2
     ↓
Updated Mask
     ↓
Frame 3
     ↓
Updated Mask
     ↓
Future Frames
```

The script executed successfully in Google Colab.

Status:

```text
PASSED ✅
```

---

# Validation Summary

All six Python examples were executed successfully in Google Colab.

```text
01_basic_mask_annotator.py              ✅ PASSED
02_mask_opacity.py                      ✅ PASSED
03_mask_and_box_annotator.py            ✅ PASSED
04_person_only_segmentation.py          ✅ PASSED
05_reusable_segmentation_function.py    ✅ PASSED
06_sam2_temporal_concept.py             ✅ PASSED
```

Visual outputs:

```text
Example 01 → 1 image
Example 02 → 1 image
Example 03 → 1 image
Example 04 → 1 image
Example 05 → 2 images
Example 06 → 0 images (conceptual)

Total validated examples: 6
Total visual outputs:     6
```

---

# Validated Results

```text
bus.jpg
├── YOLO detections: 6
├── SAM masks: 6
└── Person-only masks: 4

zidane.jpg
├── YOLO detections: 3
└── SAM masks: 3

Mask opacity:
├── 0.2 ✅
├── 0.5 ✅
└── 0.9 ✅

Examples validated: 6
Outputs generated:   6
```

---

# Generated Assets

The validated outputs are stored in:

[`assets/output/`](./assets/output/)

Asset documentation:

[`assets/README.md`](./assets/README.md)

Output documentation:

[`assets/output/README.md`](./assets/output/README.md)

Generated files:

```text
01_basic_mask_annotator_output.png
02_mask_opacity_output.png
03_mask_and_box_annotator_output.png
04_person_only_segmentation_output.png
05_reusable_bus_output.png
05_reusable_zidane_output.png
```

These files provide visual evidence of the executed examples without requiring SAM 3 inference to be rerun.

---

# Relationship to Session 06

Session 06 introduced:

```text
YOLO Detection
      ↓
Bounding Boxes
      ↓
SAM 3
      ↓
Segmentation Masks
      ↓
Mask Inspection
      ↓
Object Extraction
      ↓
Mask Analysis
```

Session 07 continues from those masks:

```text
Segmentation Masks
      ↓
MaskAnnotator
      ↓
Opacity Control
      ↓
Bounding-Box Composition
      ↓
Selective Segmentation
      ↓
Reusable Functions
      ↓
Temporal Segmentation Concepts
```

Therefore:

```text
Session 06
Generate + Analyze Masks
        ↓
Session 07
Visualize + Customize + Reuse Masks
```

---

# Technologies Used

The examples use:

- Python
- Google Colab
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- `sv.Detections`
- `sv.MaskAnnotator`
- `sv.BoxAnnotator`
- OpenCV
- NumPy
- Matplotlib

The session also introduces concepts related to:

- SAM2
- Temporal segmentation
- Temporal memory
- Video mask propagation

---

# SAM 3 Model

The SAM 3 checkpoint is not stored inside this repository because the model file is very large.

The validated Google Colab environment uses:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

Examples requiring SAM 3 verify that this model exists before segmentation begins.

During validated inference, Ultralytics displayed:

```text
WARNING ⚠️ imgsz=[1024] must be multiple of max stride 14,
updating to [1036]
```

This was a non-fatal automatic image-size adjustment.

SAM 3 inference continued successfully and generated the expected masks.

---

# Learning Progression

The six examples follow this progression:

```text
01
Basic Mask Visualization
        ↓
02
Mask Opacity
        ↓
03
Mask + Bounding Box Composition
        ↓
04
Selective Person Segmentation
        ↓
05
Reusable Segmentation Pipeline
        ↓
06
Temporal Segmentation Concept
```

The broader progression is:

```text
Object Detection
       ↓
SAM 3 Segmentation
       ↓
Pixel-Level Masks
       ↓
Mask Visualization
       ↓
Visualization Customization
       ↓
Detection Filtering
       ↓
Selective Segmentation
       ↓
Reusable Processing
       ↓
Temporal Segmentation Concepts
```

---

# Learning Goal

The purpose of these examples is to understand segmentation as more than simply generating a mask.

The examples demonstrate how segmentation results can be:

- Visualized
- Customized
- Combined with object detections
- Filtered by object class
- Reused across different images
- Organized into reusable processing functions
- Extended conceptually toward temporal video segmentation

This provides the foundation for more advanced image and video segmentation systems.

---

# Status

```text
Example scripts:      6
Examples validated:   6
Visual outputs:       6
Asset documentation:  Complete

Status: COMPLETE ✅
```
