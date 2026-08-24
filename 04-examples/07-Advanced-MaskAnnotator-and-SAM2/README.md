# 07 — Advanced MaskAnnotator and SAM2

This directory contains focused Python examples based on **Session 07 — Advanced MaskAnnotator and SAM2** from my SAM3 Computer Vision Learning Journey.

These examples continue from the segmentation concepts introduced in Session 06 and focus on how segmentation masks can be **visualized, customized, filtered, combined with detections, and reused**.

The session also introduces the conceptual transition from static-image segmentation toward temporal segmentation with SAM2.

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

## Example Structure

```text
07-Advanced-MaskAnnotator-and-SAM2/
├── README.md
├── 01_basic_mask_annotator.py
├── 02_mask_opacity.py
├── 03_mask_and_box_annotator.py
├── 04_person_only_segmentation.py
├── 05_reusable_segmentation_function.py
└── 06_sam2_temporal_concept.py
```

Each script focuses on one specific concept so the workflow can be studied step by step.

---

## 01 — Basic MaskAnnotator

File:

```text
01_basic_mask_annotator.py
```

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

The YOLO bounding boxes are then used as prompts for SAM 3.

SAM 3 generates pixel-level segmentation masks.

Finally, `MaskAnnotator` overlays those masks on the original image.

The main concept is:

```text
Detection
    ↓
Segmentation
    ↓
Mask Visualization
```

---

## 02 — Mask Opacity

File:

```text
02_mask_opacity.py
```

This example demonstrates how the `opacity` parameter changes the appearance of segmentation masks.

The session experiments with:

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
Transparent Mask
    ↓
Original Image Highly Visible
```

```text
Opacity 0.5
    ↓
Balanced Visualization
```

```text
Opacity 0.9
    ↓
Strong Mask Visualization
```

This example demonstrates that segmentation visualization can be customized depending on the purpose of the analysis.

---

## 03 — Mask + Box Annotator

File:

```text
03_mask_and_box_annotator.py
```

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
SAM Segmentation Masks
      ↓
MaskAnnotator
      ↓
YOLO Bounding Boxes
      ↓
BoxAnnotator
      ↓
Combined Visualization
```

This makes it possible to visually compare:

```text
Bounding Box
     ↓
Approximate Rectangular Region
```

with:

```text
Segmentation Mask
     ↓
Pixel-Level Object Shape
```

Bounding boxes are useful for localization.

Segmentation masks provide a more precise representation of the visible object.

---

## 04 — Person-Only Segmentation

File:

```text
04_person_only_segmentation.py
```

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
MaskAnnotator
```

The important concept is:

```text
Filter BEFORE segmentation
```

instead of:

```text
Segment everything
then filter afterwards
```

This reduces unnecessary SAM inference when only specific object classes are relevant.

---

## 05 — Reusable Segmentation Function

File:

```text
05_reusable_segmentation_function.py
```

This example restructures the YOLO + SAM workflow into reusable functions.

Instead of repeating the same processing steps for every image, the segmentation logic is encapsulated.

Conceptually:

```python
segment_image(image)
```

can perform:

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
Annotation
  ↓
Result
```

The same function can then be reused with different images.

For example:

```text
Image A
   ↓
Same Function
   ↓
Segmentation Result A

Image B
   ↓
Same Function
   ↓
Segmentation Result B
```

This demonstrates an important software-engineering principle:

```text
Reusable Pipeline
instead of
Repeated Code
```

---

## 06 — SAM2 Temporal Concept

File:

```text
06_sam2_temporal_concept.py
```

This example focuses on the conceptual transition introduced at the end of Session 07.

The validated practical in this session uses SAM 3 for static-image segmentation.

SAM2 is introduced to explain how segmentation can extend toward video.

Static segmentation works like this:

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
Future Frames
```

The key idea is:

```text
Static Segmentation
        ↓
Spatial Information
```

while:

```text
Temporal Segmentation
        ↓
Spatial Information
        +
Temporal Information
```

The purpose of this example is to make the temporal segmentation workflow easier to understand before implementing a complete video pipeline.

---

## Relationship to Session 06

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

## Technologies Used

The examples use:

- Python
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

## SAM 3 Model

The SAM 3 checkpoint is not stored inside this repository because the model file is very large.

The validated Google Colab environment uses:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

Examples requiring SAM 3 should verify that this model exists before running segmentation.

---

## Learning Progression

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

This progression moves from simple mask visualization toward reusable and more advanced segmentation workflows.

---

## Learning Goal

The purpose of these examples is to understand segmentation as more than simply generating a mask.

The complete progression is:

```text
Object Detection
       ↓
SAM Segmentation
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

These examples provide the foundation for more advanced image and video segmentation systems.
