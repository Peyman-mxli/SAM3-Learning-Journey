# Practical — Advanced MaskAnnotator and SAM2

This folder contains the validated practical implementation for **Session 07 — Advanced MaskAnnotator and SAM2**.

The practical builds directly on the segmentation workflow from Session 06 and focuses on improving how segmentation masks are **visualized, customized, filtered, compared, and reused**.

The implementation combines **YOLOv8**, **SAM 3**, **Supervision**, **OpenCV**, and **Matplotlib** in a reusable image-segmentation workflow.

---

## Practical Objective

The objective of this practical is to demonstrate:

- YOLOv8 object detection
- SAM 3 segmentation
- `sv.Detections`
- `sv.MaskAnnotator`
- `sv.BoxAnnotator`
- Mask opacity control
- Bounding-box and segmentation-mask comparison
- Combining masks with bounding boxes
- Filtering detections before SAM
- Person-only segmentation
- Reusing the same segmentation pipeline with different images
- Saving visual results
- Preparing the conceptual transition toward temporal segmentation with SAM2

---

# Main Workflow

The complete practical follows this pipeline:

```text
Input Image
     ↓
YOLOv8
     ↓
Object Detection
     ↓
sv.Detections
     ↓
Optional Class Filtering
     ↓
Bounding Boxes
     ↓
SAM 3 Prompts
     ↓
Segmentation Masks
     ↓
sv.Detections
     ↓
MaskAnnotator
     ↓
BoxAnnotator
     ↓
Visualization
     ↓
Saved Output
```

---

# Practical Structure

```text
practical/
│
├── README.md
├── advanced_mask_annotator.py
│
└── assets/
    │
    ├── README.md
    │
    ├── input/
    │   ├── README.md
    │   ├── bus.jpg
    │   └── zidane.jpg
    │
    └── output/
        ├── README.md
        ├── bounding_boxes.png
        ├── segmentation_masks.png
        ├── bbox_vs_mask.png
        ├── opacity_comparison.png
        ├── person_only_segmentation.png
        └── second_image_segmentation.png
```

---

# Input Images

The practical uses two Ultralytics sample images:

```text
bus.jpg
zidane.jpg
```

They are stored inside:

```text
assets/input/
```

The first image is used for the main experiments.

The second image is used to verify that the same segmentation pipeline can be reused with a different input.

---

# Models

## YOLOv8

The practical uses:

```text
yolov8n.pt
```

YOLOv8 performs the initial object-detection stage.

The detector provides:

```text
Class ID
Bounding Box
Confidence
```

The resulting detections are converted into:

```python
sv.Detections
```

using:

```python
sv.Detections.from_ultralytics()
```

---

## SAM 3

The practical uses the SAM 3 checkpoint stored externally in Google Drive.

Validated path:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

The checkpoint is not stored in the GitHub repository because of its large file size.

During validation, the checkpoint was successfully detected and loaded.

```text
SAM 3 exists: True
```

---

# YOLO → SAM Integration

YOLO detections provide the spatial prompts used by SAM 3.

The workflow is:

```text
Image
  ↓
YOLOv8
  ↓
Bounding Boxes
  ↓
SAM 3
  ↓
Pixel-Level Masks
```

The bounding boxes are obtained from:

```python
detections.xyxy
```

and converted into prompts using:

```python
detections.xyxy.tolist()
```

SAM 3 then generates a segmentation mask for each supplied bounding box.

---

# Experiment 1 — YOLO + SAM Segmentation

The first experiment processes:

```text
bus.jpg
```

Validated image shape:

```text
(1080, 810, 3)
```

YOLOv8 detected:

```text
4 persons
1 bus
1 stop sign
```

Total:

```text
YOLO detections: 6
```

These six bounding boxes were passed to SAM 3.

SAM 3 generated:

```text
SAM masks: 6
```

Therefore, the validated relationship was:

```text
6 YOLO detections
        ↓
6 bounding-box prompts
        ↓
6 SAM 3 masks
```

This confirms that every YOLO detection used in the experiment produced a corresponding SAM segmentation result.

---

# Experiment 2 — Bounding Boxes vs. Masks

The second experiment compares YOLO bounding boxes with SAM segmentation masks.

Three outputs are generated.

---

## Bounding Boxes

Output:

```text
assets/output/bounding_boxes.png
```

This visualization uses:

```python
sv.BoxAnnotator()
```

and represents the original YOLO detections.

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

## Segmentation Masks

Output:

```text
assets/output/segmentation_masks.png
```

This visualization uses:

```python
sv.MaskAnnotator()
```

and displays the pixel-level SAM 3 segmentation masks.

Conceptually:

```text
YOLO Bounding Boxes
        ↓
SAM 3
        ↓
Segmentation Masks
        ↓
segmentation_masks.png
```

---

## Combined Visualization

Output:

```text
assets/output/bbox_vs_mask.png
```

The segmentation masks are drawn first.

The bounding boxes are then drawn on top.

```text
Original Image
      ↓
SAM Masks
      ↓
YOLO Bounding Boxes
      ↓
Combined Visualization
```

This makes it possible to visually compare the rectangular detection regions with the more precise pixel-level masks.

---

# Bounding Box vs. Segmentation Mask

A bounding box represents an approximate rectangular region:

```text
┌──────────────────────┐
│                      │
│       OBJECT         │
│                      │
└──────────────────────┘
```

The rectangle may contain:

```text
Object Pixels
+
Background Pixels
```

A segmentation mask instead identifies pixels belonging to the object.

```text
Bounding Box
     ↓
Approximate Localization

SAM Mask
     ↓
Pixel-Level Object Shape
```

This demonstrates why segmentation provides more detailed spatial information than bounding boxes alone.

---

# Experiment 3 — Mask Opacity

The third experiment investigates the `opacity` parameter of:

```python
sv.MaskAnnotator()
```

The practical compares:

```text
0.2
0.5
0.9
```

The resulting comparison is stored in:

```text
assets/output/opacity_comparison.png
```

---

## Opacity 0.2

```text
Mask visibility: low
Original image visibility: high
```

The segmentation mask is highly transparent.

---

## Opacity 0.5

```text
Mask visibility: medium
Original image visibility: medium
```

This provides a balanced visualization.

---

## Opacity 0.9

```text
Mask visibility: high
Original image visibility: lower
```

The segmentation region becomes the dominant visual element.

---

# Why Opacity Matters

Mask opacity affects how easily the viewer can inspect both:

```text
Segmentation Result
        +
Original Image Content
```

Different applications may require different visualization priorities.

For example:

```text
Visual Inspection
      ↓
Lower / Medium Opacity

Mask-Focused Analysis
      ↓
Higher Opacity
```

---

# Experiment 4 — Filtering Before SAM

The fourth experiment demonstrates filtering detections before segmentation.

Instead of sending every YOLO detection to SAM, only detections belonging to the `person` class are retained.

COCO class:

```text
person = class_id 0
```

The filtering operation is:

```python
person_detections = yolo_detections[
    yolo_detections.class_id == 0
]
```

Validated result:

```text
Person detections: 4
```

Only those four bounding boxes are then passed to SAM 3.

---

# Person-Only Segmentation

The resulting visualization is stored in:

```text
assets/output/person_only_segmentation.png
```

The workflow is:

```text
bus.jpg
   ↓
YOLOv8
   ↓
6 Total Detections
   ↓
Class Filter
   ↓
4 Person Detections
   ↓
SAM 3
   ↓
Person Segmentation Masks
   ↓
MaskAnnotator
   ↓
BoxAnnotator
   ↓
person_only_segmentation.png
```

This demonstrates how detection filtering can be combined with segmentation.

---

# Why Filter Before SAM?

SAM segmentation is more computationally expensive than simple detection filtering.

Instead of:

```text
Detect Everything
      ↓
Segment Everything
      ↓
Discard Unwanted Objects
```

the practical demonstrates:

```text
Detect Everything
      ↓
Filter Detections
      ↓
Segment Only Relevant Objects
```

This avoids unnecessary segmentation inference.

---

# Experiment 5 — Reusable Pipeline

The fifth experiment processes:

```text
zidane.jpg
```

using the same YOLO → SAM pipeline.

No new segmentation architecture is required.

The workflow remains:

```text
zidane.jpg
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
```

---

# Second Image Validation

YOLOv8 detected:

```text
2 persons
1 tie
```

Total:

```text
YOLO detections: 3
```

Those detections were passed to SAM 3.

The resulting visualization was successfully generated and saved as:

```text
assets/output/second_image_segmentation.png
```

This validates that the practical pipeline is reusable across different images.

---

# Generated Outputs

The practical successfully generated all six expected visualization files:

```text
bounding_boxes.png
segmentation_masks.png
bbox_vs_mask.png
opacity_comparison.png
person_only_segmentation.png
second_image_segmentation.png
```

All outputs are stored in:

```text
assets/output/
```

---

# Output Summary

```text
bus.jpg
│
├── YOLO detections: 6
├── SAM masks: 6
├── Person detections: 4
│
├── bounding_boxes.png
├── segmentation_masks.png
├── bbox_vs_mask.png
├── opacity_comparison.png
└── person_only_segmentation.png


zidane.jpg
│
├── YOLO detections: 3
│
└── second_image_segmentation.png
```

---

# Validation Results

The practical was successfully executed from start to finish in Google Colab.

Validated results:

```text
SAM 3 checkpoint found:          YES

bus.jpg:
YOLO detections:                 6
SAM masks:                       6
Person detections:               4

zidane.jpg:
YOLO detections:                 3

Generated visualization files:   6
```

The execution completed with:

```text
Session 07 practical completed.
```

No fatal runtime errors occurred.

---

# SAM Image-Size Warning

During SAM 3 inference, Ultralytics displayed:

```text
WARNING ⚠️ imgsz=[1024] must be multiple of max stride 14,
updating to [1036]
```

This warning did not stop execution.

SAM automatically adjusted the requested inference size to:

```text
1036 × 1036
```

The segmentation process then completed successfully.

---

# Validated Output Files

The generated files were verified after execution:

```text
bounding_boxes.png
segmentation_masks.png
bbox_vs_mask.png
opacity_comparison.png
person_only_segmentation.png
second_image_segmentation.png
```

The output directory contained approximately:

```text
8.8 MB
```

of generated practical evidence including its documentation.

---

# Reusable Functions

The practical script separates the workflow into reusable functions.

Examples include:

```python
load_image()
run_yolo()
run_sam()
save_image()
create_bbox_vs_mask()
create_opacity_comparison()
save_opacity_figure()
```

This makes the code easier to:

- Understand
- Test
- Reuse
- Extend
- Maintain

---

# Separation of Processing Stages

The implementation separates:

```text
Input Handling
      ↓
Detection
      ↓
Segmentation
      ↓
Visualization
      ↓
Output Saving
```

This is preferable to placing the complete workflow inside one large block of code.

---

# Connection to Previous Sessions

This practical combines concepts from several previous sessions.

```text
YOLO Detection
      ↓
Supervision Detections
      ↓
Detection Filtering
      ↓
SAM Segmentation
      ↓
Mask Visualization
```

Specifically:

```text
Session 01
Supervision + Detections
        ↓
Session 02
Annotation and Visualization
        ↓
Session 03
Filtering Detections
        ↓
Session 06
SAM Segmentation
        ↓
Session 07
Advanced Mask Visualization
```

This demonstrates how the individual course concepts can be combined into increasingly complete Computer Vision workflows.

---

# SAM2 Concept

The practical implementation itself focuses on static-image segmentation with SAM 3.

The lesson also introduces the conceptual transition toward SAM2 and temporal segmentation.

Static segmentation:

```text
Image
  ↓
Prompt
  ↓
Segmentation
  ↓
Mask
```

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
Future Frames
      ↓
Mask Propagation
```

The important conceptual difference is:

```text
Static Segmentation
      ↓
Independent Image

Temporal Segmentation
      ↓
Object Information Across Time
```

---

# Technologies Used

The validated practical uses:

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
- Matplotlib

---

# Practical Validation Checklist

```text
Input directory created                ✅
bus.jpg added                          ✅
zidane.jpg added                       ✅
YOLOv8 loaded                          ✅
SAM 3 checkpoint found                 ✅
SAM 3 loaded                           ✅
YOLO detection executed                ✅
sv.Detections conversion               ✅
Bounding boxes used as SAM prompts     ✅
SAM masks generated                    ✅
MaskAnnotator executed                 ✅
BoxAnnotator executed                  ✅
Opacity 0.2 tested                     ✅
Opacity 0.5 tested                     ✅
Opacity 0.9 tested                     ✅
Person filtering executed              ✅
Person-only segmentation generated     ✅
Second image processed                 ✅
Six output images generated            ✅
Output files verified                  ✅
Outputs preserved in GitHub            ✅
```

---

# Final Result

The validated practical demonstrates the progression:

```text
Object Detection
       ↓
Bounding Boxes
       ↓
SAM 3 Segmentation
       ↓
Pixel-Level Masks
       ↓
Mask Visualization
       ↓
Opacity Customization
       ↓
Detection Filtering
       ↓
Selective Segmentation
       ↓
Reusable Pipeline
```

The final execution confirmed:

```text
bus.jpg
6 YOLO detections
6 SAM masks
4 person detections

zidane.jpg
3 YOLO detections

6 visualization outputs generated
```

The practical therefore successfully extends the basic SAM segmentation workflow from Session 06 into a more configurable and reusable segmentation-visualization pipeline.

**Status: Completed and validated successfully in Google Colab.**
