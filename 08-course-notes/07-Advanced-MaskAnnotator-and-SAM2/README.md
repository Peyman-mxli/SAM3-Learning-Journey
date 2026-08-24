# 07 — Advanced MaskAnnotator and SAM2

This session continues the segmentation workflow introduced in **Session 06 — Segmentation with SAM**.

The main focus is no longer only generating segmentation masks.

Instead, this lesson explores how to **visualize, customize, filter, compare, and reuse segmentation masks** using `sv.MaskAnnotator`, while also introducing the concept of **temporal segmentation with SAM2**.

The session includes both the original course notebook and a complete practical implementation validated in Google Colab.

---

## Session Status

```text
Session:        07 — Advanced MaskAnnotator and SAM2
Course Notes:   ✅ Completed
Class Notebook: ✅ Preserved
Class Recording:✅ Added
Practical:      ✅ Completed
Colab Test:     ✅ Passed
Outputs:        ✅ Generated and preserved
```

---

# Session Objective

The objective of this lesson is to understand how segmentation masks can be transformed from raw model outputs into clear, configurable, and reusable visual results.

The session extends the segmentation concepts from Session 06 by introducing:

- Advanced mask visualization
- Mask opacity control
- Bounding-box and mask composition
- Detection filtering before segmentation
- Selective class segmentation
- Reusable segmentation functions
- Multiple-image processing
- The conceptual transition from static segmentation to temporal segmentation

The session also introduces the transition from:

```text
Static Image Segmentation
          ↓
Temporal Video Segmentation
```

using the memory-based ideas behind SAM2.

---

# Topics Covered

This session covers:

- `sv.MaskAnnotator`
- `sv.BoxAnnotator`
- Segmentation-mask visualization
- Mask opacity
- Combining masks with bounding boxes
- Comparing YOLO boxes with SAM masks
- Filtering detections before segmentation
- Segmenting only a selected class
- Reusing the same segmentation pipeline on different images
- Reusable processing functions
- YOLOv8 + SAM 3 integration
- `sv.Detections`
- Introduction to SAM2
- Temporal segmentation
- Video object-mask propagation
- Memory-based segmentation concepts
- Practical validation in Google Colab

---

# Session Structure

```text
07-Advanced-MaskAnnotator-and-SAM2/
│
├── README.md
├── CLASS-RECORDING.md
├── 03_b_sam_mask_annotator.ipynb
│
└── practical/
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

# Class Recording

The class recording for this session is available on YouTube:

[MaskAnnotator avanzado y SAM2](https://youtu.be/GNwQl-hy8Yw)

Repository documentation:

[CLASS-RECORDING.md](./CLASS-RECORDING.md)

---

# Original Class Notebook

The original class notebook is preserved as:

[03_b_sam_mask_annotator.ipynb](./03_b_sam_mask_annotator.ipynb)

The notebook contains the original lesson experiments and serves as the source course artifact for this session.

The practical implementation expands those concepts into a structured and reusable Python workflow.

---

# Relationship to Session 06

Session 06 introduced the basic segmentation pipeline:

```text
Input Image
     ↓
YOLOv8
     ↓
Bounding Boxes
     ↓
SAM 3
     ↓
Segmentation Masks
```

Session 07 builds on those masks.

The progression becomes:

```text
YOLO Detection
      ↓
Bounding Boxes
      ↓
SAM 3 Segmentation
      ↓
Boolean Masks
      ↓
MaskAnnotator
      ↓
Visualization
      ↓
Opacity Customization
      ↓
Detection Filtering
      ↓
Selective Segmentation
      ↓
Reusable Pipeline
```

Session 06 therefore answers:

```text
How do I generate a segmentation mask?
```

Session 07 extends that question to:

```text
How do I visualize it?
How do I customize it?
How do I combine it with detections?
How do I filter objects before segmentation?
How do I reuse the workflow?
```

---

# YOLO + SAM Initialization

The lesson begins by recreating the segmentation pipeline from the previous session.

The core workflow uses:

```python
import supervision as sv
from ultralytics import YOLO, SAM
import cv2
import matplotlib.pyplot as plt
```

The models are loaded using:

```python
yolo_model = YOLO("yolov8n.pt")
sam_model = SAM(sam_path)
```

YOLO detects objects:

```python
yolo_results = yolo_model(image)[0]
```

The results are converted into Supervision detections:

```python
yolo_detections = sv.Detections.from_ultralytics(
    yolo_results
)
```

The YOLO bounding boxes are then used as SAM prompts:

```python
sam_results = sam_model(
    image,
    bboxes=yolo_detections.xyxy.tolist()
)[0]
```

Finally, the SAM results are converted into:

```python
sam_detections = sv.Detections.from_ultralytics(
    sam_results
)
```

The resulting workflow is:

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
sv.Detections
```

---

# MaskAnnotator

The main Supervision component introduced in this lesson is:

```python
sv.MaskAnnotator()
```

`MaskAnnotator` draws segmentation masks directly over the original image.

Example:

```python
mask_annotator = sv.MaskAnnotator(
    opacity=0.6
)
```

The masks are applied using:

```python
annotated_sam = mask_annotator.annotate(
    scene=image.copy(),
    detections=sam_detections
)
```

This converts raw segmentation data into a visible overlay that can be inspected and combined with other annotations.

---

# Bounding Boxes vs. Segmentation Masks

The lesson compares two different ways of representing detected objects.

## Bounding Box

YOLO produces:

```text
Rectangular Object Region
```

Conceptually:

```text
┌──────────────────────┐
│                      │
│       OBJECT         │
│                      │
└──────────────────────┘
```

The rectangle usually contains:

```text
Object Pixels
+
Background Pixels
```

A bounding box therefore provides efficient object localization but does not describe the exact object shape.

---

## Segmentation Mask

SAM produces:

```text
Pixel-Level Object Region
```

Only pixels assigned to the object are represented by the mask.

Conceptually:

```text
Bounding Box
     ↓
Approximate Region

SAM Mask
     ↓
Pixel-Level Object Shape
```

This makes segmentation more precise for object-shape and spatial analysis.

---

# Combining MaskAnnotator and BoxAnnotator

The session demonstrates how different Supervision annotators can be composed.

Example:

```python
mask_annotator = sv.MaskAnnotator(
    opacity=0.6
)

box_annotator = sv.BoxAnnotator()
```

The mask is applied first:

```python
annotated_sam = mask_annotator.annotate(
    scene=image.copy(),
    detections=sam_detections
)
```

Bounding boxes can then be drawn on the visualization.

Conceptually:

```text
Original Image
      ↓
SAM Mask
      ↓
Bounding Box
      ↓
Final Visualization
```

This makes it possible to inspect the relationship between detection localization and pixel-level segmentation.

---

# Practical Implementation

The completed practical is available here:

[practical/](./practical/)

Main Python implementation:

[advanced_mask_annotator.py](./practical/advanced_mask_annotator.py)

The practical converts the lesson concepts into a reusable pipeline containing separate functions for:

```text
Image Loading
     ↓
YOLO Detection
     ↓
SAM Segmentation
     ↓
Mask Visualization
     ↓
Opacity Experiments
     ↓
Class Filtering
     ↓
Output Generation
```

The implementation was executed and validated successfully in Google Colab.

---

# Mask Opacity

One of the most important `MaskAnnotator` parameters explored in this session is:

```python
opacity=
```

Opacity controls how strongly the segmentation mask is displayed over the original image.

Example:

```python
sv.MaskAnnotator(
    opacity=0.6
)
```

Conceptually:

```text
Higher Opacity
      ↓
Mask More Visible
      ↓
Original Object Less Visible
```

while:

```text
Lower Opacity
      ↓
Mask More Transparent
      ↓
Original Object More Visible
```

This parameter makes it possible to adapt the visualization depending on whether the original image or the segmentation result should receive more visual emphasis.

---

# Opacity Experiment

The session compares three mask-opacity values:

```text
0.2
0.5
0.9
```

The practical implementation reproduces this experiment programmatically.

```python
opacity_values = [
    0.2,
    0.5,
    0.9
]
```

For each value:

```python
annotator = sv.MaskAnnotator(
    opacity=opacity
)

annotated = annotator.annotate(
    scene=image.copy(),
    detections=sam_detections
)
```

The three results are combined into a comparison visualization.

Validated output:

[opacity_comparison.png](./practical/assets/output/opacity_comparison.png)

---

## Low Opacity — 0.2

```text
opacity = 0.2
```

Result:

```text
Original Image Visibility: High
Mask Visibility:           Low
```

The segmentation overlay remains transparent enough to clearly inspect the original image.

---

## Medium Opacity — 0.5

```text
opacity = 0.5
```

Result:

```text
Original Image Visibility: Medium
Mask Visibility:           Medium
```

This provides a balanced visualization between the original image and segmentation mask.

---

## High Opacity — 0.9

```text
opacity = 0.9
```

Result:

```text
Original Image Visibility: Lower
Mask Visibility:           High
```

The segmentation regions become the dominant visual element.

---

# Why Opacity Matters

Different Computer Vision applications may require different visualization priorities.

For example:

```text
Inspect Original Object Appearance
              ↓
       Lower Opacity
```

while:

```text
Inspect Segmentation Region
              ↓
       Higher Opacity
```

The opacity experiment demonstrates that segmentation visualization is not limited to simply drawing masks.

The presentation itself can be configured depending on the analysis goal.

---

# Practical Experiment 1 — YOLO + SAM Segmentation

The validated practical begins with:

```text
bus.jpg
```

Input location:

[bus.jpg](./practical/assets/input/bus.jpg)

The image shape during validation was:

```text
(1080, 810, 3)
```

YOLOv8 detected:

```text
4 persons
1 bus
1 stop sign
```

Total detections:

```text
YOLO detections: 6
```

The six YOLO bounding boxes were then passed to SAM 3.

SAM 3 generated:

```text
SAM masks: 6
```

The validated pipeline therefore produced:

```text
bus.jpg
   ↓
YOLOv8
   ↓
6 Detections
   ↓
6 Bounding-Box Prompts
   ↓
SAM 3
   ↓
6 Segmentation Masks
```

This confirms that each YOLO bounding-box prompt used in this experiment produced a corresponding SAM segmentation result.

---

# Practical Experiment 2 — Bounding Boxes

The first generated visualization contains the original YOLO detections.

Output:

[bounding_boxes.png](./practical/assets/output/bounding_boxes.png)

The visualization is created using:

```python
sv.BoxAnnotator()
```

Workflow:

```text
bus.jpg
   ↓
YOLOv8
   ↓
6 Object Detections
   ↓
BoxAnnotator
   ↓
bounding_boxes.png
```

This output provides the detection baseline used for comparison with segmentation.

---

# Practical Experiment 3 — Segmentation Masks

The next output visualizes the SAM 3 masks.

Output:

[segmentation_masks.png](./practical/assets/output/segmentation_masks.png)

The visualization uses:

```python
sv.MaskAnnotator(
    opacity=0.6
)
```

Workflow:

```text
bus.jpg
   ↓
YOLOv8
   ↓
Bounding Boxes
   ↓
SAM 3
   ↓
6 Segmentation Masks
   ↓
MaskAnnotator
   ↓
segmentation_masks.png
```

This visualization shows the transition from object detection to pixel-level segmentation.

---

# Practical Experiment 4 — Bounding Boxes + Masks

The practical also combines the segmentation masks and bounding boxes into one visualization.

Output:

[bbox_vs_mask.png](./practical/assets/output/bbox_vs_mask.png)

The workflow is:

```text
Original Image
      ↓
SAM 3 Masks
      ↓
MaskAnnotator
      ↓
YOLO Bounding Boxes
      ↓
BoxAnnotator
      ↓
bbox_vs_mask.png
```

This makes it easier to compare:

```text
YOLO
 ↓
Rectangular Localization
```

with:

```text
SAM 3
 ↓
Pixel-Level Segmentation
```

The combined visualization demonstrates how detection and segmentation can complement each other inside the same pipeline.

---

# Filtering Before SAM

The session also combines detection filtering with segmentation.

Instead of sending every YOLO detection to SAM, detections can first be filtered by class.

For the `person` class:

```python
person_detections = yolo_detections[
    yolo_detections.class_id == 0
]
```

COCO class:

```text
0 → person
```

The workflow becomes:

```text
Image
  ↓
YOLOv8
  ↓
All Detections
  ↓
Class Filtering
  ↓
Persons Only
  ↓
Bounding Boxes
  ↓
SAM 3
  ↓
Person Masks
```

This combines concepts from earlier detection-filtering sessions with the SAM segmentation workflow.

---

# Practical Experiment 5 — Person-Only Segmentation

During validation, `bus.jpg` produced:

```text
Total YOLO detections: 6
Person detections:     4
```

Instead of sending all six detections to SAM again, the practical sends only the four person detections.

```text
6 Total Detections
        ↓
Person Class Filter
        ↓
4 Person Detections
        ↓
SAM 3
        ↓
Person Segmentation
```

Validated output:

[person_only_segmentation.png](./practical/assets/output/person_only_segmentation.png)

The result demonstrates selective segmentation based on detection class.

---

# Why Filter Before SAM?

Filtering before segmentation avoids unnecessary processing.

Instead of:

```text
Detect Everything
      ↓
Segment Everything
      ↓
Filter Afterwards
```

the practical uses:

```text
Detect Everything
      ↓
Filter Detections
      ↓
Segment Relevant Objects
```

This is especially useful when only one class or a small subset of detected objects is relevant to the application.

The detection model therefore acts as a first-stage selector for the segmentation model.

---

# Reusing the Same Pipeline

The session also demonstrates that the segmentation workflow is not tied to `bus.jpg`.

A second image is processed:

[zidane.jpg](./practical/assets/input/zidane.jpg)

The same processing stages are reused:

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
Segmentation Masks
   ↓
MaskAnnotator
   ↓
BoxAnnotator
```

The processing logic does not need to be rewritten for the second image.

---

# Practical Experiment 6 — Second Image

During validation, YOLOv8 detected:

```text
2 persons
1 tie
```

Total:

```text
YOLO detections: 3
```

These detections were passed through the same segmentation pipeline.

Validated output:

[second_image_segmentation.png](./practical/assets/output/second_image_segmentation.png)

The experiment confirms:

```text
bus.jpg
   ↓
Same Pipeline
   ↓
Segmentation Result

zidane.jpg
   ↓
Same Pipeline
   ↓
Segmentation Result
```

This demonstrates the importance of reusable Computer Vision processing functions.

---

# Reusable Practical Functions

The practical implementation separates the workflow into reusable functions.

The main functions include:

```python
load_image()
run_yolo()
run_sam()
save_image()
create_bbox_vs_mask()
create_opacity_comparison()
save_opacity_figure()
```

Instead of placing the complete application inside one large block, the processing stages are separated into reusable components.

Conceptually:

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

This makes the practical easier to:

- Read
- Debug
- Test
- Extend
- Reuse
- Maintain

---

# Validated Output Gallery

The completed practical generated six visual outputs:

### YOLO Bounding Boxes

[View `bounding_boxes.png`](./practical/assets/output/bounding_boxes.png)

### SAM 3 Segmentation Masks

[View `segmentation_masks.png`](./practical/assets/output/segmentation_masks.png)

### Bounding Boxes + Segmentation Masks

[View `bbox_vs_mask.png`](./practical/assets/output/bbox_vs_mask.png)

### Mask Opacity Comparison

[View `opacity_comparison.png`](./practical/assets/output/opacity_comparison.png)

### Person-Only Segmentation

[View `person_only_segmentation.png`](./practical/assets/output/person_only_segmentation.png)

### Second Image Segmentation

[View `second_image_segmentation.png`](./practical/assets/output/second_image_segmentation.png)

---

# Practical Results Summary

The validated results are:

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

Total generated visualization outputs:

```text
6
```

All six expected output files were successfully generated and verified.

---

# Google Colab Validation

The complete practical was executed in Google Colab after the repository was cloned and the required dependencies were installed.

The SAM 3 checkpoint was loaded from Google Drive:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

The model validation returned:

```text
SAM 3 exists: True
Path: /content/drive/MyDrive/SAM3-Models/sam3.pt
```

Both YOLOv8 and SAM 3 loaded successfully.

```text
Loading YOLOv8...
Loading SAM 3...
Models loaded successfully.
```

The complete practical then executed from beginning to end without a fatal runtime error.

---

# Validation Results

The validated execution produced:

```text
============================================================
Session 07 — Advanced MaskAnnotator and SAM2
============================================================

bus.jpg
Image shape: (1080, 810, 3)

YOLO detections: 6
SAM masks: 6

Person detections: 4

zidane.jpg
YOLO detections: 3

Generated visualization outputs: 6

============================================================
Session 07 practical completed.
============================================================
```

The six expected output images were also verified in the output directory.

```text
bounding_boxes.png
segmentation_masks.png
bbox_vs_mask.png
opacity_comparison.png
person_only_segmentation.png
second_image_segmentation.png
```

---

# SAM 3 Image-Size Warning

During SAM 3 inference, Ultralytics displayed the following warning:

```text
WARNING ⚠️ imgsz=[1024] must be multiple of max stride 14,
updating to [1036]
```

This was not a fatal error.

SAM automatically adjusted the inference image size from:

```text
1024
```

to:

```text
1036
```

because the image size must be compatible with the model stride.

The segmentation process continued normally after the automatic adjustment.

The successful execution confirmed that this warning did not prevent the practical from generating the expected masks and output visualizations.

---

# Output Verification

After execution, the output directory contained:

```text
bbox_vs_mask.png
bounding_boxes.png
opacity_comparison.png
person_only_segmentation.png
second_image_segmentation.png
segmentation_masks.png
```

The generated practical evidence occupied approximately:

```text
8.8 MB
```

including the output documentation.

All expected visualization files were present.

---

# Practical Validation Checklist

```text
Repository cloned                       ✅
Dependencies installed                  ✅
Google Drive mounted                    ✅
SAM 3 checkpoint found                  ✅
YOLOv8 loaded                           ✅
SAM 3 loaded                            ✅
bus.jpg loaded                          ✅
zidane.jpg loaded                       ✅
YOLO detection executed                 ✅
sv.Detections conversion                ✅
Bounding boxes used as SAM prompts      ✅
SAM masks generated                     ✅
MaskAnnotator executed                  ✅
BoxAnnotator executed                   ✅
Opacity 0.2 tested                      ✅
Opacity 0.5 tested                      ✅
Opacity 0.9 tested                      ✅
Person filtering executed               ✅
Person-only segmentation generated      ✅
Second image processed                  ✅
Six visualization outputs generated     ✅
Output files verified                   ✅
Outputs preserved in repository         ✅
```

---

# Combining Previous Lessons

Session 07 brings together concepts introduced throughout the learning journey.

```text
Object Detection
      ↓
Supervision Detections
      ↓
Detection Filtering
      ↓
Bounding-Box Prompts
      ↓
SAM Segmentation
      ↓
Mask Visualization
      ↓
Reusable Pipeline
```

The progression can be understood as:

```text
Session 01
Introduction to Supervision
        ↓
sv.Detections

Session 02
Annotation and Visualization
        ↓
Annotators

Session 03
Filtering and Manipulating Detections
        ↓
Class Filtering

Session 04
Object Tracking
        ↓
Persistent Object Analysis

Session 05
Zones and Counting
        ↓
Spatial Analysis

Session 06
Segmentation with SAM
        ↓
Pixel-Level Masks

Session 07
Advanced MaskAnnotator and SAM2
        ↓
Advanced Mask Visualization
+
Selective Segmentation
+
Reusable Pipelines
+
Temporal Segmentation Concepts
```

This demonstrates how individual Computer Vision concepts can be combined into progressively more complete processing systems.

---

# SAM2 Introduction

The final conceptual part of this session introduces **SAM2** and temporal segmentation.

The practical implementation in this session uses SAM 3 for the validated static-image segmentation workflow.

SAM2 is introduced as a lesson concept to explain how segmentation can extend from individual images toward video sequences.

---

# Static Image Segmentation

Static segmentation processes an image independently.

```text
Image
  ↓
Object Prompt
  ↓
Segmentation Model
  ↓
Object Mask
```

When another image is processed, the segmentation operation starts again using that image and its prompts.

Conceptually:

```text
Image A
   ↓
Segmentation
   ↓
Mask A

Image B
   ↓
Segmentation
   ↓
Mask B
```

There is no temporal relationship required between the two images.

---

# Temporal Video Segmentation

Video introduces a sequence of related frames.

```text
Frame 1
   ↓
Object Mask
   ↓
Frame 2
   ↓
Object Mask
   ↓
Frame 3
   ↓
Object Mask
```

The challenge is no longer simply:

```text
Where is the object in this image?
```

Instead, the system must also consider:

```text
How does the same object continue through time?
```

This introduces temporal segmentation.

---

# Temporal Memory

A central concept introduced with SAM2 is memory across video frames.

Conceptually:

```text
Frame 1
   ↓
Object Segmentation
   ↓
Memory
   ↓
Frame 2
   ↓
Previous Object Information
   +
Current Frame
   ↓
Updated Mask
```

The information from previous frames can help maintain the segmentation of an object as the video progresses.

---

# Mask Propagation

Temporal segmentation allows an initial object segmentation to influence later frames.

Conceptually:

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

Instead of treating every frame as an unrelated image, the segmentation system can use information from previous frames.

---

# Static vs. Temporal Segmentation

## Static

```text
Image
  ↓
Prompt
  ↓
Segmentation
  ↓
Mask
```

Each image can be processed independently.

## Temporal

```text
Initial Frame
     ↓
Object Prompt
     ↓
Initial Mask
     ↓
Memory
     ↓
Next Frame
     ↓
Updated Object Mask
     ↓
Future Frames
```

The key conceptual difference is:

```text
Static Segmentation
        ↓
Spatial Information

Temporal Segmentation
        ↓
Spatial Information
        +
Temporal Information
```

---

# From Session 07 Toward Video Segmentation

The current validated practical focuses on:

```text
Image
  ↓
YOLOv8
  ↓
SAM 3
  ↓
Segmentation Mask
  ↓
MaskAnnotator
```

The conceptual next stage becomes:

```text
Video
  ↓
Initial Object Detection / Prompt
  ↓
Initial Segmentation
  ↓
Temporal Memory
  ↓
Frame Sequence
  ↓
Mask Propagation
  ↓
Persistent Video Segmentation
```

This creates a bridge between the static segmentation concepts practiced in this session and more advanced video-segmentation workflows.

---

# Technologies Used

The completed Session 07 work uses:

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

The lesson additionally introduces the concepts of:

- SAM2
- Temporal segmentation
- Temporal memory
- Video mask propagation

---

# Learning Outcomes

After completing this session, I understand:

- How `sv.MaskAnnotator` visualizes segmentation masks
- How mask opacity affects visualization
- How to compare bounding boxes with pixel-level masks
- How to combine `MaskAnnotator` and `BoxAnnotator`
- How YOLO detections can become SAM prompts
- How `sv.Detections` connects model outputs with Supervision
- Why detections can be filtered before segmentation
- How to segment only a selected class
- How filtering can reduce unnecessary segmentation work
- How the same segmentation pipeline can process multiple images
- How to structure segmentation logic into reusable functions
- How to save and validate visual segmentation outputs
- The difference between static and temporal segmentation
- The role of memory in temporal segmentation
- The concept of mask propagation across video frames
- How the lesson concepts prepare for more advanced video segmentation

---

# Final Practical Result

The completed practical produced:

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

Total validated visual outputs:

```text
6
```

The complete practical successfully demonstrates:

```text
Detection
    ↓
Segmentation
    ↓
Visualization
    ↓
Customization
    ↓
Filtering
    ↓
Selective Segmentation
    ↓
Pipeline Reuse
```

---

# Session Resources

## Class Recording

[MaskAnnotator avanzado y SAM2](https://youtu.be/GNwQl-hy8Yw)

## Recording Documentation

[CLASS-RECORDING.md](./CLASS-RECORDING.md)

## Original Notebook

[03_b_sam_mask_annotator.ipynb](./03_b_sam_mask_annotator.ipynb)

## Practical

[practical/README.md](./practical/README.md)

## Practical Python Script

[advanced_mask_annotator.py](./practical/advanced_mask_annotator.py)

## Input Assets

[practical/assets/input/](./practical/assets/input/)

## Generated Outputs

[practical/assets/output/](./practical/assets/output/)

---

# Session Progression

The learning progression now becomes:

```text
00 — Agentic AI Programming
        ↓
01 — Introduction to Supervision
        ↓
02 — Annotation and Visualization
        ↓
03 — Filtering and Manipulating Detections
        ↓
04 — Object Tracking
        ↓
05 — Zones and Counting
        ↓
06 — Segmentation with SAM
        ↓
07 — Advanced MaskAnnotator and SAM2
```

Session 07 builds directly on the segmentation masks introduced in Session 06 and extends the workflow into **advanced visualization, selective segmentation, reusable processing, and temporal segmentation concepts**.

---

# Status

```text
Session 07 — Advanced MaskAnnotator and SAM2

Documentation:        COMPLETE ✅
Class Notebook:       COMPLETE ✅
Class Recording:      COMPLETE ✅
Practical Code:       COMPLETE ✅
Input Assets:         COMPLETE ✅
Colab Validation:     PASSED   ✅
Generated Outputs:    6 / 6    ✅
Practical Evidence:   COMPLETE ✅
```

**Session 07 completed and validated successfully.**
