# 07 — Advanced MaskAnnotator and SAM2

This session continues the segmentation workflow introduced in **Session 06 — Segmentation with SAM**.

The main focus is no longer only generating segmentation masks.

Instead, this lesson explores how to **visualize, customize, filter, and reuse segmentation masks** using `sv.MaskAnnotator`, while also introducing the concept of **temporal segmentation with SAM2**.

---

## Session Objective

The objective of this lesson is to understand how segmentation masks can be transformed from raw model outputs into clear and configurable visual results.

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
- Segmentation-mask visualization
- Mask opacity
- Combining masks with bounding boxes
- Comparing YOLO boxes with SAM masks
- Filtering detections before segmentation
- Segmenting only a selected class
- Reusing the same segmentation pipeline on different images
- Framework-independent processing
- YOLO + SAM integration
- Introduction to SAM2
- Temporal segmentation
- Video object-mask propagation
- Memory-based segmentation concepts

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

The new progression becomes:

```text
YOLO Detection
      ↓
Bounding Boxes
      ↓
SAM Segmentation
      ↓
Boolean Masks
      ↓
MaskAnnotator
      ↓
Visualization
      ↓
Customization
      ↓
Filtering
      ↓
Reusable Segmentation Pipeline
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

Finally, SAM results are converted into:

```python
sam_detections = sv.Detections.from_ultralytics(
    sam_results
)
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

The masks are then applied using:

```python
annotated_sam = mask_annotator.annotate(
    scene=image.copy(),
    detections=sam_detections
)
```

This converts the raw segmentation data into a visible overlay.

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

The rectangle usually includes both:

```text
Object Pixels
+
Background Pixels
```

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
Actual Object Shape
```

This makes segmentation more precise for object-shape analysis.

---

# Combining MaskAnnotator and BoxAnnotator

The session demonstrates how different Supervision Annotators can be composed.

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

Then bounding boxes are drawn on top:

```python
annotated_sam = box_annotator.annotate(
    scene=annotated_sam,
    detections=sam_detections
)
```

The resulting visualization combines:

```text
Original Image
      ↓
SAM Mask
      ↓
Bounding Box
      ↓
Final Visualization
```

---

# Mask Opacity

One of the most important `MaskAnnotator` parameters is:

```python
opacity=
```

Opacity controls how strongly the segmentation overlay covers the original image.

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

---

# Opacity Experiment

The lesson compares three mask-opacity values:

```text
0.2
0.5
0.9
```

The experiment uses:

```python
for opacity in [
    0.2,
    0.5,
    0.9
]:
    ann = sv.MaskAnnotator(
        opacity=opacity
    )
```

This makes it possible to visually compare how opacity affects segmentation presentation.

---

## Low Opacity

Example:

```text
opacity = 0.2
```

Result:

```text
Original image highly visible
Mask lightly visible
```

Useful when the original object appearance is important.

---

## Medium Opacity

Example:

```text
opacity = 0.5
```

Result:

```text
Balanced mask and image visibility
```

Useful for general-purpose visualization.

---

## High Opacity

Example:

```text
opacity = 0.9
```

Result:

```text
Mask strongly visible
Original object less visible
```

Useful when the segmentation region itself is the main focus.

---

# Filtering Before SAM

The lesson also combines concepts from earlier detection-filtering sessions with segmentation.

Instead of sending every YOLO detection to SAM, detections can be filtered first.

Example:

```python
solo_personas = yolo_detections[
    yolo_detections.class_id == 0
]
```

COCO class:

```text
0 → person
```

The filtered bounding boxes are then used as SAM prompts:

```python
bboxes_personas = (
    solo_personas.xyxy.tolist()
)
```

followed by:

```python
sam_personas_results = sam_model(
    image,
    bboxes=bboxes_personas
)[0]
```

---

# Why Filter Before SAM?

The workflow becomes:

```text
Image
  ↓
YOLO
  ↓
All Detections
  ↓
Class Filtering
  ↓
Persons Only
  ↓
SAM
  ↓
Person Masks
```

This is more efficient than:

```text
Image
  ↓
YOLO
  ↓
SAM on Every Object
  ↓
Filter Masks Afterwards
```

because segmentation inference is more computationally expensive than simple detection filtering.

Filtering first reduces unnecessary segmentation work.

---

# Combining Previous Lessons

This experiment connects multiple concepts from earlier sessions.

```text
Detection
    ↓
Supervision Detections
    ↓
Class Filtering
    ↓
Bounding-Box Prompts
    ↓
SAM Segmentation
    ↓
Mask Visualization
```

This demonstrates how individual lessons begin to form larger reusable Computer Vision pipelines.

---

# Reusing the Same Pipeline

The session also tests the same segmentation workflow on a second image:

```text
zidane.jpg
```

The important observation is that the pipeline itself does not need to change.

The workflow remains:

```text
New Image
   ↓
YOLO
   ↓
sv.Detections
   ↓
Bounding Boxes
   ↓
SAM
   ↓
sv.Detections
   ↓
MaskAnnotator
```

Only the image input changes.

---

# Framework-Agnostic Workflow

The second-image experiment demonstrates an important software-engineering concept.

The processing pipeline is reusable because the code operates on common data structures.

Conceptually:

```text
Image A
   ↓
Same Pipeline
   ↓
Segmentation Result A

Image B
   ↓
Same Pipeline
   ↓
Segmentation Result B
```

This reduces duplicated code and makes the pipeline easier to reuse in larger applications.

---

# SAM2 Introduction

The final part of the lesson introduces **SAM2** and the concept of temporal segmentation.

Static-image segmentation processes one image independently:

```text
Image
  ↓
Segmentation
  ↓
Mask
```

Video segmentation introduces time:

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

A temporal segmentation system attempts to maintain information about the same object across frames.

---

# Temporal Memory

SAM2 introduces a memory mechanism designed for video segmentation.

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
Current Image
   ↓
Updated Mask
```

The memory allows object-mask information to propagate through time.

---

# Static vs. Temporal Segmentation

## Static Image Segmentation

```text
Image
  ↓
Prompt
  ↓
SAM
  ↓
Mask
```

Each image is processed independently.

---

## Temporal Video Segmentation

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

The goal is to preserve object segmentation across a video sequence.

---

# Input Assets

The lesson notebook uses two Ultralytics sample images:

```text
bus.jpg
zidane.jpg
```

These files are downloaded into:

```text
assets/
```

using Python.

The first image is used for the primary segmentation experiments.

The second image demonstrates that the same pipeline can be reused without rewriting the segmentation logic.

---

# Technologies Used

This lesson uses:

- Python
- Google Colab
- OpenCV
- NumPy
- Matplotlib
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- `sv.Detections`
- `sv.MaskAnnotator`
- `sv.BoxAnnotator`

The lesson also introduces the conceptual transition toward:

- SAM2
- Video segmentation
- Temporal memory
- Mask propagation

---

# Notebook

The original class notebook for this session is:

```text
03_b_sam_mask_annotator.ipynb
```

It should be preserved inside this session directory as the original course artifact.

Recommended structure:

```text
07-Advanced-MaskAnnotator-and-SAM2/
│
├── README.md
├── 03_b_sam_mask_annotator.ipynb
│
└── practical/
```

Additional practical files and documentation can be added after the original class notebook has been preserved.

---

# Learning Outcomes

After completing this session, I understand:

- How `sv.MaskAnnotator` visualizes segmentation masks
- How mask opacity changes visualization
- How to combine masks and bounding boxes
- The difference between bounding-box visualization and pixel-level segmentation
- Why detections can be filtered before SAM
- How class filtering reduces unnecessary segmentation inference
- How YOLO and SAM can be composed into one pipeline
- How the same segmentation code can be reused with different images
- Why reusable abstractions are important in Computer Vision
- The conceptual difference between static and temporal segmentation
- How temporal memory can support video segmentation
- Why SAM2 extends segmentation into sequential video workflows

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

Session 07 builds directly on the segmentation masks introduced in Session 06 and begins the transition from **static segmentation visualization** toward **temporal video segmentation**.
