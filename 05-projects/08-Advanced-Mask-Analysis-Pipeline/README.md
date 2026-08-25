# Project 08 — Advanced Mask Analysis Pipeline

## Overview

This project builds an advanced computer-vision pipeline that combines **YOLOv8 object detection, class filtering, SAM 3 segmentation, Supervision annotation, and mask analysis** in a reusable workflow.

The project extends the concepts developed throughout the SAM3 Computer Vision Learning Journey, especially the segmentation and advanced mask-visualization techniques practiced in Sessions 06 and 07.

The objective is to move beyond simply generating segmentation masks and create a structured pipeline capable of:

- Detecting objects
- Selecting relevant detections
- Generating segmentation masks
- Visualizing masks and bounding boxes
- Measuring mask properties
- Producing structured analytical results
- Saving visual and numerical evidence

---

## Project Goal

Build a reusable image-analysis system with the following architecture:

```text
Input Image
     ↓
YOLOv8 Detection
     ↓
Object Detections
     ↓
Optional Class Filtering
     ↓
Bounding-Box Prompts
     ↓
SAM 3 Segmentation
     ↓
Pixel-Level Masks
     ↓
Mask Analysis
     ↓
Visualization
     ↓
Structured Results
     ↓
Saved Outputs
```

The project should demonstrate how detection and segmentation can be combined with quantitative mask analysis.

---

## Core Pipeline

The main processing pipeline is:

```text
Image
  ↓
YOLOv8
  ↓
sv.Detections
  ↓
Detection Filtering
  ↓
Bounding Boxes
  ↓
SAM 3
  ↓
Segmentation Masks
  ↓
Mask Metrics
  ↓
MaskAnnotator
  ↓
BoxAnnotator
  ↓
Annotated Image
  ↓
JSON / CSV Results
```

---

## Main Features

The project will implement:

- YOLOv8 object detection
- Conversion to `sv.Detections`
- Optional class-based filtering
- YOLO bounding boxes as SAM 3 prompts
- SAM 3 pixel-level segmentation
- `sv.MaskAnnotator`
- `sv.BoxAnnotator`
- Mask-area calculation
- Bounding-box area calculation
- Mask-to-box occupancy ratio
- Detection confidence extraction
- Class-name extraction
- Reusable image-processing functions
- Annotated image generation
- Structured JSON results
- Structured CSV results
- Input/output asset organization

---

## Mask Analysis

For each segmented object, the pipeline will calculate analytical information such as:

```text
Object
├── Class
├── Confidence
├── Bounding Box
├── Bounding-Box Area
├── Mask Area
└── Mask / Box Occupancy Ratio
```

### Mask Area

The mask area represents the number of pixels belonging to the segmented object:

```text
Mask Area = Number of True Pixels in Mask
```

Conceptually:

```python
mask_area = mask.sum()
```

---

## Bounding-Box Area

For a bounding box:

```text
(x1, y1, x2, y2)
```

the area is:

```text
Box Area = (x2 - x1) × (y2 - y1)
```

---

## Mask-to-Box Occupancy Ratio

The project will compare the precise segmentation mask with its rectangular detection region.

```text
Occupancy Ratio = Mask Area / Bounding-Box Area
```

Example:

```text
Bounding-Box Area = 50,000 px
Mask Area         = 32,000 px

Occupancy Ratio   = 0.64
```

This indicates that approximately:

```text
64%
```

of the bounding-box region belongs to the segmented object.

This metric helps demonstrate the difference between:

```text
Object Detection
      ↓
Approximate Rectangular Localization
```

and:

```text
Instance Segmentation
      ↓
Pixel-Level Object Shape
```

---

## Selective Segmentation

The pipeline should support filtering detections before SAM 3 inference.

For example:

```text
YOLO Detections
      ↓
6 Objects
      ↓
Filter: person
      ↓
4 Objects
      ↓
SAM 3
      ↓
4 Segmentation Masks
```

The important principle is:

```text
Filter BEFORE SAM
```

when only specific object classes are required.

This avoids performing segmentation on irrelevant detections.

---

## Visualization

The final visualization will combine:

```text
Original Image
      +
SAM 3 Masks
      +
YOLO Bounding Boxes
```

using:

```python
sv.MaskAnnotator()
```

and:

```python
sv.BoxAnnotator()
```

The resulting image provides both:

```text
Object Localization
+
Pixel-Level Segmentation
```

---

## Structured Results

The project will save analytical information for each detected and segmented object.

Example JSON structure:

```json
{
  "image": "bus.jpg",
  "detections": [
    {
      "object_id": 1,
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "bounding_box": [
        48,
        398,
        245,
        903
      ],
      "box_area": 99485,
      "mask_area": 61234,
      "occupancy_ratio": 0.6155
    }
  ]
}
```

A CSV representation can contain:

```text
image
object_id
class_id
class_name
confidence
x1
y1
x2
y2
box_area
mask_area
occupancy_ratio
```

---

## Planned Directory Structure

```text
08-Advanced-Mask-Analysis-Pipeline/
├── README.md
├── requirements.txt
├── src/
│   ├── mask_analysis_pipeline.py
│   └── utils.py
├── data/
│   ├── input/
│   └── output/
├── results/
│   ├── json/
│   └── csv/
└── docs/
    └── RESULTS.md
```

---

## Input Data

The project will initially use static images containing multiple detectable objects.

The first validated inputs can reuse the sample images already used during Session 07:

```text
bus.jpg
zidane.jpg
```

These provide different object configurations and allow the reusable pipeline to be tested across more than one image.

---

## Output Data

The project should generate:

```text
Annotated Images
       +
JSON Analysis
       +
CSV Analysis
```

Conceptually:

```text
Input Image
     ↓
Advanced Mask Analysis Pipeline
     ↓
├── Annotated Image
├── JSON Results
└── CSV Results
```

---

## Reusable Design

The processing logic should be organized into reusable functions.

Conceptually:

```python
analyze_image(
    image,
    yolo_model,
    sam_model
)
```

The function should perform:

```text
Detection
    ↓
Filtering
    ↓
Segmentation
    ↓
Mask Analysis
    ↓
Visualization
    ↓
Structured Results
```

This makes it possible to process multiple images without duplicating the pipeline logic.

---

## Relationship to Session 07

Session 07 validated the following progression:

```text
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
Visualization
```

It also demonstrated:

```text
Mask Opacity
Mask + Box Annotation
Person-Only Segmentation
Reusable Segmentation Functions
Temporal Segmentation Concepts
```

Project 08 extends those concepts into:

```text
Detection
     ↓
Selective Segmentation
     ↓
Mask Visualization
     ↓
Quantitative Mask Analysis
     ↓
Structured Results
     ↓
Reusable Analysis Pipeline
```

---

## Relationship to Previous Projects

This project builds on earlier work from the SAM3 Learning Journey.

Previous projects introduced concepts including:

```text
Object Detection
        ↓
Detection Filtering
        ↓
Tracking
        ↓
Segmentation
        ↓
Mask Analysis
```

Project 08 combines several of these ideas into a focused segmentation-analysis system.

The emphasis of this project is not only:

```text
"What object was detected?"
```

but also:

```text
"What is the precise segmented region?"
```

and:

```text
"What quantitative information can be extracted from that mask?"
```

---

## Technologies

The project will use:

- Python
- Google Colab
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- OpenCV
- NumPy
- JSON
- CSV

Primary Supervision components:

```text
sv.Detections
sv.MaskAnnotator
sv.BoxAnnotator
```

---

## SAM 3 Model

The SAM 3 checkpoint is intentionally not stored inside the GitHub repository because of its size.

The validated Google Colab model location is:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

The pipeline should verify that the model exists before attempting segmentation.

---

## Development Stages

The project will be developed incrementally.

```text
Stage 01
Project Structure
      ↓
Stage 02
Input Image Loading
      ↓
Stage 03
YOLOv8 Detection
      ↓
Stage 04
Detection Filtering
      ↓
Stage 05
SAM 3 Segmentation
      ↓
Stage 06
Mask Metrics
      ↓
Stage 07
Visualization
      ↓
Stage 08
JSON Results
      ↓
Stage 09
CSV Results
      ↓
Stage 10
Multi-Image Validation
      ↓
Stage 11
Documentation
```

---

## Validation Criteria

The project will be considered successfully implemented when:

- The input image loads correctly.
- YOLOv8 detects objects.
- Detection results are converted to `sv.Detections`.
- Optional class filtering works correctly.
- Bounding boxes are passed to SAM 3.
- SAM 3 generates segmentation masks.
- Each mask can be associated with its detection.
- Mask area is calculated.
- Bounding-box area is calculated.
- Occupancy ratio is calculated.
- Masks are visualized correctly.
- Bounding boxes are visualized correctly.
- Annotated images are saved.
- JSON results are generated.
- CSV results are generated.
- The same pipeline works with multiple images.
- Results are documented.

---

## Expected Deliverables

The final project should contain:

```text
Source Code
    +
Input Assets
    +
Annotated Outputs
    +
JSON Results
    +
CSV Results
    +
Validation Documentation
```

These artifacts will provide both visual and numerical evidence that the pipeline works correctly.

---

## Future Extension

The current project focuses on static-image analysis.

A future extension could introduce:

```text
Video
  ↓
Initial Object Detection
  ↓
Segmentation
  ↓
Temporal Memory
  ↓
Mask Propagation
  ↓
Frame-by-Frame Analysis
```

This would connect the static-image pipeline to the temporal-segmentation concepts introduced in Session 07.

The current Project 08 does **not** claim to implement SAM2 temporal propagation.

---

## Learning Objective

The primary learning objective is to understand how segmentation masks can become analytical data rather than only visualization overlays.

The progression is:

```text
Detection
   ↓
Segmentation
   ↓
Visualization
   ↓
Measurement
   ↓
Structured Data
   ↓
Reusable Analysis
```

This creates a stronger foundation for future computer-vision systems involving segmentation, tracking, analytics, and video processing.

---

## Current Status

```text
Project: Project 08 — Advanced Mask Analysis Pipeline

Project structure:       STARTED ✅
README:                  CREATED ✅
Implementation:          PENDING
SAM 3 validation:        PENDING
Mask analysis:           PENDING
JSON results:            PENDING
CSV results:             PENDING
Multi-image validation:  PENDING
Final documentation:     PENDING

Status: IN PROGRESS
```
