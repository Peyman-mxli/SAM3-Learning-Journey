# Project 08 — Advanced Mask Analysis Pipeline

## Overview

This project implements an advanced computer-vision pipeline that combines **YOLOv8 object detection, detection filtering, SAM 3 segmentation, Supervision annotation, and quantitative mask analysis** in a reusable workflow.

The project extends the concepts developed throughout the SAM3 Computer Vision Learning Journey, especially the segmentation and advanced mask-visualization techniques practiced in Sessions 06 and 07.

The completed pipeline is capable of:

- Detecting objects with YOLOv8
- Converting detections to `sv.Detections`
- Filtering detections before segmentation
- Using YOLO bounding boxes as SAM 3 prompts
- Generating pixel-level segmentation masks
- Visualizing masks and bounding boxes
- Measuring segmentation-mask properties
- Calculating mask-to-box occupancy ratios
- Producing structured JSON results
- Producing structured CSV results
- Saving annotated visual evidence
- Processing multiple images through the same reusable pipeline

---

## Project Goal

The goal of Project 08 is to build a reusable image-analysis system with the following architecture:

```text
Input Image
     ↓
YOLOv8 Detection
     ↓
Object Detections
     ↓
Detection Filtering
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

The project demonstrates how object detection and segmentation can be combined with quantitative mask analysis.

---

## Core Pipeline

The implemented processing pipeline is:

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

The same pipeline is reused for every input image.

---

## Main Features

Project 08 implements:

- YOLOv8 object detection
- Conversion to `sv.Detections`
- Class-based detection filtering support
- YOLO bounding boxes as SAM 3 prompts
- SAM 3 pixel-level segmentation
- `sv.MaskAnnotator`
- `sv.BoxAnnotator`
- Mask-area calculation
- Bounding-box area calculation
- Mask-to-box occupancy ratio
- Detection-confidence extraction
- Class-ID extraction
- Class-name extraction
- Reusable image-processing functions
- Multi-image processing
- Annotated image generation
- Structured JSON results
- Structured CSV results
- Organized input/output assets
- Validation documentation

---

## Technologies

Project 08 uses:

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

Validated environment:

```text
OpenCV:      4.14.0
NumPy:       2.1.3
Supervision: 0.30.1
Ultralytics: 8.4.128
```

---

## Project Structure

```text
08-Advanced-Mask-Analysis-Pipeline/
├── README.md
├── requirements.txt
│
├── src/
│   ├── README.md
│   └── mask_analysis_pipeline.py
│
├── data/
│   ├── README.md
│   │
│   ├── input/
│   │   ├── README.md
│   │   ├── bus.jpg
│   │   └── zidane.jpg
│   │
│   └── output/
│       ├── README.md
│       ├── bus_analyzed.png
│       └── zidane_analyzed.png
│
├── results/
│   ├── README.md
│   │
│   ├── json/
│   │   ├── README.md
│   │   ├── bus_analysis.json
│   │   └── zidane_analysis.json
│   │
│   └── csv/
│       ├── README.md
│       ├── bus_analysis.csv
│       └── zidane_analysis.csv
│
└── docs/
    ├── README.md
    └── RESULTS.md
```

---

## Source Code

The main implementation is:

```text
src/mask_analysis_pipeline.py
```

The script contains the complete Project 08 processing workflow.

Its responsibilities include:

```text
Input Discovery
      ↓
Image Loading
      ↓
YOLOv8 Detection
      ↓
Detection Filtering
      ↓
SAM 3 Segmentation
      ↓
Mask Analysis
      ↓
Visualization
      ↓
JSON Serialization
      ↓
CSV Serialization
      ↓
Output Saving
```

The source-code directory is documented separately in:

```text
src/README.md
```

---

## Input Data

Project 08 uses two validated static images:

```text
data/input/
├── bus.jpg
└── zidane.jpg
```

These images were also useful during the Session 07 examples and provide different object configurations for validating the reusable pipeline.

### bus.jpg

Image shape:

```text
1080 × 810 × 3
```

YOLOv8 detected:

```text
4 persons
1 bus
1 stop sign
```

Total:

```text
6 objects
```

### zidane.jpg

Image shape:

```text
720 × 1280 × 3
```

YOLOv8 detected:

```text
2 persons
1 tie
```

Total:

```text
3 objects
```

---

## YOLOv8 Detection

YOLOv8 performs the initial object-detection stage.

Conceptually:

```text
Input Image
     ↓
YOLOv8
     ↓
Bounding Boxes
     +
Class IDs
     +
Confidence Scores
```

The Ultralytics result is converted into:

```python
sv.Detections
```

This provides a consistent representation for the remaining processing stages.

---

## Detection Filtering

The pipeline includes a detection-filtering stage before SAM 3 segmentation.

Conceptually:

```text
YOLO Detections
      ↓
Detection Filtering
      ↓
Selected Detections
      ↓
Bounding-Box Prompts
      ↓
SAM 3
```

During the validated Project 08 execution, all detected objects were retained.

For `bus.jpg`:

```text
YOLO detections:          6
Detections after filter:  6
```

For `zidane.jpg`:

```text
YOLO detections:          3
Detections after filter:  3
```

The filtering architecture remains reusable for workflows where only selected object classes are required.

---

## Selective Segmentation

A major design principle of the project is:

```text
Filter BEFORE SAM
```

when only specific object classes are needed.

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

This avoids performing segmentation on unwanted detections and allows the segmentation stage to focus only on relevant objects.

---

## SAM 3 Segmentation

After detection and filtering, YOLOv8 bounding boxes are used as prompts for SAM 3.

```text
YOLO Bounding Boxes
        ↓
SAM 3 Prompts
        ↓
Pixel-Level Segmentation Masks
```

During the validated execution:

```text
bus.jpg
YOLO detections: 6
SAM 3 masks:     6
```

and:

```text
zidane.jpg
YOLO detections: 3
SAM 3 masks:     3
```

Therefore, SAM 3 successfully generated one mask for every retained detection in both validation images.

---

## SAM 3 Model

The SAM 3 checkpoint is intentionally not stored inside the GitHub repository because of its size.

The validated Google Colab model location is:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

Before segmentation begins, the pipeline verifies that this checkpoint exists.

The validated environment confirmed:

```text
SAM 3 exists: True
SAM 3 path: /content/drive/MyDrive/SAM3-Models/sam3.pt
```

---

## Mask Analysis

Project 08 extends segmentation beyond visual overlays.

For every segmented object, the pipeline extracts analytical information:

```text
Object
├── Image
├── Object ID
├── Class ID
├── Class Name
├── Confidence
├── Bounding Box
├── Bounding-Box Area
├── Mask Area
└── Mask / Box Occupancy Ratio
```

This converts segmentation results into quantitative data that can be stored and analyzed.

---

## Mask Area

Mask area represents the number of pixels belonging to the segmented object.

Conceptually:

```text
Mask Area = Number of True Pixels in Mask
```

In Python, the basic operation is equivalent to:

```python
mask_area = mask.sum()
```

A segmentation mask therefore becomes a measurable pixel-level region rather than only a visualization.

---

## Bounding-Box Area

For a bounding box represented as:

```text
(x1, y1, x2, y2)
```

the area is calculated as:

```text
Box Area = (x2 - x1) × (y2 - y1)
```

The bounding-box area represents the rectangular localization produced by the object detector.

---

## Mask-to-Box Occupancy Ratio

Project 08 compares the precise segmentation region with its rectangular detection region.

```text
Occupancy Ratio = Mask Area / Bounding-Box Area
```

Conceptually:

```text
YOLO Bounding Box
        ↓
Rectangular Detection Region

SAM 3 Mask
        ↓
Pixel-Level Object Region

Mask Area
    ÷
Bounding-Box Area
    ↓
Occupancy Ratio
```

For example:

```text
Bounding-Box Area = 50,000 px
Mask Area         = 32,000 px

Occupancy Ratio   = 0.64
```

This means approximately:

```text
64%
```

of the rectangular detection region belongs to the segmented object.

This metric illustrates the difference between:

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

## Visualization

The final visual output combines:

```text
Original Image
      +
SAM 3 Segmentation Masks
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

The resulting images provide both:

```text
Object Localization
        +
Pixel-Level Segmentation
```

---

## Annotated Outputs

The validated pipeline generated:

```text
data/output/
├── bus_analyzed.png
└── zidane_analyzed.png
```

The generated file sizes during validation were approximately:

```text
bus_analyzed.png      1575.0 KB
zidane_analyzed.png    921.1 KB
```

Both files were successfully created and verified after pipeline execution.

---

## Structured Results

Project 08 saves analytical information for every detected and segmented object.

The same underlying object analysis is represented in two formats:

```text
Mask Analysis
      ↓
Structured Results
      ↓
├── JSON
└── CSV
```

This allows the results to be used both programmatically and in tabular data-analysis workflows.

---

## JSON Results

The generated JSON files are:

```text
results/json/
├── bus_analysis.json
└── zidane_analysis.json
```

Each JSON file stores image-level information and object-level analytical results.

The structure includes:

```text
Image
Object Count
Detections
    ├── Image
    ├── Object ID
    ├── Class ID
    ├── Class Name
    ├── Confidence
    ├── Bounding-Box Coordinates
    ├── Bounding-Box Area
    ├── Mask Area
    └── Occupancy Ratio
```

Conceptually:

```json
{
  "image": "bus.jpg",
  "object_count": 6,
  "detections": [
    {
      "image": "bus.jpg",
      "object_id": 1,
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "x1": 48.0,
      "y1": 398.0,
      "x2": 245.0,
      "y2": 903.0,
      "box_area": 99485.0,
      "mask_area": 61234,
      "occupancy_ratio": 0.6155
    }
  ]
}
```

The values above illustrate the result structure. The repository JSON files contain the actual generated measurements from the validated execution.

---

## CSV Results

The generated CSV files are:

```text
results/csv/
├── bus_analysis.csv
└── zidane_analysis.csv
```

Each row represents one detected and segmented object.

The columns are:

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

CSV provides a convenient representation for:

- Python
- Pandas
- Spreadsheet applications
- Statistical analysis
- Data visualization
- Future dashboards

---

## Reusable Design

The Project 08 implementation is designed around reusable processing logic.

Conceptually:

```python
analyze_image(
    image,
    yolo_model,
    sam_model
)
```

The reusable workflow performs:

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

The models are loaded once and reused across multiple input images.

This avoids duplicating the full processing logic for every image.

---

## Multi-Image Processing

The validated execution discovered two input images automatically:

```text
Input images found: 2

- bus.jpg
- zidane.jpg
```

Both were processed through the same pipeline:

```text
bus.jpg ──────────┐
                  ↓
        Advanced Mask Analysis Pipeline
                  ↓
        ├── Annotated Image
        ├── JSON Results
        └── CSV Results


zidane.jpg ───────┐
                  ↓
        Advanced Mask Analysis Pipeline
                  ↓
        ├── Annotated Image
        ├── JSON Results
        └── CSV Results
```

This validates the reusable multi-image design.

---

## Validated Execution Results

The final Project 08 execution completed successfully.

### Overall Results

| Metric | Result |
|---|---:|
| Images processed | 2 |
| Objects analyzed | 9 |
| YOLO detections | 9 |
| SAM 3 masks | 9 |
| Annotated images | 2 |
| JSON result files | 2 |
| CSV result files | 2 |

### bus.jpg

| Metric | Result |
|---|---:|
| Image shape | 1080 × 810 × 3 |
| YOLO detections | 6 |
| Detections after filtering | 6 |
| SAM 3 masks | 6 |
| Objects analyzed | 6 |

Detected classes:

```text
person × 4
bus × 1
stop sign × 1
```

Generated files:

```text
data/output/bus_analyzed.png
results/json/bus_analysis.json
results/csv/bus_analysis.csv
```

### zidane.jpg

| Metric | Result |
|---|---:|
| Image shape | 720 × 1280 × 3 |
| YOLO detections | 3 |
| Detections after filtering | 3 |
| SAM 3 masks | 3 |
| Objects analyzed | 3 |

Detected classes:

```text
person × 2
tie × 1
```

Generated files:

```text
data/output/zidane_analyzed.png
results/json/zidane_analysis.json
results/csv/zidane_analysis.csv
```

---

## Generated Artifact Validation

After execution, all expected generated artifacts were verified.

```text
data/output/bus_analyzed.png       ✅
data/output/zidane_analyzed.png    ✅

results/json/bus_analysis.json     ✅
results/json/zidane_analysis.json  ✅

results/csv/bus_analysis.csv       ✅
results/csv/zidane_analysis.csv    ✅
```

Artifact verification result:

```text
Visual outputs: 2 / 2
JSON results:   2 / 2
CSV results:    2 / 2

Total:          6 / 6
```

The two input images were also preserved in:

```text
data/input/
├── bus.jpg
└── zidane.jpg
```

---

## Output Organization

Project 08 separates source data, visual evidence, and analytical evidence.

```text
Project 08
    ↓
├── data/input/
│      ↓
│   Original Input Images
│
├── data/output/
│      ↓
│   Annotated Visual Results
│
├── results/json/
│      ↓
│   Structured Hierarchical Results
│
├── results/csv/
│      ↓
│   Structured Tabular Results
│
└── docs/
       ↓
    Validation Documentation
```

This keeps the project organized and makes each artifact type easy to locate.

---

## Relationship to Session 07

Session 07 validated the progression:

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

Session 07 also demonstrated:

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
Detection Filtering
     ↓
Segmentation
     ↓
Mask Visualization
     ↓
Quantitative Mask Analysis
     ↓
Structured Results
     ↓
Reusable Analysis Pipeline
```

The progression therefore moves from learning individual segmentation techniques to integrating them into a complete analytical project.

---

## Relationship to Previous Projects

Project 08 builds on concepts developed throughout the SAM3 Learning Journey.

Earlier work introduced:

```text
Object Detection
        ↓
Detection Filtering
        ↓
Tracking
        ↓
Segmentation
        ↓
Visualization
        ↓
Analysis
```

Project 08 combines several of these concepts into a focused segmentation-analysis system.

The emphasis is no longer only:

```text
"What object was detected?"
```

It also asks:

```text
"What is the precise segmented region?"
```

and:

```text
"What quantitative information can be extracted from that mask?"
```

---

## Development Stages

Project 08 was developed incrementally.

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

All stages were completed and validated.

---

## Validation Checklist

- [x] Project structure created
- [x] `src/` documented with its own README
- [x] `data/` documented with its own README
- [x] `data/input/` documented with its own README
- [x] `data/output/` documented with its own README
- [x] `results/` documented with its own README
- [x] `results/json/` documented with its own README
- [x] `results/csv/` documented with its own README
- [x] `docs/` documented with its own README
- [x] Input images stored in the project
- [x] Input images discovered automatically
- [x] Input images load correctly
- [x] YOLOv8 detects objects
- [x] YOLO results convert to `sv.Detections`
- [x] Detection-filtering stage executes
- [x] Bounding boxes are passed to SAM 3
- [x] SAM 3 model checkpoint is validated
- [x] SAM 3 generates segmentation masks
- [x] Each retained detection receives a segmentation mask
- [x] Mask area is calculated
- [x] Bounding-box area is calculated
- [x] Occupancy ratio is calculated
- [x] Masks are visualized with `MaskAnnotator`
- [x] Bounding boxes are visualized with `BoxAnnotator`
- [x] Annotated images are saved
- [x] JSON results are generated
- [x] CSV results are generated
- [x] The same pipeline processes multiple images
- [x] Generated artifacts are verified
- [x] Final results are documented

---

## Deliverables

The completed project contains:

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

### Source Code

```text
src/mask_analysis_pipeline.py
```

### Input Assets

```text
data/input/bus.jpg
data/input/zidane.jpg
```

### Annotated Outputs

```text
data/output/bus_analyzed.png
data/output/zidane_analyzed.png
```

### JSON Results

```text
results/json/bus_analysis.json
results/json/zidane_analysis.json
```

### CSV Results

```text
results/csv/bus_analysis.csv
results/csv/zidane_analysis.csv
```

### Documentation

```text
docs/README.md
docs/RESULTS.md
```

These artifacts provide both visual and numerical evidence that the pipeline works correctly.

---

## Reproducibility

The project can be reproduced in Google Colab using the repository source code, dependencies, input images, and a locally available SAM 3 checkpoint.

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The SAM 3 checkpoint must be available at:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

Run:

```bash
python src/mask_analysis_pipeline.py
```

When executed from the project environment, the pipeline processes the images stored in:

```text
data/input/
```

and generates outputs in:

```text
data/output/
results/json/
results/csv/
```

---

## Final Workflow

The completed Project 08 workflow is:

```text
Input Images
     ↓
YOLOv8
     ↓
Object Detection
     ↓
sv.Detections
     ↓
Detection Filtering
     ↓
Bounding-Box Prompts
     ↓
SAM 3
     ↓
Segmentation Masks
     ↓
Mask Analysis
     ↓
├── Mask Area
├── Bounding-Box Area
└── Occupancy Ratio
     ↓
Visualization
     ↓
├── MaskAnnotator
└── BoxAnnotator
     ↓
Structured Results
     ↓
├── JSON
└── CSV
     ↓
Validated Project Artifacts
```

---

## Future Extension

Project 08 intentionally focuses on static-image analysis.

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

This would connect the static-image pipeline to temporal-segmentation concepts introduced in Session 07.

Project 08 itself does **not** claim to implement SAM2 temporal propagation.

---

## Learning Objective

The primary learning objective of Project 08 is to understand how segmentation masks can become analytical data rather than only visualization overlays.

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

The project demonstrates that segmentation can provide both:

```text
Visual Understanding
        +
Quantitative Information
```

This creates a stronger foundation for future computer-vision systems involving segmentation, tracking, analytics, and video processing.

---

## Final Result

**Project 08 — Advanced Mask Analysis Pipeline** was successfully implemented and validated.

Final validated execution:

```text
Images processed:       2
Objects analyzed:       9
YOLO detections:        9
SAM 3 masks generated:  9

Annotated images:       2
JSON result files:      2
CSV result files:       2

Generated artifacts:    6 / 6 verified
```

The final project successfully integrates:

```text
YOLOv8 Detection
        +
SAM 3 Segmentation
        +
Supervision Visualization
        +
Quantitative Mask Analysis
        +
Structured JSON / CSV Results
```

into a reusable end-to-end computer-vision analysis pipeline.
