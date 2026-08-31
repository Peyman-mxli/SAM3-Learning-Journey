# SAM3 — Projects

This directory contains practical computer vision projects developed throughout my **SAM3 Computer Vision Learning Journey**.

The purpose of this section is to transform concepts studied during the course into complete, reusable projects that combine AI models, computer vision libraries, visualization tools, structured workflows, testing, evaluation, and documentation.

Unlike the smaller examples in `04-examples/`, projects in this directory combine multiple concepts into complete applications.

---

## Available Projects

### 01 — YOLO + Supervision Object Detector

[`01-YOLO-Supervision-Object-Detector/`](./01-YOLO-Supervision-Object-Detector/)

A complete object-detection pipeline using **YOLOv8**, **OpenCV**, and **Supervision**.

The project:

- Loads an input image
- Runs YOLOv8 object detection
- Converts YOLO predictions into `sv.Detections`
- Creates class and confidence labels
- Draws bounding boxes
- Adds labels to detected objects
- Saves the annotated image
- Exports predictions to JSON

### Pipeline

```text
Input Image
    ↓
OpenCV
    ↓
YOLOv8
    ↓
Predictions
    ↓
sv.Detections
    ↓
BoxAnnotator
    ↓
LabelAnnotator
    ↓
Annotated Image
    ↓
JSON Predictions
```

**Status:** Completed and tested successfully in Google Colab.

---

### 02 — Multi-Annotator Visualization Pipeline

[`02-Multi-Annotator-Visualization-Pipeline/`](./02-Multi-Annotator-Visualization-Pipeline/)

A computer vision visualization pipeline demonstrating how multiple **Supervision Annotators** can be composed as visual layers over YOLO detections.

The project:

- Loads an input image
- Runs YOLOv8 object detection
- Applies a confidence threshold
- Converts predictions into `sv.Detections`
- Generates class and confidence labels
- Applies multiple visualization layers
- Demonstrates annotation composition
- Saves the final annotated image
- Preserves input, output, and screenshots as project evidence
- Documents the complete Google Colab testing workflow

The visualization pipeline uses:

- `BoxAnnotator`
- `EllipseAnnotator`
- `DotAnnotator`
- `LabelAnnotator`

### Pipeline

```text
Input Image
    ↓
OpenCV
    ↓
YOLOv8
    ↓
Object Detection
    ↓
sv.Detections
    ↓
BoxAnnotator
    ↓
EllipseAnnotator
    ↓
DotAnnotator
    ↓
LabelAnnotator
    ↓
Annotated Image
```

### Test Result

The project was successfully tested from a fresh **Google Colab** environment.

The completed test produced:

```text
Detected objects: 13
Annotated image saved to: output/annotated_image.jpg
```

The final visualization successfully displayed the detected objects using multiple Supervision annotation layers.

### Project Evidence

Project 02 contains an organized `assets/` directory:

```text
assets/
│
├── input/
│   ├── README.md
│   └── image.png
│
├── output/
│   ├── README.md
│   └── annotated_image.jpg
│
├── screenshots/
│   ├── README.md
│   └── project screenshots
│
└── README.md
```

This preserves the original input, generated output, and screenshots documenting the development and successful execution of the project.

**Status:** Completed, tested, documented, and supported with project evidence.

---

### 03 — Detection Filtering and NMS Pipeline

[`03-Detection-Filtering-and-NMS-Pipeline/`](./03-Detection-Filtering-and-NMS-Pipeline/)

A complete object-detection post-processing pipeline demonstrating how raw YOLO predictions can be filtered and transformed into application-specific results using **Supervision** and **NumPy**.

The project:

- Loads a pedestrian street image
- Runs YOLOv8 object detection
- Converts predictions into `sv.Detections`
- Filters detections by confidence
- Filters detections by class
- Removes small bounding boxes
- Applies Non-Maximum Suppression
- Selects the Top-N most confident detections
- Calculates bounding-box center positions
- Applies spatial filtering
- Keeps detections located in the right half of the image
- Draws the image midpoint
- Generates the final annotated output
- Preserves the original input and generated output as project evidence

### Pipeline

```text
Input Image
    ↓
YOLOv8
    ↓
Raw Predictions
    ↓
sv.Detections
    ↓
Confidence Filtering
    ↓
Class Filtering
    ↓
Size Filtering
    ↓
Non-Maximum Suppression
    ↓
Top-N Selection
    ↓
Spatial Filtering
    ↓
Final Detections
    ↓
Annotated Output
```

### Test Result

Project 03 was successfully tested in **Google Colab** using:

```text
assets/input/pedestrian-plaza-detection-test.png
```

The complete filtering process produced:

```text
Initial detections:             13
        ↓
After confidence filtering:      9
        ↓
After class filtering:           8
        ↓
After size filtering:            8
        ↓
After NMS:                       8
        ↓
After Top-5 selection:           5
        ↓
After spatial filtering:         2
```

Final result:

```text
Detection Filtering Pipeline Complete

Input image:
assets/input/pedestrian-plaza-detection-test.png

Output image:
assets/output/filtered_detections.jpg

Final detections: 2
```

### Project Evidence

Project 03 contains its own organized assets:

```text
assets/
│
├── README.md
│
├── input/
│   ├── README.md
│   └── pedestrian-plaza-detection-test.png
│
└── output/
    ├── README.md
    └── filtered_detections.jpg
```

The final generated output can be viewed here:

![Project 03 Filtered Detections](./03-Detection-Filtering-and-NMS-Pipeline/assets/output/filtered_detections.jpg)

This project demonstrates the progression from raw object detection to a controlled post-processing pipeline where only detections satisfying specific application rules remain.

**Status:** Completed, tested successfully in Google Colab, documented, and supported with input/output evidence.

---
### 04 — Object Tracking with ByteTrack

[`04-Object-Tracking-with-ByteTrack/`](./04-Object-Tracking-with-ByteTrack/)

A complete object-tracking project using **YOLOv8**, **Supervision**, **ByteTrack**, and **OpenCV** to detect objects in video and maintain persistent identities across frames.

The project extends object detection from individual images into a temporal video-processing workflow.

The project:

- Loads an input video
- Reads video metadata
- Processes the video frame by frame
- Runs YOLOv8 object detection
- Converts predictions into `sv.Detections`
- Updates detections using ByteTrack
- Assigns persistent tracker IDs
- Draws bounding boxes
- Displays class names
- Displays tracker IDs
- Draws object movement traces
- Writes the processed frames to a new output video
- Preserves input and output video assets
- Documents the complete tracking workflow

### Pipeline

```text
Input Video
    ↓
Frame Extraction
    ↓
YOLOv8
    ↓
Object Detections
    ↓
sv.Detections
    ↓
ByteTrack
    ↓
Persistent Tracker IDs
    ↓
BoxAnnotator
    ↓
LabelAnnotator
    ↓
TraceAnnotator
    ↓
Annotated Frames
    ↓
Output Video
```

### Core Tracking Concept

Object detection answers:

```text
What objects exist in this frame?
```

Object tracking extends this by answering:

```text
Is this the same object that appeared in previous frames?
```

ByteTrack maintains persistent identities across frames:

```text
Frame 1
Person → ID 1

Frame 2
Same Person → ID 1

Frame 3
Same Person → ID 1
```

This transforms independent detections into temporal object trajectories.

### Visualization

The project combines:

```text
Bounding Boxes
      +
Class Labels
      +
Tracker IDs
      +
Movement Traces
```

using Supervision annotators.

The final video therefore shows not only where objects are located, but also their persistent identities and movement history.

**Status:** Completed, tested successfully in Google Colab, documented, and validated with generated tracking output.

---

### 05 — Zones and Counting

[`05-Zones-and-Counting/`](./05-Zones-and-Counting/)

A video-analysis project combining **YOLOv8**, **ByteTrack**, and **Supervision zones** to measure object occupancy and count directional crossings.

The project extends object tracking by introducing spatial rules.

The project:

- Loads an input video
- Runs YOLOv8 detection
- Tracks objects with ByteTrack
- Maintains persistent tracker IDs
- Defines a polygon region
- Measures occupancy inside the polygon
- Defines a counting line
- Detects objects crossing the line
- Counts crossings in both directions
- Annotates detections
- Visualizes the polygon zone
- Visualizes the counting line
- Combines tracking, zones, and counting in one pipeline
- Saves the processed video

### Pipeline

```text
Input Video
    ↓
YOLOv8
    ↓
Detections
    ↓
ByteTrack
    ↓
Persistent IDs
    ↓
┌───────────────────────┐
│                       │
↓                       ↓
PolygonZone          LineZone
↓                       ↓
Occupancy            Crossings
│                       │
└───────────┬───────────┘
            ↓
      Visualization
            ↓
       Output Video
```

### PolygonZone

`PolygonZone` defines a polygon-shaped region of interest.

It can answer:

```text
How many tracked objects are currently inside this region?
```

This allows the system to measure occupancy inside a selected area.

### LineZone

`LineZone` defines a virtual counting line.

Tracked objects crossing the line can be counted according to direction.

Conceptually:

```text
Object
   ↓
Tracker ID
   ↓
Crosses Virtual Line
   ↓
Direction Determined
   ↓
Counter Updated
```

### Combined Analysis

The project combines:

```text
Detection
    +
Tracking
    +
Spatial Zones
    +
Directional Counting
```

in a single video-processing pipeline.

Validated results from the practical workflow included:

```text
Final polygon occupancy: 1
Crossings Down:           3
Crossings Up:             3
```

This project demonstrates how persistent tracking information can be transformed into higher-level spatial analytics.

**Status:** Completed, tested successfully in Google Colab, documented, and validated with generated zone and counting results.

---

### 06 — Visual Tracking and Analysis System

[`06-Visual-Tracking-and-Analysis-System/`](./06-Visual-Tracking-and-Analysis-System/)

A larger integrated computer vision project designed as a **visual tracking and analysis system** for detecting, segmenting, tracking, storing, and analyzing objects or people in images and video.

Project 06 combines multiple concepts developed throughout the learning journey into a more complete application architecture.

The project includes:

- Image processing
- Object detection
- Object tracking
- Persistent tracker IDs
- Video processing
- SAM 3 segmentation
- Tracking analytics
- Trajectory analysis
- SQLite persistence
- Historical result storage
- Structured reports
- CSV-based analytics
- Performance analysis
- Visual charts
- Evaluation against manually annotated ground truth
- Project documentation

### System Concept

```text
Image / Video
      ↓
Computer Vision Pipeline
      ↓
Detection
      ↓
Segmentation
      ↓
Tracking
      ↓
Analytics
      ↓
Persistence
      ↓
Reports
      ↓
Evaluation
```

### Video Tracking Validation

The validated tracking workflow processed:

```text
75 frames
246 observations
6 tracker IDs
```

Key tracking analytics included:

```text
Average observations/frame:    3.2800
Average observations/tracker: 41.0000
Average confidence:            0.6815
Minimum confidence:            0.5124
Maximum confidence:            0.8587
Average tracker duration:      2.7333 s
Total movement distance:     693.18 px
Average distance/tracker:    115.53 px
Maximum tracker distance:    203.37 px
Average movement step:         2.9883 px
```

### Tracker Summary

The validated tracking results included:

```text
ID 1 — person
Frames:      1–75
Observations: 75
Confidence:   0.8385
Movement:   159.26 px

ID 2 — person
Frames:      1–75
Observations: 75
Confidence:   0.8587
Movement:   203.37 px

ID 3 — bus
Frames:      3–75
Observations: 59
Confidence:   0.5939
Movement:   182.36 px

ID 4 — person
Frames:      8–32
Observations: 25
Confidence:   0.7579
Movement:   139.34 px

ID 5 — person
Frames:     29–31
Observations: 3
Confidence:   0.5278
Movement:     7.91 px

ID 6 — person
Frames:     53–61
Observations: 9
Confidence:   0.5124
Movement:     0.94 px
```

### Persistence

Project 06 includes SQLite-based persistence for storing analysis sessions and results.

The database architecture supports information such as:

```text
Identifier
Timestamp
Media Information
Analysis Results
Confidence
Notes
```

This allows computer vision results to be retained for later consultation rather than existing only during runtime.

### Evaluation

Project 06 was also evaluated against manually annotated ground truth.

Evaluation dataset:

```text
Images:                  20
Ground-truth instances: 424
Predictions:            472
True Positives:         381
False Positives:         91
False Negatives:         43
```

Final metrics:

```text
Precision:    0.8072
Recall:       0.8986
Average IoU:  0.7969
Average Dice: 0.8829
```

The evaluation also includes a completed pixel confusion matrix.

### Evaluation Concept

```text
Ground Truth
     +
Predictions
     ↓
Matching
     ↓
TP / FP / FN
     ↓
Precision
Recall
IoU
Dice
```

### Documented Limitations

The final project documentation explicitly considers limitations related to:

- Lighting
- Object scale
- Occlusion
- Out-of-sample data
- False positives
- Omissions

Project 06 therefore moves beyond simply demonstrating a model and includes quantitative evaluation, persistence, analytics, and documented limitations.

**Status:** Completed according to the project MVP and Definition of Done, with tracking, segmentation, persistence, analytics, evaluation, and documentation validated.

---
### 07 — SAM 3 Segmentation Pipeline

[`07-SAM3-Segmentation-Pipeline/`](./07-SAM3-Segmentation-Pipeline/)

A complete image-segmentation project combining **YOLOv8**, **SAM 3**, **Supervision**, **OpenCV**, and **NumPy** to move from object detection to pixel-level segmentation.

The project extends the learning journey from:

```text
Detection
    ↓
Tracking
    ↓
Zones and Counting
    ↓
Pixel-Level Segmentation
```

The project:

- Loads an input image
- Runs YOLOv8 object detection
- Converts YOLO predictions into `sv.Detections`
- Uses YOLO bounding boxes as SAM 3 prompts
- Loads the SAM 3 checkpoint from Google Drive
- Generates segmentation masks
- Converts SAM 3 predictions into `sv.Detections`
- Calculates mask areas
- Compares mask area with bounding-box area
- Visualizes masks with `MaskAnnotator`
- Draws bounding boxes with `BoxAnnotator`
- Creates labels with `LabelAnnotator`
- Saves the final annotated image
- Exports structured JSON results
- Preserves generated evidence inside the project

### Pipeline

```text
Input Image
    ↓
YOLOv8
    ↓
Object Detection
    ↓
sv.Detections
    ↓
Bounding-Box Prompts
    ↓
SAM 3
    ↓
Segmentation Masks
    ↓
Mask Analysis
    ↓
MaskAnnotator
    ↓
BoxAnnotator
    ↓
LabelAnnotator
    ↓
Annotated Image
    ↓
JSON Results
```

### Detection and Segmentation

YOLOv8 provides the initial object localization:

```text
Image
  ↓
YOLOv8
  ↓
Bounding Boxes
```

Those bounding boxes are then passed to SAM 3:

```text
Bounding Boxes
      ↓
SAM 3
      ↓
Pixel-Level Masks
```

This combines the strengths of:

```text
YOLOv8
Fast Object Detection

        +

SAM 3
Pixel-Level Segmentation
```

### Mask Analysis

Project 07 introduces quantitative analysis of segmentation masks.

For every segmented object, the project can compare:

```text
Bounding-Box Area
        vs.
Segmentation-Mask Area
```

This demonstrates the difference between rectangular object localization and the object's actual pixel-level region.

### Visualization

The final visualization combines:

```text
Original Image
      +
Segmentation Masks
      +
Bounding Boxes
      +
Object Labels
```

using Supervision annotators.

This provides a clear visual representation of both object location and object shape.

### Structured Results

The project also exports segmentation information to JSON.

The structured results preserve information such as:

```text
Class
Confidence
Bounding Box
Mask Area
```

This makes segmentation results available for future analysis rather than limiting them to visual output.

**Status:** Completed, successfully tested in Google Colab, documented, and validated with real generated segmentation evidence.

---

### 08 — Advanced Mask Analysis Pipeline

[`08-Advanced-Mask-Analysis-Pipeline/`](./08-Advanced-Mask-Analysis-Pipeline/)

A reusable computer vision analysis pipeline combining **YOLOv8**, **SAM 3**, **Supervision**, **OpenCV**, and **NumPy** for object detection, pixel-level segmentation, quantitative mask analysis, visualization, and structured result export.

Project 08 extends the segmentation work developed in Project 07 by transforming segmentation masks into measurable analytical data.

The project:

- Loads multiple input images automatically
- Runs YOLOv8 object detection
- Converts YOLO predictions into `sv.Detections`
- Supports detection filtering before segmentation
- Uses YOLO bounding boxes as SAM 3 prompts
- Generates pixel-level segmentation masks
- Calculates mask area
- Calculates bounding-box area
- Calculates mask-to-box occupancy ratio
- Preserves object class information
- Preserves detection confidence
- Visualizes segmentation masks with `MaskAnnotator`
- Visualizes detections with `BoxAnnotator`
- Saves annotated output images
- Exports structured JSON results
- Exports structured CSV results
- Reuses the same analysis pipeline across multiple images
- Organizes source code, data, results, and documentation into dedicated folders

### Pipeline

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
```

### Validation Inputs

Project 08 was validated using two images:

```text
data/input/
├── bus.jpg
└── zidane.jpg
```

The same reusable pipeline processed both images.

### bus.jpg Result

YOLOv8 detected:

```text
4 persons
1 bus
1 stop sign
```

Validation result:

```text
YOLO detections:          6
Detections after filter:  6
SAM 3 masks:              6
Objects analyzed:         6
```

Generated artifacts:

```text
data/output/bus_analyzed.png
results/json/bus_analysis.json
results/csv/bus_analysis.csv
```

### zidane.jpg Result

YOLOv8 detected:

```text
2 persons
1 tie
```

Validation result:

```text
YOLO detections:          3
Detections after filter:  3
SAM 3 masks:              3
Objects analyzed:         3
```

Generated artifacts:

```text
data/output/zidane_analyzed.png
results/json/zidane_analysis.json
results/csv/zidane_analysis.csv
```

### Mask Analysis

Project 08 converts every segmentation result into quantitative information.

For each segmented object, the pipeline records information including:

```text
Image
Object ID
Class ID
Class Name
Confidence
Bounding Box
Bounding-Box Area
Mask Area
Occupancy Ratio
```

Mask area represents the number of pixels belonging to the segmented object:

```text
Mask Area = Number of True Pixels
```

Bounding-box area is calculated as:

```text
Box Area = (x2 - x1) × (y2 - y1)
```

The mask-to-box occupancy ratio is:

```text
                 Mask Area
Occupancy = ───────────────────
             Bounding-Box Area
```

This measures how much of the rectangular detection region is actually occupied by the pixel-level segmentation mask.

### Structured Results

The pipeline exports analytical results in both JSON and CSV formats.

JSON outputs:

```text
results/json/
├── bus_analysis.json
└── zidane_analysis.json
```

CSV outputs:

```text
results/csv/
├── bus_analysis.csv
└── zidane_analysis.csv
```

This allows the same segmentation information to be used for both programmatic processing and tabular analysis.

### Annotated Visual Evidence

The generated visual results are stored in:

```text
data/output/
├── bus_analyzed.png
└── zidane_analyzed.png
```

Each visualization combines:

```text
Original Image
      +
SAM 3 Segmentation Masks
      +
YOLO Bounding Boxes
```

The validated output sizes were approximately:

```text
bus_analyzed.png      1575.0 KB
zidane_analyzed.png    921.1 KB
```

### Project Structure

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
│   ├── input/
│   │   ├── README.md
│   │   ├── bus.jpg
│   │   └── zidane.jpg
│   └── output/
│       ├── README.md
│       ├── bus_analyzed.png
│       └── zidane_analyzed.png
│
├── results/
│   ├── README.md
│   ├── json/
│   │   ├── README.md
│   │   ├── bus_analysis.json
│   │   └── zidane_analysis.json
│   └── csv/
│       ├── README.md
│       ├── bus_analysis.csv
│       └── zidane_analysis.csv
│
└── docs/
    ├── README.md
    └── RESULTS.md
```

### Final Validation

The completed Project 08 execution produced:

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

The complete relationship was:

```text
9 YOLO detections
        ↓
9 retained detections
        ↓
9 bounding-box prompts
        ↓
9 SAM 3 masks
        ↓
9 analyzed objects
```

This confirms that every retained YOLO detection produced a corresponding SAM 3 segmentation mask and quantitative analysis result during the validated execution.

**Status:** Completed, successfully tested in Google Colab, documented, and validated with generated visual, JSON, and CSV evidence.

---

### 09 — SAM3 Semantic Prompt Analytics

[`09-SAM3-Semantic-Prompt-Analytics/`](./09-SAM3-Semantic-Prompt-Analytics/)

A professional semantic-segmentation analytics system using **SAM3SemanticPredictor**, natural-language prompts, **Supervision**, **OpenCV**, **NumPy**, JSON, and CSV.

Unlike detector-guided segmentation projects, Project 09 accepts concepts directly:

```text
vehicle
bus
person
wheel
```

The project:

- Discovers input images automatically
- Loads a configurable list of natural-language prompts
- Reuses one encoded image across multiple semantic queries
- Locates and segments matching object instances
- Converts predictions into `sv.Detections`
- Measures confidence and mask area
- Preserves bounding coordinates
- Classifies reliable detections using confidence and minimum-area rules
- Creates labeled filtered visualizations
- Creates multi-prompt comparison figures
- Exports detailed JSON analytics
- Exports detection-level CSV results
- Exports prompt-summary CSV results
- Documents runtime results and limitations

### Pipeline

```text
Input Image
     ↓
Configurable Text Prompts
     ↓
SAM3SemanticPredictor
     ↓
Matching Object Instances
     ↓
Segmentation Masks
     ↓
Confidence + Mask-Area Analysis
     ↓
Reliability Filtering
     ↓
Visualization
     ↓
├── Filtered Output
└── Prompt Comparison
     ↓
Structured Results
     ↓
├── JSON
└── CSV
```

### Validation Result

The standalone pipeline was executed successfully in Google Colab using a Tesla T4 GPU.

```text
Images processed:          1
Prompts processed:         4
Object records generated: 10

vehicle: 1
bus:     1
person:  6
wheel:   2

Reliable person masks:     5
Runtime errors:            0
```

Generated and verified artifacts:

```text
2 visual outputs
1 JSON report
2 CSV reports
```

**Status:** Completed, successfully executed in Google Colab, documented, and validated with real visual, JSON, and CSV evidence.

---

### 12 — SAM 3 + Muse Glimmer Vision Agent

[`12-SAM3-Muse-Glimmer-Agent/`](./12-SAM3-Muse-Glimmer-Agent/)

A validation-first agentic computer-vision project that separates reasoning, segmentation, and deterministic analysis.

```text
Natural-Language Goal
        ↓
Muse Glimmer Agent Layer
        ↓
Schema-Validated Tool Call
        ↓
SAM 3 Segmentation Adapter
        ↓
Deterministic Mask Measurement
        ↓
Structured JSON Result
```

The initial scaffold includes:

- An executable standard-library mock pipeline
- Stable request, detection, and segmentation schemas
- A mock SAM 3 adapter for control-flow testing
- An explicit real-SAM boundary that cannot be mistaken for completed inference
- Bounded retry orchestration
- JSON result export
- Unit tests
- Runtime configuration template
- Input, output, results, and validation documentation
- A checklist for the future real Muse Glimmer + SAM 3 execution

The mock backend validates software structure and orchestration only. It does not inspect images, run either model, or provide model-quality evidence.

**Status:** Project scaffold completed; mock pipeline available; real Muse Glimmer and SAM 3 integration pending practical validation.

---

# Project Progression

The projects are intentionally organized so that each one introduces additional computer vision capabilities.

```text
Project 01
Basic Object Detection
        ↓
Project 02
Multi-Layer Visualization
        ↓
Project 03
Detection Filtering + NMS
        ↓
Project 04
Object Tracking
        ↓
Project 05
Zones + Counting
        ↓
Project 06
Visual Tracking + Analysis System
        ↓
Project 07
SAM 3 Segmentation Pipeline
        ↓
Project 08
Advanced Mask Analysis Pipeline
        ↓
Project 09
SAM3 Semantic Prompt Analytics
        ↓
Project 10
SAM Encoder-Decoder Architecture
        ↓
Project 11
SAM3 Interactive Point Refinement
        ↓
Project 12
SAM 3 + Muse Glimmer Vision Agent
```

This progression moves from basic inference toward complete computer vision systems involving detection, visualization, filtering, tracking, spatial analytics, detector-guided segmentation, semantic text prompting, quantitative mask analysis, persistence, evaluation, and structured results.

---

# Skills Developed Across the Projects

The nine completed projects demonstrate practical experience with:

- Python
- Computer Vision
- YOLOv8
- Ultralytics
- Supervision
- OpenCV
- NumPy
- Object Detection
- Detection Filtering
- Confidence Thresholding
- Class Filtering
- Bounding-Box Filtering
- Non-Maximum Suppression
- Top-N Selection
- Spatial Filtering
- Annotation Composition
- Object Tracking
- ByteTrack
- Persistent Tracker IDs
- Movement Traces
- PolygonZone
- LineZone
- Occupancy Analysis
- Directional Counting
- Video Processing
- SAM 3
- Instance Segmentation
- Bounding-Box Prompting
- Natural-Language Text Prompts
- SAM3SemanticPredictor
- Semantic Segmentation
- Prompt Configuration
- Prompt-Based Object Discovery
- Reliability Filtering
- Pixel-Level Masks
- Mask Visualization
- Mask-Area Analysis
- Bounding-Box vs. Mask Comparison
- Mask-to-Box Occupancy Analysis
- Reusable Multi-Image Analysis Pipelines
- Reusable Multi-Prompt Analysis Pipelines
- Detection-Level Analytics
- Prompt-Summary Analytics
- JSON Result Export
- CSV Result Export
- SQLite Persistence
- Tracking Analytics
- Trajectory Analysis
- Quantitative Evaluation
- Precision
- Recall
- IoU
- Dice
- Ground-Truth Comparison
- Structured Project Documentation
- Google Colab Workflows

---

# Project Categories

## Object Detection

```text
Project 01
YOLO + Supervision Object Detector
```

Focus:

```text
Detection
    ↓
Visualization
    ↓
Structured Predictions
```

---

## Advanced Visualization

```text
Project 02
Multi-Annotator Visualization Pipeline
```

Focus:

```text
Detections
    ↓
Multiple Annotation Layers
    ↓
Rich Visual Output
```

---

## Detection Post-Processing

```text
Project 03
Detection Filtering and NMS Pipeline
```

Focus:

```text
Raw Detections
      ↓
Filtering
      ↓
NMS
      ↓
Top-N
      ↓
Spatial Rules
      ↓
Final Detections
```

---

## Object Tracking

```text
Project 04
Object Tracking with ByteTrack
```

Focus:

```text
Video
   ↓
Detection
   ↓
Tracking
   ↓
Persistent IDs
   ↓
Movement Traces
```

---

## Spatial Video Analytics

```text
Project 05
Zones and Counting
```

Focus:

```text
Tracking
    ↓
Polygon Zones
    +
Line Zones
    ↓
Occupancy + Counting
```

---

## Integrated Visual Analysis

```text
Project 06
Visual Tracking and Analysis System
```

Focus:

```text
Detection
    +
Segmentation
    +
Tracking
    +
Persistence
    +
Analytics
    +
Evaluation
```

---

## Pixel-Level Segmentation

```text
Project 07
SAM 3 Segmentation Pipeline
```

Focus:

```text
Detection
    ↓
Bounding-Box Prompts
    ↓
SAM 3
    ↓
Pixel-Level Masks
    ↓
Structured Segmentation Results
```

---

## Quantitative Mask Analysis

```text
Project 08
Advanced Mask Analysis Pipeline
```

Focus:

```text
Detection
    ↓
Filtering
    ↓
SAM 3 Segmentation
    ↓
Mask Measurement
    ↓
Occupancy Analysis
    ↓
JSON + CSV Results
```

---

## Semantic Prompt Analytics

```text
Project 09
SAM3 Semantic Prompt Analytics
```

Focus:

```text
Natural-Language Concepts
      ↓
SAM3SemanticPredictor
      ↓
Semantic Object Masks
      ↓
Confidence + Area Filtering
      ↓
Visual + JSON + CSV Evidence
```

---

# Repository Learning Flow

The repository follows a layered learning structure.

```text
Course Notes
     ↓
Concept Understanding
     ↓
Small Examples
     ↓
Practical Experiments
     ↓
Projects
     ↓
Integrated Computer Vision Systems
```

Within the projects, the technical progression is:

```text
Object Detection
      ↓
Visualization
      ↓
Detection Filtering
      ↓
Object Tracking
      ↓
Spatial Analytics
      ↓
Integrated Tracking Analysis
      ↓
Pixel-Level Segmentation
      ↓
Quantitative Mask Analysis
      ↓
Semantic Text-Prompt Analytics
```

Project 08 adds another important transition:

```text
Segmentation Mask
      ↓
Pixel Measurement
      ↓
Bounding-Box Comparison
      ↓
Occupancy Ratio
      ↓
Structured JSON / CSV Analytics
```

Project 09 adds the transition:

```text
Natural-Language Prompt
      ↓
Semantic Object Discovery
      ↓
Pixel-Level Masks
      ↓
Confidence + Area Filtering
      ↓
Multi-Prompt JSON / CSV Analytics
```

This structure separates learning material from reusable examples and larger integrated projects.

---

# Project Development Principles

Projects in this directory follow several common principles.

## Reproducibility

Each project should contain enough information to understand and reproduce its workflow.

This includes:

```text
README
Source Code
Requirements
Input Assets
Output Evidence
Documentation
```

---

## Organized Project Structure

Projects use dedicated folders for different responsibilities.

Depending on the project, this may include:

```text
src/
data/
assets/
results/
docs/
```

Source-code directories such as:

```text
src/
```

include their own `README.md` documentation.

---

## Evidence-Based Validation

A project is not considered complete simply because source code exists.

Validation should include evidence such as:

```text
Successful Execution
Generated Images
Generated Videos
JSON Results
CSV Results
Database Results
Evaluation Metrics
Screenshots
```

The exact evidence depends on the project.

---

## Incremental Complexity

Each project builds on concepts introduced previously.

The progression is designed to move from:

```text
Single Image
     ↓
Object Detection
     ↓
Visualization
     ↓
Filtering
     ↓
Video
     ↓
Tracking
     ↓
Spatial Analysis
     ↓
Segmentation
     ↓
Quantitative Analysis
```

This makes each project part of a larger learning path rather than an isolated exercise.

---

# Completed Projects

```text
01 — YOLO + Supervision Object Detector              ✅ Completed
02 — Multi-Annotator Visualization Pipeline          ✅ Completed
03 — Detection Filtering and NMS Pipeline            ✅ Completed
04 — Object Tracking with ByteTrack                  ✅ Completed
05 — Zones and Counting                              ✅ Completed
06 — Visual Tracking and Analysis System             ✅ Completed
07 — SAM 3 Segmentation Pipeline                     ✅ Completed
08 — Advanced Mask Analysis Pipeline                 ✅ Completed
09 — SAM3 Semantic Prompt Analytics                  ✅ Completed
```

**Total completed projects: 9**

---

# Current Technical Progression

The current project portfolio demonstrates the following technical progression:

```text
Object Detection
      ↓
Detection Visualization
      ↓
Detection Filtering
      ↓
Non-Maximum Suppression
      ↓
Spatial Filtering
      ↓
Video Object Tracking
      ↓
Persistent Tracker IDs
      ↓
Movement Traces
      ↓
Polygon Occupancy
      ↓
Directional Counting
      ↓
Tracking Analytics
      ↓
Persistence
      ↓
Ground-Truth Evaluation
      ↓
SAM 3 Segmentation
      ↓
Pixel-Level Masks
      ↓
Structured Segmentation Results
      ↓
Quantitative Mask Analysis
      ↓
Mask-to-Box Occupancy
      ↓
Reusable JSON / CSV Analytics
      ↓
Natural-Language Text Prompts
      ↓
Semantic Object Discovery
      ↓
Multi-Prompt Reliability Analytics
```

---

# Portfolio Value

The progression from Project 01 through Project 09 demonstrates increasing system complexity.

The portfolio begins with:

```text
Image
  ↓
YOLO
  ↓
Bounding Boxes
```

and progresses toward:

```text
Multiple Images
      ↓
YOLOv8 Detection
      ↓
Detection Filtering
      ↓
SAM 3 Segmentation
      ↓
Pixel-Level Masks
      ↓
Quantitative Mask Analysis
      ↓
Annotated Visual Evidence
      ↓
JSON + CSV Analytics
      ↓
Natural-Language Prompt Configuration
      ↓
Semantic Multi-Prompt Analytics
```

Along the way, the projects also demonstrate:

```text
Video Tracking
      ↓
Persistent IDs
      ↓
Spatial Zones
      ↓
Counting
      ↓
Persistence
      ↓
Evaluation
```

This progression shows development beyond basic model inference toward reusable computer vision systems that produce visual, numerical, and structured evidence.

---

# Final Summary

The `05-projects/` directory currently contains **9 completed computer vision projects**.

Together, these projects demonstrate a progression through:

```text
Object Detection
      ↓
Visualization
      ↓
Detection Filtering
      ↓
Object Tracking
      ↓
Zones and Counting
      ↓
Integrated Visual Analysis
      ↓
SAM 3 Segmentation
      ↓
Advanced Mask Analysis
      ↓
SAM3 Semantic Prompt Analytics
```

Project 09 represents the latest completed stage of this progression.

Its validated execution processed:

```text
1 input image
4 natural-language prompts
10 semantic object records

vehicle: 1
bus:     1
person:  6
wheel:   2
```

and generated:

```text
2 visual outputs
1 JSON result file
2 CSV result files
```

with all expected generated artifacts verified successfully.

The repository now demonstrates detector-guided and text-guided segmentation, including the transformation of semantic masks into measurable analytical information through:

```text
Natural-Language Prompt
    +
Confidence
    +
Mask Area
    +
Reliability Rules
    +
Structured JSON / CSV Results
```

This establishes a foundation for future work involving more advanced segmentation, temporal analysis, video segmentation, tracking, analytics, and integrated computer vision applications.

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey
