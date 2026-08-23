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
### 04 — Object Tracking

[`04-Object-Tracking/`](./04-Object-Tracking/)

A complete video object-tracking pipeline using **YOLOv8**, **Supervision**, and **ByteTrack**.

This project extends object detection from individual images into sequential video analysis.

Instead of treating every frame independently, ByteTrack assigns persistent `tracker_id` values that allow detected cars to be followed across consecutive frames.

The project:

- Loads a real traffic video
- Reads video metadata
- Runs YOLOv8 detection on every frame
- Converts predictions into `sv.Detections`
- Filters detections to the COCO `car` class
- Passes filtered detections to ByteTrack
- Assigns persistent tracker IDs
- Counts visible frames for each tracked car
- Calculates approximate visible time
- Maintains a set of unique tracker IDs
- Draws bounding boxes
- Adds tracking labels
- Draws object trajectories
- Displays a unique tracked-car counter
- Generates an annotated output video
- Preserves the real input and generated output as project evidence

### Pipeline

```text
Input Traffic Video
        ↓
VideoInfo
        ↓
Read Frame
        ↓
YOLOv8
        ↓
sv.Detections
        ↓
Car Class Filtering
        ↓
ByteTrack
        ↓
tracker_id
        ↓
Frame Counting
        ↓
Visible-Time Calculation
        ↓
Unique Tracker IDs
        ↓
BoxAnnotator
        ↓
LabelAnnotator
        ↓
TraceAnnotator
        ↓
Annotated Output Video
```

### Test Input

The final project uses:

```text
assets/input/vehicles.mp4
```

The repository version contains a short real-world traffic sample:

```text
Duration: 10 seconds
Frames: 250
Resolution: 1280 × 720
Target Class: Car
COCO Class ID: 2
```

Using a short real-world video keeps the repository lightweight while providing enough consecutive frames to demonstrate persistent object tracking.

### Test Result

Project 04 was successfully tested in **Google Colab**.

The final run processed:

```text
Processing video: 100% 250/250
```

Tracking analytics:

```text
Tracker ID 1: 146 frames | 5.84 seconds
Tracker ID 2: 54 frames  | 2.16 seconds
Tracker ID 3: 47 frames  | 1.88 seconds
Tracker ID 4: 173 frames | 6.92 seconds
Tracker ID 5: 176 frames | 7.04 seconds
Tracker ID 6: 27 frames  | 1.08 seconds
```

Unique tracker IDs:

```text
[1, 2, 3, 4, 5, 6]
```

Final result:

```text
Total unique tracked cars: 6

Object Tracking Project: SUCCESS
```

### Project Evidence

Project 04 contains:

```text
04-Object-Tracking/
│
├── assets/
│   ├── README.md
│   │
│   ├── input/
│   │   ├── README.md
│   │   └── vehicles.mp4
│   │
│   └── output/
│       ├── README.md
│       └── tracked_vehicles.mp4
│
├── object_tracking_pipeline.py
├── requirements.txt
└── README.md
```

The generated output contains:

```text
YOLO Car Detection
        ↓
Bounding Boxes
        ↓
Persistent Tracker IDs
        ↓
Frame Counts
        ↓
Visible-Time Estimates
        ↓
Tracking Trajectories
```

The final output video is stored at:

```text
04-Object-Tracking/assets/output/tracked_vehicles.mp4
```

### Important Tracking Note

A ByteTrack `tracker_id` represents an identity maintained during a tracking sequence.

It should not automatically be interpreted as a permanent real-world identity.

If an object disappears and later returns, the tracker may assign it a different ID.

Therefore:

```text
Unique Tracker IDs
```

represent identities maintained by the tracker during the processed sequence rather than guaranteed unique physical vehicles.

**Status:** Completed, successfully tested in Google Colab, documented, and supported with real input/output video evidence.

---

### 05 — Zones and Counting Analytics

[`05-Zones-and-Counting-Analytics/`](./05-Zones-and-Counting-Analytics/)

A complete spatial video analytics project using **YOLOv8s**, **Supervision**, **ByteTrack**, `PolygonZone`, and `LineZone`.

This project extends object tracking by introducing spatial regions and virtual counting boundaries.

Instead of only tracking where people move, the application can determine how many tracked people are currently inside a defined region and whether a tracked person crosses a virtual line.

The project:

- Loads a real pedestrian video
- Reads video metadata
- Runs YOLOv8s person detection on every frame
- Filters inference to the COCO `person` class
- Converts YOLO predictions into `sv.Detections`
- Tracks pedestrians using ByteTrack
- Assigns persistent `tracker_id` values
- Defines a custom PolygonZone
- Calculates current PolygonZone occupancy
- Defines a custom LineZone
- Detects confirmed line-crossing events
- Maintains directional crossing counters
- Draws person bounding boxes
- Displays `Person #ID` labels
- Visualizes the PolygonZone
- Visualizes the LineZone
- Displays current people-in-zone analytics
- Displays Crossings In / Crossings Out
- Generates a complete annotated output video
- Converts the final video to an H.264-compatible MP4 for browser playback

### Pipeline

```text
Input Pedestrian Video
        ↓
VideoInfo
        ↓
Read Frame
        ↓
YOLOv8s
        ↓
Person Detection
        ↓
sv.Detections
        ↓
ByteTrack
        ↓
Persistent Person IDs
        ↓
        ├───────────────────────┐
        ↓                       ↓
   PolygonZone               LineZone
        ↓                       ↓
Current Occupancy         Crossing Events
        ↓                       ↓
        └───────────┬───────────┘
                    ↓
               BoxAnnotator
                    ↓
               LabelAnnotator
                    ↓
             Spatial Analytics
                    ↓
              Annotated Video
```

### Detection Configuration

The final tested detector configuration is:

```text
Model: YOLOv8s
Confidence Threshold: 0.15
Inference Size: 1280
Target Class: Person
COCO Class ID: 0
```

The project initially tested YOLOv8n, but YOLOv8s was selected for the final pipeline because the real pedestrian scene contains many small and partially occluded people.

### Tracking Configuration

The final ByteTrack configuration uses:

```text
Track Activation Threshold: 0.15
Lost Track Buffer: 90
Minimum Matching Threshold: 0.70
Frame Rate: Video FPS
```

The tracker configuration was adjusted to improve tracking continuity in the crowded pedestrian scene.

### PolygonZone

The final PolygonZone coordinates are:

```text
[520, 470]
[1320, 470]
[1460, 900]
[420, 900]
```

The PolygonZone represents the primary pedestrian-crossing area and allows the application to calculate:

```text
People in Zone
```

for every processed frame.

### LineZone

Multiple line orientations were tested during development:

```text
Horizontal
Vertical
Diagonal 1
Diagonal 2
```

The final implementation uses a vertical LineZone:

```text
Start: (960, 400)
End:   (960, 920)
```

The vertical orientation was the tested configuration that produced a confirmed pedestrian crossing event.

### Test Input

The final project uses:

```text
assets/input/people_walking.mp4
```

Video information:

```text
Resolution: 1920 × 1080
FPS: 50
Frames: 763
Duration: approximately 15.26 seconds
```

The video contains a real crowded pedestrian intersection and provides a challenging test for detection, tracking, occupancy analysis, and line-crossing analytics.

### Test Result

Project 05 was successfully tested in **Google Colab**.

The final run processed:

```text
Processing video: 100%
763/763
```

Final analytics:

```text
Final people in PolygonZone: 6

Crossings In: 0
Crossings Out: 1
Total Crossings: 1
```

Because the scene contains a dense crowd and significant occlusion, the LineZone result represents **confirmed crossings maintained by the tracking pipeline**, rather than the total number of physical pedestrians visible in the intersection.

### Project Evidence

Project 05 contains:

```text
05-Zones-and-Counting-Analytics/
│
├── assets/
│   ├── input/
│   │   ├── README.md
│   │   └── people_walking.mp4
│   │
│   └── output/
│       ├── README.md
│       └── people_zones_counting_final.mp4
│
├── zones_counting_analytics.py
├── requirements.txt
└── README.md
```

The final output video is stored at:

```text
05-Zones-and-Counting-Analytics/assets/output/people_zones_counting_final.mp4
```

The generated output demonstrates:

```text
YOLOv8s Person Detection
        ↓
Bounding Boxes
        ↓
Persistent Person IDs
        ↓
PolygonZone
        ↓
Current Occupancy
        ↓
LineZone
        ↓
Directional Crossing Analytics
        ↓
Final Annotated Video
```

**Status:** Completed, successfully tested in Google Colab, documented, and supported with real input/output video evidence.

---
### 06 — Visual Tracking and Analysis System

[`06-Visual-Tracking-and-Analysis-System/`](./06-Visual-Tracking-and-Analysis-System/)

A complete computer vision system combining **YOLO**, **ByteTrack**, **SAM 3**, **Supervision**, structured persistence, trajectory analytics, session comparison, visualization, an interactive dashboard, and formal ground-truth evaluation.

Project 06 represents the integration stage of the learning journey.

Instead of focusing on one isolated computer vision technique, the project combines detection, tracking, segmentation, persistence, analytics, evaluation, and presentation into a reproducible end-to-end system.

The project:

- Processes images and recorded videos
- Detects objects using YOLO
- Converts detections into `sv.Detections`
- Tracks objects using ByteTrack
- Maintains persistent tracker IDs
- Integrates SAM 3 segmentation
- Visualizes segmentation masks
- Draws bounding boxes and labels
- Visualizes object trajectories
- Stores structured observation history
- Uses SQLite persistence
- Preserves source and session information
- Records timestamps and confidence values
- Stores tracker and class information
- Generates tracker-level analytics
- Reconstructs object trajectories
- Calculates image-space movement
- Generates analytical CSV reports
- Generates visual analytics charts
- Compares recorded processing sessions
- Provides an interactive results explorer
- Provides a tracking dashboard
- Documents system limitations and failure cases
- Evaluates false positives
- Evaluates omissions
- Measures Precision and Recall
- Measures segmentation IoU
- Measures Dice coefficient
- Generates pixel-level confusion-matrix values
- Uses manually annotated ground-truth data
- Documents lighting, scale, occlusion, and out-of-sample limitations

### System Pipeline

```text
Image / Recorded Video
          ↓
     YOLO Detection
          ↓
     sv.Detections
          ↓
       ByteTrack
          ↓
 Persistent Tracker IDs
          ↓
   SAM 3 Segmentation
          ↓
 Visualization / Masks
          ↓
 Structured Persistence
          ↓
   SQLite Observation Data
          ↓
 Tracker / Trajectory Analytics
          ↓
      CSV Reports
          ↓
 Visual Analytics / Charts
          ↓
    Session Comparison
          ↓
 Interactive Dashboard
          ↓
 Ground-Truth Evaluation
```

### Core Technologies

Project 06 integrates:

```text
Python
OpenCV
YOLO / Ultralytics
Supervision
ByteTrack
SAM 3
PyTorch
SQLite
Pandas
Matplotlib
Streamlit
COCO Segmentation
Roboflow
```

### Tracking and Analytics

A verified recorded-video processing run produced:

```text
Processed frames: 75
Recorded observations: 246
Unique tracker IDs: 6
```

The tracked objects consisted of:

```text
5 person trackers
1 bus tracker
```

Verified system-level analytics include:

| Metric | Result |
|---|---:|
| Total processed frames | 75 |
| Total observations | 246 |
| Unique tracker IDs | 6 |
| Average observations per frame | 3.2800 |
| Average observations per tracker | 41.0000 |
| Minimum tracker observations | 3 |
| Maximum tracker observations | 75 |
| Average confidence | 0.6815 |
| Minimum average confidence | 0.5124 |
| Maximum average confidence | 0.8587 |
| Average tracker duration | 2.7333 s |
| Minimum tracker duration | 0.2000 s |
| Maximum tracker duration | 5.0000 s |
| Total movement distance | 693.1800 px |
| Average movement distance per tracker | 115.5300 px |
| Maximum movement distance | 203.3700 px |
| Average step movement | 2.9883 px |

### Tracker Results

The verified tracker summary includes:

```text
Tracker 1
Class: person
Frames: 1–75
Observations: 75
Duration: 5.00 s
Average confidence: 0.8385
Movement: 159.26 px

Tracker 2
Class: person
Frames: 1–75
Observations: 75
Duration: 5.00 s
Average confidence: 0.8587
Movement: 203.37 px

Tracker 3
Class: bus
Frames: 3–75
Observations: 59
Duration: 3.93 s
Average confidence: 0.5939
Movement: 182.36 px

Tracker 4
Class: person
Frames: 8–32
Observations: 25
Duration: 1.67 s
Average confidence: 0.7579
Movement: 139.34 px

Tracker 5
Class: person
Frames: 29–31
Observations: 3
Duration: 0.20 s
Average confidence: 0.5278
Movement: 7.91 px

Tracker 6
Class: person
Frames: 53–61
Observations: 9
Duration: 0.60 s
Average confidence: 0.5124
Movement: 0.94 px
```

### Persistent Observation Storage

Project 06 includes a SQLite persistence architecture.

The database stores processing sessions and individual observations.

Session information includes:

```text
Session ID
Source
Creation timestamp
Notes
```

Observation information includes:

```text
Observation ID
Session ID
Frame number
Timestamp
Tracker ID
Class ID
Class name
Confidence
Bounding-box coordinates
Notes
```

This satisfies the project's requirement to preserve structured historical tracking evidence rather than limiting results to annotated video output.

### Session Comparison

The project supports comparison between recorded processing sessions.

Historical session data can be used to compare:

- Source media
- Observation counts
- Tracker counts
- Class distribution
- Average confidence
- Movement information
- Session-level analytical results

This allows Project 06 to move beyond single-run processing and provide historical analysis of previously processed media.

### Analytical Reports

Project 06 generates reusable reports under:

```text
reports/
```

Verified analytical outputs include:

```text
tracker_summary.csv
trajectory_summary.csv
performance_summary.csv

trajectory_visualization.png
tracker_duration_chart.png
class_observation_chart.png
movement_distance_chart.png
confidence_chart.png
performance_chart.png
```

The preserved CSV reports allow later analytics to be generated without rerunning the computationally expensive YOLO, ByteTrack, and SAM 3 pipeline.

### Visual Analytics

The generated visual analytics provide different views of system behavior.

They include:

```text
Trajectory Visualization
Tracker Duration Analysis
Class Observation Analysis
Movement Distance Analysis
Detection Confidence Analysis
System Performance Summary
```

These reports transform raw detection and tracking observations into interpretable project evidence.

### Interactive Application

Project 06 includes an application layer for exploring results.

The interface provides access to:

- Processing results
- Tracker analytics
- Confidence analytics
- Movement analytics
- Trajectory information
- Analytical charts
- Session information
- Session comparison
- Historical evidence

This satisfies the project requirement for an analysis tool and tracking dashboard.

---

## Ground-Truth Segmentation Evaluation

Project 06 includes a dedicated formal evaluation stage using manually annotated ground-truth data.

A **Roboflow Instance Segmentation** project was created specifically for evaluation.

The ground-truth dataset contains:

```text
20 evaluation images
424 manually annotated person instances
```

The images were selected from:

```text
Session 001
Session 002
```

The annotations were exported using:

```text
COCO Segmentation
```

The evaluation compares SAM 3 `person` segmentation results against the manually annotated reference masks.

### Evaluation Pipeline

```text
Evaluation Images
        ↓
Manual Ground-Truth Masks
        ↓
COCO Segmentation Dataset
        ↓
Project 06 Pipeline
        ↓
SAM 3 Person Segmentation
        ↓
Prediction / Ground-Truth Matching
        ↓
TP / FP / FN
        ↓
Precision / Recall
        ↓
IoU / Dice
        ↓
Pixel Confusion Matrix
        ↓
Evaluation Reports
```

### Ground-Truth Evaluation Results

The completed evaluation produced:

| Metric | Result |
|---|---:|
| Evaluated images | 20 |
| Ground-truth person instances | 424 |
| Predicted instances | 472 |
| True positives | 381 |
| False positives | 91 |
| False negatives / omissions | 43 |
| Precision | 0.8072 |
| Recall | 0.8986 |
| Average IoU | 0.7969 |
| Average Dice | 0.8829 |

### Pixel-Level Confusion Matrix

The evaluation also produced pixel-level confusion-matrix values:

| Pixel Classification | Count |
|---|---:|
| True-positive pixels | 1,604,567 |
| False-positive pixels | 341,396 |
| False-negative pixels | 229,371 |
| True-negative pixels | 20,864,666 |

These values provide quantitative evidence of segmentation agreement and disagreement between SAM 3 predictions and the manually annotated reference masks.

### Evaluation Outputs

The evaluation stage preserves:

```text
evaluation/
│
├── README.md
├── evaluate_ground_truth.py
├── evaluation_metrics.csv
├── evaluation_summary.json
│
└── ground_truth/
    ├── README.md
    │
    └── roboflow_export/
        ├── README.md
        ├── _annotations.coco.json
        └── evaluation images
```

The evaluation runner can reproduce the metrics from the preserved ground-truth dataset.

### Evaluation Interpretation

The measured results show:

```text
Precision:    0.8072
Recall:       0.8986
Average IoU:  0.7969
Average Dice: 0.8829
```

Recall is higher than precision in the evaluated dataset.

The system successfully matched:

```text
381 / 424
```

ground-truth person instances while producing:

```text
91 false positives
43 omissions
```

The results provide formal project-level evidence of model performance without claiming universal accuracy outside the evaluated dataset.

---

## Environmental Limitations and Failure Cases

The project explicitly documents failure conditions required by the original project proposal.

These include:

### Lighting

Prediction reliability may decrease under:

```text
Low illumination
Strong shadows
Backlighting
Overexposure
Underexposure
Sudden lighting changes
Low foreground/background contrast
```

### Scale

Small or distant people occupy fewer pixels and can be more difficult to:

```text
Detect
Track
Segment
```

This can lead to missed detections, incomplete masks, and tracker fragmentation.

### Occlusion

Partial or complete occlusion can result in:

```text
Missing detections
Incomplete segmentation
Lost tracks
Tracker fragmentation
New tracker IDs
```

Crowded scenes increase this challenge because people frequently overlap.

### Out-of-Sample Data

Performance on substantially different environments cannot be assumed to equal the measured Project 06 evaluation results.

Potential out-of-sample conditions include:

```text
Unusual camera viewpoints
Extreme lighting
Very different resolutions
Severe compression
Extreme crowd density
Strong motion blur
Infrared or non-standard imagery
Environments substantially different from tested scenes
```

The measured evaluation metrics therefore apply to the Project 06 evaluation dataset and should not be interpreted as guaranteed performance on arbitrary unseen data.

---

## Project 06 Evidence

Project 06 preserves implementation, analytics, evaluation, and documentation evidence across its organized directory structure.

Major project areas include:

```text
06-Visual-Tracking-and-Analysis-System/
│
├── analytics/
├── data/
├── docs/
├── evaluation/
├── notebooks/
├── reports/
├── src/
│
├── app.py
├── requirements.txt
└── README.md
```

The project documentation includes detailed results, limitations, evaluation methodology, architecture, and reproducibility information.

### Definition of Done

The original Project 06 proposal defines completion around three primary requirements:

```text
Process recorded videos
Register tracking history in the database
Generate exportable reports containing performance metrics
```

The completed implementation demonstrates all three.

The broader MVP also demonstrates:

```text
Image and recorded-video processing
Historical observation storage
Session querying
Session comparison
Digital results exploration
Analysis tooling
Tracking dashboard
Environmental failure documentation
False-positive evaluation
Omission evaluation
Ground-truth segmentation evaluation
```

### Final Project Status

**Status:** Completed.

Project 06 successfully integrates the major concepts developed throughout the earlier projects into a complete computer vision analysis system.

It demonstrates:

```text
Detection
    ↓
Tracking
    ↓
Segmentation
    ↓
Persistence
    ↓
Analytics
    ↓
Session Comparison
    ↓
Visualization
    ↓
Dashboard
    ↓
Ground-Truth Evaluation
    ↓
Documented Results and Limitations
```

The official Project 06 MVP requirements and Definition of Done are satisfied.

---

# Skills Demonstrated

Across the six completed projects, the following computer vision and software-development skills are demonstrated:

### Computer Vision

- Object detection
- Instance segmentation
- Multi-object tracking
- Spatial analytics
- Polygon-zone analysis
- Line-crossing analysis
- Trajectory reconstruction
- Detection filtering
- Non-Maximum Suppression
- Confidence filtering
- Class filtering
- Spatial filtering
- Ground-truth evaluation
- Mask comparison
- False-positive analysis
- Omission analysis

### YOLO / Ultralytics

- YOLOv8 inference
- YOLOv8n
- YOLOv8s
- Confidence thresholds
- Class filtering
- Prediction processing
- Video-frame inference

### Supervision

- `sv.Detections`
- `BoxAnnotator`
- `LabelAnnotator`
- `EllipseAnnotator`
- `DotAnnotator`
- `TraceAnnotator`
- `ByteTrack`
- `PolygonZone`
- `LineZone`
- Detection filtering
- Detection visualization
- Tracking visualization
- Spatial analytics

### SAM 3

- SAM 3 model integration
- Prompt-based segmentation
- Person segmentation
- Segmentation-mask generation
- Integration with detection and tracking workflows
- Ground-truth mask comparison
- Segmentation evaluation

### Evaluation

- Manual ground-truth annotation
- COCO Segmentation datasets
- True-positive analysis
- False-positive analysis
- False-negative / omission analysis
- Precision
- Recall
- Intersection over Union
- Dice coefficient
- Pixel-level confusion matrices
- Per-image evaluation metrics
- Evaluation-summary generation

### Tracking and Temporal Analytics

- Persistent tracker IDs
- Frame-level observations
- Tracker duration
- Visible-time analysis
- Trajectory reconstruction
- Image-space movement measurement
- Average step movement
- Session-level tracking analytics
- Historical session comparison

### Persistence and Data Analysis

- SQLite
- Structured session storage
- Structured observation storage
- Timestamp preservation
- Source-media preservation
- Confidence storage
- Tracker and class storage
- Bounding-box storage
- Notes
- Pandas
- CSV report generation
- Reusable analytics

### Visualization

- Bounding boxes
- Detection labels
- Tracking labels
- Segmentation masks
- Object traces
- Polygon zones
- Line zones
- Trajectory plots
- Tracker-duration charts
- Class-observation charts
- Movement-distance charts
- Confidence charts
- Performance charts

### Application Development

- Modular Python architecture
- Command-line processing
- Streamlit interface development
- Results exploration
- Tracking dashboards
- Session comparison
- Historical evidence presentation

### Engineering Workflow

- Google Colab
- CUDA GPU execution
- Git
- GitHub
- Dependency management
- Project documentation
- Reproducible testing
- Input/output evidence preservation
- H.264 video generation
- Structured project organization

---

# Project Progress

| Project | Main Topic | Status |
|---|---|---|
| 01 | YOLO + Supervision Object Detection | ✅ Completed |
| 02 | Multi-Annotator Visualization | ✅ Completed |
| 03 | Detection Filtering and NMS | ✅ Completed |
| 04 | Object Tracking with ByteTrack | ✅ Completed |
| 05 | Zones and Counting Analytics | ✅ Completed |
| 06 | Visual Tracking and Analysis System | ✅ Completed |

```text
Total completed projects: 6
```

Project 06 represents the integration and evaluation stage of the current project sequence.

---

# Technologies Used

The projects currently use:

```text
Python
OpenCV
NumPy
Pandas
Matplotlib
PyTorch
Ultralytics
YOLOv8
Supervision
ByteTrack
SAM 3
SQLite
Streamlit
Roboflow
COCO Segmentation
Google Colab
CUDA
Git
GitHub
```

Additional tools and formats are introduced when required by individual projects.

---

# Project Structure

```text
05-projects/
│
├── README.md
│
├── 01-YOLO-Supervision-Object-Detector/
│
├── 02-Multi-Annotator-Visualization-Pipeline/
│
├── 03-Detection-Filtering-and-NMS-Pipeline/
│
├── 04-Object-Tracking/
│
├── 05-Zones-and-Counting-Analytics/
│
└── 06-Visual-Tracking-and-Analysis-System/
```

Each project maintains its own implementation, documentation, dependencies, assets, and generated evidence as required.

---

# Learning Progression

The six projects demonstrate a progressive development path.

```text
Project 01
Basic Object Detection
        ↓
Project 02
Multi-Layer Visualization
        ↓
Project 03
Detection Filtering and NMS
        ↓
Project 04
Temporal Object Tracking
        ↓
Project 05
Spatial Zones and Counting
        ↓
Project 06
Integrated Tracking and Analysis System
```

The technical progression can also be represented as:

```text
Detection
    ↓
Visualization
    ↓
Filtering
    ↓
Tracking
    ↓
Spatial Analytics
    ↓
Segmentation
    ↓
Persistence
    ↓
Trajectory Analytics
    ↓
Session Comparison
    ↓
Dashboard
    ↓
Ground-Truth Evaluation
```

Project 06 brings these concepts together into a complete system rather than demonstrating them independently.

---

# Project 06 Evaluation Milestone

The completion of Project 06 adds formal quantitative evaluation to the project collection.

The manually annotated evaluation dataset contains:

```text
20 images
424 ground-truth person instances
```

The final evaluation produced:

```text
Predicted instances:            472
True positives:                 381
False positives:                 91
False negatives / omissions:     43

Precision:                     0.8072
Recall:                        0.8986
Average IoU:                   0.7969
Average Dice:                  0.8829
```

Pixel-level confusion-matrix values:

```text
True-positive pixels:       1,604,567
False-positive pixels:        341,396
False-negative pixels:        229,371
True-negative pixels:      20,864,666
```

This milestone demonstrates the transition from verifying that a computer vision pipeline **runs successfully** to quantitatively evaluating **how accurately it performs against manually annotated reference data**.

---

# Current Project Collection Status

The current project collection now demonstrates six progressively more advanced computer vision applications.

The progression begins with basic object detection and develops through:

```text
Object Detection
        ↓
Annotation
        ↓
Detection Filtering
        ↓
Object Tracking
        ↓
Zones and Counting
        ↓
SAM 3 Segmentation
        ↓
Structured Persistence
        ↓
Trajectory Analytics
        ↓
Historical Session Comparison
        ↓
Interactive Results Exploration
        ↓
Formal Ground-Truth Evaluation
```

Project 06 completes the current integration objective by combining these concepts into a documented, reproducible computer vision analysis system.

---

# Next Steps

The six current projects are complete.

Future projects may introduce additional computer vision concepts as they are covered during the learning journey.

Potential future work should remain separate from the completed Project 06 MVP.

Project 06 already documents possible extensions such as:

- Larger evaluation datasets
- Additional environmental-condition testing
- Tracking-specific metrics such as MOTA, IDF1, and HOTA
- Camera-motion compensation
- Perspective calibration
- Physical distance estimation
- Processing-time benchmarking

These are future extensions and are **not required for the completed Project 06 MVP**.

---

# Purpose of the Projects Directory

The `05-projects/` directory serves as practical portfolio evidence for the SAM3 Computer Vision Learning Journey.

Each project demonstrates the ability to move from individual concepts toward complete implementations that include:

```text
Problem Definition
        ↓
Implementation
        ↓
Testing
        ↓
Evidence
        ↓
Documentation
        ↓
Analysis
        ↓
Evaluation
```

The completed projects preserve not only source code, but also the inputs, outputs, metrics, reports, and documentation required to understand and reproduce the work.

---

# Author

**Peyman Miyandashti**

GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)

LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)

