# Project Results

This document presents the verified results produced by **Project 06 — Visual Tracking and Analysis System**.

The project combines object detection, multi-object tracking, SAM 3 segmentation, structured analytics, trajectory analysis, persistence, and performance reporting into a complete computer-vision workflow.

---

# Verified Pipeline

The completed system follows this processing architecture:

```text
Input Image / Video
        ↓
YOLO Object Detection
        ↓
ByteTrack Multi-Object Tracking
        ↓
Persistent Tracker IDs
        ↓
SAM 3 Segmentation
        ↓
Annotated Video Output
        ↓
Tracking Observations
        ↓
SQLite Persistence
        ↓
Tracker Analytics
        ↓
Trajectory Analytics
        ↓
Performance Analysis
        ↓
CSV Reports
        ↓
Visual Analytics
```

Each stage was implemented and tested as part of Project 06.

---

# Verified Video Processing

The current reference video-processing run produced:

| Metric | Result |
|---|---:|
| Processed frames | 75 |
| Recorded observations | 246 |
| Unique tracker IDs | 6 |

The system successfully maintained multiple persistent tracker IDs across the processed video sequence.

The verified H.264 output video is:

```text
sam3_tracking_output_01.mp4
```

This video demonstrates the integration of detection, tracking, and SAM 3 segmentation within the same video-processing pipeline.

---

# Verified Session 002 — Busy Street Video Run

A second independent recorded-video experiment was completed on **2026-08-22** using the same Project 06 detection, tracking, segmentation, and analytics architecture.

The source media for this experiment was:

`tracking_test_02.mp4`

The verified browser-compatible H.264 output is:

[`../assets/output/sam3_tracking_output_02.mp4`](../assets/output/sam3_tracking_output_02.mp4)

## Session 002 Results

| Metric | Result |
|---|---:|
| Processed frames | 75 |
| Recorded observations | 720 |
| Unique tracker IDs | 52 |
| Observations per frame | 9.6000 |
| Observations per tracker | 13.8462 |
| Average confidence | 0.6392 |
| Average tracker duration | 0.4617 s |
| Total movement distance | 6946.85 px |

The second session processed the same number of frames as the historical baseline while representing a substantially more active street scene.

---

# Verified Session Comparison

Project 06 now contains **two real verified processing sessions**.

| Metric | Session 001 | Session 002 | Change |
|---|---:|---:|---:|
| Processed frames | 75 | 75 | 0 |
| Total observations | 246 | 720 | +474 |
| Unique tracker IDs | 6 | 52 | +46 |
| Observations per frame | 3.2800 | 9.6000 | +6.3200 |
| Observations per tracker | 41.0000 | 13.8462 | -27.1538 |
| Average confidence | 0.6815 | 0.6392 | -0.0423 |
| Average tracker duration | 2.7333 s | 0.4617 s | -2.2716 s |
| Total movement distance | 693.18 px | 6946.85 px | +6253.67 px |

Compared with Session 001, Session 002 produced **474 additional observations** and **46 additional tracker IDs** across the same 75-frame processing window.

Average confidence decreased from **0.6815** to **0.6392**, while total measured image-space movement increased from **693.18 pixels** to **6946.85 pixels**.

These differences demonstrate how scene complexity and object activity affect the analytical outputs of the tracking system.

## Comparison Evidence

- [`../data/session_history.csv`](../data/session_history.csv)
- [`../data/session_002_observations.csv`](../data/session_002_observations.csv)
- [`../reports/session_comparison_summary.csv`](../reports/session_comparison_summary.csv)
- [`../reports/session_comparison_chart.png`](../reports/session_comparison_chart.png)
- [`../reports/tracker_summary_session_002.csv`](../reports/tracker_summary_session_002.csv)
- [`../reports/trajectory_summary_session_002.csv`](../reports/trajectory_summary_session_002.csv)

Both verified sessions can also be explored interactively through the Streamlit **Session History** page.

---

# Tracker Results

The tracker-level results are stored in:

[`../reports/tracker_summary.csv`](../reports/tracker_summary.csv)

The verified tracker results are:

| Tracker ID | Class | First Frame | Last Frame | Observations | Duration | Average Confidence |
|---:|---|---:|---:|---:|---:|---:|
| 1 | person | 1 | 75 | 75 | 5.00 s | 0.8385 |
| 2 | person | 1 | 75 | 75 | 5.00 s | 0.8587 |
| 3 | bus | 3 | 75 | 59 | 3.93 s | 0.5939 |
| 4 | person | 8 | 32 | 25 | 1.67 s | 0.7579 |
| 5 | person | 29 | 31 | 3 | 0.20 s | 0.5278 |
| 6 | person | 53 | 61 | 9 | 0.60 s | 0.5124 |

These results show that the system tracked multiple objects with significantly different visibility durations.

---

# Tracker Persistence

The longest-lived trackers were:

```text
Tracker 1 → 75 observations → 5.00 seconds
Tracker 2 → 75 observations → 5.00 seconds
```

Both trackers remained visible throughout the complete verified processing sequence.

Tracker 3 was also persistent:

```text
Tracker 3 → 59 observations → 3.93 seconds
```

Shorter tracks included:

```text
Tracker 4 → 25 observations → 1.67 seconds
Tracker 5 → 3 observations  → 0.20 seconds
Tracker 6 → 9 observations  → 0.60 seconds
```

Short-lived trackers may represent objects entering or leaving the scene, temporary visibility, occlusion, detection instability, or tracker fragmentation.

---

# Observation Results

The complete run produced:

```text
246 observations
```

across:

```text
75 frames
```

The average number of observations per frame was:

```text
3.2800
```

The average number of observations per tracker was:

```text
41.0000
```

The observation range per tracker was:

```text
Minimum: 3 observations
Maximum: 75 observations
```

---

# Detection Confidence Results

The verified tracker-level confidence analysis produced:

| Metric | Result |
|---|---:|
| Overall average confidence | 0.6815 |
| Minimum tracker average confidence | 0.5124 |
| Maximum tracker average confidence | 0.8587 |

The highest average confidence belonged to:

```text
Tracker 2
Average confidence: 0.8587
```

The lowest average confidence belonged to:

```text
Tracker 6
Average confidence: 0.5124
```

The confidence results are visualized in:

[`../reports/confidence_chart.png`](../reports/confidence_chart.png)

---

# Tracker Duration Results

The tracker-duration analysis produced:

| Metric | Result |
|---|---:|
| Average tracker duration | 2.7333 s |
| Minimum tracker duration | 0.2000 s |
| Maximum tracker duration | 5.0000 s |

Tracker duration provides a simple measurement of how long each persistent identity remained active during the analyzed video.

The results are visualized in:

[`../reports/tracker_duration_chart.png`](../reports/tracker_duration_chart.png)

---

# Trajectory Results

The trajectory analysis is stored in:

[`../reports/trajectory_summary.csv`](../reports/trajectory_summary.csv)

The verified trajectory results are:

| Tracker ID | First Frame | Last Frame | Frames Observed | Duration | Movement Distance | Average Movement |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 75 | 75 | 5.00 s | 159.26 px | 2.15 px |
| 2 | 1 | 75 | 75 | 5.00 s | 203.37 px | 2.75 px |
| 3 | 3 | 75 | 59 | 3.93 s | 182.36 px | 3.14 px |
| 4 | 8 | 32 | 25 | 1.67 s | 139.34 px | 5.81 px |
| 5 | 29 | 31 | 3 | 0.20 s | 7.91 px | 3.96 px |
| 6 | 53 | 61 | 9 | 0.60 s | 0.94 px | 0.12 px |

---

# Movement Results

The complete trajectory analysis produced:

| Metric | Result |
|---|---:|
| Total movement distance | 693.18 px |
| Average movement distance per tracker | 115.53 px |
| Maximum movement distance | 203.37 px |
| Average step movement | 2.9883 px |

The tracker with the greatest accumulated movement was:

```text
Tracker 2
203.37 pixels
```

Tracker 6 produced the smallest accumulated movement:

```text
Tracker 6
0.94 pixels
```

These distances represent movement in image coordinates.

They do not represent physical-world distance.

---

# Trajectory Visualization

The reconstructed tracker paths are visualized in:

[`../reports/trajectory_visualization.png`](../reports/trajectory_visualization.png)

The visualization provides a spatial representation of:

- Tracker movement
- Trajectory direction
- Relative movement distance
- Tracker continuity
- Object behavior across frames

The trajectory visualization complements the numerical movement measurements stored in `trajectory_summary.csv`.

---

# Movement Distance Visualization

Movement-distance comparisons are visualized in:

[`../reports/movement_distance_chart.png`](../reports/movement_distance_chart.png)

The chart makes it possible to quickly compare accumulated image-space movement between tracker IDs.

---

# Class Observation Results

The class distribution is visualized in:

[`../reports/class_observation_chart.png`](../reports/class_observation_chart.png)

The verified tracker summary contains:

```text
5 person trackers
1 bus tracker
```

The class-observation visualization represents observation frequency rather than simply counting unique tracker IDs.

This distinction is important because a tracker that remains visible for many frames contributes more observations than a short-lived tracker.

---

# Performance Analysis

System-level analytics are generated by:

[`../analytics/performance_analysis.py`](../analytics/performance_analysis.py)

The module reads:

```text
reports/tracker_summary.csv
reports/trajectory_summary.csv
```

and generates:

```text
reports/performance_summary.csv
reports/performance_chart.png
```

This allows system-level analytics to be reproduced without rerunning the computationally expensive detection, tracking, and segmentation pipeline.

---

# Performance Summary

The complete system-level report is stored in:

[`../reports/performance_summary.csv`](../reports/performance_summary.csv)

Verified results include:

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

---

# Performance Visualization

The system-level count summary is visualized in:

[`../reports/performance_chart.png`](../reports/performance_chart.png)

The chart compares:

```text
75 processed frames
246 observations
6 tracker IDs
```

This provides a compact visual summary of the verified processing run.

---

# Ground-Truth Segmentation Evaluation

A dedicated ground-truth evaluation was completed to measure SAM 3 `person` segmentation performance against manually annotated reference data.

The evaluation dataset contains:

| Dataset Metric | Result |
|---|---:|
| Evaluated images | 20 |
| Ground-truth person instances | 424 |
| SAM 3 predicted instances | 472 |

The reference dataset was manually annotated in Roboflow using an **Instance Segmentation** project and exported using **COCO Segmentation** format.

The evaluation runner is:

[`../evaluation/evaluate_ground_truth.py`](../evaluation/evaluate_ground_truth.py)

The verified evaluation outputs are:

- [`../evaluation/evaluation_metrics.csv`](../evaluation/evaluation_metrics.csv)
- [`../evaluation/evaluation_summary.json`](../evaluation/evaluation_summary.json)

---

## Segmentation Performance

The completed evaluation produced:

| Metric | Result |
|---|---:|
| True positives | 381 |
| False positives | 91 |
| False negatives / omissions | 43 |
| Precision | 0.8072 |
| Recall | 0.8986 |
| Average IoU | 0.7969 |
| Average Dice | 0.8829 |

The evaluation achieved higher recall than precision.

A recall of `0.8986` indicates that most manually annotated person instances were successfully matched by the segmentation system.

The precision of `0.8072` reflects the presence of predicted person instances that did not satisfy the evaluation matching requirement.

The average IoU of `0.7969` and average Dice coefficient of `0.8829` indicate substantial overlap between matched SAM 3 predictions and manually annotated reference masks.

---

## False Positives and Omissions

Across the complete 20-image evaluation dataset:

```text
False positives: 91
False negatives / omissions: 43
```

The evaluation therefore provides explicit measurements of both over-detection and missed ground-truth instances rather than relying only on qualitative inspection.

---

## Pixel-Level Confusion Matrix

The evaluation produced the following binary pixel-level confusion-matrix values:

| Pixel Classification | Count |
|---|---:|
| True-positive pixels | 1,604,567 |
| False-positive pixels | 341,396 |
| False-negative pixels | 229,371 |
| True-negative pixels | 20,864,666 |

These values compare the union of predicted `person` segmentation masks with the union of manually annotated `person` masks across the complete evaluation dataset.

---

## Evaluation Dataset

The ground-truth dataset contains:

```text
20 manually annotated images
424 person annotations
```

The dataset contains:

```text
10 frames from Session 001
10 frames from Session 002
```

The two sessions represent different levels of visual complexity.

Session 001 contains a relatively sparse street scene.

Session 002 contains a more crowded urban environment with:

- More visible people
- Greater variation in person scale
- Greater camera distance
- Partial visibility
- Occlusion
- Crowded regions
- Complex backgrounds

This provides evaluation evidence under varied real-world visual conditions.

---

## Ground-Truth Data

The manually annotated dataset is stored in:

[`../evaluation/ground_truth/`](../evaluation/ground_truth/)

The Roboflow COCO export is stored in:

[`../evaluation/ground_truth/roboflow_export/`](../evaluation/ground_truth/roboflow_export/)

The primary annotation file is:

[`../evaluation/ground_truth/roboflow_export/_annotations.coco.json`](../evaluation/ground_truth/roboflow_export/_annotations.coco.json)

The COCO export contains:

```text
Images: 20
Annotations: 424
Person annotations: 424
```

The exported COCO category metadata also contains an unused `object` category.

All real annotations are assigned to:

```text
person
```

No annotations are assigned to the unused `object` category.

---

## Evaluation Method

For each evaluation image:

```text
Ground-Truth Image
        ↓
Manual COCO Person Masks
        ↓
SAM 3 Person Prediction
        ↓
Predicted Segmentation Masks
        ↓
Ground Truth vs Prediction
        ↓
Instance Matching
        ↓
IoU / Dice
        ↓
Precision / Recall
        ↓
False Positives / Omissions
        ↓
Pixel Confusion Matrix
```

Predicted masks and ground-truth masks were matched using segmentation overlap.

A prediction was considered a matched instance when the mask IoU satisfied the evaluation matching threshold.

The evaluation results were generated through real SAM 3 inference using the existing Project 06 pipeline.

No metric values were manually estimated or invented.

---

## Evaluation Evidence

The evaluation stage produced:

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
        └── 20 evaluation images
```

The per-image results are stored in:

[`../evaluation/evaluation_metrics.csv`](../evaluation/evaluation_metrics.csv)

The overall evaluation summary is stored in:

[`../evaluation/evaluation_summary.json`](../evaluation/evaluation_summary.json)

---

# Processing-Time Results

The performance-analysis module supports:

- Total processing time
- Average processing time per frame
- Effective processing FPS

For the current verified run these values are:

```text
Total Processing Time: N/A
Average Processing Time per Frame: N/A
Effective Processing FPS: N/A
```

The original total processing duration was not recorded.

The project intentionally preserves these values as `N/A` rather than estimating or inventing benchmark measurements.

Future runs can provide measured processing duration to:

```text
analytics/performance_analysis.py
```

using:

```text
--processing-seconds
```

---

# Generated Analytical Reports

Project 06 currently contains the following verified analytical and evaluation outputs:

```text
reports/
│
├── README.md
│
├── tracker_summary.csv
├── trajectory_summary.csv
├── performance_summary.csv
│
├── trajectory_visualization.png
├── tracker_duration_chart.png
├── class_observation_chart.png
├── movement_distance_chart.png
├── confidence_chart.png
└── performance_chart.png

evaluation/
│
├── README.md
├── evaluate_ground_truth.py
├── evaluation_metrics.csv
├── evaluation_summary.json
│
└── ground_truth/
    ├── README.md
    └── roboflow_export/
        ├── README.md
        ├── _annotations.coco.json
        └── 20 evaluation images
```

---

# Visual Analytics

The project currently includes six analytical visualizations:

### 1. Trajectory Visualization

[`../reports/trajectory_visualization.png`](../reports/trajectory_visualization.png)

Shows reconstructed tracker movement paths.

### 2. Tracker Duration Chart

[`../reports/tracker_duration_chart.png`](../reports/tracker_duration_chart.png)

Compares tracker persistence.

### 3. Class Observation Chart

[`../reports/class_observation_chart.png`](../reports/class_observation_chart.png)

Shows observation activity by detected class.

### 4. Movement Distance Chart

[`../reports/movement_distance_chart.png`](../reports/movement_distance_chart.png)

Compares accumulated image-space movement.

### 5. Confidence Chart

[`../reports/confidence_chart.png`](../reports/confidence_chart.png)

Compares tracker-level average detection confidence.

### 6. Performance Chart

[`../reports/performance_chart.png`](../reports/performance_chart.png)

Provides a compact system-level summary of frames, observations, and tracker IDs.

---

# Persistence Results

The project successfully implemented SQLite persistence for tracking observations during the original analytics workflow.

The persistence layer enables structured storage of information such as:

- Frame number
- Tracker ID
- Object class
- Detection confidence
- Bounding-box coordinates
- Object center coordinates
- Temporal tracking information

The original temporary Colab SQLite database is not required for the current performance-analysis stage because the important aggregated information has been preserved in the project's CSV reports.

This prevents unnecessary reruns of:

```text
YOLO
ByteTrack
SAM 3
```

for later analytical tasks.

---

# H.264 Video Output

The final verified tracking video was encoded in H.264 format:

```text
sam3_tracking_output_01.mp4
```

A second verified H.264 output was also generated for Session 002:

```text
sam3_tracking_output_02.mp4
```

Using H.264 improves compatibility with:

- GitHub
- Web browsers
- Standard video players
- Documentation platforms
- Presentation software

These outputs provide direct visual evidence of the integrated detection, tracking, and segmentation pipeline.

---

# Results Architecture

The completed results workflow can be summarized as:

```text
Video
  ↓
YOLO Detection
  ↓
ByteTrack Tracking
  ↓
SAM 3 Segmentation
  ↓
H.264 Tracking Video
  ↓
Tracking Observations
  ↓
Persistence
  ↓
Tracker Summary
  ↓
Trajectory Summary
  ↓
Performance Summary
  ↓
Visual Analytics
  ↓
Historical Session Comparison
  ↓
Manual Ground Truth
  ↓
SAM 3 Evaluation
  ↓
IoU / Dice
  ↓
Precision / Recall
  ↓
False Positives / Omissions
  ↓
Confusion Matrix
  ↓
Documented Results
```

---

# What the Results Demonstrate

The verified outputs demonstrate that Project 06 can:

- Process recorded video
- Detect multiple object classes
- Maintain persistent tracker IDs
- Integrate SAM 3 segmentation into video processing
- Generate H.264-compatible annotated output
- Record structured tracking observations
- Summarize tracker behavior
- Measure tracker duration
- Analyze detection confidence
- Reconstruct object trajectories
- Measure image-space movement
- Generate reusable CSV reports
- Produce analytical visualizations
- Generate system-level performance reports
- Reuse preserved analytics without rerunning previous inference
- Preserve multiple verified processing sessions
- Compare historical session results
- Evaluate SAM 3 predictions against manually annotated ground truth
- Measure false positives
- Measure false negatives and omissions
- Calculate Precision
- Calculate Recall
- Calculate IoU
- Calculate Dice coefficient
- Produce pixel-level confusion-matrix values
- Export reproducible evaluation metrics

---

# Current Verified Project State

The current verified Project 06 state is:

```text
Image Pipeline                     COMPLETE
YOLO Detection                     COMPLETE
ByteTrack Video Tracking           COMPLETE
SAM 3 Video Integration            COMPLETE
H.264 Video Output                 COMPLETE
Video Analytics                    COMPLETE
SQLite Persistence                 COMPLETE
Tracker Summary                    COMPLETE
Trajectory Analysis                COMPLETE
Trajectory Visualization           COMPLETE
Tracker Duration Chart             COMPLETE
Class Observation Chart            COMPLETE
Movement Distance Chart            COMPLETE
Confidence Chart                   COMPLETE
Performance Analysis               COMPLETE
Performance Summary                COMPLETE
Performance Chart                  COMPLETE
Historical Session Comparison      COMPLETE
Ground-Truth Dataset               COMPLETE
Manual Person Annotation           COMPLETE
COCO Segmentation Export           COMPLETE
SAM 3 Ground-Truth Evaluation      COMPLETE
Precision Evaluation               COMPLETE
Recall Evaluation                  COMPLETE
IoU Evaluation                     COMPLETE
Dice Evaluation                    COMPLETE
False Positive Evaluation          COMPLETE
Omission Evaluation                COMPLETE
Confusion Matrix                   COMPLETE
Limitations Documentation          COMPLETE
Results Documentation              COMPLETE
```

---

# Related Documentation

- [Project README](../README.md)
- [Project Proposal](./PROJECT-PROPOSAL.md)
- [Reports Documentation](../reports/README.md)
- [Analytics Documentation](../analytics/README.md)
- [Evaluation Documentation](../evaluation/README.md)
- [Ground-Truth Documentation](../evaluation/ground_truth/README.md)
- [System Limitations](./LIMITATIONS.md)
- [Tracker Summary](../reports/tracker_summary.csv)
- [Trajectory Summary](../reports/trajectory_summary.csv)
- [Performance Summary](../reports/performance_summary.csv)
- [Evaluation Metrics](../evaluation/evaluation_metrics.csv)
- [Evaluation Summary](../evaluation/evaluation_summary.json)
- [COCO Ground Truth](../evaluation/ground_truth/roboflow_export/_annotations.coco.json)

---

# Author

**Peyman Miyandashti**

GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)

LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)
