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

Project 06 currently contains the following verified reports:

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

Using H.264 improves compatibility with:

- GitHub
- Web browsers
- Standard video players
- Documentation platforms
- Presentation software

This output provides direct visual evidence of the integrated detection, tracking, and segmentation pipeline.

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
- Reuse preserved analytics without rerunning inference

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
Limitations Documentation          COMPLETE
Results Documentation              COMPLETE
```

---

# Related Documentation

- [Project README](../README.md)
- [Reports Documentation](../reports/README.md)
- [Analytics Documentation](../analytics/README.md)
- [System Limitations](./LIMITATIONS.md)
- [Tracker Summary](../reports/tracker_summary.csv)
- [Trajectory Summary](../reports/trajectory_summary.csv)
- [Performance Summary](../reports/performance_summary.csv)

---

# Author

**Peyman Miyandashti**

GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)

LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)
