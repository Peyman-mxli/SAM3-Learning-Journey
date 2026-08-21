# Reports

This directory contains analytics reports, evaluation results, and documented findings produced during the development of the **Visual Tracking and Analysis System**.

The purpose of this directory is to preserve structured evidence about:

- Object detection
- Multi-object tracking
- Tracker-ID behavior
- Object appearance duration
- Detection confidence
- Class observations
- Segmentation
- Processing performance
- System limitations
- Failure cases

The project has now progressed beyond visual output generation and includes structured temporal analytics generated from recorded-video processing.

---

# Current Reports

```text
reports/
│
├── README.md
└── tracker_summary.csv
```

The first generated analytics report is:

[`tracker_summary.csv`](./tracker_summary.csv)

---

# Tracker Summary Report

## `tracker_summary.csv`

The tracker summary report was generated from the recorded-video test:

[`../assets/input/tracking_test_01.mp4`](../assets/input/tracking_test_01.mp4)

The video was processed using:

```text
Recorded Video
      |
      v
YOLO Object Detection
      |
      v
Supervision Detections
      |
      v
ByteTrack
      |
      v
VideoAnalytics
      |
      v
Structured Observations
      |
      v
SQLite
      |
      v
SQL Analytics Query
      |
      v
tracker_summary.csv
```

---

# Video Analytics Test

The analytics test processed the complete 75-frame video.

Verified result:

```text
Frames processed: 75
Total observations: 246
Unique tracker IDs: [1, 2, 3, 4, 5, 6]
Unique object count: 6

Class observations:
person: 187
car: 31
bus: 28

VIDEO ANALYTICS TEST: SUCCESS
```

---

# Observation Distribution

The analytics module recorded:

```text
Total observations: 246
```

These observations were distributed across the detected classes as follows:

| Class | Observations |
|---|---:|
| person | 187 |
| car | 31 |
| bus | 28 |
| **Total** | **246** |

An observation represents one tracked detection recorded for one video frame.

For example:

```text
Tracker #1 detected on Frame 1
```

represents one observation.

If the same tracker remains visible on the next frame, another observation is recorded.

This creates a temporal record of object presence throughout the video.

---

# Tracker Duration Analysis

The analytics module calculated how many frames each tracker ID remained observable.

Verified results:

| Tracker ID | First Frame | Last Frame | Frames Observed | Duration |
|---:|---:|---:|---:|---:|
| 1 | 1 | 75 | 75 | 5.00 s |
| 2 | 1 | 75 | 75 | 5.00 s |
| 3 | 3 | 75 | 59 | 3.93 s |
| 4 | 8 | 32 | 25 | 1.67 s |
| 5 | 29 | 31 | 3 | 0.20 s |
| 6 | 53 | 61 | 9 | 0.60 s |

The calculation is based on:

```text
Duration = Frames Observed / Video FPS
```

The test video uses:

```text
15 FPS
```

For example:

```text
Tracker #1

75 observed frames / 15 FPS
        =
5.00 seconds
```

---

# Tracker Stability

Trackers `#1` and `#2` remained present during all 75 frames:

```text
Tracker #1
Frames: 1 → 75
Duration: 5.00 seconds

Tracker #2
Frames: 1 → 75
Duration: 5.00 seconds
```

This demonstrates stable tracker identity across the complete test sequence for those objects.

Tracker `#3` was observed for:

```text
59 frames
3.93 seconds
```

The remaining tracker IDs had shorter lifetimes:

```text
Tracker #4 → 1.67 seconds
Tracker #5 → 0.20 seconds
Tracker #6 → 0.60 seconds
```

These shorter tracks provide useful evidence for future tracker-ID consistency and failure analysis.

---

# SQLite Tracker Query

The 246 observations were stored successfully in SQLite.

Verified database result:

```text
Session created: 1
Sessions stored: 1
Observations stored: 246

person: 187
car: 31
bus: 28

SQLITE VIDEO ANALYTICS: SUCCESS
```

A SQL query was then used to aggregate observations by tracker ID.

The query calculated:

- tracker ID
- class name
- first observed frame
- last observed frame
- number of observations
- estimated appearance duration
- average YOLO confidence

---

# Per-Tracker Analytics

The verified SQL results were:

```text
Tracker #1 | Class: person | Frames: 1-75 | Observations: 75 | Duration: 5.00s | Avg confidence: 0.8385

Tracker #2 | Class: person | Frames: 1-75 | Observations: 75 | Duration: 5.00s | Avg confidence: 0.8587

Tracker #3 | Class: bus | Frames: 3-75 | Observations: 59 | Duration: 3.93s | Avg confidence: 0.5939

Tracker #4 | Class: person | Frames: 8-32 | Observations: 25 | Duration: 1.67s | Avg confidence: 0.7579

Tracker #5 | Class: person | Frames: 29-31 | Observations: 3 | Duration: 0.20s | Avg confidence: 0.5278

Tracker #6 | Class: person | Frames: 53-61 | Observations: 9 | Duration: 0.60s | Avg confidence: 0.5124
```

---

# CSV Structure

The generated:

[`tracker_summary.csv`](./tracker_summary.csv)

contains the following columns:

```text
tracker_id
class_name
first_frame
last_frame
observations
duration_seconds
average_confidence
```

The validated CSV contains six tracker records.

Example structure:

```text
tracker_id,class_name,first_frame,last_frame,observations,duration_seconds,average_confidence
1,person,1,75,75,5.0,0.8385
2,person,1,75,75,5.0,0.8587
3,bus,3,75,59,3.93,0.5939
4,person,8,32,25,1.67,0.7579
5,person,29,31,3,0.2,0.5278
6,person,53,61,9,0.6,0.5124
```

---

# Analytics Architecture

The analytics layer introduces a new stage to the project:

```text
Computer Vision
      |
      v
YOLO Detection
      |
      v
ByteTrack
      |
      v
Tracked Detections
      |
      v
VideoAnalytics
      |
      +----------------------+
      |                      |
      v                      v
Frame Observations      Tracker Statistics
      |                      |
      v                      v
SQLite Database         Analytics Summary
      |                      |
      +----------+-----------+
                 |
                 v
             SQL Query
                 |
                 v
         tracker_summary.csv
```

---

# VideoAnalytics Module

Temporal analytics are implemented in:

[`../src/analytics.py`](../src/analytics.py)

The module collects frame-by-frame tracking information.

Each observation can contain:

```text
frame_number
timestamp_seconds
tracker_id
class_id
class_name
confidence
x1
y1
x2
y2
```

This creates a structured representation of the tracking results.

---

# Analytics Capabilities

The current analytics module supports:

- total observation count
- unique tracker-ID extraction
- unique tracker count
- per-class observation counts
- first observed frame
- last observed frame
- frames observed
- tracker appearance duration
- tracker summaries
- complete analytics summaries

---

# Database Integration

Structured observations are stored using:

[`../src/database.py`](../src/database.py)

SQLite provides a lightweight persistence layer for the project.

The validated workflow stored:

```text
1 processing session
246 object observations
```

The database can then be queried independently from the computer vision models.

This is important because expensive inference does not need to be repeated every time analytics are calculated.

---

# Why Structured Storage Matters

Without structured storage, the final output is primarily visual:

```text
Annotated Video
```

With structured storage, the project can answer questions such as:

```text
How many tracker IDs were created?

How long was each tracker visible?

What class was associated with each tracker?

What was the average detection confidence?

Which frames contained a specific tracker?

How many observations belonged to each class?
```

This transforms the project from a visualization pipeline into a basic visual analytics system.

---

# Detection Count vs. Object Count

An important distinction is:

```text
Observation Count
        !=
Physical Object Count
```

The project recorded:

```text
246 observations
```

but only:

```text
6 tracker IDs
```

An observation represents a detection associated with a tracker on one frame.

A tracker ID represents ByteTrack's attempt to maintain an object's identity across multiple frames.

Additionally:

```text
Tracker ID Count
        !=
Guaranteed Physical Object Count
```

because tracking systems can create new IDs after losing and reacquiring an object.

---

# Average Confidence

The report also calculates average YOLO detection confidence for each tracker.

Results:

| Tracker | Class | Average Confidence |
|---:|---|---:|
| 1 | person | 0.8385 |
| 2 | person | 0.8587 |
| 3 | bus | 0.5939 |
| 4 | person | 0.7579 |
| 5 | person | 0.5278 |
| 6 | person | 0.5124 |

Trackers `#5` and `#6` had the lowest average confidence and also relatively short lifetimes.

This does not by itself prove why those tracks were short, but it provides useful evidence for future tracking analysis.

---

# Evaluation Areas

The project can continue evaluating:

- object detection quality
- segmentation quality
- tracking consistency
- tracker-ID stability
- false positives
- false negatives
- processing performance
- lighting sensitivity
- occlusion handling
- object scale
- motion blur
- confidence stability
- tracker lifetime
- object trajectories
- identity switches

---

# Metrics

Depending on the experiment, reports may include:

- Precision
- Recall
- Intersection over Union
- Dice coefficient
- Average confidence
- Frames observed
- Appearance duration
- Tracker lifetime
- Observation count
- Class observation count
- Tracking consistency
- Identity stability
- Processing time

Not every metric applies to every experiment.

---

# Failure Analysis

Failure analysis is an important part of this project.

Examples include:

- missed detections
- incorrect classifications
- false detections
- tracker-ID changes
- lost tracks
- tracker reinitialization
- segmentation errors
- partial occlusion problems
- low-confidence detections
- small-object detection problems
- motion blur
- unusual camera perspectives

Failures should be documented rather than removed from the evaluation.

---

# Current Verified Analytics

```text
VIDEO
  |
  v
75 Frames
  |
  v
YOLO + ByteTrack
  |
  v
246 Structured Observations
  |
  v
6 Tracker IDs
  |
  +---------------------+
  |                     |
  v                     v
187 person           31 car
observations         observations
  |
  v
28 bus observations
  |
  v
SQLite Persistence
  |
  v
SQL Aggregation
  |
  v
tracker_summary.csv
  |
  v
SUCCESS
```

---

# Future Reports

Future reports may include:

```text
object_statistics.csv
class_summary.csv
frame_observations.csv
trajectory_report.csv
tracking_evaluation.csv
segmentation_evaluation.csv
performance_report.csv
failure_analysis.md
```

Possible visual reports include:

```text
trajectory_visualization.png
tracker_duration_chart.png
class_observation_chart.png
confidence_chart.png
segmentation_evaluation.png
```

---

# Next Analytics Milestones

The next analytics phase may introduce:

- complete frame-observation CSV export
- trajectory calculations
- object-center coordinates
- movement distance
- movement speed estimates
- per-class summaries
- confidence trends
- tracker-ID consistency analysis
- identity-switch analysis
- SQLite query utilities
- automated report generation
- visualization charts
- longer-video evaluation

---

# Important Note

Generated reports should contain reproducible analysis results and should not contain:

- authentication tokens
- Hugging Face credentials
- GitHub credentials
- private information
- model checkpoints

Large temporary files should not be committed unless they provide meaningful project evidence.

---

# Project

This directory belongs to:

[Visual Tracking and Analysis System](../README.md)

Part of the [SAM3 Learning Journey](../../../README.md).

---

## Author

**Peyman Miyandashti**

Computer Vision, Artificial Intelligence, and Software Development

[GitHub](https://github.com/Peyman-mxli)

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
