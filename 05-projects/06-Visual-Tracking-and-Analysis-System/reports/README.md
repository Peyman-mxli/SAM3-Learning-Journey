# Reports

This directory contains analytics reports, evaluation results, and documented findings produced during the development of the **Visual Tracking and Analysis System**.

The purpose of this directory is to preserve structured evidence about:

- Object detection
- Multi-object tracking
- Tracker-ID behavior
- Object appearance duration
- Detection confidence
- Class observations
- Trajectory analysis
- Image-space movement
- Segmentation
- Processing performance
- System limitations
- Failure cases

The project has progressed beyond visual output generation and now includes structured temporal analytics, SQLite persistence, tracker summaries, and trajectory reports generated from recorded-video processing.

---

# Current Reports

```text
reports/
│
├── README.md
├── tracker_summary.csv
└── trajectory_summary.csv
```

The current validated analytics reports are:

- [`tracker_summary.csv`](./tracker_summary.csv)
- [`trajectory_summary.csv`](./trajectory_summary.csv)

---

# 1. Tracker Summary Report

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

# Tracker CSV Structure

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

Example:

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

# 2. Trajectory and Movement Report

## `trajectory_summary.csv`

The second analytics report adds trajectory and movement analysis.

[`trajectory_summary.csv`](./trajectory_summary.csv)

This report uses the center point of each tracked bounding box to estimate how each tracker moves through image space.

The processing concept is:

```text
Tracked Bounding Box
        |
        v
Center-Point Calculation
        |
        v
Ordered Trajectory Points
        |
        v
Euclidean Distance
        |
        v
Accumulated Movement
        |
        v
trajectory_summary.csv
```

---

# Bounding-Box Center Point

For every tracked detection, the center point is calculated from:

```text
(x1, y1) ----------------------+
    |                          |
    |             X            |
    |       Center Point       |
    |                          |
    +---------------------- (x2, y2)
```

The center coordinates are:

```text
center_x = (x1 + x2) / 2

center_y = (y1 + y2) / 2
```

These values are stored for every frame in which the tracker is observed.

---

# Trajectory Representation

A tracker trajectory is represented as an ordered sequence of center points:

```text
Frame 1
(x1, y1)
    |
    v
Frame 2
(x2, y2)
    |
    v
Frame 3
(x3, y3)
    |
    v
...
```

These points describe the apparent motion of the tracked object through the image.

---

# Movement Distance Calculation

Movement between consecutive center points is calculated using Euclidean distance:

```text
distance = sqrt(
    (x2 - x1)^2
    +
    (y2 - y1)^2
)
```

Total movement is the sum of all consecutive distances:

```text
Total Movement
=
Distance Point 1 → Point 2
+
Distance Point 2 → Point 3
+
Distance Point 3 → Point 4
+
...
```

The result is expressed in:

```text
pixels
```

---

# Verified Trajectory Results

The complete 75-frame trajectory test produced:

```text
Frames processed: 75

Tracker #1 | Duration: 5.00s | Distance: 159.26px | Average movement: 2.15px

Tracker #2 | Duration: 5.00s | Distance: 203.37px | Average movement: 2.75px

Tracker #3 | Duration: 3.93s | Distance: 182.36px | Average movement: 3.14px

Tracker #4 | Duration: 1.67s | Distance: 139.34px | Average movement: 5.81px

Tracker #5 | Duration: 0.20s | Distance: 7.91px | Average movement: 3.96px

Tracker #6 | Duration: 0.60s | Distance: 0.94px | Average movement: 0.12px

TRAJECTORY ANALYTICS: SUCCESS
```

---

# Trajectory Summary

| Tracker ID | Duration | Movement Distance | Average Movement |
|---:|---:|---:|---:|
| 1 | 5.00 s | 159.26 px | 2.15 px |
| 2 | 5.00 s | 203.37 px | 2.75 px |
| 3 | 3.93 s | 182.36 px | 3.14 px |
| 4 | 1.67 s | 139.34 px | 5.81 px |
| 5 | 0.20 s | 7.91 px | 3.96 px |
| 6 | 0.60 s | 0.94 px | 0.12 px |

---

# Movement Interpretation

Tracker `#2` produced the largest accumulated movement:

```text
203.37 px
```

Tracker `#4` produced the largest average movement between consecutive observations:

```text
5.81 px
```

Tracker `#6` produced the smallest measured trajectory:

```text
0.94 px
```

with an average movement of:

```text
0.12 px
```

These results show that the analytics layer can distinguish different apparent movement patterns across tracker IDs.

---

# Important Measurement Limitation

Movement is measured in:

```text
image-space pixels
```

It does **not** represent:

```text
meters
kilometers
feet
real-world speed
real-world physical distance
```

The current system measures the motion of bounding-box centers in the two-dimensional image coordinate system.

For example:

```text
Tracker #2
Movement distance: 203.37 px
```

means that the center of Tracker #2's bounding box traveled approximately 203.37 pixels through the image.

---

# Why Pixel Distance Is Not Real-World Distance

Converting image-space movement into physical movement requires additional information such as:

- camera calibration
- camera height
- camera angle
- focal length
- scene geometry
- known reference measurements
- perspective transformation
- depth information

Without those values, pixel displacement should be interpreted only as:

```text
Image-Space Movement
```

---

# Effects on Movement Measurements

Movement values can be affected by:

- actual object movement
- controlled camera motion
- bounding-box size changes
- detector localization variation
- confidence changes
- partial occlusion
- tracker association changes
- perspective changes
- objects entering or leaving the frame

Therefore, trajectory distance is useful for comparative analysis but should not be described as precise physical motion.

---

# Trajectory CSV Structure

The generated:

[`trajectory_summary.csv`](./trajectory_summary.csv)

contains:

```text
tracker_id
first_frame
last_frame
frames_observed
duration_seconds
movement_distance_pixels
average_movement_pixels
```

Example:

```text
tracker_id,first_frame,last_frame,frames_observed,duration_seconds,movement_distance_pixels,average_movement_pixels
1,1,75,75,5.0,159.26,2.15
2,1,75,75,5.0,203.37,2.75
3,3,75,59,3.93,182.36,3.14
4,8,32,25,1.67,139.34,5.81
5,29,31,3,0.2,7.91,3.96
6,53,61,9,0.6,0.94,0.12
```

---

# Analytics Architecture

The analytics layer now supports both presence and movement analysis.

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
      +--------------------------+
      |                          |
      v                          v
Temporal Presence          Trajectory Points
      |                          |
      v                          v
Tracker Lifetime          Movement Distance
      |                          |
      v                          v
Class Counts              Average Movement
      |                          |
      +-------------+------------+
                    |
                    v
              Structured Data
                    |
          +---------+---------+
          |                   |
          v                   v
       SQLite             CSV Reports
                              |
                   +----------+----------+
                   |                     |
                   v                     v
          tracker_summary.csv   trajectory_summary.csv
```

---

# VideoAnalytics Module

Temporal and movement analytics are implemented in:

[`../src/analytics.py`](../src/analytics.py)

Each stored observation can now contain:

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
center_x
center_y
```

This provides the information required for both object-lifetime and trajectory analysis.

---

# Current Analytics Capabilities

The current analytics module supports:

- total observation count
- unique tracker-ID extraction
- unique tracker count
- class observation counts
- first observed frame
- last observed frame
- frames observed
- tracker duration
- center-point extraction
- tracker trajectories
- total image-space movement
- average movement between observations
- tracker summaries
- complete analytics summaries

---

# Database Integration

Structured observations are stored using:

[`../src/database.py`](../src/database.py)

SQLite provides a lightweight persistence layer for the project.

The validated database workflow stored:

```text
1 processing session
246 object observations
```

The database can then be queried independently from the computer vision models.

This is important because expensive inference does not need to be repeated every time analytics are calculated.

---

# Why Structured Storage Matters

Without structured storage, the primary result is:

```text
Annotated Video
```

With structured observations and reports, the project can answer questions such as:

```text
How many tracker IDs were created?

How long was each tracker visible?

What class was associated with each tracker?

What was the average confidence?

Which frames contained a tracker?

How many observations belonged to each class?

How far did each tracker move in image space?

Which tracker had the largest accumulated trajectory?

Which tracker had the highest average movement?
```

This transforms the project from a visualization pipeline into a more complete visual analytics system.

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

An observation represents one tracked detection on one frame.

A tracker ID represents ByteTrack's attempt to maintain an object's identity across frames.

Additionally:

```text
Tracker ID Count
        !=
Guaranteed Physical Object Count
```

because a tracking system may create a new ID after losing and reacquiring the same physical object.

---

# Average Confidence

The tracker report calculates average YOLO confidence for each identity.

| Tracker | Class | Average Confidence |
|---:|---|---:|
| 1 | person | 0.8385 |
| 2 | person | 0.8587 |
| 3 | bus | 0.5939 |
| 4 | person | 0.7579 |
| 5 | person | 0.5278 |
| 6 | person | 0.5124 |

Trackers `#5` and `#6` had relatively low average confidence and short observed lifetimes.

This does not prove causation, but it provides useful evidence for future tracker stability analysis.

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
  +----------------------------+
  |                            |
  v                            v
Temporal Presence         Trajectory Analysis
  |                            |
  v                            v
Duration                  Pixel Movement
  |                            |
  v                            v
Confidence                Average Movement
  |                            |
  +-------------+--------------+
                |
                v
             SQLite
                |
                v
            CSV Reports
                |
        +-------+-------+
        |               |
        v               v
tracker_summary.csv  trajectory_summary.csv
        |
        v
SUCCESS
```

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
- image-space trajectories
- identity switches
- movement trends

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
- Movement distance
- Average movement
- Tracking consistency
- Identity stability
- Processing time

Not every metric applies to every experiment.

---

# Failure Analysis

Failure analysis is an important part of the project.

Examples include:

- missed detections
- incorrect classifications
- false detections
- tracker-ID changes
- lost tracks
- tracker reinitialization
- segmentation errors
- partial occlusion
- low-confidence detections
- small-object detection problems
- motion blur
- unusual camera perspectives
- artificial movement caused by camera motion
- noisy center-point trajectories

Failures should be documented rather than removed from the evaluation.

---

# Future Reports

Future reports may include:

```text
object_statistics.csv
class_summary.csv
frame_observations.csv
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
movement_distance_chart.png
segmentation_evaluation.png
```

---

# Next Analytics Milestones

The next analytics phase may introduce:

- full frame-observation CSV export
- trajectory visualization
- movement-distance charts
- tracker-duration charts
- class-distribution charts
- confidence trends
- movement speed estimates
- tracker-ID consistency analysis
- identity-switch analysis
- SQLite query utilities
- automated report generation
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

Part of the:

[SAM3 Learning Journey](../../../README.md)

---

## Author

**Peyman Miyandashti**

Computer Vision, Artificial Intelligence, and Software Development

[GitHub](https://github.com/Peyman-mxli)

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
