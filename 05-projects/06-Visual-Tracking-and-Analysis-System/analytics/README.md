# Analytics

This directory contains the analytics modules used by the **Visual Tracking and Analysis System** to transform raw object-tracking observations into structured measurements, summaries, and performance reports.

The analytics layer extends the project beyond visual detection and tracking by providing quantitative information about object behavior and system performance.

---

## Purpose

The purpose of this directory is to separate analytical processing from the main computer-vision inference pipeline.

The tracking pipeline generates observations from video frames, while the analytics modules process those observations to produce meaningful statistics and reports.

The general workflow is:

```text
Video Input
    ↓
YOLO Detection
    ↓
ByteTrack Tracking
    ↓
SAM 3 Segmentation
    ↓
Tracking Observations
    ↓
SQLite Persistence
    ↓
Analytics Modules
    ↓
CSV Reports
    ↓
Visualizations
```

---

## Current Analytics Capabilities

The project currently supports analysis of:

- Tracker IDs
- Object classes
- Detection confidence
- Object appearance duration
- First and last observed frames
- Number of observations per tracker
- Object center coordinates
- Object trajectories
- Image-space movement distance
- Tracker persistence
- Class observation frequency
- Detection confidence behavior

These analytics are generated from structured observations collected during video processing.

---

## Verified Dataset

The current verified video-processing run contains:

| Metric | Result |
|---|---:|
| Processed frames | 75 |
| Recorded observations | 246 |
| Unique tracker IDs | 6 |

This dataset is used as the current reference for the project's analytics and visualization reports.

---

## Existing Analytics Outputs

The analytics pipeline currently produces the following reports:

```text
reports/
│
├── tracker_summary.csv
├── trajectory_summary.csv
├── trajectory_visualization.png
├── tracker_duration_chart.png
├── class_observation_chart.png
├── movement_distance_chart.png
└── confidence_chart.png
```

---

## Tracker Analytics

Tracker analytics convert frame-level observations into object-level summaries.

For each tracker ID, the system can determine:

- Object class
- First observed frame
- Last observed frame
- Number of observations
- Tracking duration
- Average confidence
- Temporal persistence

The resulting tracker-level information is stored in:

[`../reports/tracker_summary.csv`](../reports/tracker_summary.csv)

---

## Trajectory Analytics

Trajectory analytics measure how tracked objects move through image space.

For a bounding box:

```text
(x1, y1, x2, y2)
```

the center point is calculated as:

```text
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
```

Movement between consecutive center points is calculated using Euclidean distance:

```text
distance = sqrt(
    (center_x2 - center_x1)^2 +
    (center_y2 - center_y1)^2
)
```

The distances between consecutive observations are accumulated to estimate the total movement of each tracked object.

Because these calculations operate in image coordinates, movement distance is measured in **pixels** rather than physical-world units.

The resulting trajectory summary is stored in:

[`../reports/trajectory_summary.csv`](../reports/trajectory_summary.csv)

---

## Analytics Visualizations

The project currently includes several visual analytics reports.

### Trajectory Visualization

[`../reports/trajectory_visualization.png`](../reports/trajectory_visualization.png)

Displays reconstructed movement paths for tracked objects.

### Tracker Duration Chart

[`../reports/tracker_duration_chart.png`](../reports/tracker_duration_chart.png)

Compares how long individual tracker IDs remained active.

### Class Observation Chart

[`../reports/class_observation_chart.png`](../reports/class_observation_chart.png)

Shows the number of recorded observations associated with each detected object class.

### Movement Distance Chart

[`../reports/movement_distance_chart.png`](../reports/movement_distance_chart.png)

Compares accumulated image-space movement between tracker IDs.

### Confidence Chart

[`../reports/confidence_chart.png`](../reports/confidence_chart.png)

Visualizes detection-confidence behavior across tracked objects.

---

## SQLite Integration

Tracking observations are persisted in SQLite so analytics can be performed independently from the original video-processing loop.

Persisted observations can include:

- Frame number
- Tracker ID
- Class ID
- Class name
- Confidence
- Bounding-box coordinates
- Center coordinates
- Temporal tracking information

This architecture allows the analytics layer to query previously recorded tracking data without running object detection and segmentation again.

This is especially important because expensive computer-vision inference does not need to be repeated simply to generate a new report.

---

## Separation of Responsibilities

The project separates computer-vision inference from analytics.

### Inference Layer

Responsible for:

- Reading video frames
- Running YOLO detection
- Assigning ByteTrack tracker IDs
- Running SAM 3 segmentation
- Rendering visual annotations
- Recording observations

### Analytics Layer

Responsible for:

- Reading persisted observations
- Aggregating tracker statistics
- Calculating trajectories
- Measuring movement
- Analyzing confidence
- Generating CSV summaries
- Creating analytical visualizations
- Evaluating system performance

This separation makes the project easier to maintain and extend.

---

## Performance Analysis

The next analytics milestone introduces system-level performance evaluation.

The performance analysis module will evaluate metrics such as:

- Total processed frames
- Total observations
- Unique tracker IDs
- Average observations per frame
- Average confidence
- Tracker persistence
- Processing duration
- Average processing time per frame
- Effective processing FPS

The planned module is:

```text
analytics/performance_analysis.py
```

Planned outputs include:

```text
reports/performance_summary.csv
reports/performance_chart.png
```

These reports will extend the existing object-level analytics with system-level performance measurements.

---

## Planned Directory Structure

```text
analytics/
│
├── README.md
└── performance_analysis.py
```

As the project evolves, additional analytics modules may be added for specialized evaluation tasks.

---

## Future Analytics Extensions

Possible future extensions include:

- Per-frame object counts
- Region-of-interest analytics
- Zone occupancy
- Entry and exit events
- Dwell-time analysis
- Direction analysis
- Speed estimation
- Tracker-loss detection
- ID-switch analysis
- Segmentation-area statistics
- Processing-time analysis
- FPS benchmarking
- Detection failure analysis
- Tracking failure analysis

These extensions can build on the same persisted tracking observations already implemented in the project.

---

## Design Goal

The analytics layer is designed to transform the project from a simple computer-vision demonstration into a measurable **Visual Tracking and Analysis System**.

The complete architecture follows the principle:

```text
Detect
  ↓
Track
  ↓
Segment
  ↓
Persist
  ↓
Analyze
  ↓
Visualize
  ↓
Evaluate
```

Each stage produces information that can be inspected, measured, and documented independently.

---

## Related Documentation

- [Project README](../README.md)
- [Reports Documentation](../reports/README.md)
- [Tracker Summary](../reports/tracker_summary.csv)
- [Trajectory Summary](../reports/trajectory_summary.csv)

---

## Author

**Peyman Miyandashti**

GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)

LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)
