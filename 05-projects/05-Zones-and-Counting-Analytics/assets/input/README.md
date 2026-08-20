# Input Assets — Zones and Counting Analytics

This directory contains the source media used by the **Zones and Counting Analytics** project.

The input files are processed by the computer vision pipeline and should remain unchanged.

---

## Directory Structure

```text
input/
├── README.md
└── vehicles.mp4
```

---

## Main Input Video

The project uses:

```text
vehicles.mp4
```

The complete project-relative path is:

```text
assets/input/vehicles.mp4
```

This video contains highway traffic and provides a useful scenario for testing:

- YOLOv8 object detection
- ByteTrack object tracking
- Persistent tracker IDs
- Polygon occupancy
- Virtual line crossings
- Directional counting
- Spatial video analytics

---

## Input Path in Python

The project uses:

```python
INPUT_VIDEO = "assets/input/vehicles.mp4"
```

This path is referenced by the main project script:

```text
zones_counting_analytics.py
```

---

## Video Information

The original traffic video used during development had:

```text
Resolution: 3840 × 2160
FPS: 25
Total Frames: 538
```

A compressed repository-friendly version may be used for GitHub while preserving the same practical workflow.

---

## Reading Video Information

Supervision can inspect the video using:

```python
video_info = sv.VideoInfo.from_video_path(
    INPUT_VIDEO
)
```

Important properties include:

```python
video_info.width
video_info.height
video_info.fps
video_info.total_frames
```

These values are especially important because the PolygonZone and LineZone coordinates depend on the frame dimensions.

---

## Polygon Coordinates

The project defines a polygon relative to the video resolution.

Example:

```python
POLYGON_LEFT = np.array([
    [0,                     video_info.height // 2],
    [video_info.width // 2, video_info.height // 2],
    [video_info.width // 2, video_info.height],
    [0,                     video_info.height],
])
```

This creates a region covering approximately the lower-left section of the frame.

---

## Line Coordinates

The project also defines a horizontal virtual line across the center of the video:

```python
line_start = sv.Point(
    x=0,
    y=video_info.height // 2
)

line_end = sv.Point(
    x=video_info.width,
    y=video_info.height // 2
)
```

This allows the system to count tracked objects crossing the center boundary.

---

## Input Workflow

```text
vehicles.mp4
     ↓
Read Video Information
     ↓
Read Frame
     ↓
YOLOv8
     ↓
sv.Detections
     ↓
ByteTrack
     ↓
Tracked Objects
     ↓
PolygonZone + LineZone
     ↓
Spatial Analytics
     ↓
Annotated Output
```

---

## Do Not Overwrite the Input

The source video should remain unchanged.

The project should:

```text
READ FROM:
assets/input/
```

and:

```text
WRITE TO:
assets/output/
```

This keeps original and generated media clearly separated.

---

## Using Other Videos

The project can be adapted to other videos such as:

```text
traffic_intersection.mp4
parking_lot.mp4
store_entrance.mp4
warehouse.mp4
pedestrian_crossing.mp4
```

When replacing the input video, the following may need to be adjusted:

- Polygon coordinates
- Line coordinates
- Detection classes
- Tracking behavior
- Annotation scale
- Output filename

---

## Reproducibility

The expected input path is fixed:

```text
assets/input/vehicles.mp4
```

This allows the project to be rerun without modifying path logic.

The complete flow is:

```text
Input Video
     ↓
zones_counting_analytics.py
     ↓
YOLOv8 + ByteTrack
     ↓
PolygonZone + LineZone
     ↓
Output Video
```

---

## Related Documentation

- [Assets README](../README.md)
- [Project README](../../README.md)
- [Main Python Script](../../zones_counting_analytics.py)

---

## Author

**Peyman Miyandashti**

SAM3 Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
