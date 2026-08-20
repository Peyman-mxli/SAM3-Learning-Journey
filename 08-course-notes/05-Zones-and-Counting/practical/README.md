# Zones and Counting — Practical

This directory contains the practical implementation for **Session 06 — Zones and Counting** of my SAM3 Computer Vision Learning Journey.

The practical combines concepts from previous sessions with the new spatial analytics concepts introduced in this lesson.

The complete pipeline uses:

- YOLO for object detection
- Supervision for detection structures and visualization
- ByteTrack for object tracking
- `PolygonZone` for current occupancy
- `LineZone` for accumulated crossing counts
- OpenCV for video processing
- NumPy for polygon coordinates

The objective is to transform a normal video into a **tracked spatial analytics video**.

---

# Practical Objective

The goal of this practical is to build a video-processing pipeline capable of answering two questions simultaneously:

> How many objects are currently inside a specific region?

and:

> How many objects have crossed a specific boundary?

These correspond to:

```text
PolygonZone → Current Occupancy
LineZone    → Accumulated Flow
```

---

# Practical Architecture

The complete system follows this pipeline:

```text
Input Video
    ↓
Read Frame
    ↓
YOLO Detection
    ↓
sv.Detections
    ↓
ByteTrack
    ↓
Tracked Detections
    │
    ├───────────────┐
    ↓               ↓
PolygonZone      LineZone
    ↓               ↓
Occupancy        Crossings
    │               │
    └───────┬───────┘
            ↓
        Annotation
            ↓
       Output Video
```

---

# Directory Structure

The practical uses the following structure:

```text
practical/
│
├── README.md
│
├── zones_and_counting_practical.py
│
└── assets/
    ├── README.md
    │
    ├── input/
    │   └── vehicles.mp4
    │
    └── output/
        ├── vehicles_polygon_zone.mp4
        ├── vehicles_line_zone.mp4
        └── vehicles_combined.mp4
```

---

# Input Video

The lesson uses the Supervision example traffic video:

```text
vehicles.mp4
```

The original notebook downloads the video from:

```text
https://media.roboflow.com/supervision/video-examples/vehicles.mp4
```

For this repository, the practical input should be stored at:

```text
practical/assets/input/vehicles.mp4
```

The video contains moving vehicles and provides a useful environment for testing:

- Object detection
- Tracking
- Polygon occupancy
- Line crossings
- Directional counting

---

# Output Files

The practical can generate three output videos.

## Polygon Zone Output

```text
assets/output/vehicles_polygon_zone.mp4
```

This output demonstrates:

- Vehicle detection
- Object tracking
- Tracker IDs
- Polygon filtering
- Current polygon occupancy

---

## Line Zone Output

```text
assets/output/vehicles_line_zone.mp4
```

This output demonstrates:

- Vehicle detection
- Object tracking
- Virtual counting line
- Directional crossings
- Accumulated counts

---

## Combined Output

```text
assets/output/vehicles_combined.mp4
```

This is the final extension of the practical.

It combines:

```text
Object Detection
        +
Object Tracking
        +
PolygonZone
        +
LineZone
        +
Visualization
```

in the same output video.

---

# Required Libraries

The notebook installs the required libraries with:

```python
%pip install supervision ultralytics trackers
%pip install -q rfdetr "trackers==2.4.0"
```

The main Python imports are:

```python
import cv2
import numpy as np
import supervision as sv

from pathlib import Path
from ultralytics import YOLO
from trackers import ByteTrackTracker
```

For notebook visualization, Matplotlib can also be used:

```python
import matplotlib.pyplot as plt
```

---

# Object Detection

The lesson uses:

```python
model = YOLO("yolov8n.pt")
```

YOLO processes every video frame:

```python
results = model(
    frame,
    verbose=False
)[0]
```

The results are converted into Supervision detections:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

---

# Object Tracking

The tracker is created using:

```python
tracker = ByteTrackTracker()
```

Detections are then passed through the tracker:

```python
detections = tracker.update(
    detections
)
```

Tracking provides persistent identities through:

```python
detections.tracker_id
```

Example:

```text
Frame 100 → ID:7
Frame 101 → ID:7
Frame 102 → ID:7
```

This allows the system to reason about object movement over time.

---

# Important Processing Rule

Tracking must happen before zone analysis.

Correct:

```text
YOLO
 ↓
Detections
 ↓
ByteTrack
 ↓
Tracked Detections
 ↓
Zone Trigger
```

The practical therefore follows:

```python
results = model(
    frame,
    verbose=False
)[0]

detections = sv.Detections.from_ultralytics(
    results
)

detections = tracker.update(
    detections
)

zone.trigger(
    detections=detections
)
```

---

# Video Information

Before defining spatial zones, the practical reads the video dimensions.

```python
video_info = sv.VideoInfo.from_video_path(
    "assets/input/vehicles.mp4"
)
```

The dimensions are available through:

```python
video_info.width
video_info.height
```

These values make it possible to define zones relative to the actual video resolution.

---

# PolygonZone

The polygon used in the lesson covers approximately the lower-left section of the video.

```python
POLYGON_LEFT = np.array([
    [0,                     video_info.height // 2],
    [video_info.width // 2, video_info.height // 2],
    [video_info.width // 2, video_info.height],
    [0,                     video_info.height],
])
```

The zone is created with:

```python
zone = sv.PolygonZone(
    polygon=POLYGON_LEFT
)
```

---

# PolygonZone Trigger

The polygon evaluates the tracked detections using:

```python
inside_zone = zone.trigger(
    detections=detections
)
```

The result is a Boolean mask.

Example:

```text
[True, False, True, False]
```

This indicates which tracked detections are associated with the polygon.

---

# Filtering Objects Inside the Polygon

The mask can be applied directly:

```python
detections_inside_zone = detections[
    inside_zone
]
```

Only objects inside the zone remain.

Conceptually:

```text
All Tracked Detections
        ↓
PolygonZone.trigger()
        ↓
Boolean Mask
        ↓
Filter
        ↓
Tracked Objects Inside Zone
```

---

# Current Occupancy

The polygon automatically maintains:

```python
zone.current_count
```

This represents the current number of objects inside the region.

Example:

```text
Frame 100 → 3 vehicles
Frame 101 → 4 vehicles
Frame 102 → 2 vehicles
```

This is an **instantaneous measurement**.

---

# LineZone

The practical also defines a horizontal counting line.

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

The line zone is created with:

```python
line_zone = sv.LineZone(
    start=line_start,
    end=line_end
)
```

---

# LineZone Trigger

Tracked detections are evaluated using:

```python
line_zone.trigger(
    detections=detections
)
```

The zone maintains two directional counters:

```python
line_zone.in_count
line_zone.out_count
```

These values accumulate throughout the video.

---

# PolygonZone vs LineZone

The practical demonstrates the difference between the two measurements.

```text
PolygonZone
    ↓
Current Presence
    ↓
Occupancy
```

while:

```text
LineZone
    ↓
Accumulated Crossings
    ↓
Flow
```

A simple way to remember this is:

> PolygonZone asks: "How many are here?"

> LineZone asks: "How many passed?"

---

# Box Annotation

Bounding boxes are created using:

```python
box_annotator = sv.BoxAnnotator()
```

They can be applied with:

```python
annotated = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections
)
```

---

# Tracker ID Labels

The practical can display persistent tracker IDs.

```python
label_annotator = sv.LabelAnnotator()
```

Labels can be generated with:

```python
labels = [
    f"ID:{tracker_id}"
    for tracker_id in detections.tracker_id
]
```

Then:

```python
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels
)
```

---

# Polygon Annotation

The polygon annotator is created with:

```python
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.RED,
    thickness=4
)
```

It can then be applied with:

```python
annotated = zone_annotator.annotate(
    scene=annotated
)
```

The annotator displays the polygon and its current occupancy.

---

# Line Annotation

The line annotator is created using:

```python
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_scale=1.5,
    custom_in_text="Crossings Down",
    custom_out_text="Crossings Up"
)
```

It is applied using:

```python
annotated = line_zone_annotator.annotate(
    frame=annotated,
    line_counter=line_zone
)
```

Notice that this annotator uses:

```text
frame=
line_counter=
```

rather than the `scene=` parameter commonly used by other annotators.

---

# Combined Callback

The final practical combines both zones in the same callback.

The basic structure is:

```python
tracker = ByteTrackTracker()

def callback_combined(
    frame: np.ndarray,
    _: int
) -> np.ndarray:

    results = model(
        frame,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    detections = tracker.update(
        detections
    )

    zone.trigger(
        detections=detections
    )

    line_zone.trigger(
        detections=detections
    )

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    annotated = zone_annotator.annotate(
        scene=annotated
    )

    annotated = line_zone_annotator.annotate(
        frame=annotated,
        line_counter=line_zone
    )

    return annotated
```

---

# Combined Video Processing

The final video can be generated using:

```python
sv.process_video(
    source_path="assets/input/vehicles.mp4",
    target_path="assets/output/vehicles_combined.mp4",
    callback=callback_combined,
    show_progress=True
)
```

The resulting video combines:

```text
Bounding Boxes
      +
Tracking
      +
Polygon Occupancy
      +
Counting Line
      +
Directional Crossing Counts
```

---

# Expected Result

The final video should visually contain:

```text
┌─────────────────────────────────────────────┐
│                                             │
│      ┌─────────────────────┐                │
│      │    POLYGON ZONE     │                │
│      │                     │                │
│      │ ID:4      ID:9      │                │
│      │                     │                │
│      │ Current Count: 2    │                │
│      └─────────────────────┘                │
│                                             │
│════════════ COUNTING LINE ══════════════════│
│                                             │
│ Crossings Down: XX                          │
│ Crossings Up: XX                            │
│                                             │
└─────────────────────────────────────────────┘
```

The exact counts depend on the detections, tracking behavior, video, and zone placement.

---

# Practical Workflow

The recommended execution order is:

```text
1. Install dependencies
        ↓
2. Prepare input video
        ↓
3. Load YOLO model
        ↓
4. Read video information
        ↓
5. Define PolygonZone
        ↓
6. Define LineZone
        ↓
7. Create tracker
        ↓
8. Create annotators
        ↓
9. Process each frame
        ↓
10. Detect objects
        ↓
11. Track objects
        ↓
12. Trigger both zones
        ↓
13. Annotate results
        ↓
14. Save output video
        ↓
15. Inspect final counts
```

---

# Expected Learning Outcome

After completing this practical, I should be able to build a video analytics pipeline that:

- Detects objects with YOLO
- Converts detections into `sv.Detections`
- Tracks objects with ByteTrack
- Maintains persistent tracker IDs
- Defines polygon regions using pixel coordinates
- Detects tracked objects inside polygon zones
- Measures current occupancy
- Defines virtual counting lines
- Detects crossing events
- Measures directional flow
- Visualizes multiple analytics layers
- Generates an annotated output video

---

# Real-World Applications

The same architecture can be adapted for:

## Traffic Monitoring

```text
PolygonZone → lane occupancy
LineZone    → vehicle flow
```

## Parking Management

```text
PolygonZone → current parking occupancy
LineZone    → vehicles entering and leaving
```

## Retail Analytics

```text
PolygonZone → customers currently in an area
LineZone    → store entries and exits
```

## Security

```text
PolygonZone → restricted-area occupancy
LineZone    → boundary crossings
```

## Industrial Monitoring

```text
PolygonZone → workers inside safety zones
LineZone    → movement through checkpoints
```

---

# Key Takeaways

The practical demonstrates that:

1. Object detection identifies objects.
2. Tracking maintains object identities.
3. Polygon zones provide spatial occupancy information.
4. Line zones provide movement and crossing information.
5. The same tracked detections can feed multiple analytics components.
6. Detection and tracking do not need to be repeated for every zone.
7. Zone triggers should run before their corresponding annotations.
8. Multiple annotators can be layered onto the same frame.
9. `sv.process_video()` can apply the pipeline to an entire video.
10. Combining detection, tracking, zones, and counting creates a reusable spatial video analytics pipeline.

---

# Final Pipeline

The complete practical can be summarized as:

```text
VIDEO
  ↓
YOLO
  ↓
DETECTIONS
  ↓
BYTETRACK
  ↓
TRACKED OBJECTS
  │
  ├───────────────┐
  ↓               ↓
POLYGON ZONE    LINE ZONE
  ↓               ↓
OCCUPANCY        FLOW
  │               │
  └───────┬───────┘
          ↓
     VISUALIZATION
          ↓
    OUTPUT VIDEO
```

This practical represents the transition from simple object detection toward a more complete **spatial video analytics system**.

---

## Related Documentation

- [Session README](../README.md)
- [Concepts Overview](../concepts/README.md)
- [PolygonZone](../concepts/01-PolygonZone.md)
- [LineZone](../concepts/02-LineZone.md)
- [Occupancy vs Flow](../concepts/03-Occupancy-vs-Flow.md)
- [Tracking with Zones](../concepts/04-Tracking-with-Zones.md)
- [Combining PolygonZone and LineZone](../concepts/07-Combining-PolygonZone-and-LineZone.md)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
