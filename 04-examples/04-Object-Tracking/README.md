# Object Tracking — Code Examples

This directory contains practical Python examples based on the concepts covered in the **Object Tracking** lesson.

The examples demonstrate how object detection can be extended from individual images into video tracking using:

```python
YOLO
```

```python
Supervision
```

and:

```python
ByteTrack
```

The main objective is to understand how detected objects can receive persistent identities across multiple video frames.

---

## Learning Objectives

The examples in this directory demonstrate how to:

- Load and inspect a video
- Detect objects in video frames with YOLO
- Convert YOLO results into `sv.Detections`
- Create a ByteTrack tracker
- Assign `tracker_id` values
- Inspect tracking IDs across frames
- Process complete videos
- Display object classes and tracker IDs
- Draw tracking trajectories
- Filter detections before tracking
- Build simple tracking analytics

---

## Main Tracking Pipeline

The examples follow this general workflow:

```text
Input Video
     ↓
Video Frame
     ↓
YOLO
     ↓
sv.Detections
     ↓
Optional Filtering
     ↓
ByteTrack
     ↓
tracker_id
     ↓
Annotation
     ↓
Output Video
```

---

## Main Libraries

The examples use:

```python
import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO
```

---

## YOLO

YOLO performs object detection.

Example:

```python
model = YOLO("yolov8n.pt")
```

Then:

```python
results = model(
    frame,
    verbose=False
)[0]
```

YOLO determines:

```text
What objects are visible?
```

---

## Supervision Detections

YOLO results are converted into:

```python
sv.Detections
```

using:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

This gives us a convenient structure for working with:

```text
Bounding Boxes
Confidence Scores
Class IDs
Tracker IDs
```

---

## ByteTrack

The tracker is created using:

```python
tracker = sv.ByteTrack()
```

Detections are then passed to the tracker:

```python
detections = tracker.update_with_detections(
    detections
)
```

Before tracking:

```python
detections.tracker_id
```

is normally:

```text
None
```

After tracking, it may contain values such as:

```text
[1, 2, 3, 4]
```

---

## Detection vs. Tracking

Detection answers:

```text
What object is this?
```

Tracking answers:

```text
Which specific object is this across time?
```

For example:

```text
Detection:

car
car
truck
```

Tracking:

```text
car #1
car #2
truck #3
```

---

## Example Topics

The examples in this section focus on several practical tracking operations.

### Basic ByteTrack

Create:

```python
sv.ByteTrack()
```

and assign tracker IDs to YOLO detections.

---

### Inspect Tracker IDs

Observe:

```python
detections.tracker_id
```

before and after tracking.

---

### Tracking Across Frames

Process several consecutive frames and inspect whether IDs remain consistent.

Example:

```text
Frame 0 → [1, 2, 3]

Frame 1 → [1, 2, 3]

Frame 2 → [1, 2, 3]
```

---

### Video Processing

Use:

```python
sv.process_video()
```

to apply detection and tracking to an entire video.

---

### Tracking Labels

Display:

```text
ID:1
ID:2
ID:3
```

or:

```text
car #1
truck #2
car #3
```

using:

```python
sv.LabelAnnotator()
```

---

### Object Trajectories

Use:

```python
sv.TraceAnnotator()
```

to visualize the recent movement of tracked objects.

Conceptually:

```text
•
 \
  •
   \
    •
     \
      [car #3]
```

---

### Filtering Before Tracking

Apply filters before sending detections to ByteTrack.

For example:

```python
detections = detections[
    detections.class_id == TARGET_CLASS
]
```

Then:

```python
detections = tracker.update_with_detections(
    detections
)
```

This allows us to track only selected object classes.

---

### Tracking Analytics

Use:

```python
tracker_id
```

to associate information with individual objects.

For example:

```text
car #1 → 45 frames

car #2 → 17 frames

truck #3 → 81 frames
```

This introduces the foundation for more advanced video analytics.

---

## Example Structure

The examples are designed to progress from simple tracking operations toward a complete video tracking pipeline.

```text
Basic Detection
      ↓
ByteTrack
      ↓
tracker_id
      ↓
Multiple Frames
      ↓
Video Processing
      ↓
Tracking Annotation
      ↓
Filtering
      ↓
Tracking Analytics
```

---

## Relationship to Course Notes

These examples support the concepts documented in:

```text
08-course-notes/
└── 04-Object-Tracking/
    └── concepts/
```

The course notes explain the concepts in detail.

This directory focuses on practical Python implementations.

---

## Relationship to Previous Examples

The previous examples introduced:

```text
YOLO
 ↓
sv.Detections
 ↓
Annotation
 ↓
Filtering
```

Object tracking extends this pipeline:

```text
YOLO
 ↓
sv.Detections
 ↓
Filtering
 ↓
ByteTrack
 ↓
tracker_id
 ↓
Tracking Annotation
```

This is the transition from analyzing independent detections to following individual objects through time.

---

## Expected Input

The examples use a video such as:

```text
assets/vehicles.mp4
```

The video contains moving vehicles that can be detected and tracked across consecutive frames.

---

## Expected Outputs

Depending on the example, outputs may include videos such as:

```text
vehicles_tracked.mp4
```

```text
vehicles_class_id.mp4
```

```text
vehicles_cars.mp4
```

The exact output depends on the tracking experiment.

---

## Key Concepts Practiced

The examples reinforce:

```python
sv.Detections
```

```python
sv.ByteTrack()
```

```python
detections.tracker_id
```

```python
tracker.update_with_detections()
```

```python
tracker.reset()
```

```python
sv.BoxAnnotator()
```

```python
sv.LabelAnnotator()
```

```python
sv.TraceAnnotator()
```

```python
sv.process_video()
```

---

## Main Goal

The goal of these examples is to move from:

```text
"I can detect objects."
```

to:

```text
"I can detect an object,
assign it an identity,
and follow it across video frames."
```

This is an important step toward building complete computer vision video applications.

---

## Next Step

After creating this README, the individual Python examples can be added one at a time.

The first example will focus on the most fundamental operation:

```text
YOLO Detection
      ↓
sv.Detections
      ↓
ByteTrack
      ↓
tracker_id
```

This will provide the foundation for the remaining tracking examples.
