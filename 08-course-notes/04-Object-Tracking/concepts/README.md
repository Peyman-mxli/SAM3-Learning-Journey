# Object Tracking — Concepts

This folder contains the core concepts from the **Object Tracking** lesson.

The objective of this lesson is to understand how object detection can be extended into video tracking by assigning persistent IDs to objects across multiple frames.

In this lesson, the main tracking tool is:

```python
sv.ByteTrack()
```

The tracker receives `sv.Detections` and assigns a persistent:

```python
tracker_id
```

to each tracked object.

---

## Concepts Covered

The concepts in this lesson include:

### Detection vs. Tracking

Understand the difference between detecting objects independently in each frame and following the same object over time.

---

### `tracker_id`

Learn what `tracker_id` represents and what its value looks like before and after applying object tracking.

---

### ByteTrack

Understand how:

```python
sv.ByteTrack()
```

associates detections between consecutive video frames.

---

### Persistent Object IDs

Learn how the tracker attempts to maintain the same ID while an object moves through the video.

Example:

```text
Frame 1

car → ID 1
car → ID 2
```

```text
Frame 2

car → ID 1
car → ID 2
```

---

### Complete Tracking Pipeline

Understand the complete workflow:

```text
Video Frame
    ↓
YOLO
    ↓
sv.Detections
    ↓
ByteTrack
    ↓
tracker_id
    ↓
Annotation
```

---

### Video Processing

Learn how Supervision can process an entire video using:

```python
sv.process_video()
```

with a callback that performs detection, tracking, and annotation for every frame.

---

### Tracking Annotations

Learn how tracking information can be visualized using:

```python
sv.BoxAnnotator()
sv.LabelAnnotator()
sv.TraceAnnotator()
```

These annotators allow us to display:

- Bounding boxes
- Object classes
- Tracker IDs
- Object trajectories

---

### Filtering Before Tracking

Apply concepts from the previous lesson before sending detections to the tracker.

Example:

```text
YOLO
 ↓
Detections
 ↓
Filter
 ↓
ByteTrack
 ↓
Tracked Objects
```

This allows us to track only the objects relevant to our application.

---

### Tracking IDs Across Frames

Inspect `tracker_id` values frame by frame to observe whether an object's identity remains stable while it moves.

---

### Class ID vs. Tracker ID

Understand the difference between:

```python
class_id
```

and:

```python
tracker_id
```

`class_id` identifies the object's category.

`tracker_id` identifies the individual tracked object.

---

### Tracking-Based Analytics

Persistent IDs can later be used for tasks such as:

- Counting unique objects
- Measuring how long an object remains visible
- Following movement trajectories
- Monitoring objects entering or leaving areas
- Building more advanced video analytics

---

## Interactive Experiments

The lesson includes three main experiments.

### Experiment 1

Inspect:

```python
tracker_id
```

frame by frame.

---

### Experiment 2

Display the object's class and tracking ID together.

Example:

```text
car #1
truck #2
car #3
```

---

### Experiment 3

Filter detections before tracking them.

Example:

```text
All detections
      ↓
Cars only
      ↓
ByteTrack
      ↓
Tracked cars
```

---

## Extension Challenge

The lesson also introduces a challenge:

Track how many frames each object has remained visible.

Example:

```text
#1 (15f)
#2 (8f)
#3 (32f)
```

A dictionary can be used to store the frame count for each `tracker_id`.

---

## ByteTrack API Transition

This lesson uses:

```python
sv.ByteTrack()
```

because its API works directly with:

```python
sv.Detections
```

The lesson also explains that later notebooks migrate to:

```python
from trackers import ByteTrackTracker
```

The main tracking concepts remain the same even though the API changes.

---

## Concept Files

The individual concept files in this folder will explain each topic separately and in more detail.

The structure will follow the same documentation style used throughout the SAM3 Learning Journey.

---

## Learning Progression

This lesson continues directly from:

```text
Filtering and Manipulating Detections
                ↓
          Object Tracking
```

The complete progression is becoming:

```text
Object Detection
      ↓
sv.Detections
      ↓
Annotation
      ↓
Filtering
      ↓
Object Tracking
      ↓
Persistent IDs
      ↓
Video Analytics
```

---

## Main Goal

The main goal of this section is to understand:

> How can we recognize that an object detected in one frame is the same object appearing in the next frame?

Object tracking provides the foundation for answering that question.
