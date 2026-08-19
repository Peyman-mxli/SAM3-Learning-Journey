# Object Tracking with Supervision

Object detection tells us **what objects are present in an image or video frame**.

Object tracking goes one step further.

It allows us to determine whether an object detected in one frame is the **same object that appears in the next frame**.

In this lesson, we extend the detection and filtering concepts from the previous sections and introduce **object tracking using Supervision and ByteTrack**.

The main goal is to process a real video and assign each detected object a persistent identification number.

---

## Learning Objectives

By the end of this lesson, you should understand how to:

- Explain the difference between object detection and object tracking
- Understand the purpose of `tracker_id`
- Use `sv.ByteTrack`
- Assign persistent IDs to detected objects
- Track objects across multiple video frames
- Process videos using `sv.process_video`
- Display object classes together with tracker IDs
- Draw object trajectories using `TraceAnnotator`
- Filter detections before sending them to the tracker
- Combine filtering and tracking in the same computer vision pipeline

---

## Detection vs. Tracking

Object detection processes each frame independently.

For example:

```text
Frame 1

car
car
truck
```

In the next frame:

```text
Frame 2

car
car
truck
```

The detector knows which objects exist in each frame.

However, it does **not know whether a car in Frame 1 is the same car appearing in Frame 2**.

This is where object tracking becomes useful.

With tracking:

```text
Frame 1

car   → ID 1
car   → ID 2
truck → ID 3
```

In the next frame:

```text
Frame 2

car   → ID 1
car   → ID 2
truck → ID 3
```

Even though the objects moved, the tracker attempts to maintain their identities.

---

## What Does a Tracker Do?

A tracker compares detections between consecutive frames.

It attempts to determine whether newly detected objects correspond to objects that were already detected previously.

The tracker then assigns each object an identification number.

Conceptually:

```text
Frame N
   ↓
Object Detection
   ↓
Detections
   ↓
Tracker
   ↓
Detections + tracker_id
```

Then:

```text
Frame N+1
   ↓
Object Detection
   ↓
New Detections
   ↓
Tracker
   ↓
Same tracker_id when the object is recognized
```

This allows us to follow individual objects through a video.

---

## The `tracker_id` Property

Supervision detections can contain a property called:

```python
detections.tracker_id
```

Before tracking is performed, this value is normally:

```python
None
```

For example:

```python
results = model(frame, verbose=False)[0]

detections = sv.Detections.from_ultralytics(results)

print(detections.tracker_id)
```

Before using a tracker:

```text
tracker_id: None
```

After sending the detections through ByteTrack:

```python
detections = tracker.update_with_detections(detections)
```

the detections receive tracking IDs.

For example:

```text
tracker_id: [1 2 3 4]
```

Each number represents an object currently being tracked.

---

## ByteTrack

In this lesson, object tracking is introduced using:

```python
sv.ByteTrack()
```

Create the tracker:

```python
tracker = sv.ByteTrack()
```

Then update it using the detections from the current frame:

```python
detections = tracker.update_with_detections(detections)
```

ByteTrack attempts to associate the current detections with objects detected in previous frames.

The objective is to preserve the same `tracker_id` while an object moves through the video.

---

## Installing the Required Libraries

The lesson uses Supervision and Ultralytics:

```python
!pip install supervision ultralytics
```

Import the required libraries:

```python
import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import urllib.request
from pathlib import Path
```

---

## Preparing the Video

Create the assets directory:

```python
Path("assets").mkdir(exist_ok=True)
```

Download the sample vehicle video:

```python
urllib.request.urlretrieve(
    "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
    "assets/vehicles.mp4"
)
```

The video is stored at:

```text
assets/vehicles.mp4
```

---

## Inspecting Video Information

Supervision provides the `VideoInfo` class for inspecting video metadata.

```python
video_info = sv.VideoInfo.from_video_path(
    "assets/vehicles.mp4"
)
```

We can inspect:

```python
print(video_info.width)
print(video_info.height)
print(video_info.fps)
print(video_info.total_frames)
```

The approximate duration can be calculated with:

```python
video_info.total_frames / video_info.fps
```

---

## Loading the YOLO Model

The lesson uses YOLOv8 Nano:

```python
model = YOLO("yolov8n.pt")
```

YOLO performs object detection.

ByteTrack performs object tracking.

```text
Video Frame
    ↓
YOLO
    ↓
Object Detections
    ↓
sv.Detections
    ↓
ByteTrack
    ↓
Tracked Detections
```

---

## Inspecting `tracker_id` Before Tracking

Extract a frame from the video:

```python
cap = cv2.VideoCapture(
    "assets/vehicles.mp4"
)

ret, first_frame = cap.read()

cap.release()
```

Run YOLO:

```python
results = model(
    first_frame,
    verbose=False
)[0]
```

Convert the results:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

Inspect the tracker IDs:

```python
print(detections.tracker_id)
```

At this stage:

```text
tracker_id = None
```

This is expected because YOLO detects objects but does not assign tracking IDs.

---

## Applying ByteTrack

Create the tracker:

```python
tracker = sv.ByteTrack()
```

Pass the detections through the tracker:

```python
detections = tracker.update_with_detections(
    detections
)
```

Now:

```python
print(detections.tracker_id)
```

should return tracking IDs.

The important distinction is:

```text
YOLO
  ↓
Detects objects

ByteTrack
  ↓
Tracks individual objects
```

---

## Resetting the Tracker

The tracker stores information about previously tracked objects.

When starting a new experiment, reset it:

```python
tracker.reset()
```

This clears the previous tracking state.

---

## Creating the Annotators

The lesson uses multiple Supervision annotators:

```python
box_annotator = sv.BoxAnnotator()

label_annotator = sv.LabelAnnotator()

trace_annotator = sv.TraceAnnotator()
```

Each annotator has a different purpose.

---

### BoxAnnotator

`BoxAnnotator` draws bounding boxes around detected objects.

```python
box_annotator = sv.BoxAnnotator()
```

---

### LabelAnnotator

`LabelAnnotator` displays information about each tracked object.

For example:

```text
ID:1
ID:2
ID:3
```

We can also combine class names and tracker IDs:

```text
car #1
truck #2
car #3
```

This tells us both:

```text
WHAT is the object?
```

and:

```text
WHICH specific object is it?
```

---

### TraceAnnotator

`TraceAnnotator` visualizes the trajectory of tracked objects.

```python
trace_annotator = sv.TraceAnnotator()
```

Unlike a normal bounding box, a trace represents an object's movement across multiple frames.

Conceptually:

```text
Previous position
      •
       \
        •
         \
          •
           \
            [CAR]
```

The trace depends on `tracker_id` because Supervision needs to know which positions belong to the same object.

---

## Creating the Frame Processing Function

Video processing requires a callback function.

```python
def process_frame(
    frame: np.ndarray,
    frame_idx: int
) -> np.ndarray:

    results = model(
        frame,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    detections = tracker.update_with_detections(
        detections
    )

    labels = [
        f"ID:{tracker_id}"
        for tracker_id in detections.tracker_id
    ]

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

    annotated = trace_annotator.annotate(
        scene=annotated,
        detections=detections
    )

    return annotated
```

This callback performs the tracking process for each frame.

---

## Processing the Video

Supervision provides:

```python
sv.process_video()
```

Example:

```python
sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_tracked.mp4",
    callback=process_frame,
    show_progress=True
)
```

The processed video is saved as:

```text
assets/vehicles_tracked.mp4
```

---

## Complete Tracking Pipeline

The complete workflow is:

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
tracker_id
     ↓
BoxAnnotator
     ↓
LabelAnnotator
     ↓
TraceAnnotator
     ↓
Annotated Frame
     ↓
Output Video
```

This process repeats for every frame in the video.

---

## Inspecting IDs Across Frames

One important experiment is to inspect tracking IDs manually across several frames.

```python
tracker.reset()

cap = cv2.VideoCapture(
    "assets/vehicles.mp4"
)

for frame_num in range(3):

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    detections = tracker.update_with_detections(
        detections
    )

    print(
        f"Frame {frame_num}: "
        f"{len(detections)} objects | "
        f"IDs: {detections.tracker_id}"
    )

cap.release()
```

A possible result could look like:

```text
Frame 0 → [1, 2, 3]
Frame 1 → [1, 2, 3]
Frame 2 → [1, 2, 3]
```

This indicates that the tracker is maintaining the identities of those objects across frames.

---

## Displaying Class and Tracker ID

We can combine:

```python
class_id
```

with:

```python
tracker_id
```

Example:

```python
labels = [
    f"{results.names[class_id]} #{tracker_id}"
    for class_id, tracker_id in zip(
        detections.class_id,
        detections.tracker_id
    )
]
```

The resulting labels can look like:

```text
car #1
car #2
truck #3
bus #4
```

---

## `class_id` vs. `tracker_id`

These properties represent different information.

### `class_id`

Answers:

```text
What type of object is this?
```

Examples:

```text
person
car
truck
bus
```

### `tracker_id`

Answers:

```text
Which specific object is this?
```

For example:

```text
Object      class_id      tracker_id

Car A           2              1
Car B           2              2
Car C           2              3
```

All three objects belong to the same class.

However, each one is a different tracked object.

---

## Filtering Before Tracking

The filtering concepts from the previous lesson can be combined directly with tracking.

Suppose we only want to track cars.

In the COCO dataset:

```python
TARGET_CLASS = 2
```

represents the `car` class.

Filter the detections:

```python
detections = detections[
    detections.class_id == TARGET_CLASS
]
```

Then send those detections to ByteTrack:

```python
detections = tracker.update_with_detections(
    detections
)
```

---

## Why Filter Before Tracking?

The order of operations matters.

A useful pipeline is:

```text
YOLO
 ↓
Detections
 ↓
Filtering
 ↓
Tracking
 ↓
Annotation
```

For example, imagine YOLO detects:

```text
car
person
truck
car
bus
car
```

If our application only cares about cars, filtering produces:

```text
car
car
car
```

Only these detections are then sent to the tracker.

This directly connects this lesson with the previous lesson on filtering and manipulating detections.

---

## Tracking Only Cars

```python
TARGET_CLASS = 2

tracker.reset()

def callback_only_cars(
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

    detections = detections[
        detections.class_id == TARGET_CLASS
    ]

    detections = tracker.update_with_detections(
        detections
    )

    labels = [
        f"car #{tracker_id}"
        for tracker_id in detections.tracker_id
    ]

    scene = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    scene = label_annotator.annotate(
        scene=scene,
        detections=detections,
        labels=labels
    )

    return scene
```

Process the video:

```python
sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_cars.mp4",
    callback=callback_only_cars,
    show_progress=True
)
```

---

## Extension Challenge — Counting Visible Frames

Tracking IDs allow us to build additional analytics.

For example, we can count how many frames each object remains visible.

Create a dictionary:

```python
frame_count = {}
```

Update it:

```python
for tracker_id in detections.tracker_id:

    frame_count[tracker_id] = (
        frame_count.get(tracker_id, 0) + 1
    )
```

We could then create labels such as:

```text
#1 (15f)
#2 (8f)
#3 (32f)
```

where:

```text
f = frames
```

Persistent tracking IDs can therefore become the foundation for more advanced video analytics.

---

## Connection to the Previous Lesson

The previous lesson introduced:

**Filtering and Manipulating Detections**

We learned how to filter detections using:

- Confidence
- Class
- Object size
- Position
- Boolean masks
- NMS
- Detection combinations

Now those concepts can be used before tracking.

The learning progression becomes:

```text
Object Detection
      ↓
sv.Detections
      ↓
Filtering and Manipulation
      ↓
Object Tracking
      ↓
Persistent tracker_id
      ↓
Video Annotation
      ↓
Object Trajectories
```

This is an important transition from analyzing independent images to understanding objects **across time**.

---

## Key Concepts

### Detection

Determines:

```text
What objects exist in this frame?
```

### Tracking

Determines:

```text
Which specific object is this across multiple frames?
```

### `class_id`

Identifies the object's category.

```text
car
person
truck
bus
```

### `tracker_id`

Identifies the individual object.

```text
car #1
car #2
car #3
```

### Trace

Visualizes how a tracked object moves through the video.

---

## What I Learned

In this lesson, I learned that object detection and object tracking solve different problems.

Object detection identifies objects independently in each frame, while object tracking connects those detections across time.

I learned how `tracker_id` allows individual objects to maintain an identity while moving through a video.

I also learned how to:

- Use `sv.ByteTrack`
- Update detections with tracking information
- Inspect `tracker_id`
- Reset a tracker
- Process complete videos
- Create labels using tracker IDs
- Combine class names with tracking IDs
- Draw object trajectories
- Filter detections before tracking
- Build simple tracking-based analytics

This creates the foundation for more advanced video computer vision applications.

---

## Practical Applications

Object tracking can be used in many real-world systems.

### Traffic Analysis

Track vehicles through intersections or highways.

```text
Vehicle Detection
      ↓
Vehicle Tracking
      ↓
Traffic Analysis
```

### People Tracking

Track people moving through stores, buildings, or public spaces.

### Sports Analytics

Track players or objects during games.

### Industrial Monitoring

Track products moving through production lines.

### Security Systems

Follow detected objects across surveillance footage.

### Computer Vision Analytics

Tracking IDs can be used to calculate:

- Object counts
- Time visible
- Movement paths
- Entry and exit events
- Direction of movement
- Speed estimates
- Zone interactions

---

## Important Takeaways

1. Object detection and object tracking are different tasks.
2. YOLO detects objects but does not maintain their identity across frames.
3. ByteTrack associates detections between frames.
4. `tracker_id` identifies individual tracked objects.
5. Multiple objects can share the same `class_id` while having different `tracker_id` values.
6. `TraceAnnotator` uses tracking information to visualize object movement.
7. `sv.process_video` applies processing to every frame of a video.
8. Filtering can be performed before tracking to focus on specific objects.
9. Tracking IDs make higher-level video analytics possible.
10. Object tracking is an important step toward complete computer vision video pipelines.

---

## Learning Journey Progress

This lesson extends the concepts from the previous sections:

```text
Agentic AI Programming
        ↓
Introduction to Supervision
        ↓
Annotation and Visualization
        ↓
Filtering and Manipulating Detections
        ↓
Object Tracking
```

Each lesson adds another layer to the computer vision pipeline.

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey

- GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)
- LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)

---

## Repository

This lesson is part of the:

[SAM3 Learning Journey](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

Repository containing structured notes, examples, experiments, and projects created while learning computer vision, Supervision, object tracking, and Segment Anything Model 3.
