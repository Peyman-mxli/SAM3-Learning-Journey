# ByteTrack with Supervision

ByteTrack is the object tracker introduced in this lesson.

While YOLO detects objects in each individual frame, ByteTrack attempts to associate those detections across consecutive frames.

This allows objects to receive and maintain:

```python
tracker_id
```

values while they move through a video.

---

## Why Do We Need ByteTrack?

YOLO performs object detection.

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

YOLO detects the objects again, but detection alone does not tell us whether they are the same physical objects.

ByteTrack adds this missing information.

```text
Frame 1

car   → ID 1
car   → ID 2
truck → ID 3
```

Then:

```text
Frame 2

car   → ID 1
car   → ID 2
truck → ID 3
```

The tracker attempts to preserve the identity of each object across frames.

---

## Creating ByteTrack

In this lesson, ByteTrack is created using Supervision:

```python
tracker = sv.ByteTrack()
```

This creates a tracker that can work directly with:

```python
sv.Detections
```

The basic workflow is:

```text
YOLO
  ↓
sv.Detections
  ↓
sv.ByteTrack
  ↓
Tracked Detections
```

---

## Detection Before Tracking

First, YOLO processes a frame:

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

At this point:

```python
detections.tracker_id
```

is normally:

```text
None
```

The objects have been detected but have not yet been tracked.

---

## Updating ByteTrack

The detections are sent to ByteTrack using:

```python
detections = tracker.update_with_detections(
    detections
)
```

This is one of the most important lines in the tracking pipeline.

Before:

```text
sv.Detections

tracker_id = None
```

After:

```text
sv.Detections

tracker_id = [1, 2, 3, ...]
```

The returned detections now contain tracking information.

---

## Basic ByteTrack Example

A simple tracking operation looks like this:

```python
tracker = sv.ByteTrack()

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

print(detections.tracker_id)
```

The result may look similar to:

```text
[1 2 3 4]
```

The exact IDs depend on the objects detected in the video.

---

## ByteTrack Needs Consecutive Frames

Tracking only becomes meaningful when multiple frames are processed in sequence.

For example:

```text
Frame 0
   ↓
YOLO
   ↓
ByteTrack
   ↓
IDs assigned
```

Then:

```text
Frame 1
   ↓
YOLO
   ↓
ByteTrack
   ↓
Compare with previous tracking state
```

Then:

```text
Frame 2
   ↓
YOLO
   ↓
ByteTrack
   ↓
Continue object identities
```

The tracker maintains internal state between frames.

This state allows it to associate new detections with previously tracked objects.

---

## Why Tracker State Matters

Consider:

```text
Frame 0

car #1
car #2
truck #3
```

ByteTrack stores information about these tracked objects.

When Frame 1 arrives, it attempts to associate the new detections with:

```text
ID 1
ID 2
ID 3
```

If the associations are successful:

```text
Frame 1

car #1
car #2
truck #3
```

The IDs remain consistent.

---

## Do Not Recreate the Tracker for Every Frame

The tracker needs to maintain state across frames.

Therefore, the tracker should be created before processing the video frames.

Correct concept:

```python
tracker = sv.ByteTrack()

for frame in video:

    results = model(frame)[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    detections = tracker.update_with_detections(
        detections
    )
```

Conceptually:

```text
One Tracker
    ↓
Frame 1
    ↓
Frame 2
    ↓
Frame 3
    ↓
Frame 4
```

The same tracker processes the sequence.

---

## Resetting ByteTrack

Because ByteTrack maintains state, the lesson also uses:

```python
tracker.reset()
```

This clears the current tracking information.

For example:

```python
tracker = sv.ByteTrack()

# First experiment
...

tracker.reset()

# Second experiment
...
```

The reset is useful when restarting a video from the beginning or starting a completely new tracking experiment.

---

## Why Reset the Tracker?

Imagine that one experiment produced:

```text
ID 1
ID 2
ID 3
ID 4
```

If we begin another experiment without resetting the tracker, the previous tracking state may still influence the new sequence.

Using:

```python
tracker.reset()
```

creates a clean tracking session.

Conceptually:

```text
Old Tracking Session
        ↓
tracker.reset()
        ↓
Clean Tracker State
        ↓
New Tracking Session
```

---

## ByteTrack and YOLO Have Different Jobs

It is important not to confuse the detector and the tracker.

### YOLO

YOLO answers:

```text
What objects exist in this frame?
```

It produces:

```text
Bounding Boxes
Class IDs
Confidence Scores
```

### ByteTrack

ByteTrack answers:

```text
Which detections correspond to the same objects across frames?
```

It adds:

```python
tracker_id
```

The complete relationship is:

```text
YOLO
  ↓
Object Detection
  ↓
sv.Detections
  ↓
ByteTrack
  ↓
Object Tracking
```

---

## ByteTrack Does Not Replace YOLO

ByteTrack is not being used instead of YOLO.

The two components work together.

```text
YOLO = Detection

ByteTrack = Tracking
```

The detector first provides objects.

The tracker then associates those objects over time.

---

## Tracking Multiple Objects

ByteTrack can maintain multiple tracked objects simultaneously.

For example:

```text
Frame 1

car   → ID 1
car   → ID 2
truck → ID 3
bus   → ID 4
```

In the next frame:

```text
Frame 2

car   → ID 1
car   → ID 2
truck → ID 3
bus   → ID 4
```

If another object enters:

```text
Frame 3

car   → ID 1
car   → ID 2
truck → ID 3
bus   → ID 4
car   → ID 5
```

The new object receives another tracking ID.

---

## Tracking IDs Can Change

ByteTrack attempts to maintain stable IDs, but tracking is not perfect.

An object may:

- Leave the frame
- Become hidden
- Fail to be detected
- Become difficult to associate with previous detections

If the tracker loses the object and it later appears again, it may receive a new ID.

For example:

```text
Frame 1 → car #4
Frame 2 → car #4
Frame 3 → car #4

Object disappears

...

Frame 15 → car #9
```

This means tracking IDs should be understood within the context of the tracking sequence.

---

## ByteTrack and Filtering

The previous lesson introduced detection filtering.

Those filters can be applied before ByteTrack.

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

The pipeline becomes:

```text
YOLO
  ↓
sv.Detections
  ↓
Filtering
  ↓
ByteTrack
  ↓
Tracked Detections
```

---

## Example — Track Only Cars

Suppose:

```python
TARGET_CLASS = 2
```

represents cars.

We can write:

```python
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
```

Now ByteTrack only receives the filtered car detections.

---

## Why Filter Before ByteTrack?

Suppose YOLO detects:

```text
person
car
truck
car
bus
car
person
```

But our application only cares about cars.

Filtering gives:

```text
car
car
car
```

Then ByteTrack processes only those detections.

```text
YOLO
  ↓
7 detections
  ↓
Class Filter
  ↓
3 car detections
  ↓
ByteTrack
  ↓
3 tracked cars
```

This connects the filtering concepts from the previous lesson directly with tracking.

---

## ByteTrack and Annotation

After ByteTrack assigns IDs, those IDs can be displayed using Supervision annotators.

For example:

```python
labels = [
    f"ID:{tracker_id}"
    for tracker_id in detections.tracker_id
]
```

The labels may look like:

```text
ID:1
ID:2
ID:3
```

Then:

```python
label_annotator.annotate(
    scene=frame,
    detections=detections,
    labels=labels
)
```

can display them on the video.

---

## Combining Class and Tracker ID

We can also display:

```text
car #1
truck #2
car #3
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

This gives us both:

```text
Object Class
+
Object Identity
```

---

## ByteTrack and TraceAnnotator

Tracking IDs also allow us to visualize object trajectories.

The lesson uses:

```python
trace_annotator = sv.TraceAnnotator()
```

Then:

```python
annotated = trace_annotator.annotate(
    scene=annotated,
    detections=detections
)
```

Because detections contain `tracker_id`, Supervision can associate positions from different frames with the same object.

For example:

```text
Frame 1 → ID 3 at Position A
Frame 2 → ID 3 at Position B
Frame 3 → ID 3 at Position C
```

The trajectory becomes:

```text
A •
    \
     • B
       \
        • C
         \
          Object #3
```

---

## ByteTrack in a Video Callback

When processing an entire video, ByteTrack can be used inside a frame callback.

Example:

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

    return annotated
```

The same tracker is used as each new frame passes through the callback.

---

## Processing the Complete Video

Supervision provides:

```python
sv.process_video()
```

The callback can be applied to the complete video:

```python
sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_tracked.mp4",
    callback=process_frame,
    show_progress=True
)
```

Conceptually:

```text
Video
  ↓
Frame 1 → YOLO → ByteTrack → Annotation
  ↓
Frame 2 → YOLO → ByteTrack → Annotation
  ↓
Frame 3 → YOLO → ByteTrack → Annotation
  ↓
...
  ↓
Tracked Output Video
```

---

## Supervision ByteTrack API

The tracking API used in this lesson is:

```python
tracker = sv.ByteTrack()
```

and:

```python
detections = tracker.update_with_detections(
    detections
)
```

This approach integrates directly with:

```python
sv.Detections
```

which makes it useful for learning the fundamental tracking workflow.

---

## Transition to the `trackers` Package

The lesson also introduces an API transition.

Later material moves from:

```python
sv.ByteTrack()
```

toward:

```python
from trackers import ByteTrackTracker
```

The important point is that the tracking concept remains the same.

The API may change, but the general workflow continues to be:

```text
Detector
   ↓
Detections
   ↓
Tracker
   ↓
Tracked Detections
   ↓
Annotation
```

The purpose of this lesson is therefore to understand the tracking workflow before moving to the newer API.

---

## Complete ByteTrack Workflow

The entire concept can be summarized as:

```text
Video Frame
    ↓
YOLO
    ↓
Detection Results
    ↓
sv.Detections
    ↓
Optional Filtering
    ↓
sv.ByteTrack
    ↓
update_with_detections()
    ↓
tracker_id
    ↓
Box Annotation
    ↓
Label Annotation
    ↓
Trace Annotation
    ↓
Tracked Frame
```

The process repeats for every frame.

---

## Key Takeaways

- ByteTrack is used to associate detections across video frames.
- YOLO performs object detection while ByteTrack performs object tracking.
- The lesson creates the tracker using `sv.ByteTrack()`.
- Supervision detections are passed to `tracker.update_with_detections()`.
- ByteTrack adds `tracker_id` information to detections.
- The same tracker must maintain state while consecutive frames are processed.
- `tracker.reset()` clears the current tracking state.
- Filtering can be performed before tracking.
- Tracking IDs can be displayed using `LabelAnnotator`.
- Tracking IDs can be used by `TraceAnnotator` to visualize trajectories.
- ByteTrack and YOLO work together as parts of the same video-processing pipeline.
- Later course material transitions from `sv.ByteTrack()` to the external `ByteTrackTracker` API.

---

## Next Concept

The next concept focuses on processing complete videos with:

```python
sv.process_video()
```

and understanding how a callback function performs detection, tracking, and annotation for every frame.
