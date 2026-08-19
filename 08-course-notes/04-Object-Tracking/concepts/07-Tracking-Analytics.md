# Tracking Analytics

Object tracking becomes especially powerful when tracking IDs are used for more than visualization.

Once ByteTrack assigns a persistent:

```python
tracker_id
```

to an object, that ID can be used to collect information about the object across multiple frames.

This creates the foundation for:

```text
Tracking Analytics
```

Instead of only displaying:

```text
car #1
car #2
car #3
```

we can begin asking questions such as:

```text
How long has car #1 been visible?

How many unique objects have appeared?

Where has object #3 moved?
```

---

## From Tracking to Analytics

The basic tracking pipeline is:

```text
Video
  ↓
YOLO
  ↓
sv.Detections
  ↓
ByteTrack
  ↓
tracker_id
```

Once we have:

```python
tracker_id
```

we can add another stage:

```text
Video
  ↓
YOLO
  ↓
sv.Detections
  ↓
ByteTrack
  ↓
tracker_id
  ↓
Tracking Analytics
```

The tracking ID becomes the key that connects information about the same object across frames.

---

## Why `tracker_id` Is Important for Analytics

Imagine that a car appears in 20 consecutive frames.

Without tracking, we simply receive:

```text
Frame 1 → car
Frame 2 → car
Frame 3 → car
...
Frame 20 → car
```

We do not automatically know whether those detections belong to the same car.

With tracking:

```text
Frame 1 → car #5
Frame 2 → car #5
Frame 3 → car #5
...
Frame 20 → car #5
```

Now we know that:

```text
tracker_id = 5
```

has appeared repeatedly.

That allows us to collect information specifically for object #5.

---

## Extension Challenge — Frames Visible

One simple tracking analytics experiment is to count how many frames each tracked object remains visible.

For example:

```text
Object #1 → 15 frames
Object #2 → 8 frames
Object #3 → 32 frames
```

We can store this information using a Python dictionary.

---

## Creating the Dictionary

Start with:

```python
frame_count = {}
```

This dictionary will store:

```text
tracker_id → number of frames visible
```

Conceptually:

```text
{
    1: 15,
    2: 8,
    3: 32
}
```

---

## Updating the Frame Count

After ByteTrack processes the detections:

```python
detections = tracker.update_with_detections(
    detections
)
```

we can iterate over:

```python
detections.tracker_id
```

Example:

```python
for tracker_id in detections.tracker_id:

    frame_count[tracker_id] = (
        frame_count.get(tracker_id, 0) + 1
    )
```

Each time an object appears in another processed frame, its counter increases by one.

---

## Understanding `dict.get()`

The expression:

```python
frame_count.get(tracker_id, 0)
```

means:

```text
Get the current value for this tracker ID.

If the tracker ID does not exist yet,
start with 0.
```

Then:

```python
+ 1
```

adds one frame.

---

## Example

Suppose:

```text
Frame 1

ID 1
ID 2
ID 3
```

After processing the frame:

```text
frame_count = {
    1: 1,
    2: 1,
    3: 1
}
```

Then:

```text
Frame 2

ID 1
ID 2
ID 3
```

The dictionary becomes:

```text
frame_count = {
    1: 2,
    2: 2,
    3: 2
}
```

---

## Object Leaving the Frame

Suppose the next frame contains only:

```text
Frame 3

ID 1
ID 3
```

Now:

```text
frame_count = {
    1: 3,
    2: 2,
    3: 3
}
```

Object #2 did not appear in Frame 3, so its count remains:

```text
2
```

---

## Creating Labels with Frame Counts

The frame count can be included in the tracking label.

Instead of:

```text
car #1
```

we could display:

```text
#1 (15f)
```

where:

```text
f = frames
```

For example:

```text
#1 (15f)
#2 (8f)
#3 (32f)
```

---

## Creating the Label

A label could be constructed using:

```python
labels = [
    f"#{tracker_id} ({frame_count[tracker_id]}f)"
    for tracker_id in detections.tracker_id
]
```

If:

```python
frame_count
```

contains:

```text
{
    1: 15,
    2: 8,
    3: 32
}
```

the labels become:

```text
#1 (15f)
#2 (8f)
#3 (32f)
```

---

## Adding the Object Class

We can also combine:

```python
class_id
```

with:

```python
tracker_id
```

and the frame count.

For example:

```python
labels = [
    f"{results.names[class_id]} "
    f"#{tracker_id} "
    f"({frame_count[tracker_id]}f)"
    for class_id, tracker_id in zip(
        detections.class_id,
        detections.tracker_id
    )
]
```

The output may look like:

```text
car #1 (15f)
truck #2 (8f)
car #3 (32f)
```

Now each label contains three pieces of information:

```text
Object Class
     +
Tracker ID
     +
Frames Visible
```

---

## Complete Frame Counting Flow

The workflow becomes:

```text
Frame
  ↓
YOLO
  ↓
sv.Detections
  ↓
ByteTrack
  ↓
tracker_id
  ↓
Update frame_count
  ↓
Create Labels
  ↓
Annotate
```

---

## Example Callback

A callback could contain:

```python
frame_count = {}

def process_frame(
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

    detections = tracker.update_with_detections(
        detections
    )

    for tracker_id in detections.tracker_id:

        frame_count[tracker_id] = (
            frame_count.get(
                tracker_id,
                0
            ) + 1
        )

    labels = [
        f"#{tracker_id} "
        f"({frame_count[tracker_id]}f)"
        for tracker_id
        in detections.tracker_id
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

This transforms the tracking IDs into simple analytics.

---

## Resetting the Analytics

When starting a new tracking experiment, we should reset both:

```python
tracker
```

and the analytics data.

For example:

```python
tracker.reset()

frame_count = {}
```

Conceptually:

```text
Previous Tracking Session
        ↓
Reset Tracker
        +
Reset Analytics
        ↓
New Tracking Session
```

This prevents data from an earlier video-processing run from being mixed with a new run.

---

## Counting Unique Objects

Another possible use of tracker IDs is counting unique tracked objects.

Suppose the tracker has produced:

```text
ID 1
ID 2
ID 3
ID 4
```

We can store the IDs in a set.

For example:

```python
unique_ids = set()
```

Then:

```python
for tracker_id in detections.tracker_id:
    unique_ids.add(tracker_id)
```

The total number of unique IDs can be calculated using:

```python
len(unique_ids)
```

---

## Example

Suppose:

```text
Frame 1 → [1, 2, 3]

Frame 2 → [1, 2, 3]

Frame 3 → [1, 2, 3, 4]
```

A normal detection count across all frames could count:

```text
3 + 3 + 4 = 10 detections
```

But the unique tracking IDs are:

```text
{1, 2, 3, 4}
```

Therefore:

```text
Unique tracked IDs = 4
```

This illustrates why persistent IDs are useful for video analytics.

---

## Important Limitation of Unique ID Counting

A tracker ID is not necessarily a permanent real-world identity.

An object can sometimes receive a new ID if tracking is lost.

For example:

```text
Frame 1 → car #5
Frame 2 → car #5
Frame 3 → car #5

Object disappears

...

Frame 20 → car #11
```

A simple unique-ID counter may interpret:

```text
#5
```

and:

```text
#11
```

as two different tracked objects.

Therefore, tracker-ID counting should be understood within the limitations of the tracker.

---

## Tracking Object Movement

Tracker IDs can also be used to associate positions with specific objects.

For example:

```text
ID 7

Frame 1 → Position A
Frame 2 → Position B
Frame 3 → Position C
Frame 4 → Position D
```

These positions describe the object's movement.

Conceptually:

```text
A
•
 \
  • B
   \
    • C
     \
      • D
```

This is the same principle used by:

```python
sv.TraceAnnotator()
```

to draw object trajectories.

---

## Position History

A more advanced analytics system could store the position history for each tracker ID.

Conceptually:

```text
tracker_id 1
    ↓
Position A
Position B
Position C

tracker_id 2
    ↓
Position D
Position E
Position F
```

This information can later support:

- Movement analysis
- Direction estimation
- Path visualization
- Zone interaction analysis

---

## Tracking Duration

The frame count can also be converted into approximate time.

If the video FPS is known:

```python
video_info.fps
```

and an object has been visible for:

```text
frames_visible
```

then the approximate visible duration can be calculated as:

```text
Visible Time =
Frames Visible
──────────────
     FPS
```

In Python:

```python
visible_seconds = (
    frames_visible /
    video_info.fps
)
```

---

## Example

Suppose:

```text
frames_visible = 90

fps = 30
```

Then:

```text
90 / 30 = 3 seconds
```

The object was observed for approximately:

```text
3 seconds
```

during the processed frames.

---

## From Frame Counts to Video Analytics

The simple dictionary:

```python
frame_count = {}
```

demonstrates a much larger idea.

Once objects have persistent IDs, we can attach information to those IDs.

Conceptually:

```text
tracker_id
    ↓
Object History
    ↓
Analytics
```

For example:

```text
Tracker ID
    ↓
Frames Visible
    ↓
Time Visible
    ↓
Positions
    ↓
Trajectory
    ↓
Events
```

---

## Possible Analytics Built on Tracking

Persistent IDs can become the foundation for many types of analysis.

Examples include:

### Unique Object Counting

```text
How many different tracked objects appeared?
```

### Time Visible

```text
How long did each object remain visible?
```

### Movement Trajectories

```text
Where did each object move?
```

### Entry and Exit Events

```text
When did an object enter or leave an area?
```

### Direction Analysis

```text
Which direction did the object travel?
```

### Zone Interaction

```text
Did the tracked object enter a specific region?
```

These are natural extensions of the tracking concepts introduced in this lesson.

---

## Example — Traffic Analytics

Imagine a traffic-monitoring application.

YOLO detects vehicles:

```text
car
car
truck
bus
```

ByteTrack assigns:

```text
car #1
car #2
truck #3
bus #4
```

Analytics can then store:

```text
car #1
Frames Visible: 150

car #2
Frames Visible: 72

truck #3
Frames Visible: 210

bus #4
Frames Visible: 95
```

This is much more informative than simply knowing how many detections exist in each frame.

---

## Detection Count vs. Unique Object Count

These two concepts should not be confused.

### Detection Count

Counts objects detected in individual frames.

For example:

```text
Frame 1 → 5 detections
Frame 2 → 5 detections
Frame 3 → 5 detections
```

This gives:

```text
15 detection occurrences
```

### Unique Tracking IDs

Attempts to identify individual tracked objects.

For example:

```text
ID 1
ID 2
ID 3
ID 4
ID 5
```

This gives:

```text
5 unique tracker IDs
```

Tracking allows us to reason about objects across time instead of treating every detection as completely independent.

---

## Tracking Analytics Pipeline

The complete conceptual pipeline becomes:

```text
Input Video
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
Object History
     ↓
Frame Counts
     ↓
Unique IDs
     ↓
Movement Information
     ↓
Video Analytics
```

---

## Why This Matters for Computer Vision

Detection tells us:

```text
What exists right now?
```

Tracking tells us:

```text
Which object is which over time?
```

Tracking analytics allows us to ask:

```text
What happened to each object over time?
```

This represents an important progression:

```text
Detection
    ↓
Tracking
    ↓
Understanding Behavior
```

---

## Key Takeaways

- `tracker_id` can be used for more than displaying labels.
- Tracking IDs allow information to be associated with individual objects across frames.
- A dictionary can store the number of frames each object remains visible.
- Frame counts can be displayed directly in tracking labels.
- Tracker IDs can be stored in a set to count unique tracked IDs.
- Frame counts can be converted into approximate visible time using video FPS.
- Object positions can be associated with tracker IDs to create movement histories.
- Tracking information can become the foundation for higher-level video analytics.
- Tracker IDs are not guaranteed permanent real-world identities.
- Tracking loss can cause the same physical object to receive another ID.
- Detection counts and unique tracker-ID counts represent different information.
- Persistent IDs make it possible to move from object detection toward behavior analysis.

---

## Next Concept

The next concept focuses on the transition from the Supervision ByteTrack API:

```python
sv.ByteTrack()
```

to the newer tracking approach introduced in the course:

```python
from trackers import ByteTrackTracker
```

We will examine what changes in the API and which object-tracking concepts remain the same.
