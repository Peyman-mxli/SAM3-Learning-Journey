# Understanding `tracker_id`

One of the most important concepts in object tracking is the:

```python
tracker_id
```

A `tracker_id` is an identification number assigned to an object by a tracker.

It allows us to recognize the same object across multiple video frames.

---

## Detection Before Tracking

When YOLO detects objects in a frame, it provides information such as:

- Bounding boxes
- Class IDs
- Confidence scores

Using Supervision, YOLO results can be converted into:

```python
sv.Detections
```

Example:

```python
results = model(
    frame,
    verbose=False
)[0]

detections = sv.Detections.from_ultralytics(
    results
)
```

At this point, the objects have been detected.

However, they have not yet been tracked.

---

## Inspecting `tracker_id`

Supervision detections contain a property called:

```python
detections.tracker_id
```

We can inspect it using:

```python
print(detections.tracker_id)
```

Before applying a tracker, the result is normally:

```text
None
```

This is expected.

YOLO is an object detector.

It does not automatically assign persistent tracking IDs.

---

## Why Is `tracker_id` Initially `None`?

Consider this pipeline:

```text
Frame
  ↓
YOLO
  ↓
sv.Detections
```

At this stage, the system knows:

```text
What objects were detected?
Where are they?
What classes are they?
How confident is the model?
```

But it does not yet know:

```text
Which specific object is this across time?
```

Therefore:

```python
detections.tracker_id
```

remains:

```text
None
```

until a tracker processes the detections.

---

## Adding a Tracker

In this lesson, we use:

```python
sv.ByteTrack()
```

Create the tracker:

```python
tracker = sv.ByteTrack()
```

Then pass the detections through the tracker:

```python
detections = tracker.update_with_detections(
    detections
)
```

Now inspect:

```python
print(detections.tracker_id)
```

Instead of:

```text
None
```

we may receive something similar to:

```text
[1 2 3 4]
```

Each number represents the identity assigned to a tracked object.

---

## Before and After Tracking

The difference can be represented clearly.

### Before ByteTrack

```text
Detection 1 → car
Detection 2 → car
Detection 3 → truck

tracker_id = None
```

### After ByteTrack

```text
Detection 1 → car   → tracker_id 1
Detection 2 → car   → tracker_id 2
Detection 3 → truck → tracker_id 3
```

Now the system can distinguish between individual objects.

---

## `tracker_id` Is Associated with Each Detection

Imagine that:

```python
detections.tracker_id
```

contains:

```text
[1, 2, 3, 4]
```

This means the detections have IDs associated with them.

Conceptually:

```text
Detection 0 → tracker_id 1
Detection 1 → tracker_id 2
Detection 2 → tracker_id 3
Detection 3 → tracker_id 4
```

These IDs can then be used when creating labels or analyzing object behavior.

---

## Persistent IDs Across Frames

The real value of `tracker_id` becomes visible when we process multiple frames.

For example:

```text
Frame 0

car   → ID 1
truck → ID 2
car   → ID 3
```

In the next frame:

```text
Frame 1

car   → ID 1
truck → ID 2
car   → ID 3
```

And again:

```text
Frame 2

car   → ID 1
truck → ID 2
car   → ID 3
```

The objects have changed position, but their tracker IDs remain associated with them when the tracker successfully recognizes them.

---

## Inspecting IDs Frame by Frame

A useful experiment from this lesson is to process several video frames manually.

First reset the tracker:

```python
tracker.reset()
```

Then open the video:

```python
cap = cv2.VideoCapture(
    "assets/vehicles.mp4"
)
```

Process several frames:

```python
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
```

Finally:

```python
cap.release()
```

This allows us to observe how tracker IDs behave across consecutive frames.

---

## Example Output

The output may conceptually look like:

```text
Frame 0: 4 objects | IDs: [1 2 3 4]

Frame 1: 4 objects | IDs: [1 2 3 4]

Frame 2: 4 objects | IDs: [1 2 3 4]
```

This would indicate that the tracker successfully maintained the same identities.

The exact IDs and number of objects depend on the detections in the video.

---

## What Happens When a New Object Appears?

Suppose the tracker currently knows:

```text
ID 1
ID 2
ID 3
```

Then a new vehicle enters the scene.

The tracker may assign a new ID:

```text
Frame N

car   → ID 1
truck → ID 2
car   → ID 3
```

Then:

```text
Frame N+1

car   → ID 1
truck → ID 2
car   → ID 3
bus   → ID 4
```

The new object receives its own identity.

---

## What Happens When an Object Disappears?

Objects can leave the frame or become temporarily invisible.

For example:

```text
Frame 1

car #1
car #2
car #3
```

Later:

```text
Frame 2

car #1
car #3
```

Object #2 may have:

- Left the frame
- Been hidden by another object
- Failed to be detected
- Become difficult for the tracker to associate

The tracker manages this information internally.

---

## IDs Are Not Guaranteed Forever

A tracker attempts to maintain IDs, but this does not mean an ID will remain attached to an object forever.

For example:

```text
Frame 1 → car #7
Frame 2 → car #7
Frame 3 → car #7
```

The car then disappears for many frames.

If it later returns, the tracker may assign:

```text
car #12
```

instead of:

```text
car #7
```

Therefore:

```python
tracker_id
```

should be understood as a tracking identity within the current tracking sequence.

---

## `class_id` and `tracker_id`

These two values serve different purposes.

### `class_id`

Represents:

```text
What type of object is this?
```

Example:

```text
car
```

### `tracker_id`

Represents:

```text
Which individual car is this?
```

Example:

```text
car #4
```

Consider three cars:

```text
Object        class_id        tracker_id

Car A             2                1
Car B             2                2
Car C             2                3
```

All three share the same class.

But each has a different tracking identity.

---

## Using `tracker_id` in Labels

Tracker IDs can be displayed directly on the video.

For example:

```python
labels = [
    f"ID:{tracker_id}"
    for tracker_id in detections.tracker_id
]
```

This may produce:

```text
ID:1
ID:2
ID:3
ID:4
```

The labels can then be drawn using:

```python
sv.LabelAnnotator()
```

---

## Combining Class Name and Tracker ID

A more informative label combines the object's class and identity.

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

The output may look like:

```text
car #1
car #2
truck #3
bus #4
```

Now the annotation tells us both:

```text
WHAT the object is
```

and:

```text
WHICH individual object it is
```

---

## `tracker_id` and Object Traces

Tracker IDs are also important for drawing object trajectories.

Supervision provides:

```python
sv.TraceAnnotator()
```

The trace annotator needs tracking information to determine which historical positions belong to the same object.

For example:

```text
Frame 1 → Car #5 at Position A

Frame 2 → Car #5 at Position B

Frame 3 → Car #5 at Position C
```

Because all three detections have:

```text
tracker_id = 5
```

the system can connect those positions.

Conceptually:

```text
A •
    \
     • B
       \
        • C
         \
          Car #5
```

---

## `tracker_id` Enables Video Analytics

Once individual objects have IDs, we can build additional logic around them.

For example:

### Count Unique Objects

We can identify unique tracker IDs:

```text
ID 1
ID 2
ID 3
ID 4
```

instead of repeatedly counting the same objects in every frame.

---

### Count Frames Visible

We can maintain a dictionary:

```python
frame_count = {}
```

and update it for every tracked object:

```python
for tracker_id in detections.tracker_id:

    frame_count[tracker_id] = (
        frame_count.get(tracker_id, 0) + 1
    )
```

Then we could display:

```text
#1 (15f)
#2 (27f)
#3 (8f)
```

where:

```text
f = number of frames visible
```

---

### Follow Movement

The same ID allows us to follow an object's position:

```text
ID 3

Frame 1 → Position A
Frame 2 → Position B
Frame 3 → Position C
Frame 4 → Position D
```

This information can later be used for movement analysis.

---

## Resetting Tracker IDs

The tracker stores state from previous frames.

When beginning a new tracking experiment, the lesson uses:

```python
tracker.reset()
```

This resets the tracking state.

Conceptually:

```text
Previous Tracking Session
        ↓
tracker.reset()
        ↓
New Tracking Session
```

This is useful when restarting video processing from the beginning.

---

## Complete `tracker_id` Flow

The complete process can be summarized as:

```text
Video Frame
    ↓
YOLO
    ↓
Detections
    ↓
sv.Detections
    ↓
tracker_id = None
    ↓
ByteTrack
    ↓
update_with_detections()
    ↓
tracker_id assigned
    ↓
Tracked Detections
    ↓
Labels / Traces / Analytics
```

---

## Key Takeaways

- `tracker_id` identifies an individual tracked object.
- Before tracking, `detections.tracker_id` is normally `None`.
- YOLO performs detection but does not assign persistent tracking IDs.
- ByteTrack assigns tracking information to detections.
- The tracker attempts to preserve the same ID across consecutive frames.
- `class_id` identifies the object's category.
- `tracker_id` identifies the specific object.
- Tracker IDs can be displayed using labels.
- Tracker IDs allow object trajectories to be drawn.
- Tracker IDs can be used for higher-level video analytics.
- Tracker state can be cleared using `tracker.reset()`.
- Tracking IDs are identities within a tracking sequence, not permanent real-world identities.

---

## Next Concept

The next concept is:

```python
sv.ByteTrack()
```

We will examine how ByteTrack receives `sv.Detections`, associates objects between frames, and updates detections with persistent tracking IDs.
