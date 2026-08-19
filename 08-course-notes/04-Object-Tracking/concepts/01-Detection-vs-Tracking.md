# Detection vs. Object Tracking

Object detection and object tracking are related computer vision tasks, but they solve different problems.

Understanding this difference is the first step toward working with tracking in video.

---

## Object Detection

Object detection analyzes an image or video frame and determines:

- What objects are visible
- Where those objects are located
- What class each object belongs to
- How confident the model is about each detection

For example, YOLO may analyze a frame and detect:

```text
Frame 1

car
car
truck
person
```

Using Supervision, the YOLO results can be converted into:

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

The detections contain information such as:

```text
Bounding Boxes
Class IDs
Confidence Scores
```

However, detection alone does not maintain an object's identity across multiple frames.

---

## The Problem with Detection in Video

Consider two consecutive video frames.

```text
Frame 1

car
truck
car
```

Then:

```text
Frame 2

car
truck
car
```

The detector knows that both frames contain cars and trucks.

But it does not automatically know:

```text
Is the first car in Frame 1
the same car in Frame 2?
```

Each frame is being analyzed as a new detection problem.

This is where object tracking becomes necessary.

---

## Object Tracking

Object tracking connects detections across multiple video frames.

Instead of only identifying:

```text
car
car
truck
```

the system can assign individual identities:

```text
car   → ID 1
car   → ID 2
truck → ID 3
```

When the next frame is processed, the tracker attempts to maintain those identities:

```text
Frame 1

car   → ID 1
car   → ID 2
truck → ID 3
```

```text
Frame 2

car   → ID 1
car   → ID 2
truck → ID 3
```

The objects may move to different positions, but their IDs can remain associated with them.

---

## Basic Tracking Workflow

The tracking process can be represented as:

```text
Frame N
   ↓
Object Detection
   ↓
Detections without IDs
   ↓
Tracker
   ↓
Detections with tracker_id
```

Then the next frame is processed:

```text
Frame N+1
   ↓
Object Detection
   ↓
New Detections
   ↓
Tracker
   ↓
Persistent tracker_id
```

The tracker attempts to determine which detections correspond to objects that were already visible.

---

## Detection Answers "What?"

Object detection primarily answers:

```text
What object is this?
```

Examples:

```text
person
car
truck
bus
```

The object's category is represented by:

```python
class_id
```

For example:

```text
class_id = 2
```

can represent a car in the COCO dataset.

---

## Tracking Answers "Which One?"

Tracking answers a different question:

```text
Which specific object is this?
```

For example:

```text
car #1
car #2
car #3
```

All three objects belong to the same class:

```text
car
```

But each one represents a different tracked object.

---

## `class_id` vs. `tracker_id`

This is one of the most important distinctions in object tracking.

### `class_id`

The `class_id` tells us the object's category.

Example:

```text
Car A   → class_id 2
Car B   → class_id 2
Truck A → class_id 7
```

### `tracker_id`

The `tracker_id` tells us which individual object we are tracking.

Example:

```text
Car A   → tracker_id 1
Car B   → tracker_id 2
Truck A → tracker_id 3
```

Together:

```text
Object        class_id        tracker_id

Car A             2                1
Car B             2                2
Truck A           7                3
```

Two objects can therefore have the same `class_id` while having different `tracker_id` values.

---

## Tracking Introduces Time

Object detection can work with a single image.

```text
Image
  ↓
YOLO
  ↓
Detections
```

Tracking requires multiple frames.

```text
Frame 1
   ↓
Frame 2
   ↓
Frame 3
   ↓
Frame 4
   ↓
...
```

The tracker uses information from consecutive frames to maintain object identities.

---

## Detection Pipeline

Without tracking:

```text
Video
  ↓
Frame
  ↓
YOLO
  ↓
sv.Detections
  ↓
Annotation
```

Each frame is treated independently.

---

## Tracking Pipeline

With tracking:

```text
Video
  ↓
Frame
  ↓
YOLO
  ↓
sv.Detections
  ↓
Tracker
  ↓
tracker_id
  ↓
Annotation
```

The tracker adds information about object identity across time.

---

## Why Persistent IDs Are Useful

Persistent IDs allow us to perform tasks that are difficult with detection alone.

For example, imagine a traffic video.

Detection may report:

```text
Frame 1 → 8 vehicles
Frame 2 → 9 vehicles
Frame 3 → 8 vehicles
```

But this does not tell us whether the same vehicles are being detected repeatedly.

Tracking allows us to follow individual vehicles:

```text
Vehicle #1
Vehicle #2
Vehicle #3
Vehicle #4
...
```

This makes more advanced video analysis possible.

---

## Tracking and Object Trajectories

Once objects have IDs, their positions can be followed over time.

For example:

```text
Frame 1 → Car #3 at Position A
Frame 2 → Car #3 at Position B
Frame 3 → Car #3 at Position C
```

These positions form a trajectory.

Conceptually:

```text
Position A
    •
     \
      • Position B
       \
        • Position C
         \
          [Car #3]
```

Supervision can visualize these trajectories using:

```python
sv.TraceAnnotator()
```

---

## Detection and Tracking Work Together

Tracking does not replace object detection.

The detector first finds the objects.

The tracker then attempts to maintain their identities.

```text
Object Detector
      ↓
Find Objects
      ↓
Tracker
      ↓
Maintain Identities
```

In this lesson:

```text
YOLO
      ↓
Object Detection
      ↓
sv.Detections
      ↓
ByteTrack
      ↓
tracker_id
```

YOLO performs detection.

ByteTrack performs tracking.

---

## Tracking IDs Are Not Permanent Identities

A `tracker_id` represents an object's identity during a tracking sequence.

It is not a permanent real-world identity.

For example:

```text
Frame 1 → car #5
Frame 2 → car #5
Frame 3 → car #5
```

If the car disappears from the video for enough time, the tracker may lose it.

If it appears again later, it may receive another ID.

For example:

```text
Frame 1  → car #5
Frame 2  → car #5
Frame 3  → car #5

Object disappears

...

Frame 20 → car #11
```

Therefore, tracking IDs depend on the tracker maintaining the association between detections.

---

## Connection to Filtering

The previous lesson introduced filtering and manipulating detections.

For example:

```python
detections = detections[
    detections.class_id == TARGET_CLASS
]
```

Those concepts can now be combined with tracking.

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

For example, if YOLO detects:

```text
person
car
truck
car
bus
car
```

and we only want to track cars, we can filter first:

```text
car
car
car
```

Then those detections are sent to the tracker.

This directly connects:

```text
03 — Filtering and Manipulating Detections
```

with:

```text
04 — Object Tracking
```

---

## Simple Way to Remember

The difference can be summarized as:

```text
Detection:

"What objects are here?"
```

Tracking:

```text
"Which object is which over time?"
```

Or even more simply:

```text
Detection = WHAT

Tracking = WHICH ONE + OVER TIME
```

---

## Key Takeaways

- Object detection analyzes objects in individual frames.
- Detection alone does not maintain object identity across frames.
- Object tracking connects detections over time.
- Trackers assign IDs to individual objects.
- `class_id` represents the object's category.
- `tracker_id` represents the individual tracked object.
- Multiple objects can have the same `class_id` but different `tracker_id` values.
- Tracking requires information from consecutive video frames.
- YOLO performs detection while ByteTrack performs tracking.
- Detection and tracking work together in a video-processing pipeline.
- Persistent IDs make more advanced video analytics possible.

---

## Next Concept

The next concept is:

```python
detections.tracker_id
```

We will examine what `tracker_id` contains before tracking and how its value changes after detections are processed by ByteTrack.
