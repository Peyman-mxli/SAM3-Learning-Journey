# Filtering Before Tracking

Object tracking does not require us to track every object detected by the model.

In many computer vision applications, we are interested in tracking only specific objects.

For example:

- Only cars
- Only people
- Only trucks
- Only objects above a confidence threshold
- Only objects inside a particular region

The filtering concepts from the previous lesson can therefore be combined directly with object tracking.

The important idea is:

```text
Detect
   ↓
Filter
   ↓
Track
```

rather than tracking every detection automatically.

---

## Connection to the Previous Lesson

In the previous lesson, we learned how to manipulate:

```python
sv.Detections
```

using boolean masks.

For example:

```python
detections = detections[
    detections.class_id == TARGET_CLASS
]
```

This allows us to keep only detections belonging to a specific class.

Now we can place ByteTrack after that operation:

```python
detections = tracker.update_with_detections(
    detections
)
```

The complete pipeline becomes:

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

## Why Filter Before Tracking?

Imagine YOLO detects:

```text
person
car
truck
car
bus
car
person
bicycle
```

Suppose our application is designed to analyze only cars.

There is no need to send every detected object to the tracker.

Instead, we can filter the detections first.

```text
Original Detections

person
car
truck
car
bus
car
person
bicycle
```

After filtering:

```text
Filtered Detections

car
car
car
```

Then ByteTrack receives:

```text
car
car
car
```

instead of all detected classes.

---

## Basic Filtering Pipeline

The general workflow is:

```text
Frame
  ↓
YOLO
  ↓
Detection Results
  ↓
sv.Detections
  ↓
Apply Filter
  ↓
Filtered Detections
  ↓
ByteTrack
  ↓
Tracked Objects
```

This combines two important parts of the learning journey:

```text
Filtering and Manipulating Detections
                +
          Object Tracking
```

---

## Filtering by Class

One of the simplest filters is a class filter.

Suppose we want to track cars.

For the COCO dataset:

```python
TARGET_CLASS = 2
```

represents:

```text
car
```

We can filter the detections using:

```python
detections = detections[
    detections.class_id == TARGET_CLASS
]
```

Then apply tracking:

```python
detections = tracker.update_with_detections(
    detections
)
```

---

## Complete Class Filtering Example

The basic sequence is:

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

The order is important.

```text
1. Detect
2. Convert
3. Filter
4. Track
```

---

## Before Filtering

Imagine YOLO produces:

```text
Detection 1 → person
Detection 2 → car
Detection 3 → truck
Detection 4 → car
Detection 5 → bus
Detection 6 → car
```

At this point:

```python
detections.class_id
```

contains several different classes.

---

## Applying the Boolean Mask

The expression:

```python
detections.class_id == TARGET_CLASS
```

creates a boolean mask.

Conceptually:

```text
person → False
car    → True
truck  → False
car    → True
bus    → False
car    → True
```

The mask might look like:

```text
[False, True, False, True, False, True]
```

Then:

```python
detections[
    detections.class_id == TARGET_CLASS
]
```

keeps only the detections where the mask is:

```text
True
```

---

## After Filtering

The resulting detections contain only:

```text
car
car
car
```

These detections are then passed to:

```python
tracker.update_with_detections()
```

ByteTrack now tracks only the selected class.

---

## Tracking Only Cars

A callback for tracking only cars can look like:

```python
TARGET_CLASS = 2

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

---

## Processing the Filtered Video

Before starting the new tracking sequence:

```python
tracker.reset()
```

Then process the video:

```python
sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_cars.mp4",
    callback=callback_only_cars,
    show_progress=True
)
```

The resulting video contains tracking annotations only for the selected class.

---

## Why Filtering Happens Before ByteTrack

Consider two possible pipelines.

### Pipeline A

```text
Detection
   ↓
Tracking
   ↓
Filtering
```

### Pipeline B

```text
Detection
   ↓
Filtering
   ↓
Tracking
```

For this lesson, the useful approach is:

```text
Detection
   ↓
Filtering
   ↓
Tracking
```

The tracker receives only the objects relevant to the application.

---

## Filtering Reduces the Tracking Scope

Suppose YOLO detects 15 objects:

```text
15 Total Detections
```

but only 5 are cars.

Filtering first gives:

```text
15 detections
     ↓
Car Filter
     ↓
5 detections
     ↓
ByteTrack
```

ByteTrack then focuses on those selected detections.

---

## Combining Filtering and Tracking IDs

After filtering, each tracked car can receive its own:

```python
tracker_id
```

For example:

```text
car #1
car #2
car #3
car #4
```

All of them have the same object class:

```text
car
```

but different tracking identities.

Conceptually:

```text
Object        class_id        tracker_id

Car A             2                1
Car B             2                2
Car C             2                3
Car D             2                4
```

---

## Filtering Does Not Replace Tracking

Filtering and tracking solve different problems.

### Filtering

Answers:

```text
Which detections do I want to keep?
```

### Tracking

Answers:

```text
Which individual object is this across frames?
```

Together:

```text
Filtering
   ↓
Choose relevant objects

Tracking
   ↓
Maintain their identities
```

---

## Filtering by Confidence

The filtering concepts are not limited to object classes.

For example, detections can also be filtered by confidence:

```python
CONFIDENCE_THRESHOLD = 0.50

detections = detections[
    detections.confidence >= CONFIDENCE_THRESHOLD
]
```

Then:

```python
detections = tracker.update_with_detections(
    detections
)
```

The tracker now receives only detections with sufficient confidence.

---

## Confidence Filtering Pipeline

Conceptually:

```text
YOLO
  ↓
All Detections
  ↓
Confidence >= 0.50
  ↓
Reliable Detections
  ↓
ByteTrack
```

This can help focus the tracking pipeline on detections that meet the application's confidence requirement.

---

## Combining Class and Confidence Filters

Filters can also be combined.

For example:

```python
TARGET_CLASS = 2
CONFIDENCE_THRESHOLD = 0.50
```

Create a mask:

```python
mask = (
    (detections.class_id == TARGET_CLASS)
    &
    (detections.confidence >= CONFIDENCE_THRESHOLD)
)
```

Apply it:

```python
detections = detections[
    mask
]
```

Then track:

```python
detections = tracker.update_with_detections(
    detections
)
```

The pipeline becomes:

```text
All Detections
      ↓
Class = Car
      +
Confidence >= 0.50
      ↓
Filtered Cars
      ↓
ByteTrack
```

---

## Reusing Previous Filtering Knowledge

The previous lesson introduced several filtering strategies.

Those same concepts can be placed before tracking.

Examples include:

```text
Class Filtering
Confidence Filtering
Size Filtering
Position Filtering
Combined Boolean Masks
```

The general idea remains:

```python
detections = detections[
    FILTER_CONDITION
]

detections = tracker.update_with_detections(
    detections
)
```

---

## Filtering by Object Size

For some applications, we may want to ignore very small detections.

For example:

```python
widths = (
    detections.xyxy[:, 2] -
    detections.xyxy[:, 0]
)

heights = (
    detections.xyxy[:, 3] -
    detections.xyxy[:, 1]
)

areas = widths * heights
```

Then:

```python
MIN_AREA = 5000

detections = detections[
    areas >= MIN_AREA
]
```

After filtering:

```python
detections = tracker.update_with_detections(
    detections
)
```

The tracker receives only objects that satisfy the size requirement.

---

## Filtering by Position

We can also select objects based on their position in the frame.

For example, suppose we only want detections whose center is on the right half of the image.

Conceptually:

```text
Video Frame

┌────────────────────────────┐
│            │               │
│   Ignore   │     Track     │
│            │               │
│            │               │
└────────────────────────────┘
             ↑
        Frame Center
```

The detections can first be filtered by position and then passed to ByteTrack.

---

## Multiple Filters Before Tracking

A more advanced pipeline could use several filters.

For example:

```text
YOLO
  ↓
Confidence Filter
  ↓
Class Filter
  ↓
Size Filter
  ↓
Position Filter
  ↓
ByteTrack
```

This allows us to define exactly which detections should participate in tracking.

---

## Example Application

Imagine we are building a traffic-monitoring system.

Our goal is:

```text
Track only cars
with confidence >= 0.50
```

The pipeline becomes:

```text
Traffic Video
     ↓
YOLO
     ↓
All Detections
     ↓
Class = Car
     ↓
Confidence >= 0.50
     ↓
ByteTrack
     ↓
Tracked Cars
     ↓
car #1
car #2
car #3
...
```

This is more focused than tracking every object in the scene.

---

## Filtering and Annotation

After filtering and tracking, the remaining detections can be annotated normally.

```python
labels = [
    f"car #{tracker_id}"
    for tracker_id in detections.tracker_id
]
```

Then:

```python
scene = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections
)
```

and:

```python
scene = label_annotator.annotate(
    scene=scene,
    detections=detections,
    labels=labels
)
```

Only the filtered and tracked objects appear in the final annotation.

---

## Adding Traces

Because the filtered detections have been processed by ByteTrack, they contain:

```python
tracker_id
```

This means we can also use:

```python
sv.TraceAnnotator()
```

Example:

```python
scene = trace_annotator.annotate(
    scene=scene,
    detections=detections
)
```

Now the output can show the trajectories of only the selected objects.

---

## Complete Filtered Tracking Pipeline

A complete workflow can therefore look like:

```text
Input Video
     ↓
Current Frame
     ↓
YOLO
     ↓
sv.Detections
     ↓
Class Filter
     ↓
Confidence Filter
     ↓
Filtered Detections
     ↓
ByteTrack
     ↓
tracker_id
     ↓
Create Labels
     ↓
BoxAnnotator
     ↓
LabelAnnotator
     ↓
TraceAnnotator
     ↓
Output Frame
     ↓
Output Video
```

---

## Why This Combination Is Important

This lesson connects two major computer vision skills.

The previous lesson taught us how to control:

```text
Which detections are selected?
```

The current lesson teaches us how to determine:

```text
Which selected detection belongs to which object over time?
```

Together:

```text
Detection
   ↓
Manipulation
   ↓
Tracking
   ↓
Video Analytics
```

This is an important step toward building practical computer vision pipelines.

---

## Practical Applications

Filtering before tracking can be useful in many applications.

### Traffic Monitoring

Track only:

```text
cars
trucks
buses
```

while ignoring unrelated classes.

### People Analytics

Track only:

```text
person
```

detections.

### Industrial Monitoring

Track only the product or component relevant to the production process.

### Region Monitoring

Track objects only after they enter a specific part of the image.

### Confidence-Based Tracking

Ignore detections that do not meet a chosen confidence threshold.

---

## Key Takeaways

- We do not need to track every object detected by YOLO.
- `sv.Detections` can be filtered before being passed to ByteTrack.
- Class filtering can be used to track only selected object categories.
- Confidence filtering can remove detections below a chosen threshold.
- Previous filtering concepts can be reused directly in tracking pipelines.
- Multiple filters can be combined before tracking.
- ByteTrack receives the filtered detections and assigns `tracker_id` values.
- Filtering determines which objects we care about.
- Tracking determines which individual object is which across frames.
- Filtered tracked objects can still use boxes, labels, and traces.
- Combining filtering with tracking creates more focused video-analysis pipelines.

---

## Next Concept

The next concept focuses on using persistent tracking IDs for simple analytics.

We will examine how:

```python
tracker_id
```

can be used to count how many frames an individual object remains visible and how tracking information can become the foundation for higher-level video analytics.
