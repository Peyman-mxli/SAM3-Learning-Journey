# Tracking with Zones

## Introduction

Object tracking and spatial zones work together to transform individual detections into meaningful video analytics.

Object detection answers:

> What objects are visible?

Object tracking answers:

> Which object is which across multiple frames?

Zones answer:

> Where are those tracked objects?

Combining these concepts allows us to understand how individual objects interact with specific areas and boundaries over time.

The basic idea is:

```text
Detection
    +
Tracking
    +
Zones
    =
Spatial Object Analytics
```

---

# 1. Why Tracking Is Important

Object detectors such as YOLO analyze each frame independently.

For example:

```text
Frame 100 → car
Frame 101 → car
Frame 102 → car
Frame 103 → car
```

Without tracking, the system does not automatically know whether these detections represent:

```text
one car across four frames
```

or:

```text
four different cars
```

Object tracking solves this problem by assigning persistent identities.

---

# 2. Persistent Object Identity

After tracking, the same object can maintain the same ID across multiple frames.

Example:

```text
Frame 100 → Car ID:7
Frame 101 → Car ID:7
Frame 102 → Car ID:7
Frame 103 → Car ID:7
```

Now the system understands that all four detections represent the same physical object.

This persistent identity is stored in:

```python
detections.tracker_id
```

---

# 3. ByteTrack

In this session, object tracking is performed using ByteTrack.

The tracker is created with:

```python
from trackers import ByteTrackTracker

tracker = ByteTrackTracker()
```

The detections are then passed to the tracker:

```python
detections = tracker.update(
    detections
)
```

After this step, tracked detections can contain persistent tracker IDs.

---

# 4. Detection Before Tracking

Tracking requires detections.

The detector first identifies objects in the current frame.

Example:

```python
results = model(
    frame,
    verbose=False
)[0]
```

YOLO produces detection results.

These results are converted into Supervision detections:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

The tracker then processes them:

```python
detections = tracker.update(
    detections
)
```

---

# 5. Correct Processing Order

One of the most important rules from this session is:

> Tracking should happen before zone analysis.

The correct pipeline is:

```text
Video Frame
    ↓
YOLO Detection
    ↓
Supervision Detections
    ↓
ByteTrack
    ↓
Tracked Detections
    ↓
Zone Analysis
    ↓
Counting
    ↓
Annotation
```

In code:

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

# 6. Why the Tracker Comes Before `trigger()`

Zone analysis becomes more useful when detections contain persistent identities.

For example:

```text
ID:7
```

may move through the scene like this:

```text
Frame 1

ID:7
  ↓

┌──────────────────────┐
│     POLYGON ZONE     │
│                      │
└──────────────────────┘
```

Later:

```text
Frame 20

┌──────────────────────┐
│     POLYGON ZONE     │
│        ID:7          │
│          ↓           │
└──────────────────────┘
```

The system knows that the object entering the polygon is still `ID:7`.

---

# 7. Tracking with PolygonZone

`PolygonZone` determines whether tracked detections are currently inside a polygon.

Example:

```python
inside_zone = zone.trigger(
    detections=detections
)
```

The returned Boolean mask can then filter the tracked detections:

```python
detections_inside_zone = detections[
    inside_zone
]
```

This creates a collection containing only tracked objects associated with the zone.

---

# 8. Tracking IDs Inside a Polygon

Suppose the tracker produces:

```text
ID:2
ID:5
ID:8
ID:11
```

The polygon trigger returns:

```text
[True, False, True, False]
```

The result becomes:

```text
Objects inside zone:

ID:2
ID:8
```

The spatial filter therefore preserves the identities of the selected objects.

---

# 9. Creating Tracker Labels

Tracked objects can be labeled using their IDs.

Example:

```python
labels = [
    f"ID:{tracker_id}"
    for tracker_id in detections_inside_zone.tracker_id
]
```

The final visualization might display:

```text
┌───────────────┐
│     Car       │
│     ID:7      │
└───────────────┘
```

This makes it easier to observe how specific objects move through the zone.

---

# 10. Tracking with LineZone

Tracking becomes especially important when using `LineZone`.

A line crossing is a movement event.

The system must determine whether the same object moved from one side of the line to the other.

Consider:

```text
Frame 100

       ID:12
         ↓

════════════════════
    COUNTING LINE
════════════════════
```

Then:

```text
Frame 105

════════════════════
    COUNTING LINE
════════════════════

         ↓
       ID:12
```

Because the tracker maintained the identity:

```text
ID:12
```

the system can interpret this as a crossing event.

---

# 11. Without Tracking

Without tracking, the frames might look conceptually like:

```text
Frame 100 → car
Frame 101 → car
Frame 102 → car
Frame 103 → car
```

The system lacks persistent identity.

This makes movement-based reasoning much harder because it cannot easily determine whether the detections belong to the same object.

---

# 12. With Tracking

With tracking:

```text
Frame 100 → Car ID:12
Frame 101 → Car ID:12
Frame 102 → Car ID:12
Frame 103 → Car ID:12
```

Now movement can be analyzed over time.

Conceptually:

```text
Object Position at Time A
        +
Same Object Position at Time B
        ↓
Movement
        ↓
Possible Crossing Event
```

---

# 13. LineZone Trigger

Once tracking has been applied:

```python
line_zone.trigger(
    detections=detections
)
```

can evaluate the tracked objects.

The counters are then available through:

```python
line_zone.in_count
line_zone.out_count
```

These represent accumulated crossing events.

---

# 14. One Tracker Per Video Sequence

A tracker maintains state across frames.

For this reason, the same tracker instance should normally process the entire video sequence.

Example:

```python
tracker = ByteTrackTracker()

def callback(frame, frame_index):

    detections = ...

    detections = tracker.update(
        detections
    )

    return frame
```

The tracker is created outside the callback.

---

# 15. Why Not Recreate the Tracker Every Frame?

This would be incorrect:

```python
def callback(frame, frame_index):

    tracker = ByteTrackTracker()

    detections = tracker.update(
        detections
    )
```

Creating a new tracker for every frame destroys the temporal tracking history.

The tracker would repeatedly start from a new state.

Instead:

```python
tracker = ByteTrackTracker()

def callback(frame, frame_index):

    detections = tracker.update(
        detections
    )
```

allows the tracker to maintain identities across the sequence.

---

# 16. Tracker State

Tracking is a **stateful** process.

Conceptually:

```text
Frame 1
   ↓
Tracker State
   ↓
Frame 2
   ↓
Updated Tracker State
   ↓
Frame 3
   ↓
Updated Tracker State
   ↓
...
```

The tracker remembers information about previously detected objects.

This allows persistent identities to be maintained.

---

# 17. Zone State

Zones may also maintain state.

For example:

```python
zone.current_count
```

stores the current polygon occupancy.

Meanwhile:

```python
line_zone.in_count
line_zone.out_count
```

store accumulated crossing information.

Therefore, a video analytics pipeline contains multiple stateful components.

---

# 18. Stateful Video Pipeline

A simplified architecture is:

```text
                  VIDEO FRAMES
                       ↓
                     YOLO
                       ↓
                   Detections
                       ↓
                 ┌───────────┐
                 │  Tracker  │
                 │   State   │
                 └─────┬─────┘
                       ↓
              Tracked Detections
                       ↓
          ┌────────────┴────────────┐
          │                         │
          ↓                         ↓
   PolygonZone                  LineZone
       State                      State
          │                         │
          ↓                         ↓
   Current Count              Crossing Count
```

This is more advanced than processing each image independently.

---

# 19. Tracking + PolygonZone Pipeline

A complete polygon pipeline can be represented as:

```text
Frame
  ↓
YOLO
  ↓
Detections
  ↓
ByteTrack
  ↓
Tracked Detections
  ↓
PolygonZone.trigger()
  ↓
Boolean Mask
  ↓
Filter Detections
  ↓
Tracked Objects Inside Zone
  ↓
Annotation
```

---

# 20. Tracking + LineZone Pipeline

A complete line counting pipeline can be represented as:

```text
Frame
  ↓
YOLO
  ↓
Detections
  ↓
ByteTrack
  ↓
Tracked Detections
  ↓
LineZone.trigger()
  ↓
Crossing Analysis
  ↓
Update in_count / out_count
  ↓
Annotation
```

---

# 21. Combining Both Zones

The same tracked detections can be used by both zone systems.

Example:

```python
detections = tracker.update(
    detections
)

zone.trigger(
    detections=detections
)

line_zone.trigger(
    detections=detections
)
```

There is no need to detect or track the objects separately for each zone.

---

# 22. Efficient Pipeline Design

A better architecture is:

```text
                 YOLO
                  ↓
              Detections
                  ↓
               Tracker
                  ↓
          Tracked Detections
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
  PolygonZone           LineZone
```

Rather than:

```text
YOLO → Tracker → PolygonZone

YOLO → Tracker → LineZone
```

The shared tracked detections can feed multiple analytics components.

---

# 23. Multiple Zones

The architecture can be extended further.

For example:

```text
                  Tracked Detections
                         ↓
       ┌─────────────────┼─────────────────┐
       ↓                 ↓                 ↓
 PolygonZone A     PolygonZone B       LineZone
       ↓                 ↓                 ↓
 Lane A Count       Lane B Count      Crossings
```

This allows a single camera to monitor several regions simultaneously.

---

# 24. Tracking Errors

Tracking is powerful, but it is not perfect.

Possible problems include:

- ID switches
- Lost tracks
- Occlusion
- Fast-moving objects
- Objects leaving and re-entering the frame
- Weak detections
- Crowded scenes

These issues can affect zone and crossing analytics.

---

# 25. ID Switches

An ID switch occurs when the tracker assigns a different identity to the same physical object.

Example:

```text
Frame 100 → Car ID:5
Frame 101 → Car ID:5
Frame 102 → Car ID:9
```

The physical car did not change, but its tracker identity changed.

This can potentially affect event counting.

---

# 26. Occlusion

Objects can temporarily disappear behind other objects.

For example:

```text
Car A
   ↓
Truck
   ↓
Camera
```

If the tracker loses the car during the occlusion, it may receive a different ID when it becomes visible again.

Stable tracking therefore improves zone analytics.

---

# 27. Detection Quality Matters

Tracking depends on object detections.

Conceptually:

```text
Poor Detection
      ↓
Poor Tracking
      ↓
Poor Zone Analysis
      ↓
Poor Counting
```

A reliable system therefore depends on the entire pipeline.

---

# 28. Real-World Traffic Example

Suppose a traffic camera detects vehicles.

YOLO might detect:

```text
car
car
truck
bus
```

ByteTrack adds:

```text
Car ID:4
Car ID:8
Truck ID:11
Bus ID:15
```

PolygonZone adds:

```text
ID:4 → inside left lane
ID:8 → outside
ID:11 → inside left lane
ID:15 → outside
```

LineZone may later record:

```text
ID:4 → crossed
ID:11 → crossed
```

This creates structured traffic analytics from raw video.

---

# 29. From Detection to Behavior

Tracking and zones allow us to move beyond simple detection.

```text
Detection
    ↓
"There is a car."

Tracking
    ↓
"This is Car ID:7."

PolygonZone
    ↓
"Car ID:7 is inside Zone A."

LineZone
    ↓
"Car ID:7 crossed Line B."
```

This represents an important progression toward understanding object behavior.

---

# 30. Key Takeaways

The most important concepts are:

1. Object detection identifies objects independently in each frame.
2. Object tracking maintains identity across frames.
3. ByteTrack assigns persistent tracker IDs.
4. `detections.tracker_id` contains tracked identities.
5. Tracking should happen before zone analysis.
6. `PolygonZone` can filter tracked objects by spatial region.
7. `LineZone` uses tracked movement to detect crossings.
8. The tracker should normally persist across the complete video sequence.
9. Recreating the tracker every frame destroys tracking history.
10. The same tracked detections can feed multiple zones.
11. Tracking quality directly affects zone analytics.
12. Detection, tracking, and zones together enable behavioral video analysis.

---

# Summary

Tracking provides the temporal information required to understand how objects interact with spatial zones.

The complete progression is:

```text
Object Detection
      ↓
What is visible?

Object Tracking
      ↓
Which object is which?

PolygonZone
      ↓
Where is the object?

LineZone
      ↓
Did the object cross a boundary?

Counting
      ↓
Generate useful analytics
```

Combining these components transforms a basic object detector into a **stateful spatial video analytics system** capable of understanding both object presence and movement.

---

## Related Concepts

- [Concepts Overview](./README.md)
- [PolygonZone](./01-PolygonZone.md)
- [LineZone](./02-LineZone.md)
- [Occupancy vs Flow](./03-Occupancy-vs-Flow.md)
- [PolygonZone Trigger and Filtering](./05-PolygonZone-Trigger-and-Filtering.md)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
