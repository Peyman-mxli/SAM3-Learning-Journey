# Combining PolygonZone and LineZone

## Introduction

`PolygonZone` and `LineZone` provide two different types of spatial analysis.

`PolygonZone` measures **current occupancy**:

> How many objects are inside this area right now?

`LineZone` measures **accumulated flow**:

> How many objects have crossed this boundary?

Using both in the same video allows a computer vision system to analyze **presence and movement simultaneously**.

The central idea is:

```text
PolygonZone + LineZone
        ↓
Occupancy + Flow
        ↓
Spatial Video Analytics
```

---

# 1. Why Combine the Two Zones?

A single metric does not always provide enough information about a scene.

For example, imagine a parking lot.

Knowing:

```text
Current vehicles inside = 18
```

is useful.

But we may also want to know:

```text
Vehicles entered = 52
Vehicles exited  = 34
```

These measurements describe different aspects of the same environment.

`PolygonZone` provides the first type of information.

`LineZone` provides the second.

---

# 2. Different Questions

The two components answer different questions.

```text
PolygonZone
     ↓
How many objects are currently here?

LineZone
     ↓
How many objects crossed this boundary?
```

Together:

```text
Where are the objects?
        +
How are they moving?
```

---

# 3. Shared Detection Pipeline

We do not need to run YOLO separately for each zone.

A single detection operation can generate the objects needed by both systems.

```text
Video Frame
     ↓
YOLO
     ↓
Detections
```

These detections are then tracked.

```text
Detections
     ↓
ByteTrack
     ↓
Tracked Detections
```

The same tracked detections can then be passed to both zones.

---

# 4. Shared Tracked Detections

The architecture becomes:

```text
                 Video Frame
                      ↓
                    YOLO
                      ↓
                 Detections
                      ↓
                  ByteTrack
                      ↓
             Tracked Detections
                      │
             ┌────────┴────────┐
             │                 │
             ↓                 ↓
       PolygonZone         LineZone
             │                 │
             ↓                 ↓
        Occupancy             Flow
```

This is efficient because detection and tracking are performed only once per frame.

---

# 5. Triggering Both Zones

After tracking:

```python
detections = tracker.update(
    detections
)
```

the same detections can be passed to both triggers:

```python
zone.trigger(
    detections=detections
)

line_zone.trigger(
    detections=detections
)
```

Each component performs its own spatial analysis.

---

# 6. PolygonZone Result

The polygon updates:

```python
zone.current_count
```

This represents the number of objects currently associated with the polygon.

Example:

```text
Current polygon occupancy = 4
```

This value may change every frame.

---

# 7. LineZone Result

The line updates:

```python
line_zone.in_count
line_zone.out_count
```

These represent accumulated crossing events.

Example:

```text
Crossings Down = 15
Crossings Up   = 9
```

These counters describe movement rather than current presence.

---

# 8. Different Temporal Behavior

The values behave differently over time.

## PolygonZone

```text
Frame 1 → 3
Frame 2 → 5
Frame 3 → 4
Frame 4 → 2
Frame 5 → 6
```

The value can increase or decrease.

---

## LineZone

```text
Frame 1   → 0 crossings
Frame 50  → 1 crossing
Frame 100 → 2 crossings
Frame 180 → 3 crossings
```

The counters accumulate crossing events.

---

# 9. Creating the Polygon

A polygon can be defined using NumPy coordinates.

Example:

```python
POLYGON_LEFT = np.array([
    [0,                     video_info.height // 2],
    [video_info.width // 2, video_info.height // 2],
    [video_info.width // 2, video_info.height],
    [0,                     video_info.height],
])
```

Then:

```python
zone = sv.PolygonZone(
    polygon=POLYGON_LEFT
)
```

---

# 10. Creating the Counting Line

A horizontal counting line can be defined using:

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

Then:

```python
line_zone = sv.LineZone(
    start=line_start,
    end=line_end
)
```

Now the same frame contains two spatial structures:

```text
Polygon Area
     +
Counting Line
```

---

# 11. Creating the Annotators

The polygon annotator can be created with:

```python
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.RED,
    thickness=4
)
```

The line annotator can be created with:

```python
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_scale=1.5,
    custom_in_text="Crossings Down",
    custom_out_text="Crossings Up"
)
```

These components visualize both analytics systems.

---

# 12. Extension Challenge

The session includes an extension challenge:

> Display the red `PolygonZone` and the `LineZone` in the same video at the same time.

The important requirement is that both triggers must execute inside the same callback.

Conceptually:

```text
Tracked Detections
       │
       ├──→ PolygonZone.trigger()
       │
       └──→ LineZone.trigger()
```

---

# 13. Combined Callback

A solution can follow this structure:

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

This callback combines detection, tracking, occupancy analysis, crossing analysis, and visualization.

---

# 14. Processing Order

The processing order is important.

```text
Frame
  ↓
YOLO
  ↓
sv.Detections
  ↓
ByteTrack
  ↓
Tracked Detections
  ↓
PolygonZone.trigger()
  ↓
LineZone.trigger()
  ↓
Box Annotation
  ↓
Polygon Annotation
  ↓
Line Annotation
  ↓
Output Frame
```

Tracking occurs before both zone triggers.

---

# 15. Why Trigger Before Annotation?

The annotators visualize information stored by the zones.

For example:

```python
zone.current_count
```

must first be updated by:

```python
zone.trigger()
```

Similarly:

```python
line_zone.in_count
line_zone.out_count
```

must be updated by:

```python
line_zone.trigger()
```

Therefore:

```text
Trigger
   ↓
Update State
   ↓
Annotate
```

is the correct conceptual order.

---

# 16. Annotation Order

The challenge suggests applying the annotators in this order:

```python
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
```

Each annotator receives the output of the previous annotation step.

---

# 17. Layered Output

Conceptually:

```text
Original Frame
      ↓
Bounding Boxes
      ↓
Polygon Boundary + Count
      ↓
Counting Line + Crossing Counts
      ↓
Final Annotated Frame
```

The final frame contains information from multiple analytics components.

---

# 18. Processing the Combined Video

The callback can be used with:

```python
sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_combined.mp4",
    callback=callback_combined,
    show_progress=True
)
```

Supervision applies the callback to every frame and creates the output video.

---

# 19. Complete Conceptual Pipeline

The complete system can be represented as:

```text
                    INPUT VIDEO
                         ↓
                     Read Frame
                         ↓
                        YOLO
                         ↓
                    Detections
                         ↓
                     ByteTrack
                         ↓
                 Tracked Objects
                         │
              ┌──────────┴──────────┐
              │                     │
              ↓                     ↓
        PolygonZone              LineZone
              │                     │
              ↓                     ↓
       Current Occupancy       Crossing Events
              │                     │
              └──────────┬──────────┘
                         ↓
                    Annotation
                         ↓
                  OUTPUT VIDEO
```

---

# 20. Example Output

A combined frame might conceptually display:

```text
┌─────────────────────────────────────────────┐
│                                             │
│      ┌─────────────────────┐                │
│      │   POLYGON ZONE      │                │
│      │                     │                │
│      │ ID:4       ID:9     │                │
│      │                     │                │
│      │ Current Count: 2    │                │
│      └─────────────────────┘                │
│                                             │
│════════════ COUNTING LINE ══════════════════│
│                                             │
│ Crossings Down: 15                          │
│ Crossings Up: 9                             │
│                                             │
└─────────────────────────────────────────────┘
```

The viewer can understand both current presence and accumulated movement.

---

# 21. Multiple Analytics from One Camera

This demonstrates an important computer vision architecture principle:

> One video stream can support multiple analytics tasks.

The same camera can provide:

```text
Object Detection
Object Tracking
Lane Occupancy
Crossing Counts
Directional Flow
Restricted Area Monitoring
```

without requiring a separate detector for each task.

---

# 22. Extending to Multiple Polygon Zones

The architecture can be expanded.

For example:

```text
                    Tracked Objects
                          │
       ┌──────────────────┼──────────────────┐
       ↓                  ↓                  ↓
 PolygonZone A      PolygonZone B        LineZone
       ↓                  ↓                  ↓
 Left Lane           Right Lane          Crossings
 Occupancy            Occupancy
```

This could monitor multiple lanes simultaneously.

---

# 23. Extending to Multiple Lines

Multiple `LineZone` objects could also represent different checkpoints.

Conceptually:

```text
                 Tracked Objects
                       │
       ┌───────────────┼───────────────┐
       ↓               ↓               ↓
    Line A          Line B          Line C
       ↓               ↓               ↓
 Entrance          Checkpoint         Exit
```

Each line can maintain its own crossing counters.

---

# 24. Combining Multiple Zones and Lines

A larger analytics system could contain:

```text
Camera
  ↓
Detector
  ↓
Tracker
  ↓
Tracked Objects
  │
  ├── Polygon A → Lane Occupancy
  ├── Polygon B → Parking Occupancy
  ├── Polygon C → Restricted Area
  │
  ├── Line A → Entrance Count
  ├── Line B → Exit Count
  └── Line C → Traffic Flow
```

This demonstrates how simple components can be combined into more advanced systems.

---

# 25. Real-World Traffic Analytics

A traffic camera could use:

```text
PolygonZone A
    ↓
Vehicles currently in left lane

PolygonZone B
    ↓
Vehicles currently in right lane

LineZone
    ↓
Vehicles crossing intersection
```

Together, these metrics provide more useful traffic information than object detection alone.

---

# 26. Parking Analytics

A parking system could use:

```text
PolygonZone
    ↓
Current parking occupancy

LineZone
    ↓
Vehicles entering / leaving
```

This allows the system to analyze both current capacity and traffic movement.

---

# 27. Retail Analytics

A store camera could use:

```text
PolygonZone
    ↓
Customers currently in a department

LineZone
    ↓
Customers entering / leaving
```

This provides both occupancy and visitor flow.

---

# 28. Security Monitoring

A security system could use:

```text
PolygonZone
    ↓
Objects currently inside restricted area

LineZone
    ↓
Objects crossing security boundary
```

Tracking maintains the identities of the objects interacting with those regions.

---

# 29. Industrial Monitoring

In an industrial environment:

```text
PolygonZone
    ↓
Workers currently inside safety area

LineZone
    ↓
Vehicles or products crossing checkpoint
```

These analytics can help monitor movement and occupancy around important areas.

---

# 30. Avoiding Duplicate Detection Work

An important optimization is to reuse detections.

Inefficient design:

```text
YOLO → PolygonZone

YOLO → LineZone
```

This performs object detection twice.

Better design:

```text
          YOLO
           ↓
      Detections
           ↓
        Tracker
           ↓
   Tracked Detections
        ↙       ↘
PolygonZone   LineZone
```

Detection and tracking are performed only once.

---

# 31. Shared State and Independent State

Although both zones receive the same detections, they maintain different information.

`PolygonZone` maintains:

```python
zone.current_count
```

`LineZone` maintains:

```python
line_zone.in_count
line_zone.out_count
```

Therefore:

```text
Shared Input
    +
Independent Analytics State
```

allows multiple measurements to be produced from the same video stream.

---

# 32. From Detection to Scene Understanding

The progression now becomes:

```text
YOLO
 ↓
"What objects are visible?"

ByteTrack
 ↓
"Which object is which?"

PolygonZone
 ↓
"Where are the objects?"

LineZone
 ↓
"Which objects crossed a boundary?"

Counting
 ↓
"What is happening in this scene?"
```

This is an important progression toward higher-level computer vision systems.

---

# 33. Key Takeaways

The most important concepts are:

1. `PolygonZone` and `LineZone` measure different spatial behaviors.
2. `PolygonZone` measures current occupancy.
3. `LineZone` measures accumulated crossings.
4. Both can use the same tracked detections.
5. YOLO only needs to run once per frame.
6. Tracking only needs to run once per frame.
7. Both zone triggers should run before their corresponding annotations.
8. Multiple annotators can be layered on the same frame.
9. One camera can support multiple analytics tasks.
10. Multiple polygons and lines can extend the architecture further.
11. Shared detections make the pipeline more efficient.
12. Combining zones creates a more complete understanding of object behavior.

---

# Summary

The combined system follows this pattern:

```text
Detect
  ↓
Track
  ↓
Analyze Spatial Position
  │
  ├── PolygonZone → Occupancy
  │
  └── LineZone → Flow
  ↓
Visualize
  ↓
Generate Spatial Analytics
```

`PolygonZone` tells us **where objects currently are**.

`LineZone` tells us **how objects move through a boundary**.

Together, they transform a basic object detection pipeline into a more complete **spatial video analytics system**.

---

## Related Concepts

- [Concepts Overview](./README.md)
- [PolygonZone](./01-PolygonZone.md)
- [LineZone](./02-LineZone.md)
- [Occupancy vs Flow](./03-Occupancy-vs-Flow.md)
- [Tracking with Zones](./04-Tracking-with-Zones.md)
- [PolygonZone Trigger and Filtering](./05-PolygonZone-Trigger-and-Filtering.md)
- [Zone Annotation and Visualization](./06-Zone-Annotation-and-Visualization.md)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
