# LineZone

## Introduction

`LineZone` is a component provided by the **Supervision** computer vision library for counting tracked objects that cross a virtual line.

Unlike `PolygonZone`, which measures how many objects are currently inside an area, `LineZone` measures **crossing events accumulated over time**.

The main question answered by `LineZone` is:

> How many objects have crossed this line?

This makes `LineZone` useful for measuring **flow**.

---

## Basic Concept

A `LineZone` can be imagined as a virtual turnstile placed across part of a video.

```text
              VIDEO FRAME

┌───────────────────────────────────────┐
│                                       │
│          Vehicle A                    │
│              ↓                        │
│                                       │
│══════════ COUNTING LINE ══════════════│
│                                       │
│              ↓                        │
│                                       │
└───────────────────────────────────────┘
```

When the tracked vehicle crosses the line, the crossing counter is updated.

For example:

```text
Vehicle A crosses → count = 1
Vehicle B crosses → count = 2
Vehicle C crosses → count = 3
```

The count is accumulated as the video is processed.

---

## PolygonZone vs LineZone

The easiest way to understand `LineZone` is to compare it with `PolygonZone`.

```text
PolygonZone
    ↓
"How many objects are here NOW?"

LineZone
    ↓
"How many objects have CROSSED?"
```

Therefore:

```text
PolygonZone → Presence / Occupancy

LineZone → Crossings / Flow
```

---

## Defining a Counting Line

A line requires two points:

```text
Start Point
     ↓
     ─────────────────────
                           ↑
                       End Point
```

Each point contains an `(x, y)` coordinate.

In Supervision, points can be created using:

```python
sv.Point()
```

---

## Horizontal Line Example

In this session, a horizontal line is placed approximately in the middle of the video.

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

The line begins at the left side of the frame and ends at the right side.

Conceptually:

```text
(0, height/2) ─────────────────────→ (width, height/2)
```

---

## Creating LineZone

After defining the start and end points, the zone can be created with:

```python
line_zone = sv.LineZone(
    start=line_start,
    end=line_end
)
```

The resulting object keeps track of crossing events.

---

## Direction Matters

A line has two possible crossing directions.

Conceptually:

```text
           Direction A
               ↓

════════════════════════════
        COUNTING LINE
════════════════════════════

               ↑
           Direction B
```

Supervision maintains separate counters for these directions.

They are available through:

```python
line_zone.in_count
line_zone.out_count
```

---

## `in_count`

The property:

```python
line_zone.in_count
```

stores accumulated crossings in one direction.

For example:

```text
Object 1 crosses → in_count = 1
Object 2 crosses → in_count = 2
Object 3 crosses → in_count = 3
```

The value continues increasing as new crossing events occur.

---

## `out_count`

The property:

```python
line_zone.out_count
```

stores accumulated crossings in the opposite direction.

For example:

```text
Object 4 crosses opposite direction → out_count = 1
Object 5 crosses opposite direction → out_count = 2
```

Together, the two counters provide directional flow information.

---

## Triggering the Line

The crossing logic is updated with:

```python
line_zone.trigger(
    detections=detections
)
```

This should be called for every processed frame.

Conceptually:

```text
Tracked Detections
        ↓
LineZone.trigger()
        ↓
Compare Object Movement
        ↓
Detect Crossing Event
        ↓
Update Directional Counter
```

---

## Why Tracking Is Important

A crossing is not simply determined by checking whether an object touches a line in one frame.

The system needs to understand the object's movement across multiple frames.

Consider this sequence:

```text
Frame 100

      ID 7
        ↓

--------------------
   COUNTING LINE
--------------------
```

Then:

```text
Frame 101

--------------------
   COUNTING LINE
--------------------

        ↓
      ID 7
```

Because the object has the same tracker ID in both frames, the system can determine that:

```text
ID 7 moved from one side of the line
to the other side.
```

This represents a crossing event.

---

## Without Tracking

Without tracking, the detector might produce:

```text
Frame 100 → car
Frame 101 → car
Frame 102 → car
```

The system would not reliably know whether these detections represent:

```text
one car across three frames
```

or:

```text
three different cars
```

Tracking solves this problem by assigning persistent IDs.

---

## Persistent Tracker IDs

After ByteTrack processes the detections, objects can contain:

```python
detections.tracker_id
```

Example:

```text
Frame 100 → ID 7
Frame 101 → ID 7
Frame 102 → ID 7
```

The persistent identity allows `LineZone` to reason about object movement.

---

## Correct Processing Order

The processing order used in this session is:

```text
Video Frame
    ↓
YOLO Detection
    ↓
sv.Detections
    ↓
ByteTrack
    ↓
Tracked Detections
    ↓
LineZone.trigger()
    ↓
Annotation
    ↓
Output Frame
```

Tracking should occur before the line trigger.

---

## Example Processing Code

The basic pattern is:

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

line_zone.trigger(
    detections=detections
)
```

After this call, the counters can be inspected:

```python
print(line_zone.in_count)
print(line_zone.out_count)
```

---

## LineZoneAnnotator

Supervision provides:

```python
sv.LineZoneAnnotator
```

to visualize the line and its counters.

Example from this session:

```python
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_scale=1.5,
    custom_in_text="Crossings Down",
    custom_out_text="Crossings Up"
)
```

This allows the final video to display the accumulated crossing counts.

---

## Annotating the Frame

`LineZoneAnnotator` uses a slightly different annotation API than some other Supervision annotators.

Example:

```python
annotated = line_zone_annotator.annotate(
    frame=annotated,
    line_counter=line_zone
)
```

Notice the arguments:

```text
frame=
line_counter=
```

rather than:

```text
scene=
```

This is an important implementation detail when using the annotator.

---

## Complete Callback Example

A simplified callback based on the session is:

```python
def callback_line(frame: np.ndarray, _: int) -> np.ndarray:

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

    line_zone.trigger(
        detections=detections
    )

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    annotated = line_zone_annotator.annotate(
        frame=annotated,
        line_counter=line_zone
    )

    return annotated
```

The callback is executed for every video frame.

---

## Accumulated Counting

The most important difference between `LineZone` and `PolygonZone` is how the count behaves over time.

### PolygonZone

```text
Frame 1 → current_count = 3
Frame 2 → current_count = 5
Frame 3 → current_count = 2
Frame 4 → current_count = 4
```

The value represents current occupancy.

### LineZone

```text
Frame 1   → in_count = 0
Frame 50  → in_count = 1
Frame 120 → in_count = 2
Frame 200 → in_count = 3
```

The counter accumulates crossing events.

---

## Flow

The concept measured by `LineZone` is called:

```text
Flow
```

Flow answers questions such as:

> How many vehicles passed this point?

> How many people entered the building?

> How many customers left the store?

> How many objects moved between production areas?

This is fundamentally different from measuring occupancy.

---

## Entrance and Exit Counting

A common application is monitoring an entrance.

```text
             BUILDING

          ↑ Exit
          │
══════════════════════
    COUNTING LINE
══════════════════════
          │
          ↓ Entry
```

The two directional counters can represent:

```text
in_count  → entries
out_count → exits
```

The exact interpretation depends on the orientation of the line and the camera.

---

## Traffic Monitoring

A line can be placed across a road.

```text
      Vehicle
         ↓

=========================
     COUNTING LINE
=========================

         ↓
```

Every tracked vehicle crossing the line contributes to the accumulated traffic count.

This can be used to measure:

- Vehicle flow
- Traffic volume
- Directional traffic
- Entrance and exit counts

---

## Retail Analytics

A counting line can be placed across a store entrance.

The system can estimate:

```text
Customers Entering
Customers Leaving
```

This information can support:

- Visitor analytics
- Store traffic analysis
- Peak-hour detection
- Capacity estimation

---

## Parking Systems

A `LineZone` can monitor the entrance to a parking lot.

For example:

```text
in_count  → vehicles entering
out_count → vehicles leaving
```

Combined with other logic, this can help estimate parking occupancy.

---

## Industrial Applications

Counting lines can also monitor movement in industrial environments.

Examples include:

- Products crossing conveyor boundaries
- Vehicles entering loading areas
- Workers moving between sections
- Objects passing inspection points

---

## Security Applications

A line can represent a virtual security boundary.

Tracked objects crossing that line can be counted or used to trigger additional logic.

Examples:

```text
Gate crossing
Restricted entrance
Perimeter crossing
Checkpoint
```

---

## Line Position Matters

The position of the counting line affects the quality of the result.

A useful counting line should generally:

- Cross the expected movement path
- Be visible clearly in the camera
- Avoid areas with excessive occlusion
- Avoid locations where objects frequently stop directly on the line
- Be placed where tracking is reasonably stable

Poor line placement can make crossing analysis less reliable.

---

## Direction Depends on Line Orientation

The meaning of:

```python
in_count
```

and:

```python
out_count
```

depends on the orientation of the line.

Changing:

```text
start → end
```

can affect how the two sides of the line are interpreted.

Therefore, after creating a `LineZone`, the directional behavior should be tested with the actual video.

---

## PolygonZone and LineZone Together

A `LineZone` does not need to operate alone.

It can be combined with a `PolygonZone`.

For example:

```text
PolygonZone
    ↓
How many vehicles are currently in the area?

LineZone
    ↓
How many vehicles crossed the boundary?
```

Together they provide:

```text
Occupancy + Flow
```

This creates a more complete spatial analytics system.

---

## Combined Example

The same tracked detections can be sent to both zones:

```python
zone.trigger(
    detections=detections
)

line_zone.trigger(
    detections=detections
)
```

Then both visualizations can be added to the frame.

```python
annotated = zone_annotator.annotate(
    scene=annotated
)

annotated = line_zone_annotator.annotate(
    frame=annotated,
    line_counter=line_zone
)
```

---

## Key Takeaways

The most important concepts about `LineZone` are:

1. `LineZone` defines a virtual counting boundary.
2. The line is defined using a start point and an end point.
3. `line_zone.trigger()` evaluates tracked objects for crossing events.
4. `in_count` stores accumulated crossings in one direction.
5. `out_count` stores accumulated crossings in the opposite direction.
6. LineZone measures flow rather than current occupancy.
7. Object tracking provides the persistent IDs needed for reliable crossing analysis.
8. Line position and orientation affect the meaning of the results.
9. `LineZoneAnnotator` visualizes the line and counters.
10. `PolygonZone` and `LineZone` can operate together.

---

## Summary

`LineZone` adds **movement-based counting** to a computer vision pipeline.

The basic idea is:

```text
Detect Objects
      ↓
Track Objects
      ↓
Define Counting Line
      ↓
Analyze Movement
      ↓
Detect Crossing
      ↓
Update Directional Count
```

While `PolygonZone` answers:

> How many objects are here now?

`LineZone` answers:

> How many objects passed through here?

This makes `LineZone` particularly useful for **traffic monitoring, entrance counting, parking systems, retail analytics, security monitoring, and industrial automation**.

---

## Related Concepts

- [Concepts Overview](./README.md)
- [PolygonZone](./01-PolygonZone.md)
- [Occupancy vs Flow](./03-Occupancy-vs-Flow.md)
- [Tracking with Zones](./04-Tracking-with-Zones.md)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
