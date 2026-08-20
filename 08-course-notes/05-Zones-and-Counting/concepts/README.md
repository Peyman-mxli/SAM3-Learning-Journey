# Zones and Counting — Concepts

This directory contains the core concepts covered in **Session 06 — Zones and Counting** of my SAM3 Computer Vision Learning Journey.

The session extends object detection and object tracking by introducing **spatial analysis**.

Instead of only asking:

> What objects are visible?

or:

> Which object is which?

we can now ask:

> Where are the objects?

> How many objects are currently inside a specific area?

> How many objects crossed a specific boundary?

These concepts are implemented primarily using **Supervision**, **YOLO**, and **ByteTrack**.

---

## Concepts Covered

### 01 — PolygonZone

[`01-PolygonZone.md`](./01-PolygonZone.md)

Introduces polygon-based spatial regions and explains how they can be used to determine which objects are currently inside a specific area.

Main concepts:

- Polygon coordinates
- Image coordinate system
- `sv.PolygonZone`
- `PolygonZone.trigger()`
- Boolean detection masks
- `current_count`
- Zone visualization

---

### 02 — LineZone

[`02-LineZone.md`](./02-LineZone.md)

Explains how virtual lines can be used to count objects crossing a boundary.

Main concepts:

- Virtual counting lines
- `sv.LineZone`
- Crossing detection
- Directional counting
- `in_count`
- `out_count`
- Accumulated flow

---

### 03 — Occupancy vs Flow

[`03-Occupancy-vs-Flow.md`](./03-Occupancy-vs-Flow.md)

Explains the fundamental difference between measuring **current presence** and **accumulated movement**.

```text
PolygonZone → Occupancy
LineZone    → Flow
```

This distinction is important when designing real-world computer vision systems.

---

### 04 — Tracking with Zones

[`04-Tracking-with-Zones.md`](./04-Tracking-with-Zones.md)

Explains why object tracking is important when working with zones and counting.

Main concepts:

- Persistent `tracker_id`
- Detection across multiple frames
- ByteTrack
- Tracking before zone analysis
- Reliable crossing detection
- Object identity

The processing order is:

```text
Frame
  ↓
Object Detection
  ↓
sv.Detections
  ↓
Object Tracking
  ↓
Zone Analysis
  ↓
Counting
  ↓
Annotation
```

---

### 05 — PolygonZone Trigger and Filtering

[`05-PolygonZone-Trigger-and-Filtering.md`](./05-PolygonZone-Trigger-and-Filtering.md)

Explains how `PolygonZone.trigger()` produces a Boolean mask and how that mask can be used to filter detections.

Example:

```python
inside_zone = zone.trigger(
    detections=detections
)

detections_inside_zone = detections[
    inside_zone
]
```

Conceptually:

```text
Detections
    ↓
PolygonZone.trigger()
    ↓
Boolean Mask
    ↓
Filter Detections
    ↓
Objects Inside Zone
```

---

### 06 — Zone Annotation and Visualization

[`06-Zone-Annotation-and-Visualization.md`](./06-Zone-Annotation-and-Visualization.md)

Explains how Supervision annotators can visualize spatial analysis results.

Main components include:

- `BoxAnnotator`
- `LabelAnnotator`
- `PolygonZoneAnnotator`
- `LineZoneAnnotator`

These annotators allow the final video to display:

- Bounding boxes
- Tracker IDs
- Polygon boundaries
- Current occupancy
- Counting lines
- Directional crossing totals

---

### 07 — Combining PolygonZone and LineZone

[`07-Combining-PolygonZone-and-LineZone.md`](./07-Combining-PolygonZone-and-LineZone.md)

Explains how both zone systems can operate simultaneously using the same tracked detections.

Example:

```python
zone.trigger(
    detections=detections
)

line_zone.trigger(
    detections=detections
)
```

This allows a single video analytics pipeline to measure both:

```text
Current Occupancy
        +
Accumulated Flow
```

---

## PolygonZone vs LineZone

The two zone types answer different questions.

| Feature | PolygonZone | LineZone |
|---|---|---|
| Purpose | Monitor an area | Monitor a boundary |
| Measurement | Current presence | Accumulated crossings |
| Main question | How many are here now? | How many passed? |
| Main counter | `current_count` | `in_count` / `out_count` |
| Typical use | Occupancy | Flow |
| Tracking | Useful for object identity | Important for crossing detection |

A simple way to remember the difference is:

```text
PolygonZone
    ↓
"How many objects are HERE NOW?"

LineZone
    ↓
"How many objects PASSED HERE?"
```

---

## Core Processing Pipeline

The main pipeline introduced in this session is:

```text
Input Video
     ↓
Read Frame
     ↓
YOLO Detection
     ↓
Supervision Detections
     ↓
ByteTrack
     ↓
Tracked Objects
     ↓
 ┌────────────────────────┐
 │                        │
 ↓                        ↓
PolygonZone           LineZone
 │                        │
 ↓                        ↓
Occupancy              Crossings
 │                        │
 └───────────┬────────────┘
             ↓
         Annotation
             ↓
        Output Video
```

---

## Important Processing Order

Tracking should happen before zone analysis.

Correct:

```text
Detection
    ↓
Tracking
    ↓
Zone Trigger
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

detections = tracker.update(
    detections
)

inside_zone = zone.trigger(
    detections=detections
)
```

This gives the zone access to tracked objects with persistent identities.

---

## Important Properties

### PolygonZone

```python
zone.current_count
```

Represents the number of objects currently inside the polygon.

---

### LineZone

```python
line_zone.in_count
line_zone.out_count
```

Represent accumulated crossings in each direction.

---

## Boolean Zone Masks

`PolygonZone.trigger()` returns a Boolean mask.

Example:

```text
[True, False, True, False]
```

The mask can be used to select only detections inside the zone:

```python
detections_inside_zone = detections[
    inside_zone
]
```

This demonstrates an important pattern used throughout computer vision:

```text
Detect
  ↓
Evaluate Condition
  ↓
Create Mask
  ↓
Filter
  ↓
Analyze Selected Objects
```

---

## Real-World Applications

Zones and counting are useful for many computer vision applications.

### Traffic Analytics

```text
PolygonZone → vehicles currently occupying a lane
LineZone    → vehicles crossing a road boundary
```

### Parking Management

```text
PolygonZone → vehicles currently inside parking areas
LineZone    → vehicles entering or leaving
```

### Retail Analytics

```text
PolygonZone → customers currently inside an area
LineZone    → customers entering or leaving a store
```

### Security Monitoring

```text
PolygonZone → objects inside restricted areas
LineZone    → objects crossing security boundaries
```

### Industrial Monitoring

```text
PolygonZone → workers or equipment inside safety areas
LineZone    → movement between production zones
```

---

## Key Concepts

The most important concepts from this session are:

- Spatial regions can be defined using pixel coordinates.
- `PolygonZone` measures current occupancy.
- `LineZone` measures accumulated flow.
- `trigger()` evaluates tracked detections against a zone.
- Boolean masks can filter detections.
- Tracking provides persistent object identities.
- `tracker_id` allows objects to be followed across frames.
- Multiple zones can analyze the same detections.
- Annotators visualize spatial analysis results.
- Detection, tracking, zones, and counting can be combined into one video analytics pipeline.

---

## From Detection to Spatial Analytics

The progression of concepts throughout the course can now be understood as:

```text
Object Detection
        ↓
"What objects exist?"

Object Tracking
        ↓
"Which object is which?"

Zones
        ↓
"Where are the objects?"

Counting
        ↓
"How many are present or moving?"

Spatial Video Analytics
```

This is an important transition from basic object detection toward more advanced real-world computer vision systems.

---

## Technologies

The concepts in this directory use:

- Python
- NumPy
- OpenCV
- Matplotlib
- Ultralytics YOLO
- Supervision
- ByteTrack
- Google Colab

---

## Related Course Material

- [SAM3 Course](https://sam-3-vision-computacional-5tao.vercel.app/)
- [Session 06 — Zones and Counting](https://sam-3-vision-computacional-5tao.vercel.app/modulo/06)
- [Main Session README](../README.md)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
