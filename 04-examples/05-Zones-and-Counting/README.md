# Zones and Counting — Code Examples

This directory contains practical Python examples based on the concepts covered in **Session 06 — Zones and Counting** of my SAM3 Computer Vision Learning Journey.

The examples demonstrate how object detection and tracking can be extended with spatial analysis using:

- YOLOv8
- Supervision
- ByteTrack
- PolygonZone
- LineZone
- OpenCV
- NumPy

The main goal is to understand two important computer vision questions:

```text
PolygonZone:
How many objects are inside this area RIGHT NOW?

LineZone:
How many objects have CROSSED this line IN TOTAL?
```

---

## Learning Progression

The examples are organized from basic zone creation to a complete spatial video analytics pipeline.

```text
01 — Basic PolygonZone
        ↓
02 — PolygonZone Current Count
        ↓
03 — Filter Detections Inside Zone
        ↓
04 — Tracking with PolygonZone
        ↓
05 — Basic LineZone
        ↓
06 — LineZone Crossing Count
        ↓
07 — Tracking with LineZone
        ↓
08 — PolygonZone + LineZone
        ↓
09 — Complete Zones and Counting Pipeline
```

---

## Available Examples

### 01 — Basic PolygonZone

[`01-basic-polygon-zone.py`](./01-basic-polygon-zone.py)

Introduces the creation of a polygonal region using Supervision.

Concepts:

- Polygon coordinates
- NumPy arrays
- `sv.PolygonZone`
- `sv.PolygonZoneAnnotator`
- Spatial regions
- Zone visualization

Conceptually:

```text
Video Frame
     ↓
Polygon Coordinates
     ↓
PolygonZone
     ↓
Visualized Region
```

---

### 02 — PolygonZone Current Count

[`02-polygon-zone-current-count.py`](./02-polygon-zone-current-count.py)

Demonstrates how `PolygonZone` measures the number of objects currently located inside a region.

Concepts:

- `zone.trigger()`
- Boolean zone masks
- `zone.current_count`
- Instantaneous occupancy
- Frame-by-frame counting

Conceptually:

```text
Detections
     ↓
PolygonZone
     ↓
zone.trigger()
     ↓
current_count
```

`PolygonZone` answers:

```text
How many objects are inside this area RIGHT NOW?
```

---

### 03 — Filter Detections Inside Zone

[`03-filter-detections-inside-zone.py`](./03-filter-detections-inside-zone.py)

Demonstrates how the Boolean mask returned by `PolygonZone.trigger()` can be used to keep only detections located inside a region.

Concepts:

- Boolean masks
- Detection filtering
- Spatial filtering
- Zone membership
- Supervision `Detections`

Example:

```python
zone_mask = zone.trigger(
    detections=detections
)

detections_inside_zone = detections[
    zone_mask
]
```

Conceptually:

```text
All Detections
      ↓
PolygonZone
      ↓
Boolean Mask
      ↓
Filter
      ↓
Detections Inside Zone
```

---

### 04 — Tracking with PolygonZone

[`04-tracking-with-polygon-zone.py`](./04-tracking-with-polygon-zone.py)

Combines object detection, ByteTrack, and `PolygonZone`.

Concepts:

- YOLOv8 detection
- ByteTrack tracking
- Persistent tracker IDs
- Confirmed tracks
- Polygon occupancy
- Zone-based tracking

The processing order is important:

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
PolygonZone
  ↓
Occupancy
```

Tracking is performed before zone analysis so the spatial system can work with tracked objects.

---

### 05 — Basic LineZone

[`05-basic-line-zone.py`](./05-basic-line-zone.py)

Introduces a virtual counting line using Supervision.

Concepts:

- `sv.Point`
- Line start point
- Line end point
- `sv.LineZone`
- `sv.LineZoneAnnotator`
- Virtual boundaries

Conceptually:

```text
Start Point
     +
End Point
     ↓
LineZone
     ↓
Virtual Counting Boundary
```

---

### 06 — LineZone Crossing Count

[`06-line-zone-crossing-count.py`](./06-line-zone-crossing-count.py)

Demonstrates how `LineZone` accumulates crossing events.

Concepts:

- `line_zone.trigger()`
- Directional counting
- `in_count`
- `out_count`
- Accumulated flow

The important counters are:

```python
line_zone.in_count
line_zone.out_count
```

Unlike `PolygonZone`, these values represent accumulated events.

Conceptually:

```text
Tracked Object
      ↓
Crosses Line
      ↓
LineZone
      ↓
Directional Counter
```

`LineZone` answers:

```text
How many objects have CROSSED this line?
```

---

### 07 — Tracking with LineZone

[`07-tracking-with-line-zone.py`](./07-tracking-with-line-zone.py)

Combines YOLOv8, ByteTrack, and `LineZone` to count tracked objects crossing a virtual boundary.

Concepts:

- Object detection
- Object tracking
- Persistent IDs
- Crossing detection
- Directional traffic flow
- Accumulated counts

Pipeline:

```text
Video Frame
     ↓
YOLOv8
     ↓
Detections
     ↓
ByteTrack
     ↓
Persistent IDs
     ↓
LineZone
     ↓
Crossing Events
```

Tracking is essential because the system needs to understand the movement of the same object across multiple frames.

---

### 08 — PolygonZone + LineZone

[`08-polygon-and-line-zone.py`](./08-polygon-and-line-zone.py)

Combines occupancy analysis and crossing analysis in the same video.

Concepts:

- Multiple spatial zones
- Shared tracked detections
- Polygon occupancy
- Line crossings
- Multiple annotators
- Spatial video analytics

Both zone systems receive the same tracked detections:

```python
polygon_zone.trigger(
    detections=detections
)

line_zone.trigger(
    detections=detections
)
```

Conceptually:

```text
               Tracked Objects
                     ↓
            ┌────────┴────────┐
            ↓                 ↓
      PolygonZone          LineZone
            ↓                 ↓
        Occupancy             Flow
            │                 │
            └────────┬────────┘
                     ↓
                Visualization
```

---

### 09 — Complete Zones and Counting Pipeline

[`09-complete-zones-counting-pipeline.py`](./09-complete-zones-counting-pipeline.py)

Combines the main concepts from the session into one complete video analytics example.

The pipeline includes:

- YOLOv8 object detection
- Supervision detections
- ByteTrack object tracking
- Persistent tracker IDs
- Confirmed track filtering
- PolygonZone
- Current occupancy
- LineZone
- Directional crossing counts
- Bounding-box annotation
- Tracker-ID labels
- Zone visualization
- Video processing
- Output video generation

Complete pipeline:

```text
Input Video
     ↓
Read Frame
     ↓
YOLOv8
     ↓
Supervision Detections
     ↓
ByteTrack
     ↓
Confirmed Tracked Objects
     ↓
┌─────────────────────────┐
│                         │
↓                         ↓
PolygonZone            LineZone
↓                         ↓
Occupancy                Flow
│                         │
└────────────┬────────────┘
             ↓
         Annotation
             ↓
        Output Video
```

---

## PolygonZone vs LineZone

The fundamental difference between the two zone types is:

| Feature | PolygonZone | LineZone |
|---|---|---|
| Represents | Area | Boundary |
| Measures | Presence | Crossings |
| Count type | Instantaneous | Accumulated |
| Main property | `current_count` | `in_count` / `out_count` |
| Typical use | Occupancy | Flow |
| Example | Cars currently in parking area | Cars entering parking area |

A simple way to remember this is:

```text
PolygonZone = WHERE objects are

LineZone = WHERE objects PASS
```

---

## Why Tracking Matters

Object detection alone processes objects independently in every frame.

Tracking adds persistent identities:

```text
Frame 100 → ID 4
Frame 101 → ID 4
Frame 102 → ID 4
Frame 103 → ID 4
```

This allows the system to understand movement over time.

Without tracking:

```text
Detection
    ↓
Object exists
```

With tracking:

```text
Detection
    ↓
Persistent Identity
    ↓
Movement
    ↓
Spatial Event
```

This is especially important for `LineZone`, where the system must determine whether the same object actually crossed a boundary.

---

## Practical Results

The complete pipeline was tested using a traffic video.

The original test video contained:

```text
Resolution:   3840 × 2160
FPS:          25
Total Frames: 538
```

The final combined pipeline produced:

```text
Final polygon occupancy: 1
Crossings Down: 3
Crossings Up: 3
```

---

## Related Course Notes

The complete lesson documentation is available here:

[`../../08-course-notes/05-Zones-and-Counting/`](../../08-course-notes/05-Zones-and-Counting/)

The tested practical implementation is available here:

[`../../08-course-notes/05-Zones-and-Counting/practical/`](../../08-course-notes/05-Zones-and-Counting/practical/)

---

## Class Recording

The completed Zones and Counting practical is available on YouTube:

[Watch — SAM3: Zonas y Conteo | PolygonZone, LineZone y ByteTrack](https://youtu.be/43i0z9b81Z4)

---

## Technologies

The examples use:

```text
Python
OpenCV
NumPy
Ultralytics YOLOv8
Supervision
ByteTrack
```

---

## Key Concepts

After completing these examples, the main concepts to understand are:

1. A polygon defines a spatial region using `(x, y)` coordinates.
2. `PolygonZone` measures current occupancy.
3. `zone.trigger()` determines which detections are inside a polygon.
4. `zone.current_count` stores the current occupancy.
5. `LineZone` detects objects crossing a virtual boundary.
6. `line_zone.in_count` and `line_zone.out_count` accumulate crossings.
7. Tracking should occur before zone analysis.
8. Persistent tracker IDs allow movement to be analyzed across frames.
9. Polygon and line zones can use the same tracked detections.
10. Occupancy and flow represent different spatial analytics measurements.

---

## Learning Outcome

These examples demonstrate the progression:

```text
Detection
    ↓
Tracking
    ↓
Spatial Analysis
    ↓
Occupancy + Flow
    ↓
Video Analytics
```

Together, object detection, tracking, and spatial zones provide the foundation for applications such as:

- Traffic monitoring
- Parking management
- Retail analytics
- People counting
- Security monitoring
- Restricted-area detection
- Entrance and exit counting
- Industrial safety systems

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
