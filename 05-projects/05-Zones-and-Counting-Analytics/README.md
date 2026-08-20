# Zones and Counting Analytics

This project implements a complete **spatial video analytics pipeline** using YOLOv8, Supervision, ByteTrack, PolygonZone, and LineZone.

It builds on the previous Object Tracking project by adding spatial reasoning to tracked objects.

Instead of only asking:

```text
What objects are visible?
```

or:

```text
Which object is which across multiple frames?
```

this project also asks:

```text
How many tracked objects are inside a specific area?
```

and:

```text
How many tracked objects crossed a specific boundary?
```

---

## Project Goal

The goal of this project is to build a reusable video analytics application capable of:

- Detecting objects with YOLOv8
- Converting predictions to `sv.Detections`
- Tracking objects across video frames
- Assigning persistent tracker IDs
- Defining a polygonal region of interest
- Measuring current zone occupancy
- Defining a virtual counting line
- Detecting directional line crossings
- Displaying tracker IDs
- Visualizing spatial zones
- Generating an annotated output video
- Reporting final spatial analytics

---

## Core Technologies

This project uses:

- Python
- OpenCV
- NumPy
- Ultralytics YOLOv8
- Supervision
- ByteTrack
- PolygonZone
- LineZone
- Google Colab

---

## Computer Vision Pipeline

The complete architecture is:

```text
Input Video
     ↓
Read Frame
     ↓
YOLOv8
     ↓
Object Detections
     ↓
sv.Detections
     ↓
ByteTrack
     ↓
Persistent Tracker IDs
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
       Visualization
             ↓
        Output Video
```

---

## Object Detection

YOLOv8 detects objects in every video frame.

```python
results = model(
    frame,
    verbose=False
)[0]
```

The predictions are converted into Supervision detections:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

This produces a standardized detection structure containing information such as:

```text
Bounding Boxes
Class IDs
Confidence Scores
```

---

## Object Tracking

Detections are passed to ByteTrack:

```python
detections = tracker.update(
    detections
)
```

Tracking adds persistent identities to detected objects.

Conceptually:

```text
Frame 100 → Vehicle ID 4
Frame 101 → Vehicle ID 4
Frame 102 → Vehicle ID 4
Frame 103 → Vehicle ID 4
```

The system can therefore analyze the movement of individual objects through time.

---

## PolygonZone

A `PolygonZone` represents an area inside the video frame.

Example:

```python
POLYGON_LEFT = np.array([
    [0, video_info.height // 2],
    [video_info.width // 2, video_info.height // 2],
    [video_info.width // 2, video_info.height],
    [0, video_info.height],
])
```

The zone is created with:

```python
polygon_zone = sv.PolygonZone(
    polygon=POLYGON_LEFT
)
```

---

## Polygon Occupancy

The polygon is triggered using tracked detections:

```python
polygon_zone.trigger(
    detections=detections
)
```

The current number of objects inside the region is available through:

```python
polygon_zone.current_count
```

This value represents **instantaneous occupancy**.

It answers:

```text
How many objects are inside this area right now?
```

---

## LineZone

A `LineZone` represents a virtual counting boundary.

The line is defined using two points:

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

The counting line is created with:

```python
line_zone = sv.LineZone(
    start=line_start,
    end=line_end
)
```

---

## Crossing Detection

Tracked detections are passed to:

```python
line_zone.trigger(
    detections=detections
)
```

LineZone maintains directional counters:

```python
line_zone.in_count
line_zone.out_count
```

Unlike polygon occupancy, these values accumulate throughout the video.

---

## PolygonZone vs. LineZone

| Feature | PolygonZone | LineZone |
|---|---|---|
| Represents | Area | Boundary |
| Measures | Presence | Crossings |
| Count type | Current | Accumulated |
| Main value | `current_count` | `in_count`, `out_count` |
| Analytics | Occupancy | Flow |

The difference can be summarized as:

```text
PolygonZone:
How many objects are HERE NOW?

LineZone:
How many objects PASSED HERE?
```

---

## Why Tracking Is Required

Spatial events become much more useful when detections have persistent identities.

Without tracking:

```text
Frame 1 → Vehicle
Frame 2 → Vehicle
Frame 3 → Vehicle
```

The application cannot reliably determine whether these detections represent the same object.

With tracking:

```text
Frame 1 → ID 7
Frame 2 → ID 7
Frame 3 → ID 7
Frame 4 → ID 7 crosses line
```

The crossing can now be treated as an event associated with a specific tracked object.

---

## Annotation Pipeline

The project combines several visualization layers:

```text
Original Frame
      ↓
Bounding Boxes
      ↓
Tracker ID Labels
      ↓
PolygonZone
      ↓
LineZone
      ↓
Final Annotated Frame
```

Supervision provides the main visualization components:

```python
sv.BoxAnnotator()
sv.LabelAnnotator()
sv.PolygonZoneAnnotator()
sv.LineZoneAnnotator()
```

---

## Input Video

The project uses:

```text
assets/input/vehicles.mp4
```

The video used during development contains highway traffic and provides a useful scenario for testing:

- Vehicle detection
- Vehicle tracking
- Zone occupancy
- Directional crossing counts

---

## Tested Video Information

The practical video used during development has:

```text
Resolution: 3840 × 2160
FPS: 25
Total Frames: 538
```

---

## Expected Output

The project generates:

```text
assets/output/zones_counting_analytics.mp4
```

The output video contains:

- Detected objects
- Bounding boxes
- Persistent tracker IDs
- PolygonZone visualization
- Current polygon occupancy
- LineZone visualization
- Directional crossing counters

---

## Tested Results

During the course practical, the combined spatial analytics pipeline produced:

```text
Final polygon occupancy: 1
Crossings Down: 3
Crossings Up: 3
Total Crossings: 6
```

These values demonstrate the difference between occupancy and accumulated flow.

---

## Project Structure

```text
05-Zones-and-Counting-Analytics/
│
├── README.md
│
├── zones_counting_analytics.py
│
└── assets/
    │
    ├── README.md
    │
    ├── input/
    │   ├── README.md
    │   └── vehicles.mp4
    │
    └── output/
        └── README.md
```

---

## Processing Logic

The application follows this sequence for every frame:

```text
1. Read Video Frame
        ↓
2. Run YOLOv8
        ↓
3. Convert Results to sv.Detections
        ↓
4. Update ByteTrack
        ↓
5. Keep Confirmed Tracker IDs
        ↓
6. Trigger PolygonZone
        ↓
7. Trigger LineZone
        ↓
8. Create Tracker Labels
        ↓
9. Draw Bounding Boxes
        ↓
10. Draw Tracker IDs
        ↓
11. Draw PolygonZone
        ↓
12. Draw LineZone
        ↓
13. Write Annotated Frame
```

---

## Applications

This type of pipeline can be adapted for:

- Traffic monitoring
- Vehicle counting
- Parking analytics
- Building entrances
- People counting
- Retail analytics
- Queue monitoring
- Restricted-area monitoring
- Warehouse analytics
- Industrial safety
- Pedestrian flow analysis
- Transportation systems

---

## Relationship to Previous Projects

This project builds directly on the earlier repository projects.

```text
01 — Object Detection
        ↓
02 — Visualization
        ↓
03 — Detection Filtering
        ↓
04 — Object Tracking
        ↓
05 — Zones and Counting Analytics
```

Each project introduces another layer of computer vision reasoning.

The progression is:

```text
Detection
    ↓
Visualization
    ↓
Filtering
    ↓
Tracking
    ↓
Spatial Analytics
```

---

## Related Course Material

Course notes:

[`../../08-course-notes/05-Zones-and-Counting/`](../../08-course-notes/05-Zones-and-Counting/)

Code examples:

[`../../04-examples/05-Zones-and-Counting/`](../../04-examples/05-Zones-and-Counting/)

Class recording:

[Watch — Zones and Counting](https://youtu.be/43i0z9b81Z4)

---

## Learning Outcome

After completing this project, the complete conceptual workflow becomes:

```text
Video
  ↓
Detection
  ↓
Filtering
  ↓
Tracking
  ↓
Persistent Identity
  ↓
Spatial Reasoning
  ↓
Occupancy
  +
Crossing Events
  ↓
Video Analytics
```

This project demonstrates how individual computer vision techniques can be combined into a practical **spatial video analytics system**.

---

## Author

**Peyman Miyandashti**

SAM3 Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
