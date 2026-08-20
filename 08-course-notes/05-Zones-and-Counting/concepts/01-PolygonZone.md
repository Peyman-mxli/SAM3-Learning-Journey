# PolygonZone

## Introduction

`PolygonZone` is a component provided by the **Supervision** computer vision library that allows us to define a polygon-shaped region inside an image or video frame.

It determines which detected or tracked objects are currently located inside that region.

The main question answered by `PolygonZone` is:

> How many objects are inside this area right now?

This makes `PolygonZone` especially useful for measuring **occupancy** or **current presence**.

---

## Basic Concept

A polygon zone can be imagined as a virtual area placed over a video.

```text
┌─────────────────────────────────────────────┐
│                                             │
│               VIDEO FRAME                   │
│                                             │
│     ┌───────────────────┐                   │
│     │   POLYGON ZONE    │                   │
│     │                   │                   │
│     │   Car A   Car B   │      Car C        │
│     │                   │                   │
│     └───────────────────┘                   │
│                                             │
└─────────────────────────────────────────────┘
```

In this example:

```text
Objects detected: 3
Objects inside PolygonZone: 2
Current count: 2
```

`Car C` is detected by the object detector, but it is outside the monitored region.

---

## Defining a Polygon

A polygon is represented by a collection of `(x, y)` coordinates.

Example from this session:

```python
POLYGON_LEFT = np.array([
    [0,                     video_info.height // 2],
    [video_info.width // 2, video_info.height // 2],
    [video_info.width // 2, video_info.height],
    [0,                     video_info.height],
])
```

Each coordinate represents one vertex of the polygon.

Conceptually:

```text
[x, y]
```

The vertices are connected together in order to form the monitored region.

---

## Image Coordinate System

Image coordinates normally begin at the top-left corner.

```text
(0,0) ───────────────────────────→ X
  │
  │
  │
  │
  ↓
  Y
```

This means:

- `x` increases toward the right.
- `y` increases toward the bottom.

For a video with a resolution of:

```text
Width  = 1280
Height = 720
```

the approximate corner coordinates are:

```text
Top-left     → (0, 0)
Top-right    → (1280, 0)
Bottom-left  → (0, 720)
Bottom-right → (1280, 720)
```

Understanding this coordinate system is important when designing custom zones.

---

## Using the Video Dimensions

Instead of hardcoding pixel values, zones can be defined relative to the video dimensions.

For example:

```python
video_info.width // 2
```

represents half of the video width.

Similarly:

```python
video_info.height // 2
```

represents half of the video height.

This makes the zone definition easier to understand and adapt.

For example:

```python
POLYGON_LEFT = np.array([
    [0,                     video_info.height // 2],
    [video_info.width // 2, video_info.height // 2],
    [video_info.width // 2, video_info.height],
    [0,                     video_info.height],
])
```

creates a polygon covering approximately the lower-left section of the video.

---

## Creating the PolygonZone

After defining the coordinates, the zone can be created using:

```python
zone = sv.PolygonZone(
    polygon=POLYGON_LEFT
)
```

The `polygon` parameter contains the vertices defining the monitored region.

Conceptually:

```text
Polygon Coordinates
        ↓
sv.PolygonZone
        ↓
Spatial Monitoring Area
```

---

## Polygon Shapes

A polygon is not limited to rectangles.

It can represent many different shapes.

For example:

```text
Rectangle
Triangle
Trapezoid
Irregular Polygon
```

This is useful because real-world regions are often not perfectly rectangular.

Examples include:

- Road lanes
- Parking areas
- Sidewalks
- Building entrances
- Restricted areas
- Industrial safety zones

---

## Visualizing the Zone

Supervision provides:

```python
sv.PolygonZoneAnnotator
```

to visualize the zone.

Example:

```python
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.RED,
    thickness=4
)
```

The zone can then be drawn on a frame:

```python
frame_with_zone = zone_annotator.annotate(
    scene=frame.copy()
)
```

This allows us to visually confirm that the polygon is positioned correctly.

---

## Why Visual Verification Matters

Polygon coordinates are defined in pixel space.

A small mistake in the coordinates can place the zone in the wrong part of the image.

A useful workflow is:

```text
Define Coordinates
        ↓
Create PolygonZone
        ↓
Read First Video Frame
        ↓
Draw Polygon
        ↓
Display Frame
        ↓
Verify Position
        ↓
Adjust Coordinates if Necessary
```

This should normally be done before processing an entire video.

---

## Using `trigger()`

The main method used with `PolygonZone` is:

```python
zone.trigger(
    detections=detections
)
```

This evaluates the detections and determines which objects are inside the polygon.

Example:

```python
inside_zone = zone.trigger(
    detections=detections
)
```

The returned value is a Boolean mask.

---

## Boolean Mask

Suppose four objects were detected.

The zone might return:

```text
[True, False, True, False]
```

This means:

```text
Detection 0 → inside
Detection 1 → outside
Detection 2 → inside
Detection 3 → outside
```

Therefore:

```text
Total detections = 4
Inside zone      = 2
Outside zone     = 2
```

---

## Filtering Detections

The Boolean mask can be applied directly to `sv.Detections`.

Example:

```python
inside_zone = zone.trigger(
    detections=detections
)

detections_inside_zone = detections[
    inside_zone
]
```

The resulting object contains only detections associated with the polygon.

Conceptually:

```text
All Detections
      ↓
PolygonZone.trigger()
      ↓
Boolean Mask
      ↓
Filter
      ↓
Detections Inside Zone
```

---

## Understanding `current_count`

`PolygonZone` maintains a property called:

```python
zone.current_count
```

This represents the number of objects currently inside the polygon.

For example:

```python
inside_zone = zone.trigger(
    detections=detections
)

print(inside_zone.sum())
print(zone.current_count)
```

The two values should represent the same current occupancy.

---

## Instantaneous Count

An important characteristic of `PolygonZone` is that its count represents the **current frame**.

For example:

```text
Frame 1 → 2 objects
Frame 2 → 3 objects
Frame 3 → 5 objects
Frame 4 → 4 objects
Frame 5 → 1 object
```

The count changes as objects enter and leave the polygon.

It is not intended to represent the total number of unique objects that have ever visited the area.

That is an important distinction.

---

## Occupancy

The concept measured by `PolygonZone` can be described as:

```text
Occupancy
```

Occupancy answers:

> How many objects are currently occupying this region?

Examples include:

```text
How many cars are currently in this lane?

How many people are currently inside this room?

How many vehicles are currently waiting at this intersection?

How many workers are currently inside this safety zone?
```

---

## PolygonZone and Tracking

In this session, tracking is performed before triggering the polygon.

The correct processing order is:

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
Zone Filtering
  ↓
Annotation
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

---

## Tracker IDs

After tracking, detections can contain persistent identifiers:

```python
detections.tracker_id
```

For example:

```text
ID:4
ID:8
ID:15
```

These identifiers allow the system to recognize the same object across multiple frames.

After filtering by the polygon, labels can be created with:

```python
labels = [
    f"ID:{tracker_id}"
    for tracker_id in detections_inside_zone.tracker_id
]
```

Now only tracked objects inside the zone are displayed.

---

## Complete Zone Callback Concept

The basic processing logic used in the session is:

```python
def callback_zone(frame: np.ndarray, _: int) -> np.ndarray:

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

    detections_inside_zone = detections[
        inside_zone
    ]

    labels = [
        f"ID:{tracker_id}"
        for tracker_id in detections_inside_zone.tracker_id
    ]

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections_inside_zone
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections_inside_zone,
        labels=labels
    )

    annotated = zone_annotator.annotate(
        scene=annotated
    )

    return annotated
```

This combines:

```text
Detection
    +
Tracking
    +
Zone Analysis
    +
Filtering
    +
Visualization
```

---

## Changing the Polygon

One of the experiments in the session demonstrates that changing the polygon coordinates completely changes the monitored region.

For example, a right-side zone can be created with:

```python
POLYGON_RIGHT = np.array([
    [video_info.width // 2, video_info.height // 2],
    [video_info.width,      video_info.height // 2],
    [video_info.width,      video_info.height],
    [video_info.width // 2, video_info.height],
])
```

Then:

```python
zone_right = sv.PolygonZone(
    polygon=POLYGON_RIGHT
)
```

This allows different parts of the same scene to be analyzed independently.

---

## Multiple Zones

The same idea can be extended to multiple polygons.

For example:

```text
                VIDEO FRAME

┌──────────────────────────────────────┐
│                                      │
│                                      │
│   ┌───────────┐     ┌───────────┐    │
│   │  ZONE A   │     │  ZONE B   │    │
│   │           │     │           │    │
│   └───────────┘     └───────────┘    │
│                                      │
└──────────────────────────────────────┘
```

This could represent:

```text
Zone A → left traffic lane
Zone B → right traffic lane
```

Each zone can maintain its own current occupancy.

---

## Real-World Applications

### Traffic Monitoring

A polygon can represent a road lane.

The system can measure how many vehicles are currently occupying that lane.

---

### Parking Management

A polygon can represent:

- One parking space
- A parking section
- An entire parking area

The system can monitor current occupancy.

---

### Retail Analytics

A polygon can represent a department or section of a store.

The system can measure how many customers are currently inside that area.

---

### Security

A polygon can represent a restricted region.

Tracked objects entering that area can be detected and analyzed.

---

### Industrial Safety

A polygon can define a dangerous machine area.

The system can determine whether workers or vehicles are currently inside the restricted zone.

---

## PolygonZone vs Object Detection

Object detection alone answers:

```text
"What objects are visible?"
```

PolygonZone adds spatial context:

```text
"Which of those objects are inside this specific area?"
```

Therefore:

```text
Detection
    ↓
Object Awareness

Detection + PolygonZone
    ↓
Spatial Awareness
```

---

## Key Takeaways

The most important concepts about `PolygonZone` are:

1. A polygon defines a spatial region using `(x, y)` coordinates.
2. `sv.PolygonZone` creates the monitored region.
3. `zone.trigger()` evaluates detections against the polygon.
4. `trigger()` returns a Boolean mask.
5. The Boolean mask can filter `sv.Detections`.
6. `zone.current_count` represents current occupancy.
7. The count changes from frame to frame.
8. Polygon zones measure presence rather than accumulated crossings.
9. Tracking can provide persistent identities for objects inside the zone.
10. Polygon zones can represent real-world regions such as lanes, rooms, parking areas, and restricted spaces.

---

## Summary

`PolygonZone` transforms object detection into spatial analysis.

Instead of analyzing every detected object equally, we can focus on objects interacting with a specific region.

The basic idea is:

```text
Detect Objects
      ↓
Track Objects
      ↓
Define Spatial Region
      ↓
Check Zone Membership
      ↓
Filter Objects
      ↓
Count Current Occupancy
```

This provides the foundation for applications involving **occupancy monitoring, traffic analysis, security, retail analytics, parking management, and industrial safety**.

---

## Related Concepts

- [Concepts Overview](./README.md)
- [LineZone](./02-LineZone.md)
- [Occupancy vs Flow](./03-Occupancy-vs-Flow.md)
- [Tracking with Zones](./04-Tracking-with-Zones.md)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
