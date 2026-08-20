# Session 06 — Zones and Counting

This session introduces **spatial zones and object counting** using computer vision, object detection, and object tracking.

The main objective is to understand how detected and tracked objects can be analyzed based on their position inside specific regions of a video.

Two important Supervision components are introduced:

- `PolygonZone`
- `LineZone`

These tools make it possible to answer two different questions:

> How many objects are inside a specific area right now?

and

> How many objects have crossed a specific line over time?

---

## Session Information

**Course:** SAM3 — Computer Vision with Segment Anything Model 3  
**Session:** 06  
**Module:** 02  
**Notebook:** `02_c_zonas_conteo.ipynb`  
**Topic:** Zones and Counting  
**Estimated notebook time:** 30 minutes  
**Estimated class duration:** 1 hour

---

## Learning Objectives

By the end of this session, I learned how to:

- Define polygonal regions inside an image or video
- Detect whether tracked objects are inside a region
- Count objects currently present inside a zone
- Create virtual counting lines
- Count objects crossing a line
- Understand the difference between occupancy and accumulated flow
- Combine object detection with object tracking
- Combine tracking with `PolygonZone`
- Combine tracking with `LineZone`
- Visualize zones and counters directly on video frames
- Use multiple counting mechanisms in the same video pipeline

---

# 1. Why Use Zones?

Object detection tells us:

> What objects are visible?

Object tracking tells us:

> Which object is which across multiple frames?

Zones add another layer of information:

> Where are those objects located?

This allows a computer vision system to understand activity inside specific areas.

For example, a traffic monitoring system could define different regions for:

- Left lane
- Right lane
- Parking area
- Intersection
- Pedestrian crossing
- Building entrance
- Restricted area

The system can then analyze only the objects interacting with those regions.

---

# 2. Two Types of Zones

In this session, two main types of zones are used:

1. `PolygonZone`
2. `LineZone`

Although both are used for spatial analysis, they answer different questions.

---

## PolygonZone

A `PolygonZone` represents an area of the image.

It answers:

> How many objects are inside this area right now?

The count can change from frame to frame.

Conceptually:

```text
PolygonZone → current presence / occupancy
```

Example:

```text
Frame 1 → 3 vehicles inside
Frame 2 → 4 vehicles inside
Frame 3 → 2 vehicles inside
```

The important property is:

```python
zone.current_count
```

This represents the number of detected or tracked objects currently inside the polygon.

---

## LineZone

A `LineZone` works differently.

Instead of representing an area, it represents a virtual line.

It answers:

> How many objects have crossed this line?

Conceptually:

```text
LineZone → accumulated crossings / flow
```

For example:

```text
Vehicle 1 crosses → count = 1
Vehicle 2 crosses → count = 2
Vehicle 3 crosses → count = 3
```

Unlike `PolygonZone`, the count is accumulated throughout the video.

The main properties are:

```python
line_zone.in_count
line_zone.out_count
```

These counters represent crossings in the two possible directions.

---

# 3. PolygonZone vs LineZone

The easiest way to understand the difference is:

```text
PolygonZone:
"How many objects are HERE NOW?"

LineZone:
"How many objects PASSED HERE?"
```

Another useful analogy is:

```text
PolygonZone → parking area occupancy

LineZone → entrance turnstile
```

A polygon measures **presence**.

A line measures **movement across a boundary**.

---

# 4. Defining a Polygon

A polygon is defined using a collection of `(x, y)` coordinates.

In the practical, the lower-left half of the video was used:

```python
POLYGON_LEFT = np.array([
    [0,                     video_info.height // 2],
    [video_info.width // 2, video_info.height // 2],
    [video_info.width // 2, video_info.height],
    [0,                     video_info.height],
])
```

Each coordinate represents one vertex of the polygon.

The points are connected together to create the region.

Polygons can represent many shapes, including:

- Rectangles
- Triangles
- Trapezoids
- Irregular regions

This makes polygon zones useful when the monitored area does not have a simple rectangular shape.

---

# 5. Creating a PolygonZone

Once the polygon coordinates are defined, the zone can be created with:

```python
zone = sv.PolygonZone(
    polygon=POLYGON_LEFT
)
```

The zone can then be visualized using:

```python
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.RED,
    thickness=4
)
```

The annotator draws the polygon on the frame and displays its current object count.

---

# 6. Detecting Objects Inside the Zone

The important operation is:

```python
inside_zone = zone.trigger(
    detections=detections
)
```

`trigger()` returns a Boolean mask.

Example:

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

The mask can then be used to filter the detections:

```python
detections_inside_zone = detections[
    inside_zone
]
```

Only the objects currently inside the polygon remain.

---

# 7. Understanding `current_count`

Every time `trigger()` is called, the zone updates:

```python
zone.current_count
```

For example:

```python
mask = zone.trigger(
    detections=detections
)

print(mask.sum())
print(zone.current_count)
```

Both values represent the number of objects currently inside the zone.

Conceptually:

```text
mask.sum()
    ↓
Number of True values

zone.current_count
    ↓
Current occupancy stored by PolygonZone
```

---

# 8. Tracking Before Zone Analysis

One of the most important concepts from this session is the processing order.

The tracker should be applied before the zone is triggered.

Correct order:

```text
Video Frame
    ↓
YOLO Detection
    ↓
Supervision Detections
    ↓
Object Tracking
    ↓
Zone Trigger
    ↓
Filtering / Counting
    ↓
Annotation
    ↓
Output Frame
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

Tracking provides persistent object identities that can be used by the zone logic.

---

# 9. Tracker IDs

After tracking, objects can contain a persistent:

```python
tracker_id
```

During practical testing, new detections initially received:

```text
-1
```

until ByteTrack established confirmed tracks.

Confirmed objects then received persistent IDs such as:

```text
0
1
2
4
5
```

For the final practical pipeline, unconfirmed `-1` tracks were filtered before zone analysis:

```python
if detections.tracker_id is not None:
    confirmed_mask = detections.tracker_id != -1
    detections = detections[confirmed_mask]
```

Labels can then be generated using:

```python
labels = [
    f"ID:{tracker_id}"
    for tracker_id in detections.tracker_id
]
```

Instead of simply seeing:

```text
car
car
car
```

we can identify individual tracked objects:

```text
ID:3
ID:7
ID:12
```

This connects the concepts from the previous Object Tracking session directly with zone analysis.

---

# 10. LineZone

`LineZone` creates a virtual counting boundary.

In the practical, a horizontal line across the middle of the video was defined using:

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

For the tested 3840 × 2160 video, this produced:

```text
Start: (0, 1080)
End:   (3840, 1080)
```

Objects crossing this line can now be counted.

---

# 11. Triggering LineZone

The crossing logic is updated with:

```python
line_zone.trigger(
    detections=detections
)
```

Unlike `PolygonZone`, we inspect:

```python
line_zone.in_count
line_zone.out_count
```

These values accumulate during video processing.

The practical ultimately produced:

```text
Crossings Down: 3
Crossings Up:   3
```

This provides directional traffic information.

---

# 12. Why Tracking Matters for LineZone

Without tracking, the system sees detections independently in every frame.

For example:

```text
Frame 100 → car
Frame 101 → car
Frame 102 → car
Frame 103 → car
```

Without tracking, it is difficult to determine whether these detections represent four different cars or the same car across four frames.

Tracking solves this by assigning a persistent identity.

Example:

```text
Frame 100 → ID 7
Frame 101 → ID 7
Frame 102 → ID 7
Frame 103 → ID 7
```

Now the system can determine when **ID 7 actually crosses the line**.

---

# 13. Combining PolygonZone and LineZone

Both types of zones can be used in the same video pipeline.

The same tracked detections can trigger both systems:

```python
polygon_zone.trigger(
    detections=detections
)

line_zone.trigger(
    detections=detections
)
```

The frame can then display:

- Bounding boxes
- Persistent tracker IDs
- Polygon region
- Current polygon occupancy
- Counting line
- Incoming crossings
- Outgoing crossings

This creates a more complete traffic-analysis pipeline.

---

# 14. Combined Processing Pipeline

The complete pipeline implemented and tested during this session was:

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
Confirmed Tracked Detections
    ↓
┌─────────────────────────┐
│                         │
↓                         ↓
PolygonZone            LineZone
│                         │
↓                         ↓
Current Occupancy      Crossing Count
│                         │
└────────────┬────────────┘
             ↓
        Annotation
             ↓
       Output Video
```

This is an important pattern for building real-world computer vision systems.

---

# 15. Practical Implementation

A complete practical implementation was created for this session and tested in Google Colab.

The implementation combines:

- YOLOv8 object detection
- Supervision detections
- ByteTrack object tracking
- Confirmed tracker IDs
- `PolygonZone`
- `LineZone`
- Bounding-box annotation
- Tracker-ID labels
- Occupancy counting
- Directional crossing counting
- Full video processing

The implementation is available here:

[View `zones_and_counting_practical.py`](./practical/zones_and_counting_practical.py)

The complete practical documentation is available here:

[View Practical README](./practical/README.md)

---

## Tested Environment

The practical was successfully tested with:

```text
OpenCV:       4.14.0
NumPy:        2.2.2
Supervision:  0.30.0
Ultralytics:  8.4.128
YOLO Model:   yolov8n.pt
Tracker:      ByteTrackTracker
```

The environment test completed successfully before processing the video.

---

## Input Video

The practical uses:

```text
practical/assets/input/vehicles.mp4
```

The original video used during testing had:

```text
Resolution:   3840 × 2160
FPS:          25.0
Total Frames: 538
Duration:     approximately 21.5 seconds
```

A compressed repository-friendly version is stored in the practical input assets.

[View Input Assets](./practical/assets/input/README.md)

---

## PolygonZone Practical Result

The red polygon was positioned over the lower-left portion of the highway.

During the single-frame tracking test, the system successfully produced:

```text
Confirmed tracked objects: 3
Objects inside zone: 2
zone.current_count: 2
IDs inside zone: [4 5]
```

The complete PolygonZone video processing successfully reached:

```text
538 / 538 frames
```

and produced a final occupancy value of:

```text
Final current_count: 1
```

---

## LineZone Practical Result

The LineZone was positioned horizontally at:

```text
y = 1080
```

across the complete width of the frame:

```text
Start: (0, 1080)
End:   (3840, 1080)
```

The complete LineZone video processing successfully reached:

```text
538 / 538 frames
```

with:

```text
Crossings Down: 3
Crossings Up:   3
```

---

## Final Combined Result

The extension challenge was completed by combining `PolygonZone` and `LineZone` in the same video-processing callback.

The final combined processing completed successfully:

```text
Processing video: 100%
538 / 538 frames

Combined processing completed.

Final polygon occupancy: 1
Crossings Down: 3
Crossings Up: 3
```

The final output is available here:

[Watch the Final Combined Output](./practical/assets/output/vehicles_combined.mp4)

The output demonstrates:

```text
YOLOv8 Detection
        +
ByteTrack Tracking
        +
Persistent IDs
        +
PolygonZone Occupancy
        +
LineZone Crossings
        =
Spatial Video Analytics
```

---

# 16. Real-World Applications

Zone-based counting can be applied to many real-world problems.

## Traffic Monitoring

Use polygon zones to measure:

- Lane occupancy
- Congestion
- Vehicles waiting at intersections

Use line zones to measure:

- Vehicles entering a road
- Vehicles leaving a road
- Traffic flow

## Retail Analytics

Polygon zones can measure:

- Customers inside departments
- People near displays
- Queue occupancy

Line zones can measure:

- Store entrances
- Store exits
- Customer traffic

## Parking Systems

Polygon zones can represent:

- Parking spaces
- Parking sections
- Restricted areas

The system can determine whether each area is occupied.

## Security Systems

Zones can monitor:

- Restricted areas
- Building entrances
- Gates
- Perimeters

Tracking makes it possible to follow the same object while it interacts with multiple zones.

## Industrial Monitoring

Zone analysis can help detect:

- Workers entering dangerous areas
- Vehicles crossing safety boundaries
- Objects moving between production areas
- Equipment occupying restricted regions

---

# 17. Occupancy vs Flow

One of the most important concepts from this session is the distinction between **occupancy** and **flow**.

## Occupancy

Measures how many objects are currently present.

```text
PolygonZone
```

Example question:

> How many vehicles are currently inside this parking area?

## Flow

Measures how many objects moved through a boundary.

```text
LineZone
```

Example question:

> How many vehicles entered the parking lot?

These are different metrics and should not be confused.

---

# 18. Combined Concept

The main concepts from this session can be summarized as:

```text
Object Detection
        +
Object Tracking
        +
Spatial Zones
        +
Counting
        =
Spatial Video Analytics
```

Detection identifies objects.

Tracking maintains their identity.

Zones define where events matter.

Counting converts those events into useful measurements.

---

# 19. Extension Challenge

The session challenge was to display both a `PolygonZone` and a `LineZone` in the same video.

Both triggers were called using the same tracked detections:

```python
polygon_zone.trigger(
    detections=detections
)

line_zone.trigger(
    detections=detections
)
```

The annotators were then applied to the same frame:

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

This challenge was successfully completed.

The final implementation combines **occupancy analysis** and **flow analysis** in a single computer vision pipeline.

---

# 20. Concepts Documentation

Detailed concept notes for this session are available in the `concepts` directory.

- [Concepts Overview](./concepts/README.md)
- [PolygonZone](./concepts/01-PolygonZone.md)
- [LineZone](./concepts/02-LineZone.md)
- [Occupancy vs Flow](./concepts/03-Occupancy-vs-Flow.md)
- [Tracking with Zones](./concepts/04-Tracking-with-Zones.md)
- [PolygonZone Trigger and Filtering](./concepts/05-PolygonZone-Trigger-and-Filtering.md)
- [Zone Annotation and Visualization](./concepts/06-Zone-Annotation-and-Visualization.md)
- [Combining PolygonZone and LineZone](./concepts/07-Combining-PolygonZone-and-LineZone.md)

---

# 21. Practical Resources

The practical directory contains the complete tested implementation and assets.

- [Practical README](./practical/README.md)
- [Python Implementation](./practical/zones_and_counting_practical.py)
- [Assets Documentation](./practical/assets/README.md)
- [Input Assets](./practical/assets/input/README.md)
- [Output Assets](./practical/assets/output/README.md)
- [Final Combined Video](./practical/assets/output/vehicles_combined.mp4)

Directory structure:

```text
practical/
├── README.md
├── zones_and_counting_practical.py
└── assets/
    ├── README.md
    ├── input/
    │   ├── README.md
    │   └── vehicles.mp4
    └── output/
        ├── README.md
        └── vehicles_combined.mp4
```

---

# 22. Class Recording

The completed Zones and Counting practical is available on YouTube:

[Watch — SAM3: Zonas y Conteo | PolygonZone, LineZone y ByteTrack](https://youtu.be/43i0z9b81Z4)

Additional recording documentation:

[View `CLASS-RECORDING.md`](./CLASS-RECORDING.md)

---

# 23. Key Takeaways

From this session, I learned that:

1. `PolygonZone` measures current object presence inside an area.
2. `LineZone` measures accumulated crossings.
3. Polygon zones are defined using pixel coordinates.
4. `zone.trigger()` returns a Boolean mask.
5. `zone.current_count` represents current occupancy.
6. `line_zone.in_count` and `line_zone.out_count` represent directional crossings.
7. Tracking should happen before zone analysis.
8. Persistent `tracker_id` values make movement analysis possible.
9. Unconfirmed tracker IDs can be filtered before zone analysis.
10. Polygon and line zones can operate simultaneously.
11. Occupancy and flow are different spatial analytics metrics.
12. YOLO, ByteTrack, and Supervision can be combined into a complete video analytics pipeline.

---

# Session Summary

This session extended the object-tracking pipeline by introducing **spatial reasoning**.

Instead of only detecting and tracking objects, the system can now understand whether objects:

- Enter an area
- Leave an area
- Remain inside an area
- Cross a virtual boundary

The complete idea is:

```text
Detect
   ↓
Track
   ↓
Analyze Position
   ↓
Count
   ↓
Visualize
```

This creates the foundation for more advanced computer vision applications such as:

- Traffic analytics
- Occupancy monitoring
- Security systems
- Retail analytics
- Parking management
- Industrial automation

The practical implementation was successfully tested from beginning to end using a real traffic video.

---

## Session Status

**Session 06 — Zones and Counting: COMPLETED**

```text
Concept Documentation          ✓
PolygonZone                    ✓
LineZone                       ✓
YOLOv8 Detection               ✓
ByteTrack Tracking             ✓
Persistent Tracker IDs         ✓
Confirmed Track Filtering      ✓
Occupancy Counting             ✓
Directional Crossing Counting ✓
Combined Pipeline              ✓
Google Colab Testing           ✓
Input Video                    ✓
Final Combined Output Video    ✓
Python Implementation          ✓
Extension Challenge            ✓
Class Recording                ✓
GitHub Documentation           ✓
```

Final tested results:

```text
Final polygon occupancy: 1
Crossings Down: 3
Crossings Up: 3
```

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Ultralytics YOLOv8
- Supervision
- ByteTrack
- Google Colab

---

## Related Course Material

- [SAM3 Course](https://sam-3-vision-computacional-5tao.vercel.app/)
- [Session 06](https://sam-3-vision-computacional-5tao.vercel.app/modulo/06)
- [Class Recording](https://youtu.be/43i0z9b81Z4)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
