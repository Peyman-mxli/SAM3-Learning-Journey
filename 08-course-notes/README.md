# SAM3 Course Notes

This directory contains my organized notes, concepts, practical exercises, notebooks, class recordings, and supporting material from the **SAM3 — Computer Vision with Segment Anything Model 3** course.

The purpose of this section is to document each course session in a structured way, transforming the original class material into a reusable computer vision learning reference.

---

## Course Sessions

| # | Session | Status |
|---|---|---|
| 00 | [Agentic AI Programming](./00-Agentic-AI-Programming/) | Completed |
| 01 | [Introduction to Supervision](./01-Introduction-to-Supervision/) | Completed |
| 02 | [Annotation and Visualization](./02-Annotation-and-Visualization/) | Completed |
| 03 | [Filtering and Manipulating Detections](./03-Filtering-and-Manipulating-Detections/) | Completed |
| 04 | [Object Tracking](./04-Object-Tracking/) | Completed |
| 05 | [Zones and Counting](./05-Zones-and-Counting/) | Completed |

---

# Session 00 — Agentic AI Programming

[`00-Agentic-AI-Programming/`](./00-Agentic-AI-Programming/)

This session introduces AI-assisted programming methodologies and the use of AI tools during software and computer vision development.

Topics include:

- Agentic AI programming
- AI-assisted code generation
- Prompt engineering
- AI-assisted debugging
- Error analysis
- OpenCV
- Computer vision development workflows

**Status:** Completed

---

# Session 01 — Introduction to Supervision

[`01-Introduction-to-Supervision/`](./01-Introduction-to-Supervision/)

This session introduces the **Supervision** library and its role in computer vision workflows.

Topics include:

- Supervision
- YOLOv8
- Ultralytics
- OpenCV
- `sv.Detections`
- Bounding boxes
- Class IDs
- Confidence scores
- Confidence thresholds
- Detection visualization
- Model comparison
- JSON prediction export

The session establishes the basic workflow:

```text
Input Image
     ↓
YOLO
     ↓
Predictions
     ↓
sv.Detections
     ↓
Processing
     ↓
Visualization
```

**Status:** Completed

---

# Session 02 — Annotation and Visualization

[`02-Annotation-and-Visualization/`](./02-Annotation-and-Visualization/)

This session explores how object-detection results can be transformed into clear and useful visual representations using **Supervision Annotators**.

Topics include:

- `BoxAnnotator`
- `RoundBoxAnnotator`
- `HaloAnnotator`
- `BlurAnnotator`
- `BoxCornerAnnotator`
- `EllipseAnnotator`
- `DotAnnotator`
- `LabelAnnotator`
- Annotation customization
- Color palettes
- Bounding-box thickness
- Label text scale
- Class and confidence labels
- Annotation layers
- Annotation order
- Multi-Annotator visualization pipelines
- Detection vs. visualization

The session develops the workflow:

```text
Input Image
     ↓
YOLOv8
     ↓
Detection Results
     ↓
sv.Detections
     ↓
Supervision Annotators
     ↓
Annotation Layers
     ↓
Final Visualization
```

This lesson also connects directly to the practical project:

```text
05-projects/
└── 02-Multi-Annotator-Visualization-Pipeline/
```

**Status:** Completed

---

# Session 03 — Filtering and Manipulating Detections

[`03-Filtering-and-Manipulating-Detections/`](./03-Filtering-and-Manipulating-Detections/)

This session focuses on **post-processing object detections** and selecting exactly which predictions should continue through a computer vision pipeline.

Instead of using every raw prediction produced by YOLO, detections can be filtered according to confidence, class, size, position, and other conditions.

Topics include:

- `sv.Detections` filtering
- NumPy-style Boolean masks
- Confidence filtering
- Class filtering
- Combining multiple conditions
- Element-wise Boolean operators
- Excluding specific classes
- Bounding-box area filtering
- Detection merging
- `sv.Detections.merge()`
- Non-Maximum Suppression (NMS)
- Intersection over Union (IoU)
- NMS thresholds
- Duplicate detection removal
- Sorting detections by confidence
- `np.argsort()`
- Top-N detection selection
- Bounding-box coordinates
- Bounding-box center calculation
- Spatial filtering
- Left/right image filtering
- Region-based detection logic

The session develops the post-processing workflow:

```text
Input Image
     ↓
YOLO
     ↓
Raw Detections
     ↓
sv.Detections
     ↓
Confidence Filtering
     ↓
Class Filtering
     ↓
Size Filtering
     ↓
Merge + NMS
     ↓
Confidence Ranking
     ↓
Spatial Filtering
     ↓
Final Relevant Detections
```

## Detection Filtering Example

A computer vision application can define increasingly specific requirements:

```text
All Detections
      ↓
Keep Only People
      ↓
Confidence > 60%
      ↓
Area > 5000 px²
      ↓
Remove Duplicate Boxes
      ↓
Keep Only Required Image Region
      ↓
Final Detections
```

This demonstrates that model inference is only one part of an object-detection system.

**Post-processing determines which predictions are actually useful for the application.**

## Lesson Materials

- [Main Lesson Documentation](./03-Filtering-and-Manipulating-Detections/README.md)
- [Concept Documentation](./03-Filtering-and-Manipulating-Detections/concepts/)
- [Practical Exercises](./03-Filtering-and-Manipulating-Detections/practical-exercises/)
- [Original Course Notebook](./03-Filtering-and-Manipulating-Detections/02_a_filtrado_detecciones.ipynb)
- [Class Recording](./03-Filtering-and-Manipulating-Detections/CLASS-RECORDING.md)

**Status:** Completed

---

# Session 04 — Object Tracking

[`04-Object-Tracking/`](./04-Object-Tracking/)

This session extends computer vision from detecting objects in individual images to **tracking objects across consecutive video frames**.

Instead of treating every detection independently, object tracking attempts to maintain the identity of each object while it moves through a video.

The session introduces **ByteTrack** together with Supervision and explores how persistent tracking IDs can be used to follow objects, visualize trajectories, and prepare tracking analytics.

Topics include:

- Object tracking
- Video processing
- Frame-by-frame processing
- Multi-object tracking
- Supervision
- `sv.Detections`
- ByteTrack
- `sv.ByteTrack`
- Object association
- Persistent tracking IDs
- `tracker_id`
- Tracking labels
- Bounding-box tracking
- `BoxAnnotator`
- `LabelAnnotator`
- `TraceAnnotator`
- Object trajectories
- Tracking visualization
- Tracking analytics
- OpenCV video processing
- Video export
- H.264 conversion
- FFmpeg
- Google Colab testing

The conceptual transition is:

```text
Object Detection
       ↓
Independent Detections
       ↓
Object Association
       ↓
ByteTrack
       ↓
Persistent tracker_id
       ↓
Object Tracking
       ↓
Movement History
       ↓
Tracking Analytics
```

## Detection vs. Tracking

Object detection answers:

```text
What objects are visible in this frame?
```

Object tracking adds another question:

```text
Is this the same object that appeared in previous frames?
```

For example:

```text
Frame 1 → person #1
Frame 2 → person #1
Frame 3 → person #1
Frame 4 → person #1
```

Although the object changes position, the tracker attempts to maintain the same identity.

---

## Object Tracking Workflow

The complete conceptual workflow developed in this session is:

```text
Input Video
     ↓
Video Frames
     ↓
Object Detection
     ↓
sv.Detections
     ↓
Detection Filtering
     ↓
ByteTrack
     ↓
tracker_id
     ↓
Tracking Annotations
     ↓
Object Trajectories
     ↓
Tracking Analytics
     ↓
Processed Video
```

---

## ByteTrack

ByteTrack associates detections between consecutive frames and assigns persistent tracking identities.

Conceptually:

```text
Frame N Detections
        ↓
ByteTrack
        ↓
Compare With Previous Tracks
        ↓
Object Association
        ↓
tracker_id
```

This allows multiple objects of the same class to be distinguished.

Example:

```text
person #1
person #2
person #3
```

---

## Object Trajectories

Once an object has a persistent tracking identity, its previous positions can be stored and visualized.

Conceptually:

```text
Previous Positions
        ↓
● → ● → ● → ●
            ↑
      Current Position
```

Object trajectories can support applications such as:

- Movement analysis
- Direction analysis
- Line crossing
- Entrance and exit monitoring
- Object counting
- Traffic analysis
- Behavior analysis
- Tracking analytics

---

## Object Tracking Practical

A complete practical exercise was created for this session.

The practical uses a custom **10-second synthetic demonstration video** containing multiple moving objects.

The exercise focuses directly on tracking behavior by generating known detections and passing them through:

```text
Synthetic Detections
        ↓
sv.Detections
        ↓
ByteTrack
        ↓
tracker_id
        ↓
Bounding Boxes
        ↓
Tracking Labels
        ↓
Object Trajectories
        ↓
Processed Video
```

This approach isolates the tracking stage from object-classification uncertainty and makes ByteTrack behavior easier to study.

---

## Practical Input

The input video is:

```text
practical/assets/input/tracking_demo.mp4
```

Video properties:

```text
Duration: 10 seconds
Resolution: 960 × 540
Frame Rate: 30 FPS
Total Frames: 300
Format: MP4
```

---

## Practical Implementation

The main implementation is:

[`object_tracking_practical.py`](./04-Object-Tracking/practical/object_tracking_practical.py)

The script performs:

1. Video loading
2. Frame-by-frame processing
3. Synthetic detection generation
4. `sv.Detections` creation
5. ByteTrack processing
6. `tracker_id` assignment
7. Bounding-box annotation
8. Tracking label generation
9. Object trajectory visualization
10. Frame information annotation
11. Video export

---

## Practical Result

The practical successfully tracks three moving objects:

```text
object_a #1
object_b #2
object_c #3
```

The tracking pipeline processed:

```text
Width: 960
Height: 540
FPS: 30.0
Frames: 300
```

and completed successfully.

The initial OpenCV output was converted to H.264 for browser compatibility.

Final output:

[`tracked_demo_h264.mp4`](./04-Object-Tracking/practical/assets/output/tracked_demo_h264.mp4)

---

## Google Colab Validation

The practical was executed and verified in Google Colab.

Environment:

```text
OpenCV: 5.0.0
NumPy: 2.0.2
Supervision: 0.30.0
```

Validation result:

```text
Environment test: SUCCESS
```

The final H.264 video was displayed and visually inspected directly inside Google Colab.

---

## Lesson Materials

- [Main Lesson Documentation](./04-Object-Tracking/README.md)
- [Concept Documentation](./04-Object-Tracking/concepts/)
- [Practical Exercises](./04-Object-Tracking/practical/)
- [Practical Documentation](./04-Object-Tracking/practical/README.md)
- [Practical Python Script](./04-Object-Tracking/practical/object_tracking_practical.py)
- [Input Assets](./04-Object-Tracking/practical/assets/input/)
- [Output Assets](./04-Object-Tracking/practical/assets/output/)
- [Final Tracking Video](./04-Object-Tracking/practical/assets/output/tracked_demo_h264.mp4)
- [Class Recording Documentation](./04-Object-Tracking/CLASS-RECORDING.md)
- [Watch the Object Tracking Class Recording on YouTube](https://youtu.be/UXN0l33NqF4)

**Status:** Completed

---

# Session 05 — Zones and Counting

[`05-Zones-and-Counting/`](./05-Zones-and-Counting/)

This session extends object tracking into **spatial video analytics** by defining regions inside a video frame and analyzing how tracked objects interact with those regions.

The session introduces two different types of spatial analysis:

- **PolygonZone** — measures how many objects are currently inside an area.
- **LineZone** — measures how many tracked objects have crossed a virtual line.

The fundamental distinction is:

```text
PolygonZone:
How many objects are inside this area RIGHT NOW?

LineZone:
How many objects have CROSSED this line IN TOTAL?
```

This transforms object tracking into a system capable of measuring both **occupancy** and **flow**.

Topics include:

- Spatial video analytics
- Polygon coordinates
- NumPy polygon definitions
- `sv.PolygonZone`
- `sv.PolygonZoneAnnotator`
- Polygon occupancy
- `zone.trigger()`
- Boolean zone masks
- `zone.current_count`
- Spatial detection filtering
- Tracking with PolygonZone
- Virtual counting lines
- `sv.Point`
- `sv.LineZone`
- `sv.LineZoneAnnotator`
- Line crossing detection
- Directional crossings
- `line_zone.trigger()`
- `line_zone.in_count`
- `line_zone.out_count`
- Accumulated crossing counts
- Tracking with LineZone
- PolygonZone and LineZone composition
- Occupancy analysis
- Flow analysis
- YOLOv8
- ByteTrack
- Persistent tracker IDs
- Video processing
- Spatial event detection

---

## From Tracking to Spatial Analytics

Object tracking provides persistent identities.

For example:

```text
Frame 100 → ID 4
Frame 101 → ID 4
Frame 102 → ID 4
Frame 103 → ID 4
```

Once the same object can be followed through multiple frames, the system can begin asking spatial questions.

```text
Where is ID 4?
        ↓
Is ID 4 inside this region?
        ↓
Did ID 4 cross this boundary?
```

The progression becomes:

```text
Object Detection
       ↓
Object Tracking
       ↓
Persistent Identity
       ↓
Movement
       ↓
Spatial Interaction
       ↓
Occupancy + Crossing Events
       ↓
Spatial Video Analytics
```

---

## PolygonZone

`PolygonZone` defines an area of interest inside the image.

A polygon is represented by a collection of `(x, y)` pixel coordinates.

Example:

```python
POLYGON_LEFT = np.array([
    [0, video_info.height // 2],
    [video_info.width // 2, video_info.height // 2],
    [video_info.width // 2, video_info.height],
    [0, video_info.height],
])
```

The polygon can represent:

- A traffic lane
- A parking area
- A restricted region
- A store section
- A waiting area
- A work zone
- Any custom spatial region

Conceptually:

```text
Video Frame
     ↓
Polygon Coordinates
     ↓
PolygonZone
     ↓
Region of Interest
```

---

## PolygonZone Trigger

The zone evaluates detections using:

```python
zone.trigger(
    detections=detections
)
```

The result is a Boolean mask.

Conceptually:

```text
Detection 1 → True
Detection 2 → False
Detection 3 → True
Detection 4 → False
```

where:

```text
True  = object is inside the polygon
False = object is outside the polygon
```

The mask can then filter the detection collection:

```python
zone_mask = zone.trigger(
    detections=detections
)

detections_inside_zone = detections[
    zone_mask
]
```

This creates a spatial filtering workflow:

```text
All Detections
      ↓
PolygonZone
      ↓
Boolean Mask
      ↓
Spatial Filtering
      ↓
Objects Inside Zone
```

---

## PolygonZone Current Count

`PolygonZone` maintains:

```python
zone.current_count
```

This represents the number of objects currently inside the polygon.

Conceptually:

```text
Frame 1 → 2 objects inside
Frame 2 → 3 objects inside
Frame 3 → 3 objects inside
Frame 4 → 1 object inside
```

The value is therefore **instantaneous**.

It answers:

```text
How many objects are inside this area right now?
```

This makes `PolygonZone` useful for **occupancy analysis**.

---

## Tracking with PolygonZone

Tracking can be performed before triggering the zone.

The workflow is:

```text
Video Frame
     ↓
YOLOv8
     ↓
Detections
     ↓
ByteTrack
     ↓
Persistent Tracker IDs
     ↓
PolygonZone
     ↓
Current Occupancy
```

This allows spatial information to be connected with persistent object identities.

Instead of only knowing:

```text
3 objects are inside the zone
```

the system can understand which tracked objects are inside:

```text
ID 3
ID 7
ID 11
```

---

## LineZone

`LineZone` represents a virtual boundary rather than an area.

A line is defined using two points:

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

The line is then created with:

```python
line_zone = sv.LineZone(
    start=line_start,
    end=line_end
)
```

Conceptually:

```text
Start Point
     +
End Point
     ↓
Virtual Boundary
     ↓
LineZone
```

---

## LineZone Crossing Detection

`LineZone` uses:

```python
line_zone.trigger(
    detections=detections
)
```

to analyze tracked objects crossing the virtual boundary.

Unlike `PolygonZone`, it does not primarily measure how many objects are currently located somewhere.

Instead, it records **crossing events**.

Conceptually:

```text
Tracked Object
      ↓
Approaches Line
      ↓
Crosses Boundary
      ↓
LineZone Event
      ↓
Directional Counter Updated
```

---

## Directional Crossing Counts

`LineZone` maintains two counters:

```python
line_zone.in_count
line_zone.out_count
```

These represent accumulated crossings in opposite directions.

For example:

```text
Crossings Down: 3
Crossings Up:   3
```

The total number of crossing events can be calculated as:

```python
total_crossings = (
    line_zone.in_count
    +
    line_zone.out_count
)
```

Unlike `PolygonZone.current_count`, these values accumulate as the video is processed.

---

## PolygonZone vs. LineZone

The fundamental difference between the two systems is:

| Feature | PolygonZone | LineZone |
|---|---|---|
| Represents | Area | Boundary |
| Measures | Presence | Crossing |
| Count type | Instantaneous | Accumulated |
| Main value | `current_count` | `in_count`, `out_count` |
| Analytics type | Occupancy | Flow |
| Example | Cars currently in an area | Cars crossing an entrance |

The easiest way to remember the distinction is:

```text
PolygonZone = WHERE objects are

LineZone = WHERE objects PASS
```

Or:

```text
PolygonZone:
How many are HERE NOW?

LineZone:
How many PASSED HERE?
```

---

## Why Tracking Matters

Line crossing requires information about movement over time.

A single object detection only tells us:

```text
An object exists at position X.
```

Tracking provides:

```text
Frame 1 → ID 7 at position A
Frame 2 → ID 7 at position B
Frame 3 → ID 7 at position C
Frame 4 → ID 7 crosses the line
```

Therefore:

```text
Detection
    ↓
Object Exists
```

becomes:

```text
Detection
    ↓
Tracking
    ↓
Persistent Identity
    ↓
Movement
    ↓
Spatial Event
```

This is why the processing order is important:

```text
YOLO
  ↓
Detections
  ↓
ByteTrack
  ↓
Tracked Detections
  ↓
PolygonZone / LineZone
```

---

## Combining PolygonZone and LineZone

Both spatial systems can operate on the same tracked detections.

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

This allows one video analytics pipeline to measure both:

```text
Occupancy
+
Traffic Flow
```

at the same time.

---

## Complete Zones and Counting Workflow

The complete workflow developed during this session is:

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
     Supervision Annotators
             ↓
        Output Video
```

This represents an important progression from simple detection toward real-world video analytics.

---

## Zones and Counting Practical

The practical implementation uses a traffic video containing multiple moving vehicles.

The source video is processed frame by frame using:

```text
Traffic Video
     ↓
YOLOv8
     ↓
sv.Detections
     ↓
ByteTrackTracker
     ↓
Tracked Vehicles
     ↓
PolygonZone + LineZone
     ↓
Bounding Boxes + Tracker IDs
     ↓
Occupancy + Crossing Counters
     ↓
Processed Video
```

---

## Practical Video Information

The traffic video used during the practical has:

```text
Resolution: 3840 × 2160
FPS: 25
Total Frames: 538
```

---

## Practical Results

The combined zones and counting pipeline produced:

```text
Final polygon occupancy: 1
Crossings Down: 3
Crossings Up: 3
Total Crossings: 6
```

These results demonstrate the difference between the two types of measurements.

The polygon value represents the occupancy at the end of processing:

```text
Final polygon occupancy: 1
```

while the line counters represent accumulated events:

```text
3 crossings down
+
3 crossings up
=
6 total crossings
```

---

## Zones and Counting Code Examples

The concepts from this session were also transformed into small reusable examples inside:

```text
04-examples/
└── 05-Zones-and-Counting/
```

The examples progress through:

```text
01-basic-polygon-zone.py
        ↓
02-polygon-zone-current-count.py
        ↓
03-filter-detections-inside-zone.py
        ↓
04-tracking-with-polygon-zone.py
        ↓
05-basic-line-zone.py
        ↓
06-line-zone-crossing-count.py
        ↓
07-tracking-with-line-zone.py
        ↓
08-polygon-and-line-zone.py
        ↓
09-complete-zones-counting-pipeline.py
```

This creates a progression from basic spatial regions to a complete video analytics pipeline.

---

## Practical Applications

Zones and counting can be used for:

- Traffic monitoring
- Vehicle counting
- Parking management
- Entrance and exit monitoring
- Retail analytics
- People counting
- Crowd monitoring
- Security systems
- Restricted-area monitoring
- Industrial safety
- Warehouse analytics
- Queue monitoring
- Transportation analytics
- Building occupancy
- Pedestrian flow analysis

---

## Lesson Materials

- [Main Lesson Documentation](./05-Zones-and-Counting/README.md)
- [Concept Documentation](./05-Zones-and-Counting/concepts/)
- [Practical Exercises](./05-Zones-and-Counting/practical/)
- [Practical Documentation](./05-Zones-and-Counting/practical/README.md)
- [Class Recording Documentation](./05-Zones-and-Counting/CLASS-RECORDING.md)
- [Watch the Zones and Counting Class Recording on YouTube](https://youtu.be/43i0z9b81Z4)

**Status:** Completed

---

# Organization

Each course session may contain:

```text
session-name/
│
├── concepts/
│   └── Detailed theoretical explanations
│
├── practical/
│   ├── README.md
│   ├── Practical Python implementations
│   │
│   └── assets/
│       ├── input/
│       └── output/
│
├── course-notebook.ipynb
│   └── Original Jupyter / Google Colab notebook
│
├── CLASS-RECORDING.md
│   └── Class recording and related information
│
└── README.md
    └── Main session documentation
```

Some earlier sessions use:

```text
practical-exercises/
```

instead of:

```text
practical/
```

The exact structure may vary depending on the material covered during each class.

---

# Learning Workflow

The course material is organized into a progressive learning workflow:

```text
Class Session
      ↓
Class Recording
      ↓
Original Notebook
      ↓
Course Notes
      ↓
Concept Documentation
      ↓
Code Examples
      ↓
Practical Exercises
      ↓
Testing and Validation
      ↓
Complete Projects
```

This allows the original course material to evolve into a structured and reusable learning resource.

---

# Course Notes vs. Examples vs. Projects

The repository separates different types of learning material.

## Course Notes

```text
08-course-notes/
```

Contains:

- Detailed explanations
- Concepts
- Class material
- Practical exercises
- Original notebooks
- Class recordings
- Practical assets
- Tested outputs

## Code Examples

```text
04-examples/
```

Contains small, focused, runnable Python examples demonstrating individual concepts.

The examples currently progress through:

```text
Agentic AI Programming
        ↓
Object Detection
        ↓
Annotation
        ↓
Detection Filtering
        ↓
Object Tracking
        ↓
Zones and Counting
```

## Projects

```text
05-projects/
```

Contains larger applications that combine multiple concepts into complete computer vision workflows.

The relationship is:

```text
Course Notes
     ↓
Understand the Concept
     ↓
Code Examples
     ↓
Practice the Concept
     ↓
Practical Exercises
     ↓
Experiment
     ↓
Test and Validate
     ↓
Projects
     ↓
Build Complete Applications
```

---

# Technologies and Concepts

The course notes currently cover technologies and concepts including:

- Python
- Computer Vision
- OpenCV
- NumPy
- Matplotlib
- Ultralytics
- YOLOv8
- Supervision
- `sv.Detections`
- Object detection
- Object tracking
- Multi-object tracking
- ByteTrack
- `sv.ByteTrack`
- `ByteTrackTracker`
- `tracker_id`
- Object association
- Persistent tracking identities
- Bounding boxes
- Confidence scores
- Confidence thresholds
- Detection labels
- Tracking labels
- Boolean masks
- Detection filtering
- Class filtering
- Size filtering
- Spatial filtering
- Bounding-box area
- Bounding-box center calculation
- Detection merging
- Non-Maximum Suppression (NMS)
- Intersection over Union (IoU)
- Top-N detection selection
- Supervision Annotators
- `BoxAnnotator`
- `LabelAnnotator`
- `TraceAnnotator`
- `PolygonZoneAnnotator`
- `LineZoneAnnotator`
- Annotation customization
- Annotation layers
- Multi-Annotator pipelines
- Detection post-processing
- Video processing
- Frame-by-frame processing
- Object trajectories
- Tracking visualization
- Tracking analytics
- Spatial video analytics
- Polygon coordinates
- Spatial regions
- `PolygonZone`
- Polygon occupancy
- `zone.trigger()`
- `zone.current_count`
- Boolean zone masks
- Zone-based filtering
- Virtual counting boundaries
- `sv.Point`
- `LineZone`
- Line crossing detection
- `line_zone.trigger()`
- `line_zone.in_count`
- `line_zone.out_count`
- Directional counting
- Accumulated crossing counts
- Occupancy analysis
- Flow analysis
- Video export
- H.264
- FFmpeg
- Google Colab
- AI-assisted programming

As the course progresses, this list will expand toward more advanced computer vision and **Segment Anything Model 3 (SAM3)** workflows.

---

# Current Progress

| # | Course Session | Notes | Concepts | Exercises | Status |
|---|---|---|---|---|---|
| 00 | Agentic AI Programming | Completed | Completed | Completed | Completed |
| 01 | Introduction to Supervision | Completed | Completed | Completed | Completed |
| 02 | Annotation and Visualization | Completed | Completed | Completed | Completed |
| 03 | Filtering and Manipulating Detections | Completed | Completed | Completed | Completed |
| 04 | Object Tracking | Completed | Completed | Completed | Completed |
| 05 | Zones and Counting | Completed | Completed | Completed | Completed |

---

# Progress Overview

```text
00 — Agentic AI Programming                 Completed
01 — Introduction to Supervision            Completed
02 — Annotation and Visualization           Completed
03 — Filtering and Manipulating Detections  Completed
04 — Object Tracking                        Completed
05 — Zones and Counting                     Completed
06 — Next Course Session                    Upcoming
```

**Completed course sessions: 6**

---

# Purpose

The goal of these notes is not only to preserve the course material, but also to document my learning process and build a reusable reference for:

- Computer Vision
- Supervision
- YOLO
- Segment Anything Model 3 (SAM3)
- AI-assisted development
- Model inference and evaluation
- Computer vision pipelines
- Detection visualization
- Annotation systems
- Detection filtering
- Detection post-processing
- Non-Maximum Suppression
- Spatial analysis
- Object tracking
- Multi-object tracking
- Persistent object identities
- Object trajectories
- Tracking analytics
- Video processing
- Polygon regions
- Zone-based filtering
- Occupancy monitoring
- Line crossing
- Directional counting
- Traffic flow analysis
- Spatial video analytics
- Practical AI development

Each completed session expands the repository from basic concepts toward complete computer vision applications.

---

# Repository Learning Progression

The overall learning progression can now be represented as:

```text
Agentic AI Programming
        ↓
Introduction to Supervision
        ↓
Annotation and Visualization
        ↓
Filtering and Manipulating Detections
        ↓
Object Tracking
        ↓
Zones and Counting
        ↓
Advanced Computer Vision Concepts
        ↓
SAM3 Workflows
        ↓
Complete AI Applications
```

The first six sessions establish an increasingly complete computer vision pipeline:

```text
AI-Assisted Development
        ↓
Object Detection
        ↓
Detection Visualization
        ↓
Detection Filtering
        ↓
Object Tracking
        ↓
Persistent Object Identity
        ↓
Spatial Zones
        ↓
Occupancy + Flow
        ↓
Spatial Video Analytics
```

---

# From Detection to Spatial Video Analytics

The course progression now demonstrates how individual computer vision concepts can be connected into a complete system.

It begins with detection:

```text
Image
  ↓
YOLO
  ↓
Detections
```

Then visualization is added:

```text
Detections
    ↓
Supervision Annotators
    ↓
Visual Output
```

Filtering makes detections application-specific:

```text
Raw Detections
      ↓
Confidence
      ↓
Class
      ↓
Size
      ↓
NMS
      ↓
Spatial Filtering
      ↓
Relevant Detections
```

Tracking introduces time:

```text
Relevant Detections
        ↓
ByteTrack
        ↓
Persistent tracker_id
        ↓
Movement Across Frames
```

Zones introduce spatial reasoning:

```text
Tracked Objects
      ↓
Spatial Zones
      ↓
┌───────────────┐
↓               ↓
PolygonZone   LineZone
↓               ↓
Occupancy       Flow
```

Together, these concepts form:

```text
Detection
    +
Visualization
    +
Filtering
    +
Tracking
    +
Spatial Analysis
    =
Spatial Video Analytics
```

---

# Current Computer Vision Pipeline

After completing Zones and Counting, the repository now documents the following end-to-end conceptual pipeline:

```text
Real-World Video
        ↓
Frame Extraction
        ↓
YOLOv8
        ↓
Raw Object Detections
        ↓
sv.Detections
        ↓
Detection Filtering
        ↓
ByteTrack
        ↓
Persistent tracker_id
        ↓
Tracked Objects
        ↓
┌──────────────────────────────┐
│                              │
↓                              ↓
PolygonZone                 LineZone
│                              │
↓                              ↓
Current Occupancy          Crossing Events
│                              │
└──────────────┬───────────────┘
               ↓
       Spatial Analytics
               ↓
       Supervision Annotators
               ↓
        Processed Video
               ↓
     Application-Level Data
```

This combines the major concepts studied throughout the course so far into a single video-based computer vision architecture.

---

# Next Learning Direction

After completing **Zones and Counting**, the repository has progressed from simple object detection to spatial video analytics.

The next course session can build on the current pipeline:

```text
Detection
    ↓
Filtering
    ↓
Tracking
    ↓
Persistent IDs
    ↓
Spatial Zones
    ↓
Occupancy + Crossing Events
    ↓
Advanced Computer Vision Analysis
    ↓
SAM3 Workflows
```

Future sessions can extend this foundation with additional computer vision concepts introduced by the course.

The repository will continue to document each new lesson using the same progression:

```text
Course Material
      ↓
Concept Documentation
      ↓
Practical Implementation
      ↓
Reusable Code Examples
      ↓
Testing
      ↓
Complete Computer Vision Workflows
```

---

# Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
