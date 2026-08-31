# SAM3 Course Notes

This directory contains my organized notes, concepts, practical exercises, notebooks, class recordings, validated outputs, and supporting material from the **SAM3 — Computer Vision with Segment Anything Model 3** learning journey.

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
| 06 | [Segmentation with SAM](./06-Segmentation-with-SAM/) | Completed |
| 07 | [Advanced MaskAnnotator and SAM2](./07-Advanced-MaskAnnotator-and-SAM2/) | Completed |
| 08 | [SAM 3 Text Prompts](./08-SAM3-Text-Prompts/) | Completed |
| 09 | [SAM Encoder-Decoder](./09-SAM-Encoder-Decoder/) | Completed |
| 10 | [SAM 3 Point Prompts](./10-SAM3-Point-Prompts/) | Completed |
| 11 | [SAM 3 Video Segmentation](./11-SAM3-Video-Segmentation/) | Completed |
| 12 | [Muse Glimmer and SAM 3 Agents](./12-Muse-Glimmer-and-SAM3-Agents/) | Documentation completed |

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
             ┌───────┴───────┐
             ↓               ↓
       PolygonZone         LineZone
             ↓               ↓
         Occupancy           Flow
             │               │
             └───────┬───────┘
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

# Session 06 — Segmentation with SAM

[`06-Segmentation-with-SAM/`](./06-Segmentation-with-SAM/)

This session introduces **pixel-level object segmentation with SAM 3** and extends the previous object-detection workflow from rectangular bounding boxes to precise segmentation masks.

Instead of representing an object only with:

```text
Bounding Box
```

the segmentation workflow identifies:

```text
Individual Pixels Belonging to the Object
```

The session combines **YOLOv8**, **SAM 3**, **Supervision**, **NumPy**, and **OpenCV** to create a complete detection-to-segmentation pipeline.

Topics include:

- Segment Anything Model
- SAM 3
- YOLOv8 + SAM integration
- Bounding-box prompting
- Segmentation masks
- Boolean NumPy masks
- `sv.Detections`
- Mask inspection
- Pixel-level segmentation
- Mask area
- Object extraction
- Bounding-box area vs. mask area
- Mask serialization
- `np.packbits`
- `np.unpackbits`
- Base64 encoding
- JSON-compatible mask storage
- Segmentation visualization
- Reusable segmentation workflows

---

## From Detection to Segmentation

Previous sessions primarily represented objects using bounding boxes.

For example:

```text
Input Image
     ↓
YOLOv8
     ↓
Object Detection
     ↓
Bounding Box
```

A bounding box identifies the approximate rectangular region containing an object.

Session 06 extends this workflow:

```text
Input Image
     ↓
YOLOv8
     ↓
Bounding Boxes
     ↓
SAM 3
     ↓
Segmentation Masks
```

YOLO therefore answers:

```text
Where is the object approximately?
```

while SAM provides:

```text
Which pixels actually belong to the object?
```

---

## Bounding Boxes as SAM Prompts

YOLOv8 first detects the objects in the image.

The detection results are converted into:

```python
sv.Detections
```

The bounding boxes are available through:

```python
detections.xyxy
```

These boxes can then be used as prompts for SAM 3.

Conceptually:

```text
YOLO Detection
      ↓
Bounding Box
      ↓
SAM Prompt
      ↓
Segmentation Mask
```

This creates a useful combination:

```text
YOLO
 ↓
Fast Object Localization

SAM 3
 ↓
Precise Pixel Segmentation
```

---

## Segmentation Masks

A segmentation mask represents the object at pixel level.

Conceptually:

```text
True  → pixel belongs to the object
False → pixel belongs to the background
```

A SAM segmentation mask can therefore be represented as a Boolean NumPy array.

For an image with dimensions:

```text
1080 × 810
```

the corresponding mask can have the shape:

```text
(1080, 810)
```

Each position in the mask corresponds to a pixel in the original image.

---

## Mask Inspection

Once a segmentation mask has been generated, it can be analyzed as a NumPy data structure.

Useful properties include:

```text
Mask type
Mask shape
Mask dtype
Unique values
Object pixels
Total pixels
Object coverage
```

For a Boolean mask:

```python
mask.sum()
```

counts the number of pixels belonging to the segmented object.

Conceptually:

```text
Segmentation Mask
       ↓
Count True Pixels
       ↓
Object Area in Pixels
```

This allows segmentation to provide geometric information that cannot be obtained as precisely from a rectangular bounding box.

---

## Object Extraction

A segmentation mask can also be used to isolate the detected object from the original image.

The key operation demonstrated in the session is:

```python
object_image[~mask] = 0
```

The inverse mask:

```python
~mask
```

selects pixels outside the segmented object.

Those pixels are then set to black.

Conceptually:

```text
Original Image
      +
SAM Mask
      ↓
Keep Object Pixels
      ↓
Remove Background Pixels
      ↓
Extracted Object
```

This produces a pixel-level extraction rather than a rectangular crop.

---

## Mask Area vs. Bounding-Box Area

The session compares:

```text
Segmentation Mask Area
```

with:

```text
Bounding-Box Area
```

The bounding-box area can be calculated from:

```text
width × height
```

while the mask area is calculated from the number of `True` pixels.

The comparison percentage is:

```python
percentage = (
    mask_area
    / bounding_box_area
) * 100
```

This answers:

```text
How much of the rectangular bounding box
is actually occupied by the segmented object?
```

A lower percentage means the bounding box contains more background.

A higher percentage means more of the rectangular region corresponds to the actual segmented object.

---

## Validated Mask Area Comparison

The validated `bus.jpg` experiment produced six objects.

```text
Object 0
Class: bus
Mask area: 265686 px
Bounding-box area: 411059.31 px
Mask / box: 64.63%

Object 1
Class: person
Mask area: 46648 px
Bounding-box area: 99214.33 px
Mask / box: 47.02%

Object 2
Class: person
Mask area: 20935 px
Bounding-box area: 67998.79 px
Mask / box: 30.79%

Object 3
Class: person
Mask area: 32911 px
Bounding-box area: 55768.55 px
Mask / box: 59.01%

Object 4
Class: person
Mask area: 10715 px
Bounding-box area: 20346.07 px
Mask / box: 52.66%

Object 5
Class: stop sign
Mask area: 1878 px
Bounding-box area: 2288.43 px
Mask / box: 82.07%
```

This demonstrates that bounding boxes can contain significantly more area than the actual segmented object.

---

## Mask Serialization

Segmentation masks can contain hundreds of thousands or millions of Boolean values.

Storing the complete Boolean array directly in JSON would be inefficient.

The session therefore demonstrates a compact serialization workflow:

```text
Boolean Mask
     ↓
Flatten
     ↓
np.packbits()
     ↓
Packed Bytes
     ↓
Base64 Encoding
     ↓
JSON-Compatible String
```

The encoded data can later be reconstructed:

```text
Base64 String
     ↓
Base64 Decode
     ↓
np.frombuffer()
     ↓
np.unpackbits()
     ↓
Trim Extra Bits
     ↓
Reshape
     ↓
Original Boolean Mask
```

---

## Validated Serialization Result

The validated example used a mask with:

```text
Original mask shape: (1080, 810)
Original mask dtype: bool
Object pixels: 265686
```

The serialization process produced:

```text
Boolean pixels:     874800
Packed bytes:       109350
Base64 characters:  145800
```

The mask was then decoded.

Validation returned:

```text
Decoded mask shape: (1080, 810)
Decoded mask dtype: bool

Decoded mask matches original: True
```

This confirms that the serialized representation could reconstruct the original segmentation mask correctly.

---

## SAM 3 Image-Size Adjustment

During validated SAM 3 inference, Ultralytics displayed:

```text
WARNING ⚠️ imgsz=[1024] must be multiple of max stride 14,
updating to [1036]
```

This warning did not stop execution.

The requested size:

```text
1024
```

was automatically adjusted to:

```text
1036
```

and segmentation continued successfully.

---

## Session 06 Code Examples

The lesson concepts were transformed into six focused examples inside:

```text
04-examples/
└── 06-Segmentation-with-SAM/
```

The examples are:

```text
01_yolo_detection.py
02_sam_bbox_segmentation.py
03_mask_inspection.py
04_object_extraction.py
05_mask_area_comparison.py
06_mask_serialization.py
```

The progression is:

```text
YOLO Detection
      ↓
SAM Bounding-Box Segmentation
      ↓
Mask Inspection
      ↓
Object Extraction
      ↓
Mask Area Comparison
      ↓
Mask Serialization
```

---

## Example 01 — YOLO Detection

The first example establishes the detection stage.

It demonstrates:

- Loading `bus.jpg`
- Running YOLOv8
- Converting predictions into `sv.Detections`
- Inspecting bounding boxes
- Inspecting class IDs
- Inspecting confidence scores

These bounding boxes become the prompts used by SAM 3.

---

## Example 02 — SAM Bounding-Box Segmentation

The second example combines YOLO and SAM.

```text
bus.jpg
   ↓
YOLOv8
   ↓
6 Detections
   ↓
Bounding Boxes
   ↓
SAM 3
   ↓
6 Segmentation Masks
```

This demonstrates how object detection and segmentation can be combined inside the same workflow.

---

## Example 03 — Mask Inspection

The third example focuses on the mask itself.

It demonstrates how to inspect:

- Shape
- Data type
- Unique values
- Object pixels
- Total pixels
- Image coverage

The mask is treated as a Boolean NumPy array rather than only as a visualization.

---

## Example 04 — Object Extraction

The fourth example uses a SAM mask to extract an object.

Validated result for the selected object:

```text
Selected mask shape: (1080, 810)
Object pixels: 265686
```

The extraction uses:

```python
object_image[~mask] = 0
```

Pixels outside the mask become black while pixels inside the mask remain unchanged.

---

## Example 05 — Mask Area Comparison

The fifth example compares the area of every generated mask with its corresponding YOLO bounding box.

The experiment demonstrates that:

```text
Bounding Box
     ↓
Approximate Rectangular Area
```

while:

```text
Segmentation Mask
     ↓
Actual Pixel-Level Region
```

This provides a quantitative demonstration of the additional geometric precision available through segmentation.

---

## Example 06 — Mask Serialization

The sixth example demonstrates storing segmentation masks in a JSON-compatible representation.

The workflow uses:

- `np.packbits`
- Raw bytes
- Base64
- JSON
- Base64 decoding
- `np.frombuffer`
- `np.unpackbits`
- NumPy reshape
- `np.array_equal`

The final validation returned:

```text
Decoded mask matches original: True
```

---

## SAM 3 Model

The SAM 3 checkpoint is not stored inside the repository because the model file is very large.

The validated Google Colab environment uses:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

This keeps the repository lightweight while allowing the segmentation examples to use the external checkpoint.

---

## Technologies Used

Session 06 uses:

- Python
- Google Colab
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- NumPy
- OpenCV
- Base64
- JSON

---

## Session 06 Learning Progression

The complete conceptual progression is:

```text
Object Detection
       ↓
Bounding Boxes
       ↓
SAM 3 Prompts
       ↓
Segmentation Masks
       ↓
Boolean Mask Analysis
       ↓
Object Extraction
       ↓
Geometric Analysis
       ↓
Mask Serialization
```

This creates the foundation required for the advanced mask visualization concepts introduced in Session 07.

---

## Connection to Session 07

Session 06 answers:

```text
How do I generate and analyze segmentation masks?
```

Session 07 continues with:

```text
How do I visualize those masks?
How do I customize their appearance?
How do I filter objects before segmentation?
How do I reuse the segmentation pipeline?
How can segmentation extend toward video?
```

Therefore:

```text
Session 06
Segmentation with SAM
       ↓
Generate + Inspect Masks
       ↓
Session 07
Advanced MaskAnnotator and SAM2
       ↓
Visualize + Customize + Reuse Masks
```

---

## Lesson Materials

- [Main Lesson Documentation](./06-Segmentation-with-SAM/README.md)
- [Session 06 Folder](./06-Segmentation-with-SAM/)
- [Segmentation Code Examples](../04-examples/06-Segmentation-with-SAM/)
- [Code Examples Documentation](../04-examples/06-Segmentation-with-SAM/README.md)

**Status:** Completed

---

# Session 07 — Advanced MaskAnnotator and SAM2

[`07-Advanced-MaskAnnotator-and-SAM2/`](./07-Advanced-MaskAnnotator-and-SAM2/)

This session continues the segmentation workflow introduced in **Session 06 — Segmentation with SAM**.

The focus moves beyond generating segmentation masks and explores how those masks can be **visualized, customized, filtered, compared, and reused** using Supervision.

The session also introduces the conceptual transition from static image segmentation toward **temporal video segmentation with SAM2**.

Topics include:

- `sv.MaskAnnotator`
- `sv.BoxAnnotator`
- Advanced mask visualization
- Mask opacity
- Bounding-box and mask composition
- YOLOv8 + SAM 3 integration
- Bounding-box prompting
- `sv.Detections`
- Detection filtering before SAM
- Person-only segmentation
- Selective segmentation
- Reusable segmentation functions
- Multiple-image processing
- Output visualization
- SAM2 concepts
- Temporal segmentation
- Temporal memory
- Video mask propagation

---

## From Session 06 to Session 07

Session 06 established:

```text
Input Image
     ↓
YOLOv8
     ↓
Bounding Boxes
     ↓
SAM 3
     ↓
Segmentation Masks
```

Session 07 extends the workflow:

```text
YOLO Detection
      ↓
Bounding Boxes
      ↓
SAM 3
      ↓
Segmentation Masks
      ↓
MaskAnnotator
      ↓
Visualization
      ↓
Opacity Customization
      ↓
Detection Filtering
      ↓
Selective Segmentation
      ↓
Reusable Pipeline
```

The progression can be summarized as:

```text
Session 06
Generate + Analyze Masks
        ↓
Session 07
Visualize + Customize + Reuse Masks
```

---

## MaskAnnotator

The main Supervision component introduced in this session is:

```python
sv.MaskAnnotator()
```

`MaskAnnotator` overlays segmentation masks on the original image.

Example:

```python
mask_annotator = sv.MaskAnnotator(
    opacity=0.6
)
```

The annotation operation follows:

```python
annotated_image = mask_annotator.annotate(
    scene=image.copy(),
    detections=sam_detections
)
```

Conceptually:

```text
Original Image
      +
SAM Masks
      ↓
MaskAnnotator
      ↓
Visual Segmentation Result
```

---

## Mask Opacity

The lesson explores how the `opacity` parameter changes segmentation visualization.

The practical compares:

```text
0.2
0.5
0.9
```

Conceptually:

```text
Opacity 0.2
    ↓
Transparent Mask
    ↓
Original Image Highly Visible
```

```text
Opacity 0.5
    ↓
Balanced Visualization
```

```text
Opacity 0.9
    ↓
Strong Mask
    ↓
Segmentation Region Highly Visible
```

The validated comparison is stored as:

[`opacity_comparison.png`](./07-Advanced-MaskAnnotator-and-SAM2/practical/assets/output/opacity_comparison.png)

---

## Combining Masks and Bounding Boxes

`MaskAnnotator` and `BoxAnnotator` can be combined.

The workflow is:

```text
Original Image
      ↓
SAM Masks
      ↓
MaskAnnotator
      ↓
YOLO Bounding Boxes
      ↓
BoxAnnotator
      ↓
Combined Visualization
```

This allows direct comparison between:

```text
YOLO Bounding Box
        ↓
Approximate Rectangular Localization
```

and:

```text
SAM Mask
        ↓
Pixel-Level Object Region
```

Validated output:

[`bbox_vs_mask.png`](./07-Advanced-MaskAnnotator-and-SAM2/practical/assets/output/bbox_vs_mask.png)

---

## Filtering Before SAM

The session demonstrates that detections can be filtered **before** segmentation.

For example:

```python
person_detections = yolo_detections[
    yolo_detections.class_id == 0
]
```

where:

```text
COCO class 0 = person
```

The pipeline becomes:

```text
Input Image
     ↓
YOLOv8
     ↓
All Detections
     ↓
Class Filter
     ↓
Person Detections
     ↓
SAM 3
     ↓
Person Masks
```

This avoids sending irrelevant detections through the segmentation stage.

---

## Why Selective Segmentation Matters

Instead of:

```text
Detect Everything
      ↓
Segment Everything
      ↓
Discard Unwanted Objects
```

the workflow can use:

```text
Detect Everything
      ↓
Filter
      ↓
Segment Relevant Objects
```

This demonstrates how concepts from **Session 03 — Filtering and Manipulating Detections** can be reused inside a segmentation pipeline.

---

# Session 07 Practical

The complete practical implementation is stored in:

[`07-Advanced-MaskAnnotator-and-SAM2/practical/`](./07-Advanced-MaskAnnotator-and-SAM2/practical/)

Main Python script:

[`advanced_mask_annotator.py`](./07-Advanced-MaskAnnotator-and-SAM2/practical/advanced_mask_annotator.py)

The practical uses two images:

```text
bus.jpg
zidane.jpg
```

Input assets:

[`practical/assets/input/`](./07-Advanced-MaskAnnotator-and-SAM2/practical/assets/input/)

---

## Practical Experiment — bus.jpg

The first image was:

```text
bus.jpg
```

Validated image shape:

```text
(1080, 810, 3)
```

YOLOv8 detected:

```text
4 persons
1 bus
1 stop sign
```

Total:

```text
YOLO detections: 6
```

The six bounding boxes were supplied to SAM 3.

Validated SAM result:

```text
SAM masks: 6
```

Therefore:

```text
bus.jpg
   ↓
YOLOv8
   ↓
6 Detections
   ↓
6 Bounding-Box Prompts
   ↓
SAM 3
   ↓
6 Masks
```

---

## Person-Only Segmentation

The six detections from `bus.jpg` contained:

```text
4 persons
```

After filtering:

```text
Total detections:  6
Person detections: 4
```

Only the four person detections were passed to SAM.

The resulting visualization is:

[`person_only_segmentation.png`](./07-Advanced-MaskAnnotator-and-SAM2/practical/assets/output/person_only_segmentation.png)

The validated workflow was:

```text
6 YOLO Detections
        ↓
Person Filter
        ↓
4 Person Detections
        ↓
SAM 3
        ↓
Person Segmentation Masks
```

---

## Reusable Pipeline — zidane.jpg

The practical then applies the same processing logic to:

```text
zidane.jpg
```

YOLOv8 detected:

```text
2 persons
1 tie
```

Total:

```text
YOLO detections: 3
```

The same pipeline was reused:

```text
zidane.jpg
    ↓
YOLOv8
    ↓
sv.Detections
    ↓
Bounding Boxes
    ↓
SAM 3
    ↓
Segmentation Masks
    ↓
MaskAnnotator
    ↓
BoxAnnotator
```

Validated output:

[`second_image_segmentation.png`](./07-Advanced-MaskAnnotator-and-SAM2/practical/assets/output/second_image_segmentation.png)

This confirms that the implementation is reusable and not tied to one input image.

---

## Generated Outputs

The practical successfully generated six visualization files:

```text
bounding_boxes.png
segmentation_masks.png
bbox_vs_mask.png
opacity_comparison.png
person_only_segmentation.png
second_image_segmentation.png
```

Output directory:

[`practical/assets/output/`](./07-Advanced-MaskAnnotator-and-SAM2/practical/assets/output/)

---

## Validated Practical Results

The final validated results were:

```text
bus.jpg
├── YOLO detections: 6
├── SAM masks: 6
└── Person detections: 4

zidane.jpg
└── YOLO detections: 3

Generated visualization outputs: 6
```

The complete practical finished with:

```text
Session 07 practical completed.
```

---

## SAM 3 Image-Size Warning

During SAM 3 inference, Ultralytics displayed:

```text
WARNING ⚠️ imgsz=[1024] must be multiple of max stride 14,
updating to [1036]
```

The warning did not interrupt execution.

SAM automatically adjusted the inference size to:

```text
1036 × 1036
```

and segmentation completed successfully.

---

## Reusable Functions

The practical organizes processing into reusable functions including:

```python
load_image()
run_yolo()
run_sam()
save_image()
create_bbox_vs_mask()
create_opacity_comparison()
save_opacity_figure()
```

The architecture separates:

```text
Input
  ↓
Detection
  ↓
Segmentation
  ↓
Visualization
  ↓
Output
```

This makes the implementation easier to maintain and extend.

---

# Introduction to SAM2

The lesson also introduces the conceptual transition toward **SAM2** and temporal segmentation.

Static image segmentation follows:

```text
Image
  ↓
Prompt
  ↓
Segmentation
  ↓
Mask
```

Video segmentation introduces a temporal dimension.

```text
Initial Frame
      ↓
Object Prompt
      ↓
Initial Mask
      ↓
Temporal Memory
      ↓
Future Frames
      ↓
Mask Propagation
```

---

## Temporal Memory

For static images, each image can be processed independently.

For video, information from previous frames can help maintain object segmentation over time.

Conceptually:

```text
Frame 1
   ↓
Object Mask
   ↓
Memory
   ↓
Frame 2
   ↓
Updated Mask
   ↓
Memory
   ↓
Frame 3
   ↓
Updated Mask
```

The important conceptual progression is:

```text
Static Segmentation
        ↓
Spatial Information
```

to:

```text
Temporal Segmentation
        ↓
Spatial Information
        +
Temporal Information
```

---

## Mask Propagation

Temporal segmentation allows an initial mask to influence segmentation in future frames.

```text
Initial Object Prompt
        ↓
Initial Mask
        ↓
Temporal Memory
        ↓
Next Frame
        ↓
Mask Propagation
        ↓
Future Frames
```

This introduces the foundation for persistent object segmentation in video.

---

## Session 07 Learning Outcomes

After completing this session, I understand:

- How `sv.MaskAnnotator` visualizes segmentation masks
- How mask opacity changes visualization
- How to combine masks with bounding boxes
- How YOLO bounding boxes can prompt SAM 3
- How to filter detections before segmentation
- How to perform selective class segmentation
- Why filtering before SAM can avoid unnecessary processing
- How to reuse a segmentation pipeline on different images
- How to organize segmentation code into reusable functions
- How to save and validate segmentation visualizations
- The difference between static and temporal segmentation
- The concept of temporal memory
- The concept of mask propagation across video frames

---

## Lesson Materials

- [Main Lesson Documentation](./07-Advanced-MaskAnnotator-and-SAM2/README.md)
- [Class Recording Documentation](./07-Advanced-MaskAnnotator-and-SAM2/CLASS-RECORDING.md)
- [Original Class Notebook](./07-Advanced-MaskAnnotator-and-SAM2/03_b_sam_mask_annotator.ipynb)
- [Practical Documentation](./07-Advanced-MaskAnnotator-and-SAM2/practical/README.md)
- [Practical Python Script](./07-Advanced-MaskAnnotator-and-SAM2/practical/advanced_mask_annotator.py)
- [Input Assets](./07-Advanced-MaskAnnotator-and-SAM2/practical/assets/input/)
- [Generated Outputs](./07-Advanced-MaskAnnotator-and-SAM2/practical/assets/output/)
- [Watch the Class Recording on YouTube](https://youtu.be/GNwQl-hy8Yw)

**Status:** Completed

---

# Course Progression

The course notes now document eight completed learning sessions:

```text
00 — Agentic AI Programming
        ↓
01 — Introduction to Supervision
        ↓
02 — Annotation and Visualization
        ↓
03 — Filtering and Manipulating Detections
        ↓
04 — Object Tracking
        ↓
05 — Zones and Counting
        ↓
06 — Segmentation with SAM
        ↓
07 — Advanced MaskAnnotator and SAM2
```

Each session builds on concepts introduced earlier in the learning journey.

---

# Learning Progression

The technical progression across the completed sessions is:

```text
AI-Assisted Programming
        ↓
Object Detection
        ↓
Detection Representation
        ↓
Visualization
        ↓
Detection Filtering
        ↓
Object Tracking
        ↓
Spatial Analytics
        ↓
Pixel-Level Segmentation
        ↓
Advanced Mask Visualization
        ↓
Selective Segmentation
        ↓
Temporal Segmentation Concepts
```

This progression moves from basic computer vision inference toward increasingly complete and reusable analysis pipelines.

---

# Computer Vision Pipeline Progression

The sessions can also be understood as the gradual construction of a larger Computer Vision system.

```text
Input Image / Video
        ↓
YOLOv8
        ↓
Object Detection
        ↓
sv.Detections
        ↓
Filtering
        ↓
Visualization
        ↓
ByteTrack
        ↓
Persistent Object IDs
        ↓
PolygonZone / LineZone
        ↓
Spatial Analytics
        ↓
SAM 3
        ↓
Pixel-Level Segmentation
        ↓
MaskAnnotator
        ↓
Selective Segmentation
        ↓
Reusable Vision Pipelines
```

The SAM2 concepts introduced in Session 07 extend this progression toward:

```text
Video Frames
     ↓
Initial Segmentation
     ↓
Temporal Memory
     ↓
Mask Propagation
     ↓
Persistent Video Segmentation
```

---

# Completed Sessions

Current completed course sessions:

```text
8
```

Completed:

```text
00 — Agentic AI Programming
01 — Introduction to Supervision
02 — Annotation and Visualization
03 — Filtering and Manipulating Detections
04 — Object Tracking
05 — Zones and Counting
06 — Segmentation with SAM
07 — Advanced MaskAnnotator and SAM2
```

Progress:

```text
Completed Sessions: 8
Latest Completed Session: 07 — Advanced MaskAnnotator and SAM2
```

---

# Practical Work Completed

The course notes include practical implementations covering:

```text
AI-Assisted Programming
        ↓
YOLO Object Detection
        ↓
Supervision Detections
        ↓
Annotation Pipelines
        ↓
Detection Filtering
        ↓
ByteTrack Object Tracking
        ↓
Object Trajectories
        ↓
Polygon Occupancy
        ↓
Line Crossing Counts
        ↓
YOLO + SAM Segmentation
        ↓
Mask Inspection
        ↓
Object Extraction
        ↓
Mask Area Analysis
        ↓
Mask Serialization
        ↓
Advanced Mask Visualization
        ↓
Opacity Experiments
        ↓
Selective Person Segmentation
        ↓
Reusable Segmentation Pipelines
```

---

# Validated Session Results

Several sessions include practical results that were executed and validated.

## Session 04 — Object Tracking

```text
Input video:
960 × 540
30 FPS
300 frames

Tracking:
3 moving objects
Persistent tracker IDs

Validation:
Google Colab successful
Final H.264 video generated
```

---

## Session 05 — Zones and Counting

```text
Input video:
3840 × 2160
25 FPS
538 frames

Final polygon occupancy: 1
Crossings Down: 3
Crossings Up: 3
Total Crossings: 6
```

---

## Session 06 — Segmentation with SAM

Validated `bus.jpg` segmentation:

```text
YOLO detections: 6
SAM masks generated: 6
```

Validated selected mask:

```text
Mask shape: (1080, 810)
Object pixels: 265686
```

Validated serialization:

```text
Boolean pixels: 874800
Packed bytes: 109350
Base64 characters: 145800

Decoded mask matches original: True
```

---

## Session 07 — Advanced MaskAnnotator and SAM2

Validated `bus.jpg` results:

```text
YOLO detections: 6
SAM masks: 6
Person detections: 4
```

Validated `zidane.jpg` results:

```text
YOLO detections: 3
```

Generated outputs:

```text
bounding_boxes.png
segmentation_masks.png
bbox_vs_mask.png
opacity_comparison.png
person_only_segmentation.png
second_image_segmentation.png
```

Total:

```text
6 validated visualization outputs
```

---

# Session 08 — SAM 3 Text Prompts

[`08-SAM3-Text-Prompts/`](./08-SAM3-Text-Prompts/)

This session introduces semantic segmentation with natural-language prompts in SAM 3.

Topics include text prompts, open-vocabulary concepts, prompt comparison, confidence analysis, mask visualization, and practical semantic segmentation.

**Status:** Completed

---

# Session 09 — SAM Encoder-Decoder

[`09-SAM-Encoder-Decoder/`](./09-SAM-Encoder-Decoder/)

This session explores the internal encoder-decoder architecture behind SAM, including image embeddings, prompt encoding, mask decoding, and the relationship between model components.

**Status:** Completed

---

# Session 10 — SAM 3 Point Prompts

[`10-SAM3-Point-Prompts/`](./10-SAM3-Point-Prompts/)

This session explores positive and negative point prompts for interactive SAM 3 segmentation, including coordinate selection, point labels, mask refinement, and visual feedback.

**Status:** Completed

---

# Session 11 — SAM 3 Video Segmentation

[`11-SAM3-Video-Segmentation/`](./11-SAM3-Video-Segmentation/)

This session extends SAM 3 into video using two pipelines:

```text
YOLO → ByteTrack → SAM 3
```

and:

```text
Text Prompts → SAM3VideoSemanticPredictor
```

Topics include frame-by-frame segmentation, persistent tracker IDs, attribute transfer, mask and trace visualization, polygon-zone filtering, confidence-based opacity, temporal mask-area analysis, streaming inference, and direct semantic video prompts.

Class recording:

[SAM 3 en Video — Segmentación y Tracking](https://youtu.be/_EuNGCYS35k)

**Status:** Completed

---

# Session 12 — Muse Glimmer and SAM 3 Agents

[`12-Muse-Glimmer-and-SAM3-Agents/`](./12-Muse-Glimmer-and-SAM3-Agents/)

This extension connects the course's computer-vision work with local agentic AI. It documents how Meta Muse Glimmer can provide multimodal reasoning, planning, tool calling, orchestration, and failure recovery while SAM 3 remains the specialized perception and segmentation component.

The proposed division of responsibilities is:

```text
Muse Glimmer → Reasoning + Planning + Tool Orchestration
SAM 3        → Visual Perception + Pixel-Level Segmentation
Python       → Deterministic Measurement + Validation
```

Topics include:

- Open-weight and local model deployment
- Multimodal text-and-image reasoning
- Agentic task completion
- Tool and function calling
- Failure recovery
- SAM 3 tool contracts
- Schema-validated vision results
- Deterministic mask measurement
- Hardware and quantization planning
- Security, permissions, observability, and bounded retries
- A validation-first integration roadmap

The module contains architecture, installation, hardware, workflow, and official-reference documentation. It does not claim that the combined runtime has already been installed or tested.

**Status:** Documentation completed; practical integration not yet validated

---

# Course Notes Organization

Each course session is documented independently.

Depending on the lesson, a session may contain:

```text
README.md
CLASS-RECORDING.md
Original Course Notebook
Concept Documentation
Practical Exercises
Python Scripts
Input Assets
Output Assets
Validated Results
```

This structure preserves the original course material while also documenting my own practical implementation and validation work.

---

# Repository Connections

The course notes are connected with other sections of the repository.

```text
08-course-notes/
      ↓
Lesson Documentation

04-examples/
      ↓
Small Reusable Code Examples

05-projects/
      ↓
Larger Integrated Projects

09-assets/
      ↓
Repository Visual Assets
```

The general learning workflow is:

```text
Course Lesson
     ↓
Concept Understanding
     ↓
Focused Code Example
     ↓
Practical Exercise
     ↓
Integrated Project
     ↓
Validation
     ↓
Documentation
```

---

# Technologies Used Across the Course Notes

The completed sessions currently use concepts and tools including:

- Python
- Google Colab
- OpenCV
- NumPy
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- `sv.Detections`
- `sv.BoxAnnotator`
- `sv.LabelAnnotator`
- `sv.TraceAnnotator`
- `sv.MaskAnnotator`
- ByteTrack
- PolygonZone
- LineZone
- Matplotlib
- FFmpeg
- Base64
- JSON

The course also introduces concepts related to:

- SAM2
- Temporal segmentation
- Temporal memory
- Mask propagation

---

# Current Learning Status

```text
SAM3 Computer Vision Learning Journey

Course Sessions Documented: 13

00 Agentic AI Programming                 ✅
01 Introduction to Supervision            ✅
02 Annotation and Visualization           ✅
03 Filtering and Manipulating Detections  ✅
04 Object Tracking                        ✅
05 Zones and Counting                     ✅
06 Segmentation with SAM                  ✅
07 Advanced MaskAnnotator and SAM2        ✅
08 SAM 3 Text Prompts                     ✅
09 SAM Encoder-Decoder                    ✅
10 SAM 3 Point Prompts                    ✅
11 SAM 3 Video Segmentation               ✅
12 Muse Glimmer and SAM 3 Agents          📘
```

The repository currently documents the learning progression through **Session 12 — Muse Glimmer and SAM 3 Agents**.

---

# Next Steps

The next practical milestone is to validate a minimal Muse Glimmer tool-calling environment before creating the corresponding project in `05-projects/`.

Current progression:

```text
Detection
    ↓
Tracking
    ↓
Spatial Analytics
    ↓
Image Segmentation
    ↓
Semantic and Point Prompts
    ↓
Video Segmentation
    ↓
Temporal Analysis
    ↓
Agentic Vision Orchestration
```

---

# Purpose of These Notes

These notes are designed to serve as:

- A structured record of my SAM3 learning journey
- A reference for Computer Vision concepts
- Documentation of practical experiments
- Evidence of tested implementations
- A connection between course lessons and larger projects
- A reusable technical reference for future Computer Vision work

The objective is not only to preserve course material, but to demonstrate the progression from individual concepts toward complete Computer Vision pipelines.

---

# Status

```text
08-course-notes/

Sessions documented:  13
Course sessions completed: 12
Extension modules documented: 1
Latest module:        12 — Muse Glimmer and SAM 3 Agents

Status: UP TO DATE ✅
```
