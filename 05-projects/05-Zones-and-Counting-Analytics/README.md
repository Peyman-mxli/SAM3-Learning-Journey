# Zones and Counting Analytics

This project implements a complete **people detection, tracking, zone monitoring, and crossing analytics pipeline** using **Ultralytics YOLOv8**, **Supervision**, **ByteTrack**, `PolygonZone`, and `LineZone`.

The project extends the object-tracking concepts from the previous project by adding **spatial analytics**.

Instead of only asking:

> Where is each detected person?

the pipeline can also answer:

- How many people are currently inside a defined region?
- Which tracked people are inside the region?
- Has a tracked person crossed a virtual line?
- In which direction did the crossing occur?
- How many crossing events have been recorded?

The final implementation was tested using a real crowded pedestrian-crossing video in **Google Colab**.

---

## Project Objectives

The main objectives of this project are to:

- Detect people in video using YOLOv8
- Convert YOLO predictions into `sv.Detections`
- Track detected people using ByteTrack
- Maintain persistent `tracker_id` values
- Define a custom `PolygonZone`
- Measure current zone occupancy
- Define a custom `LineZone`
- Detect line-crossing events
- Count crossing directions
- Annotate people with bounding boxes
- Display persistent tracker IDs
- Visualize spatial zones
- Display real-time analytics
- Process an entire video
- Generate an annotated output video
- Validate the complete pipeline in Google Colab

---

## Technologies

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
- FFmpeg
- Git
- GitHub

---

## Project Structure

```text
05-Zones-and-Counting-Analytics/
│
├── assets/
│   │
│   ├── input/
│   │   ├── README.md
│   │   └── people_walking.mp4
│   │
│   └── output/
│       ├── README.md
│       └── people_zones_counting_final.mp4
│
├── zones_counting_analytics.py
├── requirements.txt
└── README.md
```

---

## Input Video

The final project uses:

```text
assets/input/people_walking.mp4
```

The video contains a real-world pedestrian crossing with many people moving through a large intersection.

Video information:

```text
Resolution: 1920 × 1080
FPS: 50
Frames: 763
Duration: approximately 15.26 seconds
```

The fixed-camera perspective makes the video suitable for experimenting with:

- Person detection
- Multi-object tracking
- Polygon occupancy
- Virtual counting lines
- Pedestrian movement analytics

The scene is intentionally challenging because it contains a large number of pedestrians, significant crowd density, partial occlusion, and people moving in multiple directions.

---

## Detection Model

The final implementation uses:

```python
MODEL_NAME = "yolov8s.pt"
```

The smaller `yolov8n.pt` model was initially tested.

However, because the pedestrian scene contains many small and partially occluded people, the final project uses **YOLOv8s** to improve person detection in the crowded scene.

The inference configuration is:

```python
CONFIDENCE_THRESHOLD = 0.15
PERSON_CLASS_ID = 0
INFERENCE_SIZE = 1280
```

COCO class ID `0` represents:

```text
person
```

Therefore, only person detections are passed into the tracking and spatial-analysis stages.

---

## Detection Pipeline

Each video frame is processed by YOLOv8s.

The detection workflow is:

```text
Video Frame
     ↓
YOLOv8s
     ↓
Person Detection
     ↓
Ultralytics Results
     ↓
sv.Detections
```

The conversion is performed using:

```python
detections = sv.Detections.from_ultralytics(
    result
)
```

This allows the detections to be processed using Supervision's tracking, zone, and annotation tools.

---

## Object Tracking

Detection alone processes individual frames independently.

To perform meaningful zone and crossing analytics, the application must know whether a detected person in one frame is the same person observed in later frames.

This project therefore uses:

```python
sv.ByteTrack()
```

The final tracker configuration is:

```python
tracker = sv.ByteTrack(
    track_activation_threshold=0.15,
    lost_track_buffer=90,
    minimum_matching_threshold=0.70,
    frame_rate=video_info.fps
)
```

This configuration was selected after testing the pipeline on the crowded pedestrian scene.

The longer lost-track buffer helps the tracker maintain identities when pedestrians become temporarily difficult to detect because of crowding or occlusion.

---

## Persistent Tracker IDs

After ByteTrack processes the detections, tracked people receive:

```text
tracker_id
```

Example labels may appear as:

```text
Person #4
Person #18
Person #27
Person #32
```

These IDs allow the system to reason about movement across consecutive frames.

The processing architecture becomes:

```text
Frame
  ↓
YOLOv8s
  ↓
Person Detections
  ↓
ByteTrack
  ↓
Persistent tracker_id
```

---

## Important Tracker ID Note

A `tracker_id` represents an identity maintained by ByteTrack during the tracking sequence.

It should not automatically be interpreted as a permanent real-world identity.

In crowded scenes, a person may:

- Become occluded
- Leave the camera view
- Become temporarily undetected
- Lose the current tracking association
- Receive another tracker ID later

Therefore, tracker IDs represent **tracking identities maintained by the algorithm**, not guaranteed permanent identities of physical people.

This distinction is especially important in the crowded pedestrian video used by this project.

---

# PolygonZone

One of the main concepts demonstrated by this project is `PolygonZone`.

A PolygonZone defines a custom polygon-shaped region inside the video frame.

The final polygon used by the project is:

```python
POLYGON = np.array([
    [520, 470],
    [1320, 470],
    [1460, 900],
    [420, 900]
], dtype=np.int32)
```

The zone is created using:

```python
polygon_zone = sv.PolygonZone(
    polygon=POLYGON
)
```

---

## Why Use a Polygon?

Real-world areas of interest are not always rectangular.

For example:

- Pedestrian crossings
- Entrances
- Parking areas
- Store sections
- Restricted areas
- Roads
- Loading zones
- Production areas

may have irregular shapes.

A polygon allows the analysis region to follow the geometry of the real scene more accurately.

---

## PolygonZone in This Project

The PolygonZone covers the main pedestrian-crossing region visible in the video.

Conceptually:

```text
Camera Frame
│
├── Outside Zone
│
│
└── PolygonZone
      │
      ├── Person #4
      ├── Person #18
      ├── Person #27
      └── ...
```

The zone is evaluated on every frame:

```python
polygon_zone.trigger(
    detections=detections
)
```

Supervision then maintains:

```python
polygon_zone.current_count
```

which represents the current number of tracked detections inside the polygon.

---

## Polygon Occupancy

The application displays the current zone occupancy directly on the video:

```text
People in Zone: N
```

For example:

```text
People in Zone: 11
```

The value changes as pedestrians enter and leave the defined region.

This demonstrates the difference between:

```text
Total detections in frame
```

and:

```text
Detections currently inside a specific spatial region
```

---

# LineZone

The second major spatial-analysis concept is `LineZone`.

A LineZone creates a virtual line inside the video.

When a tracked object moves from one side of the line to the other, Supervision can register a crossing event.

---

## Final LineZone

Several line positions and orientations were tested during development.

The final implementation uses a vertical line through the central pedestrian flow:

```python
line_start = sv.Point(
    x=960,
    y=400
)

line_end = sv.Point(
    x=960,
    y=920
)
```

The LineZone is created using:

```python
line_zone = sv.LineZone(
    start=line_start,
    end=line_end
)
```

---

## Why a Vertical Line?

An initial horizontal counting line was tested.

Several alternatives were then evaluated:

```text
Horizontal
Vertical
Diagonal 1
Diagonal 2
```

The crowded scene contains pedestrians moving in multiple directions, and the horizontal and diagonal configurations did not produce reliable crossing events with the available tracking sequence.

The vertical orientation was the configuration that produced a confirmed crossing event during testing.

This demonstrates an important computer vision principle:

> Zone geometry should be designed according to the movement patterns and camera perspective of the actual scene.

---

## Crossing Direction

The LineZone maintains two directional counters:

```python
line_zone.in_count
line_zone.out_count
```

The video displays these values as:

```text
Crossings In
Crossings Out
```

A crossing is only registered when the tracker maintains enough information to determine that the same tracked object moved from one side of the virtual line to the other.

---

## Why Tracking Is Required for LineZone

A single detection does not contain movement history.

For example:

```text
Frame 1
Person detected left of line
```

does not tell us whether that person crossed anything.

But with tracking:

```text
Frame 1
Person #12 → Left Side

Frame 2
Person #12 → Near Line

Frame 3
Person #12 → Right Side
```

the system can determine:

```text
Person #12 crossed the LineZone
```

Therefore:

```text
Detection
    ↓
Tracking
    ↓
Persistent ID
    ↓
Movement History
    ↓
Line Crossing
```

---

# Complete Pipeline

The final project architecture is:

```text
people_walking.mp4
        ↓
Read Video Frame
        ↓
YOLOv8s
        ↓
Person Detection
        ↓
sv.Detections
        ↓
ByteTrack
        ↓
Persistent Person IDs
        ↓
        ├───────────────────────┐
        ↓                       ↓
   PolygonZone               LineZone
        ↓                       ↓
Current Occupancy         Crossing Events
        ↓                       ↓
        └───────────┬───────────┘
                    ↓
             BoxAnnotator
                    ↓
             LabelAnnotator
                    ↓
          Spatial Annotations
                    ↓
            Analytics Overlay
                    ↓
          Annotated Video Frame
                    ↓
             Output Video
```

---

## Annotation Pipeline

The final visualization combines several layers.

### Bounding Boxes

```python
sv.BoxAnnotator()
```

draws bounding boxes around tracked pedestrians.

### Tracker Labels

```python
sv.LabelAnnotator()
```

displays labels such as:

```text
Person #32
```

### Polygon Visualization

```python
sv.PolygonZoneAnnotator()
```

draws the PolygonZone over the pedestrian-crossing area.

### Line Visualization

```python
sv.LineZoneAnnotator()
```

draws the virtual counting line and displays:

```text
Crossings In
Crossings Out
```

### Additional Analytics

OpenCV is used to display:

```text
People in Zone
Frame
```

on the generated video.

---

# Installation

Install the project dependencies using:

```bash
pip install -r requirements.txt
```

The project requires:

```text
ultralytics
supervision
opencv-python
numpy
```

These dependencies are defined in:

```text
requirements.txt
```

---

# Running the Project

From the project directory:

```bash
python zones_counting_analytics.py
```

The application expects the input video at:

```text
assets/input/people_walking.mp4
```

and generates:

```text
assets/output/people_zones_counting_final.mp4
```

---

# Google Colab Testing

The project was developed and validated in **Google Colab**.

The repository was cloned using:

```bash
git clone https://github.com/Peyman-mxli/SAM3-Learning-Journey.git
```

The working directory was changed to:

```python
%cd /content/SAM3-Learning-Journey/05-projects/05-Zones-and-Counting-Analytics
```

The project was then tested against the real pedestrian video stored in the repository.

---

# Development and Testing Process

The final implementation was reached through several experiments rather than selecting arbitrary zone coordinates.

The development process included:

```text
Load Real Pedestrian Video
        ↓
Inspect Video Metadata
        ↓
Inspect First Frame
        ↓
Design PolygonZone
        ↓
Preview Polygon Geometry
        ↓
Test YOLO Person Detection
        ↓
Test ByteTrack IDs
        ↓
Preview Tracking + Zones
        ↓
Test Horizontal LineZone
        ↓
Evaluate Crossing Results
        ↓
Inspect Pedestrian Flow
        ↓
Test Alternative Line Positions
        ↓
Compare Horizontal / Vertical / Diagonal Lines
        ↓
Tune Detection and Tracking
        ↓
Select Vertical LineZone
        ↓
Process Complete Video
        ↓
Verify Final Output
```

This process demonstrates why spatial analytics should be validated against actual movement in the scene.

---

# Detector and Tracker Tuning

The crowded pedestrian scene presented a more difficult tracking problem than the traffic video used in the previous Object Tracking project.

The initial configuration used:

```text
YOLOv8n
Default ByteTrack configuration
```

Testing showed that many pedestrians were:

- Small relative to the full frame
- Partially occluded
- Close together
- Moving in different directions

The final configuration therefore uses:

```text
YOLOv8s
Confidence threshold: 0.15
Inference size: 1280
Track activation threshold: 0.15
Lost-track buffer: 90
Minimum matching threshold: 0.70
```

This improved the ability of the pipeline to detect and track pedestrians in the crowded scene.

---

# Final Test

The complete final pipeline was successfully executed in Google Colab.

Video information:

```text
Resolution: 1920 × 1080
FPS: 50.0
Frames: 763
```

Processing completed successfully:

```text
Processing video: 100%
763/763
```

Final analytics:

```text
Final people in PolygonZone: 6
Crossings In: 0
Crossings Out: 1
Total Crossings: 1
```

The final result confirms that:

- YOLOv8s successfully detected pedestrians
- ByteTrack assigned tracking IDs
- PolygonZone measured pedestrian occupancy
- LineZone registered a confirmed crossing
- The complete 763-frame video was processed successfully
- The annotated output video was generated successfully

---

# Understanding the Crossing Result

The final result contains:

```text
Total Crossings: 1
```

This number should be interpreted carefully.

The scene contains a very large crowd with substantial occlusion.

LineZone crossing analytics depend on the same tracker ID being maintained while a person moves from one side of the line to the other.

In a dense crowd, tracking continuity can be interrupted.

Therefore, the crossing counter represents:

```text
Confirmed crossings observed by the tracking pipeline
```

rather than:

```text
The total number of physical pedestrians that crossed the intersection
```

This is an important distinction when designing real-world computer vision analytics systems.

---

# Final Output

The final processed video is stored at:

```text
assets/output/people_zones_counting_final.mp4
```

The video contains:

```text
Original Pedestrian Video
        ↓
YOLO Person Detections
        ↓
Bounding Boxes
        ↓
Persistent Person IDs
        ↓
PolygonZone
        ↓
People-in-Zone Counter
        ↓
Vertical LineZone
        ↓
Crossings In / Out
        ↓
Frame Counter
        ↓
Final Annotated Video
```

The final video was also converted to an **H.264-compatible MP4** during Google Colab testing to ensure reliable browser playback.

---

# Example Output Information

The generated video displays information similar to:

```text
People in Zone: 11
Frame: 0

Crossings In: 0
Crossings Out: 0
```

As the video progresses, these values are updated according to the tracked detections and zone events.

The verified final crossing result is:

```text
Crossings In: 0
Crossings Out: 1
```

---

# Input and Output Assets

## Input

```text
assets/input/people_walking.mp4
```

Contains the real pedestrian video used to test the application.

## Output

```text
assets/output/people_zones_counting_final.mp4
```

Contains the final processed and annotated video.

Each asset directory also contains its own `README.md` explaining the purpose of the files stored there.

---

# Key Concepts Demonstrated

This project demonstrates:

- Video processing
- YOLO object detection
- Person-class filtering
- `sv.Detections`
- Multi-object tracking
- ByteTrack
- Persistent `tracker_id`
- Spatial reasoning
- Polygon regions
- Polygon occupancy
- `PolygonZone`
- `PolygonZoneAnnotator`
- Virtual counting lines
- `LineZone`
- `LineZoneAnnotator`
- Directional crossing counts
- Bounding-box annotations
- Tracker labels
- Frame-by-frame analytics
- Crowded-scene tracking limitations
- Detector tuning
- Tracker tuning
- Real-world pipeline validation

---

# What This Project Adds to the Learning Journey

Previous projects focused on:

```text
Project 01
Object Detection
        ↓
Project 02
Visualization
        ↓
Project 03
Detection Filtering
        ↓
Project 04
Object Tracking
```

Project 05 adds:

```text
Object Tracking
      ↓
Spatial Zones
      ↓
Occupancy Analytics
      ↓
Crossing Detection
      ↓
Video Analytics
```

The progression is therefore:

```text
Detection
    ↓
Visualization
    ↓
Filtering
    ↓
Tracking
    ↓
Zones
    ↓
Counting
    ↓
Spatial Analytics
```

---

# Real-World Applications

The concepts demonstrated by this project can be extended to applications such as:

- Pedestrian counting
- Store occupancy monitoring
- Entrance and exit analytics
- Crowd monitoring
- Queue analysis
- Restricted-area monitoring
- Traffic analytics
- Parking analytics
- Warehouse monitoring
- Factory safety systems
- Smart-city analytics
- Security-camera analytics

---

# Limitations

This project also demonstrates several real-world limitations of computer vision systems.

## Crowd Density

Dense crowds make object tracking more difficult because people frequently overlap.

## Occlusion

A pedestrian may disappear behind another pedestrian or object.

## Small Objects

People farther from the camera occupy fewer pixels and may be harder for the detector to recognize consistently.

## Tracker Identity Changes

If ByteTrack loses a person and later detects them again, the person may receive another `tracker_id`.

## LineZone Dependence on Tracking

A line crossing can only be confirmed when sufficient tracking continuity exists across the line.

## Camera-Specific Coordinates

The PolygonZone and LineZone coordinates used in this project are designed specifically for:

```text
people_walking.mp4
1920 × 1080
```

A different video or resolution may require different coordinates.

---

# Lessons Learned

This project demonstrates that building video analytics requires more than simply running an object detector.

A successful pipeline requires:

```text
Detection
+
Tracking
+
Spatial Geometry
+
Temporal Information
+
Visualization
+
Testing
```

The project also showed that zone placement should not be selected blindly.

The PolygonZone and LineZone were visually inspected and tested against actual pedestrian movement.

The crowded scene also demonstrated why persistent tracking is more difficult than frame-by-frame detection.

---

# Project Status

**Completed and successfully tested in Google Colab.**

Final verified processing result:

```text
Frames processed: 763

Final people in PolygonZone: 6

Crossings In: 0
Crossings Out: 1
Total Crossings: 1
```

Final output:

```text
assets/output/people_zones_counting_final.mp4
```

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
