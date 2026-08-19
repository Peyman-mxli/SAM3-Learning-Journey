# Object Tracking with YOLO, Supervision, and ByteTrack

This project builds a complete **object tracking pipeline** using YOLO, Supervision, and ByteTrack.

The project extends the concepts from previous detection, annotation, and filtering lessons into video analysis.

Instead of detecting objects independently in every frame, the system assigns each tracked object a persistent:

```python
tracker_id
```

This allows the same object to be followed across consecutive video frames.

---

## Project Goal

The objective of this project is to process a real traffic video and create an annotated output containing:

- Object detections
- Persistent tracking IDs
- Object class names
- Bounding boxes
- Tracking trajectories
- Filtering before tracking
- Frame visibility counts
- Estimated visible time
- Unique tracked-object counts

The final pipeline combines the main concepts from the **Object Tracking** lesson into one reusable computer vision project.

---

## Main Pipeline

The complete project follows this workflow:

```text
Input Video
     ↓
Read Frame
     ↓
YOLO Detection
     ↓
sv.Detections
     ↓
Detection Filtering
     ↓
ByteTrack
     ↓
tracker_id
     ↓
Tracking Analytics
     ↓
Bounding Boxes
     ↓
Labels
     ↓
Object Traces
     ↓
Annotated Frame
     ↓
Output Video
```

---

## Technologies Used

The project uses:

```text
Python
OpenCV
Ultralytics YOLO
Supervision
ByteTrack
NumPy
```

---

## Project Structure

```text
04-Object-Tracking/
│
├── README.md
│
├── object_tracking_pipeline.py
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
        ├── README.md
        └── tracked_vehicles.mp4
```

---

## Input

The project uses a real traffic video stored at:

```text
assets/input/vehicles.mp4
```

For the final GitHub version, the input video was prepared as a short real-world traffic sample.

The final test input contains:

```text
Duration: 10 seconds
Frames: 250
Resolution: 1280 × 720
Content: Real highway traffic
Target Object: Cars
```

A short input video keeps the repository lightweight while still providing enough consecutive frames to demonstrate object tracking.

The original course sample comes from:

```text
https://media.roboflow.com/supervision/video-examples/vehicles.mp4
```

The project script can also automatically download the sample video if the expected input file does not exist.

---

## Output

The processed tracking video is saved to:

```text
assets/output/tracked_vehicles.mp4
```

The output contains:

- Detected cars
- Bounding boxes
- Persistent tracker IDs
- Tracking trajectories
- Frame visibility counts
- Visible-time estimates
- Unique tracked-car counter

Example labels look like:

```text
car #1 | 44f | 1.76s
car #2 | 44f | 1.76s
car #3 | 29f | 1.16s
```

where:

```text
car
```

represents the detected class,

```text
#1
```

represents the tracking ID,

```text
44f
```

represents the number of frames the object has been visible,

and:

```text
1.76s
```

represents the estimated visible time.

---

## Installing Dependencies

Install the required libraries:

```bash
pip install supervision ultralytics opencv-python numpy
```

---

## Imports

The project uses:

```python
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO
```

---

## Loading the YOLO Model

The project uses:

```python
YOLO("yolov8n.pt")
```

Example:

```python
model = YOLO(
    "yolov8n.pt"
)
```

YOLO performs object detection on every video frame.

---

## Target Object Class

The project tracks cars.

In the COCO dataset:

```python
TARGET_CLASS_ID = 2
```

represents:

```text
car
```

This means YOLO can detect many different object classes, but only cars are passed into the tracking stage.

---

## Creating the Tracker

The project uses:

```python
sv.ByteTrack()
```

Example:

```python
tracker = sv.ByteTrack()
```

ByteTrack receives the filtered detections and attempts to maintain the identity of each car across consecutive frames.

---

## Resetting the Tracker

Before processing a new video:

```python
tracker.reset()
```

This clears any previous tracking state.

The tracker must then maintain its state during the complete processing sequence.

---

## Detection

For every frame:

```python
results = model(
    frame,
    verbose=False
)[0]
```

YOLO provides information such as:

```text
Bounding Boxes
Class IDs
Confidence Scores
```

---

## Converting to `sv.Detections`

YOLO results are converted into Supervision detections:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

This provides access to properties such as:

```python
detections.xyxy
detections.confidence
detections.class_id
detections.tracker_id
```

---

## Detection Before Tracking

Before ByteTrack processes the detections:

```python
detections.tracker_id
```

is normally:

```text
None
```

YOLO detects objects but does not assign persistent tracking identities.

ByteTrack is responsible for adding those identities.

---

## Filtering Before Tracking

The project filters detections before sending them to ByteTrack.

```python
detections = detections[
    detections.class_id == TARGET_CLASS_ID
]
```

Because:

```python
TARGET_CLASS_ID = 2
```

only cars remain.

The resulting detections are then passed to:

```python
detections = tracker.update_with_detections(
    detections
)
```

---

## Why Filter Before Tracking?

The project follows:

```text
YOLO
 ↓
Detections
 ↓
Filter
 ↓
ByteTrack
```

instead of:

```text
YOLO
 ↓
Detections
 ↓
ByteTrack
 ↓
Filter
```

Filtering first means ByteTrack only needs to manage objects relevant to the application.

For example:

```text
YOLO detects:

person
car
truck
car
bus
car
```

After filtering:

```text
car
car
car
```

Only those cars are tracked.

---

## Applying ByteTrack

Tracking is applied using:

```python
detections = tracker.update_with_detections(
    detections
)
```

After tracking:

```python
detections.tracker_id
```

may contain values such as:

```text
[1, 2, 3, 4]
```

Each value represents an individual tracked object.

---

## `class_id` vs. `tracker_id`

These values represent different information.

### `class_id`

Answers:

```text
What type of object is this?
```

Examples:

```text
car
truck
bus
person
```

### `tracker_id`

Answers:

```text
Which individual object is this?
```

Examples:

```text
car #1
car #2
car #3
```

Several objects can belong to the same class while having different tracker IDs.

---

## Tracking Labels

A simple tracker label can be created with:

```python
labels = [
    f"ID:{tracker_id}"
    for tracker_id
    in detections.tracker_id
]
```

Example:

```text
ID:1
ID:2
ID:3
```

The final project uses more informative labels containing:

```text
Class Name
Tracker ID
Visible Frames
Visible Time
```

For example:

```text
car #4 | 173f | 6.92s
```

---

## Bounding Box Annotation

Bounding boxes are created using:

```python
box_annotator = sv.BoxAnnotator()
```

and applied with:

```python
annotated_frame = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections
)
```

---

## Label Annotation

Labels are created using:

```python
label_annotator = sv.LabelAnnotator()
```

and applied with:

```python
annotated_frame = label_annotator.annotate(
    scene=annotated_frame,
    detections=detections,
    labels=labels
)
```

---

## Object Trajectories

The project uses:

```python
trace_annotator = sv.TraceAnnotator()
```

and:

```python
annotated_frame = trace_annotator.annotate(
    scene=annotated_frame,
    detections=detections
)
```

`TraceAnnotator` uses:

```python
tracker_id
```

to associate positions belonging to the same object across multiple frames.

This creates a visual trajectory showing how the tracked vehicle moves through the scene.

---

## Tracking Analytics

The project uses persistent tracker IDs to create simple analytics.

A dictionary stores how many frames each tracked car remains visible:

```python
frame_count = {}
```

For each tracker ID:

```python
frame_count[tracker_id] = (
    frame_count.get(
        tracker_id,
        0
    ) + 1
)
```

---

## Frame Visibility Count

Suppose:

```text
car #5
```

appears in 60 processed frames.

The dictionary may contain:

```text
{
    5: 60
}
```

The object can then be labeled:

```text
car #5 | 60f
```

where:

```text
f = frames
```

---

## Visible Time

The approximate visible time is calculated using the video's FPS.

```text
Visible Time =
Frames Visible / FPS
```

In Python:

```python
visible_seconds = (
    frames_visible /
    VIDEO_FPS
)
```

For example:

```text
Frames Visible = 146
FPS = 25

146 / 25 = 5.84 seconds
```

The label can therefore display:

```text
car #1 | 146f | 5.84s
```

---

## Unique Tracker IDs

The project stores every observed tracker ID inside a Python set:

```python
unique_tracker_ids = set()
```

New IDs are added using:

```python
unique_tracker_ids.add(
    tracker_id
)
```

The total number of observed tracker IDs is:

```python
len(
    unique_tracker_ids
)
```

---

## Final Test Results

The completed project was tested successfully in **Google Colab** using the real 10-second traffic video stored in the repository.

The final processing run completed:

```text
Processing video: 100% 250/250
```

The tracking analytics were:

```text
Tracker ID 1: 146 frames | 5.84 seconds
Tracker ID 2: 54 frames  | 2.16 seconds
Tracker ID 3: 47 frames  | 1.88 seconds
Tracker ID 4: 173 frames | 6.92 seconds
Tracker ID 5: 176 frames | 7.04 seconds
Tracker ID 6: 27 frames  | 1.08 seconds
```

Unique tracker IDs:

```text
[1, 2, 3, 4, 5, 6]
```

Total unique tracked cars:

```text
6
```

The pipeline successfully generated:

```text
assets/output/tracked_vehicles.mp4
```

The final GitHub output video was converted to a browser-compatible H.264 MP4 and verified visually.

The video displays:

```text
Car Detection
     ↓
Bounding Box
     ↓
Persistent Tracker ID
     ↓
Frame Count
     ↓
Visible Time
     ↓
Tracking Trace
```

The final test completed with:

```text
Object Tracking Project: SUCCESS
```

---

## Important Tracking Limitation

A tracker ID is not a permanent real-world identity.

For example:

```text
Frame 1 → car #5
Frame 2 → car #5
Frame 3 → car #5
```

If the tracker loses the object and it later returns:

```text
Frame 20 → car #11
```

the same physical vehicle may receive a different ID.

Therefore:

```text
unique tracker IDs
```

should not automatically be interpreted as perfectly accurate unique real-world object counts.

They represent identities maintained by ByteTrack during the tracking sequence.

---

## Video Information

The project uses:

```python
sv.VideoInfo.from_video_path()
```

Example:

```python
video_info = sv.VideoInfo.from_video_path(
    INPUT_VIDEO
)
```

Useful properties include:

```python
video_info.width
video_info.height
video_info.fps
video_info.total_frames
```

The duration can be calculated using:

```python
video_info.total_frames / video_info.fps
```

---

## Video Processing Callback

The project processes frames using a callback.

Simplified structure:

```python
def process_frame(
    frame: np.ndarray,
    frame_idx: int
) -> np.ndarray:

    results = model(
        frame,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    detections = detections[
        detections.class_id == TARGET_CLASS_ID
    ]

    detections = tracker.update_with_detections(
        detections
    )

    return frame
```

The complete callback additionally performs tracking analytics and annotations.

---

## Processing the Complete Video

Supervision provides:

```python
sv.process_video()
```

The project uses:

```python
sv.process_video(
    source_path=INPUT_VIDEO,
    target_path=OUTPUT_VIDEO,
    callback=process_frame,
    show_progress=True
)
```

For every frame:

```text
Read Frame
    ↓
Run Callback
    ↓
Return Annotated Frame
    ↓
Write Frame to Output Video
```

---

## Complete Project Flow

The full project can be represented as:

```text
vehicles.mp4
     ↓
VideoInfo
     ↓
YOLO
     ↓
sv.Detections
     ↓
Car Filter
     ↓
ByteTrack
     ↓
tracker_id
     ↓
Frame Counter
     ↓
Visible-Time Calculation
     ↓
Unique Tracker IDs
     ↓
BoxAnnotator
     ↓
LabelAnnotator
     ↓
TraceAnnotator
     ↓
tracked_vehicles.mp4
```

---

## Running the Project

From:

```text
05-projects/04-Object-Tracking/
```

install the dependencies if necessary:

```bash
pip install supervision ultralytics opencv-python numpy
```

Then run:

```bash
python object_tracking_pipeline.py
```

The script will:

```text
1. Create the required directories
2. Locate the input traffic video
3. Download the sample video if necessary
4. Read the video information
5. Load YOLOv8n
6. Create ByteTrack
7. Process every video frame
8. Detect objects
9. Filter cars
10. Track cars
11. Maintain tracker IDs
12. Count visible frames
13. Calculate visible time
14. Record unique tracker IDs
15. Draw boxes
16. Draw labels
17. Draw trajectories
18. Generate the output video
```

---

## Successful Console Output

A successful final run includes:

```text
Processing video: 100% 250/250

Unique tracker IDs:
[1, 2, 3, 4, 5, 6]

Total unique tracked cars:
6

Output video created successfully.

Object Tracking Project: SUCCESS
```

---

## Learning Outcomes

This project demonstrates how multiple computer vision concepts can be combined into one complete video-processing pipeline.

It connects:

```text
Object Detection
        ↓
Supervision Detections
        ↓
Class Filtering
        ↓
Object Tracking
        ↓
Persistent IDs
        ↓
Tracking Annotation
        ↓
Tracking Analytics
        ↓
Video Output
```

---

## Concepts Practiced

This project reinforces:

```python
YOLO()
```

```python
sv.Detections.from_ultralytics()
```

```python
sv.ByteTrack()
```

```python
tracker.update_with_detections()
```

```python
detections.tracker_id
```

```python
tracker.reset()
```

```python
sv.VideoInfo
```

```python
sv.process_video()
```

```python
sv.BoxAnnotator()
```

```python
sv.LabelAnnotator()
```

```python
sv.TraceAnnotator()
```

---

## Connection to Previous Projects

The previous projects focused on:

```text
Object Detection
      ↓
Annotation
      ↓
Detection Filtering
```

This project adds:

```text
Object Tracking
```

The progression becomes:

```text
YOLO Detection
      ↓
sv.Detections
      ↓
Annotation
      ↓
Filtering
      ↓
ByteTrack
      ↓
Persistent tracker_id
      ↓
Tracking Analytics
      ↓
Video Processing
```

---

## Future Extensions

The concepts introduced in this project can later be extended into:

```text
Zone Monitoring
Entry / Exit Counting
Direction Analysis
Traffic Analytics
People Tracking
Movement Analysis
Object Speed Estimation
SAM-Based Video Workflows
```

These applications all build on the ability to maintain object identities across consecutive video frames.

---

## Project Status

```text
Course Topic: Object Tracking
Project Type: Video Computer Vision
Detector: YOLOv8n
Tracking: ByteTrack
Detection Representation: sv.Detections
Target Class: Car (COCO Class ID 2)
Filtering: Class-based
Annotation: Boxes + Labels + Traces
Analytics: Frame Count + Visible Time + Unique IDs
Input: Real 10-second traffic video
Input Frames: 250
Final Tracked Cars: 6
Output: tracked_vehicles.mp4
Environment: Google Colab
Test Result: SUCCESS
Status: Completed and successfully tested
```

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey

- GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)
- LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)

---

## Repository

This project is part of the:

[SAM3 Learning Journey](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

The repository documents the progression from fundamental computer vision concepts toward complete detection, tracking, segmentation, and SAM3 workflows.
