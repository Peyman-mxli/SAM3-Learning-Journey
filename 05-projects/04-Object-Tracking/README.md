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

The objective of this project is to process a real video and create an annotated output containing:

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
    ├── input/
    │   └── vehicles.mp4
    │
    └── output/
        └── tracked_vehicles.mp4
```

---

## Input

The project expects an input video at:

```text
assets/input/vehicles.mp4
```

The course lesson uses the sample vehicle video:

```text
https://media.roboflow.com/supervision/video-examples/vehicles.mp4
```

The script can automatically download the video if it does not already exist.

---

## Output

The processed tracking video is saved to:

```text
assets/output/tracked_vehicles.mp4
```

The output video contains tracking information for the selected objects.

Example labels may look like:

```text
car #1 | 25f | 0.83s

car #2 | 47f | 1.57s

car #3 | 91f | 3.03s
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
25f
```

represents the number of frames visible,

and:

```text
0.83s
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

YOLO performs object detection on each video frame.

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

The tracker receives detections and attempts to maintain object identities between consecutive frames.

---

## Resetting the Tracker

Before processing a new video:

```python
tracker.reset()
```

This clears the previous tracking state.

The tracker must maintain state during one complete video-processing sequence.

---

## Detection

For every frame:

```python
results = model(
    frame,
    verbose=False
)[0]
```

YOLO provides detection information such as:

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

This gives the pipeline access to properties such as:

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

This is because YOLO detects objects but does not assign persistent tracking identities.

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

Each value identifies an individual tracked object.

---

## `class_id` vs. `tracker_id`

These two values represent different information.

### `class_id`

Answers:

```text
What type of object is this?
```

For example:

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

For example:

```text
car #1
car #2
car #3
```

Several objects may belong to the same class while having different tracking IDs.

---

## Filtering Before Tracking

This project also applies filtering before ByteTrack.

The lesson demonstrates tracking only cars.

In the COCO dataset:

```python
TARGET_CLASS = 2
```

represents:

```text
car
```

The detections are filtered using:

```python
detections = detections[
    detections.class_id == TARGET_CLASS
]
```

Then the filtered detections are sent to ByteTrack:

```python
detections = tracker.update_with_detections(
    detections
)
```

---

## Why Filter Before Tracking?

The order is:

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

Filtering first means ByteTrack only manages objects relevant to the application.

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

Only those objects are tracked.

---

## Tracking Labels

Tracker IDs can be displayed using:

```python
sv.LabelAnnotator()
```

A simple tracking label can be created with:

```python
labels = [
    f"ID:{tracker_id}"
    for tracker_id
    in detections.tracker_id
]
```

This produces:

```text
ID:1
ID:2
ID:3
```

---

## Class and Tracker ID Labels

A more informative label combines the class name with the tracker ID.

Example:

```python
labels = [
    f"{results.names[int(class_id)]} "
    f"#{tracker_id}"
    for class_id, tracker_id in zip(
        detections.class_id,
        detections.tracker_id
    )
]
```

Example output:

```text
car #1
car #2
truck #3
```

---

## Bounding Box Annotation

Bounding boxes are drawn using:

```python
sv.BoxAnnotator()
```

Create the annotator:

```python
box_annotator = sv.BoxAnnotator()
```

Apply it:

```python
annotated_frame = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections
)
```

---

## Label Annotation

Labels are drawn using:

```python
sv.LabelAnnotator()
```

Example:

```python
annotated_frame = label_annotator.annotate(
    scene=annotated_frame,
    detections=detections,
    labels=labels
)
```

---

## Object Trajectories

The project also uses:

```python
sv.TraceAnnotator()
```

Create it:

```python
trace_annotator = sv.TraceAnnotator()
```

Apply it:

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

---

## Tracking Analytics

The project uses persistent tracking IDs to create simple analytics.

A dictionary stores how many frames each object remains visible:

```python
frame_count = {}
```

For every tracker ID:

```python
for tracker_id in detections.tracker_id:

    tracker_id = int(
        tracker_id
    )

    frame_count[tracker_id] = (
        frame_count.get(
            tracker_id,
            0
        ) + 1
    )
```

---

## Frame Visibility Count

Suppose object:

```text
#5
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

The approximate visible time can be calculated using the video's FPS.

The formula is:

```text
Visible Time =
Frames Visible
──────────────
     FPS
```

In Python:

```python
visible_seconds = (
    frames_visible /
    video_fps
)
```

For example:

```text
frames visible = 90

FPS = 30
```

Then:

```text
90 / 30 = 3 seconds
```

The label could display:

```text
car #5 | 90f | 3.00s
```

---

## Unique Tracker IDs

The project can also store every tracker ID inside a Python set.

```python
unique_tracker_ids = set()
```

Then:

```python
unique_tracker_ids.add(
    tracker_id
)
```

The number of unique tracker IDs is:

```python
len(
    unique_tracker_ids
)
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

the same physical vehicle may now have a different ID.

Therefore:

```text
unique tracker IDs
```

should not automatically be interpreted as perfectly accurate unique real-world object counts.

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

Example structure:

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
        detections.class_id == TARGET_CLASS
    ]

    detections = tracker.update_with_detections(
        detections
    )

    return frame
```

The callback contains the main computer vision logic.

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
Class Filter
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

From the project directory:

```bash
python object_tracking_pipeline.py
```

The script will:

```text
1. Create required directories
2. Download the sample video if needed
3. Load YOLO
4. Create ByteTrack
5. Read video information
6. Process every video frame
7. Filter selected objects
8. Track them
9. Generate tracking analytics
10. Create an annotated output video
```

---

## Expected Console Output

The exact values depend on the tracking results, but the console may display information such as:

```text
Video information:

Resolution: ...
FPS: ...
Total frames: ...
Duration: ...

Starting object tracking...

Tracking analytics
----------------------------------------

Tracker ID 1: 75 frames | 2.50 seconds
Tracker ID 2: 41 frames | 1.37 seconds
Tracker ID 3: 98 frames | 3.27 seconds

Total unique tracked objects: 3

Output video saved to:
assets/output/tracked_vehicles.mp4
```

---

## Learning Outcomes

This project demonstrates how multiple computer vision concepts can be combined into one pipeline.

It connects:

```text
Object Detection
        ↓
Supervision Detections
        ↓
Filtering
        ↓
Object Tracking
        ↓
Tracking Annotation
        ↓
Tracking Analytics
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
Tracking
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
Video Analytics
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

These applications all build on the ability to maintain object identities across video frames.

---

## Project Status

```text
Course Topic: Object Tracking
Project Type: Video Computer Vision
Detector: YOLOv8n
Tracking: ByteTrack
Detection Representation: sv.Detections
Filtering: Class-based
Annotation: Boxes + Labels + Traces
Analytics: Frame Count + Visible Time + Unique IDs
Status: Ready for implementation and testing
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
