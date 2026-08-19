# Video Processing with Supervision

Object tracking requires us to process a sequence of video frames.

Unlike image detection, where we analyze a single image, video tracking repeatedly performs detection and tracking while preserving information between frames.

Supervision provides useful tools for working with videos, including:

```python
sv.VideoInfo
```

and:

```python
sv.process_video()
```

These tools allow us to inspect video information and apply a processing function to every frame.

---

## From Images to Video

With a single image, the basic detection workflow is:

```text
Image
  ↓
YOLO
  ↓
sv.Detections
  ↓
Annotation
  ↓
Output Image
```

Video processing extends this workflow.

A video contains many consecutive frames:

```text
Video
  ↓
Frame 0
Frame 1
Frame 2
Frame 3
...
```

Each frame must be processed.

For tracking, the same tracker must also preserve its state between those frames.

---

## Video Tracking Pipeline

The general workflow becomes:

```text
Input Video
     ↓
Read Frame
     ↓
YOLO
     ↓
sv.Detections
     ↓
ByteTrack
     ↓
tracker_id
     ↓
Annotation
     ↓
Processed Frame
     ↓
Output Video
```

This process repeats until all frames have been processed.

---

## Preparing the Video

The lesson uses a sample vehicle video.

First, create an assets directory:

```python
from pathlib import Path

Path("assets").mkdir(exist_ok=True)
```

Then download the sample video:

```python
import urllib.request

urllib.request.urlretrieve(
    "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
    "assets/vehicles.mp4"
)
```

The source video is stored at:

```text
assets/vehicles.mp4
```

---

## Inspecting Video Information

Supervision provides:

```python
sv.VideoInfo
```

This allows us to inspect information about a video before processing it.

Create a `VideoInfo` object:

```python
video_info = sv.VideoInfo.from_video_path(
    "assets/vehicles.mp4"
)
```

The object contains useful video metadata.

---

## Video Width

The width of the video can be accessed with:

```python
video_info.width
```

For example:

```python
print(video_info.width)
```

This represents the horizontal resolution of each frame.

---

## Video Height

The height can be accessed with:

```python
video_info.height
```

Example:

```python
print(video_info.height)
```

Together:

```text
width × height
```

represent the resolution of the video.

---

## Frames Per Second

The video's frame rate is available through:

```python
video_info.fps
```

Example:

```python
print(video_info.fps)
```

FPS means:

```text
Frames Per Second
```

For example:

```text
30 FPS
```

means approximately 30 frames are displayed every second.

---

## Total Frames

The total number of frames is available using:

```python
video_info.total_frames
```

Example:

```python
print(video_info.total_frames)
```

This tells us how many frames must be processed.

---

## Calculating Video Duration

The approximate duration of a video can be calculated using:

```python
video_info.total_frames / video_info.fps
```

Conceptually:

```text
Duration =
Total Frames
────────────
    FPS
```

For example:

```python
duration = (
    video_info.total_frames /
    video_info.fps
)

print(duration)
```

The result represents the approximate duration in seconds.

---

## Why Video Information Is Useful

Video metadata helps us understand the processing workload.

For example:

```text
Resolution
    ↓
How large each frame is

FPS
    ↓
How many frames occur each second

Total Frames
    ↓
How many frames must be processed
```

This information can become important when working with large videos or expensive computer vision models.

---

## Reading a Video with OpenCV

A video can also be opened manually using OpenCV.

```python
import cv2

cap = cv2.VideoCapture(
    "assets/vehicles.mp4"
)
```

Read one frame:

```python
ret, frame = cap.read()
```

Then release the video:

```python
cap.release()
```

---

## Understanding `cap.read()`

The following line:

```python
ret, frame = cap.read()
```

returns two values.

### `ret`

Indicates whether a frame was successfully read.

For example:

```python
if not ret:
    break
```

### `frame`

Contains the actual video frame as an image array.

That frame can then be passed to YOLO.

---

## Processing a Single Video Frame

Once a frame has been extracted, YOLO can process it normally.

```python
results = model(
    frame,
    verbose=False
)[0]
```

Convert the result into Supervision detections:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

Then apply the tracker:

```python
detections = tracker.update_with_detections(
    detections
)
```

The single-frame pipeline becomes:

```text
Video
  ↓
Frame
  ↓
YOLO
  ↓
sv.Detections
  ↓
ByteTrack
  ↓
Tracked Detections
```

---

## Why Manual Frame Processing Is Useful

Reading frames manually is useful when we want to inspect tracking behavior.

For example, the lesson processes several frames and prints their IDs.

```python
tracker.reset()

cap = cv2.VideoCapture(
    "assets/vehicles.mp4"
)

for frame_num in range(3):

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    detections = tracker.update_with_detections(
        detections
    )

    print(
        f"Frame {frame_num}: "
        f"{len(detections)} objects | "
        f"IDs: {detections.tracker_id}"
    )

cap.release()
```

This makes it possible to inspect how IDs behave between consecutive frames.

---

## Processing an Entire Video

Manually writing a loop is useful for experiments.

However, Supervision also provides:

```python
sv.process_video()
```

This function can process the entire video for us.

The basic structure is:

```python
sv.process_video(
    source_path=SOURCE_VIDEO,
    target_path=TARGET_VIDEO,
    callback=process_frame,
    show_progress=True
)
```

---

## Source Path

The source path identifies the input video.

Example:

```python
source_path="assets/vehicles.mp4"
```

This is the video that will be processed.

---

## Target Path

The target path identifies where the processed video will be saved.

Example:

```python
target_path="assets/vehicles_tracked.mp4"
```

The original video remains the input.

The annotated result is written to a new video file.

---

## Callback Function

The most important part of:

```python
sv.process_video()
```

is the callback.

Example:

```python
callback=process_frame
```

The callback is a function that receives each frame.

It performs the computer vision operations and returns the processed frame.

Conceptually:

```text
Frame
  ↓
Callback
  ↓
Processed Frame
```

---

## Basic Callback Structure

A callback can be defined like this:

```python
def process_frame(
    frame: np.ndarray,
    frame_idx: int
) -> np.ndarray:

    # Process frame here

    return frame
```

The function receives:

```python
frame
```

and:

```python
frame_idx
```

Then it returns the processed frame.

---

## `frame`

The first callback argument contains the current video frame.

```python
frame
```

This is the image that will be sent to YOLO.

Example:

```python
results = model(
    frame,
    verbose=False
)[0]
```

---

## `frame_idx`

The second callback argument contains the current frame index.

```python
frame_idx
```

Conceptually:

```text
Frame 0
Frame 1
Frame 2
Frame 3
...
```

The frame index can be useful for:

- Debugging
- Logging
- Frame-specific processing
- Experiments

If it is not needed, it can also be represented using:

```python
_
```

Example:

```python
def callback(
    frame: np.ndarray,
    _: int
) -> np.ndarray:
```

---

## Tracking Inside the Callback

For object tracking, the callback can perform:

1. Object detection
2. Conversion to `sv.Detections`
3. Tracking
4. Label creation
5. Annotation

Example:

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

    detections = tracker.update_with_detections(
        detections
    )

    labels = [
        f"ID:{tracker_id}"
        for tracker_id in detections.tracker_id
    ]

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

    return annotated
```

---

## Why the Tracker Is Created Outside the Callback

The tracker needs to maintain information between frames.

Therefore, it should exist before the callback starts processing the video.

Example:

```python
tracker = sv.ByteTrack()
```

Then:

```python
def process_frame(
    frame: np.ndarray,
    frame_idx: int
) -> np.ndarray:

    ...

    detections = tracker.update_with_detections(
        detections
    )

    ...
```

Conceptually:

```text
One ByteTrack Instance
        ↓
     Frame 0
        ↓
     Frame 1
        ↓
     Frame 2
        ↓
     Frame 3
```

The same tracker processes the complete sequence.

---

## Why We Should Not Create ByteTrack Inside the Callback

If we created:

```python
tracker = sv.ByteTrack()
```

again for every frame, the tracker would not preserve its tracking history correctly.

Conceptually, that would become:

```text
Frame 0 → New Tracker

Frame 1 → New Tracker

Frame 2 → New Tracker

Frame 3 → New Tracker
```

The purpose of tracking is to connect information across frames.

Therefore, we need one tracker that maintains state throughout the video sequence.

---

## Reset Before Processing

Before starting a new complete tracking run, the tracker can be reset:

```python
tracker.reset()
```

Then process the video:

```python
sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_tracked.mp4",
    callback=process_frame,
    show_progress=True
)
```

This starts the video with a clean tracking state.

---

## Showing Processing Progress

The lesson uses:

```python
show_progress=True
```

Example:

```python
sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_tracked.mp4",
    callback=process_frame,
    show_progress=True
)
```

This allows us to see the processing progress while Supervision works through the video.

---

## Complete Example

A complete tracking setup can be organized like this:

```python
import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
```

Create the callback:

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

    detections = tracker.update_with_detections(
        detections
    )

    labels = [
        f"ID:{tracker_id}"
        for tracker_id in detections.tracker_id
    ]

    scene = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    scene = label_annotator.annotate(
        scene=scene,
        detections=detections,
        labels=labels
    )

    scene = trace_annotator.annotate(
        scene=scene,
        detections=detections
    )

    return scene
```

Reset the tracker:

```python
tracker.reset()
```

Process the video:

```python
sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_tracked.mp4",
    callback=process_frame,
    show_progress=True
)
```

---

## What Happens for Every Frame?

For every frame, Supervision calls our callback.

Conceptually:

```text
Frame 0
  ↓
process_frame()
  ↓
Tracked Frame 0

Frame 1
  ↓
process_frame()
  ↓
Tracked Frame 1

Frame 2
  ↓
process_frame()
  ↓
Tracked Frame 2

...

Final Output Video
```

The callback defines what happens to each frame.

---

## Detection, Tracking, and Annotation

Inside the callback, we can clearly separate the different stages.

### Stage 1 — Detection

```python
results = model(
    frame,
    verbose=False
)[0]
```

### Stage 2 — Convert to Supervision

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

### Stage 3 — Tracking

```python
detections = tracker.update_with_detections(
    detections
)
```

### Stage 4 — Labels

```python
labels = [
    f"ID:{tracker_id}"
    for tracker_id in detections.tracker_id
]
```

### Stage 5 — Annotation

```python
scene = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections
)
```

Then:

```python
scene = label_annotator.annotate(
    scene=scene,
    detections=detections,
    labels=labels
)
```

And optionally:

```python
scene = trace_annotator.annotate(
    scene=scene,
    detections=detections
)
```

Finally:

```python
return scene
```

---

## Adding Filtering to Video Processing

The concepts from the previous lesson can also be inserted into the callback.

For example:

```python
detections = sv.Detections.from_ultralytics(
    results
)

detections = detections[
    detections.class_id == TARGET_CLASS
]

detections = tracker.update_with_detections(
    detections
)
```

The pipeline becomes:

```text
Video Frame
    ↓
YOLO
    ↓
sv.Detections
    ↓
Filter
    ↓
ByteTrack
    ↓
Annotation
    ↓
Output Frame
```

This allows the video pipeline to track only selected objects.

---

## Example — Track Cars Only

Suppose:

```python
TARGET_CLASS = 2
```

represents cars.

The callback can contain:

```python
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
```

Now the tracker receives only car detections.

---

## Manual Loop vs. `sv.process_video`

Both approaches are useful.

### Manual OpenCV Loop

Example:

```python
cap = cv2.VideoCapture(
    "assets/vehicles.mp4"
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Process frame

cap.release()
```

This approach gives us direct control over each frame.

It is useful for experiments and debugging.

### `sv.process_video`

Example:

```python
sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_tracked.mp4",
    callback=process_frame
)
```

This provides a convenient way to apply the same processing pipeline to the complete video.

---

## Complete Video Processing Flow

The full lesson workflow can be summarized as:

```text
vehicles.mp4
      ↓
sv.process_video()
      ↓
Read Current Frame
      ↓
Callback
      ↓
YOLO Detection
      ↓
sv.Detections
      ↓
Optional Filtering
      ↓
ByteTrack
      ↓
tracker_id
      ↓
BoxAnnotator
      ↓
LabelAnnotator
      ↓
TraceAnnotator
      ↓
Return Processed Frame
      ↓
Next Frame
      ↓
...
      ↓
vehicles_tracked.mp4
```

---

## Key Takeaways

- Videos are sequences of individual frames.
- Object tracking requires consecutive frames to be processed in order.
- `sv.VideoInfo` provides information about a video.
- Useful video properties include width, height, FPS, and total frames.
- OpenCV can be used to manually read frames with `cv2.VideoCapture`.
- `sv.process_video()` provides a convenient way to process an entire video.
- A callback function defines what happens to each frame.
- YOLO performs detection inside the callback.
- YOLO results are converted into `sv.Detections`.
- ByteTrack updates the detections with tracking IDs.
- The same tracker should maintain state across consecutive frames.
- `tracker.reset()` can be used before beginning a new tracking run.
- Annotation is performed after detection and tracking.
- Filtering can be inserted before tracking.
- The processed frames are written to a new output video.

---

## Next Concept

The next concept focuses on visualizing tracking results using:

```python
sv.BoxAnnotator()
sv.LabelAnnotator()
sv.TraceAnnotator()
```

and understanding how bounding boxes, tracking labels, and object trajectories work together in the final tracked video.
