```python
"""
08-visible-time.py

Estimate How Long Each Tracked Object Remains Visible

This example extends the tracking frame counter by converting
the number of visible frames into approximate visible time.

The calculation is:

    visible_seconds = frames_visible / video_fps

The example:

1. Loads the sample video
2. Reads the video FPS
3. Detects objects with YOLO
4. Tracks objects with ByteTrack
5. Counts how many frames each tracker_id appears
6. Converts frame count into seconds
7. Displays class, tracker ID, frames, and visible time
8. Saves the annotated output video
"""

from pathlib import Path
import urllib.request

import numpy as np
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "yolov8n.pt"

VIDEO_URL = (
    "https://media.roboflow.com/"
    "supervision/video-examples/vehicles.mp4"
)

SOURCE_VIDEO = "assets/vehicles.mp4"

OUTPUT_VIDEO = "assets/vehicles_visible_time.mp4"


# --------------------------------------------------
# Create assets directory
# --------------------------------------------------

Path("assets").mkdir(exist_ok=True)


# --------------------------------------------------
# Download sample video if needed
# --------------------------------------------------

if not Path(SOURCE_VIDEO).exists():

    print("Downloading sample video...")

    urllib.request.urlretrieve(
        VIDEO_URL,
        SOURCE_VIDEO
    )

    print(
        f"Video saved to: {SOURCE_VIDEO}"
    )


# --------------------------------------------------
# Load video information
# --------------------------------------------------

video_info = sv.VideoInfo.from_video_path(
    SOURCE_VIDEO
)

VIDEO_FPS = video_info.fps


print(
    f"Video FPS: {VIDEO_FPS}"
)


# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

model = YOLO(MODEL_NAME)


# --------------------------------------------------
# Create ByteTrack tracker
# --------------------------------------------------

tracker = sv.ByteTrack()


# --------------------------------------------------
# Reset tracker
# --------------------------------------------------

tracker.reset()


# --------------------------------------------------
# Create annotators
# --------------------------------------------------

box_annotator = sv.BoxAnnotator()

label_annotator = sv.LabelAnnotator()

trace_annotator = sv.TraceAnnotator()


# --------------------------------------------------
# Store frame counts
#
# Structure:
#
# tracker_id -> frames visible
#
# Example:
#
# {
#     1: 45,
#     2: 71,
#     3: 20
# }
# --------------------------------------------------

frame_count = {}


# --------------------------------------------------
# Frame processing callback
# --------------------------------------------------

def process_frame(
    frame: np.ndarray,
    frame_idx: int
) -> np.ndarray:

    # ----------------------------------------------
    # Run YOLO detection
    # ----------------------------------------------

    results = model(
        frame,
        verbose=False
    )[0]


    # ----------------------------------------------
    # Convert YOLO results to Supervision
    # ----------------------------------------------

    detections = sv.Detections.from_ultralytics(
        results
    )


    # ----------------------------------------------
    # Apply ByteTrack
    # ----------------------------------------------

    detections = tracker.update_with_detections(
        detections
    )


    # ----------------------------------------------
    # Update frame counters
    # ----------------------------------------------

    if detections.tracker_id is not None:

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


    # ----------------------------------------------
    # Create labels
    #
    # Example:
    #
    # car #1 | 45f | 1.50s
    # ----------------------------------------------

    labels = []

    if detections.tracker_id is not None:

        for class_id, tracker_id in zip(
            detections.class_id,
            detections.tracker_id
        ):

            class_id = int(
                class_id
            )

            tracker_id = int(
                tracker_id
            )

            class_name = results.names[
                class_id
            ]

            frames_visible = frame_count[
                tracker_id
            ]

            visible_seconds = (
                frames_visible /
                VIDEO_FPS
            )

            label = (
                f"{class_name} "
                f"#{tracker_id} | "
                f"{frames_visible}f | "
                f"{visible_seconds:.2f}s"
            )

            labels.append(
                label
            )


    # ----------------------------------------------
    # Draw bounding boxes
    # ----------------------------------------------

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )


    # ----------------------------------------------
    # Draw labels
    # ----------------------------------------------

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )


    # ----------------------------------------------
    # Draw trajectories
    # ----------------------------------------------

    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )


    # ----------------------------------------------
    # Return processed frame
    # ----------------------------------------------

    return annotated_frame


# --------------------------------------------------
# Process complete video
# --------------------------------------------------

sv.process_video(
    source_path=SOURCE_VIDEO,
    target_path=OUTPUT_VIDEO,
    callback=process_frame,
    show_progress=True
)


# --------------------------------------------------
# Print final analytics
# --------------------------------------------------

print(
    "\nVisible time per tracked object:"
)

print(
    "-" * 50
)

for tracker_id, frames_visible in sorted(
    frame_count.items()
):

    visible_seconds = (
        frames_visible /
        VIDEO_FPS
    )

    print(
        f"Tracker ID {tracker_id}: "
        f"{frames_visible} frames "
        f"-> {visible_seconds:.2f} seconds"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print(
    f"\nOutput video saved to: "
    f"{OUTPUT_VIDEO}"
)

print(
    "Visible time was estimated using:"
)

print(
    "visible_seconds = "
    "frames_visible / video_fps"
)

print(
    "\nExample label:"
)

print(
    "car #1 | 45f | 1.50s"
)

print(
    "\nVisible-time tracking completed successfully."
)
```
