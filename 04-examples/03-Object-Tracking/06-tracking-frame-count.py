```python
"""
06-tracking-frame-count.py

Count How Many Frames Each Object Remains Visible

This example extends the object tracking pipeline by using
tracker_id values for simple tracking analytics.

The example:

1. Detects objects with YOLO
2. Converts results to sv.Detections
3. Tracks objects with ByteTrack
4. Counts how many frames each tracker_id appears
5. Displays the frame count in each object's label
6. Draws bounding boxes and trajectories
7. Saves the annotated output video

Example label:

    car #1 (25f)

where:

    car = detected class
    #1  = tracker ID
    25f = object visible for 25 processed frames
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

OUTPUT_VIDEO = "assets/vehicles_frame_count.mp4"


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
# Tracking analytics
#
# Dictionary structure:
#
# tracker_id -> frames visible
#
# Example:
#
# {
#     1: 25,
#     2: 17,
#     3: 42
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
    # Update frame counts
    # ----------------------------------------------

    if detections.tracker_id is not None:

        for tracker_id in detections.tracker_id:

            tracker_id = int(tracker_id)

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
    # car #1 (25f)
    # truck #2 (17f)
    # ----------------------------------------------

    labels = []

    if detections.tracker_id is not None:

        for class_id, tracker_id in zip(
            detections.class_id,
            detections.tracker_id
        ):

            class_id = int(class_id)

            tracker_id = int(tracker_id)

            class_name = results.names[
                class_id
            ]

            visible_frames = frame_count[
                tracker_id
            ]

            label = (
                f"{class_name} "
                f"#{tracker_id} "
                f"({visible_frames}f)"
            )

            labels.append(label)


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
    # Draw object trajectories
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
# Print tracking analytics
# --------------------------------------------------

print(
    "\nTracking frame counts:"
)

print(
    "-" * 40
)

for tracker_id, frames_visible in sorted(
    frame_count.items()
):

    print(
        f"Tracker ID {tracker_id}: "
        f"{frames_visible} frames"
    )


# --------------------------------------------------
# Count unique tracker IDs
# --------------------------------------------------

unique_objects = len(
    frame_count
)


print(
    "\nUnique tracker IDs:"
)

print(
    unique_objects
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print(
    f"\nOutput video saved to: "
    f"{OUTPUT_VIDEO}"
)

print(
    "Each tracked object's label contains "
    "its class, tracker ID, and number "
    "of frames visible."
)

print(
    "\nExample:"
)

print(
    "car #1 (25f)"
)

print(
    "\nTracking analytics completed successfully."
)
```
