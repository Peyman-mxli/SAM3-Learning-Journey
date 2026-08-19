
```python
"""
02-tracker-ids-across-frames.py

Tracking IDs Across Video Frames

This example demonstrates how ByteTrack maintains object
identities across consecutive video frames.

The example:

1. Loads a YOLO model
2. Opens a sample video
3. Processes consecutive frames
4. Detects objects with YOLO
5. Converts results to sv.Detections
6. Applies ByteTrack
7. Prints tracker IDs for every processed frame

The main objective is to observe whether the same objects
maintain the same tracker_id values as they move.
"""

from pathlib import Path
import urllib.request

import cv2
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

VIDEO_PATH = "assets/vehicles.mp4"

FRAMES_TO_PROCESS = 10


# --------------------------------------------------
# Create assets directory
# --------------------------------------------------

Path("assets").mkdir(exist_ok=True)


# --------------------------------------------------
# Download sample video if needed
# --------------------------------------------------

if not Path(VIDEO_PATH).exists():

    print("Downloading sample video...")

    urllib.request.urlretrieve(
        VIDEO_URL,
        VIDEO_PATH
    )

    print(
        f"Video saved to: {VIDEO_PATH}"
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
# Open video
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video: "
        f"{VIDEO_PATH}"
    )


# --------------------------------------------------
# Process consecutive frames
# --------------------------------------------------

print(
    "\nTracking IDs across consecutive frames:\n"
)

for frame_num in range(FRAMES_TO_PROCESS):

    # ----------------------------------------------
    # Read frame
    # ----------------------------------------------

    ret, frame = cap.read()

    if not ret or frame is None:

        print(
            "No more frames available."
        )

        break


    # ----------------------------------------------
    # Run YOLO detection
    # ----------------------------------------------

    results = model(
        frame,
        verbose=False
    )[0]


    # ----------------------------------------------
    # Convert to Supervision detections
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
    # Print frame information
    # ----------------------------------------------

    print(
        f"Frame {frame_num}: "
        f"{len(detections)} tracked objects"
    )

    print(
        f"Tracker IDs: "
        f"{detections.tracker_id}"
    )


    # ----------------------------------------------
    # Print individual tracked objects
    # ----------------------------------------------

    if (
        detections.tracker_id
        is not None
    ):

        for class_id, tracker_id in zip(
            detections.class_id,
            detections.tracker_id
        ):

            class_name = results.names[
                int(class_id)
            ]

            print(
                f"  {class_name} "
                f"-> ID {tracker_id}"
            )


    print(
        "-" * 50
    )


# --------------------------------------------------
# Release video
# --------------------------------------------------

cap.release()


# --------------------------------------------------
# Summary
# --------------------------------------------------

print(
    "\nTracking experiment completed."
)

print(
    "Compare the tracker IDs between frames."
)

print(
    "When ByteTrack successfully associates "
    "the same physical object across frames, "
    "its tracker_id remains the same."
)
```
