```python
"""
04-class-and-tracker-id.py

Display Object Class and Tracker ID

This example demonstrates how to combine the detected
object class with the persistent tracker ID.

Instead of displaying only:

    ID:1

the annotation displays:

    car #1
    truck #2
    bus #3

This allows us to see both:

1. WHAT the object is
2. WHICH tracked object it is
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

OUTPUT_VIDEO = "assets/vehicles_class_id.mp4"


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
    # Create class + tracker ID labels
    # ----------------------------------------------

    labels = [
        f"{results.names[int(class_id)]} "
        f"#{tracker_id}"
        for class_id, tracker_id in zip(
            detections.class_id,
            detections.tracker_id
        )
    ]


    # ----------------------------------------------
    # Draw bounding boxes
    # ----------------------------------------------

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )


    # ----------------------------------------------
    # Draw class + tracker ID labels
    # ----------------------------------------------

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
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
# Summary
# --------------------------------------------------

print(
    f"\nTracked video saved to: "
    f"{OUTPUT_VIDEO}"
)

print(
    "Each tracked object is labeled "
    "with its class and tracker ID."
)

print(
    "Example: car #1, truck #2, bus #3"
)
```
