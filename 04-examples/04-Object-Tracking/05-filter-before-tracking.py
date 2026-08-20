```python
"""
05-filter-before-tracking.py

Filter Detections Before Object Tracking

This example demonstrates how to combine detection
filtering with ByteTrack.

The pipeline is:

1. Detect objects with YOLO
2. Convert results to sv.Detections
3. Filter detections by class
4. Send only filtered detections to ByteTrack
5. Assign tracker IDs
6. Annotate the tracked objects
7. Save the output video

In this example, only cars are tracked.
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

OUTPUT_VIDEO = "assets/vehicles_cars.mp4"

# COCO class ID:
# 2 = car
TARGET_CLASS = 2


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
    # Filter detections
    #
    # Keep only cars:
    # class_id == 2
    # ----------------------------------------------

    detections = detections[
        detections.class_id == TARGET_CLASS
    ]


    # ----------------------------------------------
    # Apply ByteTrack AFTER filtering
    # ----------------------------------------------

    detections = tracker.update_with_detections(
        detections
    )


    # ----------------------------------------------
    # Create labels
    # ----------------------------------------------

    labels = [
        f"car #{tracker_id}"
        for tracker_id
        in detections.tracker_id
    ]


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
# Summary
# --------------------------------------------------

print(
    f"\nFiltered tracking video saved to: "
    f"{OUTPUT_VIDEO}"
)

print(
    f"Target class ID: {TARGET_CLASS}"
)

print(
    "Only car detections were passed "
    "to ByteTrack."
)

print(
    "Pipeline completed successfully:"
)

print(
    "YOLO -> Filter -> ByteTrack -> Annotation"
)
```
