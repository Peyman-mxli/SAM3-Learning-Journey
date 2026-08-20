"""
06-line-zone-crossing-count.py

LineZone Crossing Count Example

This example demonstrates how to:

- Detect objects with YOLOv8
- Track objects with ByteTrack
- Keep only confirmed tracker IDs
- Trigger a LineZone
- Count directional crossing events
- Process an entire video
- Display the final crossing counts

Concepts:
- YOLOv8 detection
- ByteTrack tracking
- Persistent tracker IDs
- Confirmed track filtering
- LineZone.trigger()
- in_count
- out_count
- Accumulated flow
"""

from pathlib import Path

import numpy as np
import supervision as sv

from ultralytics import YOLO
from trackers import ByteTrackTracker


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "yolov8n.pt"

INPUT_VIDEO = "assets/input/vehicles.mp4"

OUTPUT_VIDEO = "line_zone_crossing_count.mp4"


# --------------------------------------------------
# Validate Input
# --------------------------------------------------

if not Path(INPUT_VIDEO).exists():
    raise FileNotFoundError(
        f"Input video not found: {INPUT_VIDEO}"
    )


# --------------------------------------------------
# Load YOLO Model
# --------------------------------------------------

model = YOLO(
    MODEL_NAME
)


# --------------------------------------------------
# Read Video Information
# --------------------------------------------------

video_info = sv.VideoInfo.from_video_path(
    INPUT_VIDEO
)


# --------------------------------------------------
# Define Counting Line
# --------------------------------------------------

line_start = sv.Point(
    x=0,
    y=video_info.height // 2
)

line_end = sv.Point(
    x=video_info.width,
    y=video_info.height // 2
)


# --------------------------------------------------
# Create LineZone
# --------------------------------------------------

line_zone = sv.LineZone(
    start=line_start,
    end=line_end
)


# --------------------------------------------------
# Create Tracker
# --------------------------------------------------

tracker = ByteTrackTracker()


# --------------------------------------------------
# Create Annotators
# --------------------------------------------------

box_annotator = sv.BoxAnnotator()

label_annotator = sv.LabelAnnotator()

line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_scale=1.5,
    custom_in_text="Crossings Down",
    custom_out_text="Crossings Up"
)


# --------------------------------------------------
# Video Processing Callback
# --------------------------------------------------

def callback(
    frame: np.ndarray,
    frame_index: int
) -> np.ndarray:

    # ----------------------------------------------
    # Detect Objects
    # ----------------------------------------------

    results = model(
        frame,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    # ----------------------------------------------
    # Track Objects
    # ----------------------------------------------

    detections = tracker.update(
        detections
    )

    # ----------------------------------------------
    # Keep Confirmed Tracks
    # ----------------------------------------------

    if detections.tracker_id is not None:

        confirmed_mask = (
            detections.tracker_id != -1
        )

        detections = detections[
            confirmed_mask
        ]

    # ----------------------------------------------
    # Trigger LineZone
    # ----------------------------------------------

    line_zone.trigger(
        detections=detections
    )

    # ----------------------------------------------
    # Create Tracker ID Labels
    # ----------------------------------------------

    if detections.tracker_id is not None:

        labels = [
            f"ID:{tracker_id}"
            for tracker_id
            in detections.tracker_id
        ]

    else:

        labels = []

    # ----------------------------------------------
    # Draw Bounding Boxes
    # ----------------------------------------------

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    # ----------------------------------------------
    # Draw Tracker IDs
    # ----------------------------------------------

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # ----------------------------------------------
    # Draw LineZone
    # ----------------------------------------------

    annotated_frame = line_zone_annotator.annotate(
        frame=annotated_frame,
        line_counter=line_zone
    )

    return annotated_frame


# --------------------------------------------------
# Process Video
# --------------------------------------------------

print(
    "Processing LineZone crossing example..."
)

sv.process_video(
    source_path=INPUT_VIDEO,
    target_path=OUTPUT_VIDEO,
    callback=callback,
    show_progress=True
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print(
    "\nLineZone processing completed."
)

print(
    f"Crossings Down: "
    f"{line_zone.in_count}"
)

print(
    f"Crossings Up: "
    f"{line_zone.out_count}"
)

print(
    f"Total crossings: "
    f"{line_zone.in_count + line_zone.out_count}"
)

print(
    f"Saved: "
    f"{OUTPUT_VIDEO}"
)
