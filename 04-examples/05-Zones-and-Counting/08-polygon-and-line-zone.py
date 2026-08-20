"""
08-polygon-and-line-zone.py

PolygonZone + LineZone

This example combines occupancy analysis and crossing analysis
using the same tracked detections.

It demonstrates how to:

- Detect objects with YOLOv8
- Track objects with ByteTrack
- Keep only confirmed tracker IDs
- Trigger PolygonZone
- Trigger LineZone
- Display tracker IDs
- Display polygon occupancy
- Display directional crossing counts
- Process both spatial analytics systems in one video

Concepts:
- Shared tracked detections
- PolygonZone
- LineZone
- Occupancy
- Flow
- Persistent IDs
- Multiple annotators
- Spatial video analytics
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

OUTPUT_VIDEO = "polygon_and_line_zone.mp4"


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

print(
    f"Resolution: "
    f"{video_info.width} x {video_info.height}"
)

print(
    f"FPS: "
    f"{video_info.fps}"
)

print(
    f"Total frames: "
    f"{video_info.total_frames}"
)


# --------------------------------------------------
# Define PolygonZone
# --------------------------------------------------

POLYGON_LEFT = np.array([
    [
        0,
        video_info.height // 2
    ],
    [
        video_info.width // 2,
        video_info.height // 2
    ],
    [
        video_info.width // 2,
        video_info.height
    ],
    [
        0,
        video_info.height
    ],
])

polygon_zone = sv.PolygonZone(
    polygon=POLYGON_LEFT
)


# --------------------------------------------------
# Define LineZone
# --------------------------------------------------

line_start = sv.Point(
    x=0,
    y=video_info.height // 2
)

line_end = sv.Point(
    x=video_info.width,
    y=video_info.height // 2
)

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

polygon_annotator = sv.PolygonZoneAnnotator(
    zone=polygon_zone,
    color=sv.Color.RED,
    thickness=4
)

line_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_scale=1.5,
    custom_in_text="Crossings Down",
    custom_out_text="Crossings Up"
)


# --------------------------------------------------
# Processing Callback
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
    # Keep Only Confirmed Tracks
    # ----------------------------------------------

    if detections.tracker_id is not None:

        confirmed_mask = (
            detections.tracker_id != -1
        )

        detections = detections[
            confirmed_mask
        ]

    # ----------------------------------------------
    # Trigger PolygonZone
    # ----------------------------------------------

    polygon_zone.trigger(
        detections=detections
    )

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
    # Draw PolygonZone
    # ----------------------------------------------

    annotated_frame = polygon_annotator.annotate(
        scene=annotated_frame
    )

    # ----------------------------------------------
    # Draw LineZone
    # ----------------------------------------------

    annotated_frame = line_annotator.annotate(
        frame=annotated_frame,
        line_counter=line_zone
    )

    return annotated_frame


# --------------------------------------------------
# Process Video
# --------------------------------------------------

print(
    "\nProcessing PolygonZone + LineZone..."
)

sv.process_video(
    source_path=INPUT_VIDEO,
    target_path=OUTPUT_VIDEO,
    callback=callback,
    show_progress=True
)


# --------------------------------------------------
# Final Results
# --------------------------------------------------

print(
    "\nPolygonZone + LineZone processing completed."
)

print(
    f"Final polygon occupancy: "
    f"{polygon_zone.current_count}"
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
    f"Total Crossings: "
    f"{line_zone.in_count + line_zone.out_count}"
)

print(
    f"Output video: "
    f"{OUTPUT_VIDEO}"
)
