"""
09-complete-zones-counting-pipeline.py

Complete Zones and Counting Pipeline

This example combines the full spatial video analytics workflow:

- YOLOv8 object detection
- Supervision detections
- ByteTrack object tracking
- Confirmed tracker ID filtering
- PolygonZone occupancy analysis
- LineZone crossing analysis
- Tracker ID labels
- Bounding-box annotation
- Polygon visualization
- Line visualization
- Full video processing
- Final statistics

Input:
    assets/input/vehicles.mp4

Output:
    complete_zones_counting_pipeline.mp4
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

OUTPUT_VIDEO = "complete_zones_counting_pipeline.mp4"


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

print(
    f"Loading model: "
    f"{MODEL_NAME}"
)

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
    # 1. Detect Objects
    # ----------------------------------------------

    results = model(
        frame,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    # ----------------------------------------------
    # 2. Track Objects
    # ----------------------------------------------

    detections = tracker.update(
        detections
    )

    # ----------------------------------------------
    # 3. Keep Only Confirmed Tracks
    # ----------------------------------------------

    if detections.tracker_id is not None:

        confirmed_mask = (
            detections.tracker_id != -1
        )

        detections = detections[
            confirmed_mask
        ]

    # ----------------------------------------------
    # 4. Trigger PolygonZone
    # ----------------------------------------------

    polygon_zone.trigger(
        detections=detections
    )

    # ----------------------------------------------
    # 5. Trigger LineZone
    # ----------------------------------------------

    line_zone.trigger(
        detections=detections
    )

    # ----------------------------------------------
    # 6. Create Tracker ID Labels
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
    # 7. Draw Bounding Boxes
    # ----------------------------------------------

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    # ----------------------------------------------
    # 8. Draw Tracker IDs
    # ----------------------------------------------

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # ----------------------------------------------
    # 9. Draw PolygonZone
    # ----------------------------------------------

    annotated_frame = polygon_annotator.annotate(
        scene=annotated_frame
    )

    # ----------------------------------------------
    # 10. Draw LineZone
    # ----------------------------------------------

    annotated_frame = line_annotator.annotate(
        frame=annotated_frame,
        line_counter=line_zone
    )

    return annotated_frame


# --------------------------------------------------
# Process Complete Video
# --------------------------------------------------

print(
    "\nStarting complete Zones and Counting pipeline..."
)

sv.process_video(
    source_path=INPUT_VIDEO,
    target_path=OUTPUT_VIDEO,
    callback=callback,
    show_progress=True
)


# --------------------------------------------------
# Final Statistics
# --------------------------------------------------

final_occupancy = (
    polygon_zone.current_count
)

crossings_down = (
    line_zone.in_count
)

crossings_up = (
    line_zone.out_count
)

total_crossings = (
    crossings_down
    +
    crossings_up
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print(
    "\nComplete pipeline finished successfully."
)

print(
    f"Final polygon occupancy: "
    f"{final_occupancy}"
)

print(
    f"Crossings Down: "
    f"{crossings_down}"
)

print(
    f"Crossings Up: "
    f"{crossings_up}"
)

print(
    f"Total Crossings: "
    f"{total_crossings}"
)

print(
    f"Output video: "
    f"{OUTPUT_VIDEO}"
)


# --------------------------------------------------
# Pipeline Summary
# --------------------------------------------------

print(
    "\nPipeline:"
)

print(
    "YOLOv8"
    " -> ByteTrack"
    " -> PolygonZone"
    " + LineZone"
    " -> Annotation"
    " -> Output Video"
)
