```python
"""
zones_counting_analytics.py

People Zones and Counting Analytics

Final tested Project 05 implementation.

This project combines:

- YOLOv8s person detection
- Supervision detections
- ByteTrack object tracking
- Persistent tracker IDs
- PolygonZone occupancy analysis
- LineZone crossing analysis
- Bounding-box annotations
- Tracker ID labels
- Spatial video analytics
- H.264-ready output workflow

SAM3 Learning Journey
"""

from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "yolov8s.pt"

INPUT_VIDEO = "assets/input/people_walking.mp4"

OUTPUT_VIDEO = (
    "assets/output/people_zones_counting_final.mp4"
)

CONFIDENCE_THRESHOLD = 0.15
PERSON_CLASS_ID = 0
INFERENCE_SIZE = 1280


# --------------------------------------------------
# Create Required Directories
# --------------------------------------------------

Path("assets/output").mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Validate Input Video
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

print("\nVideo Information")

print(
    f"Resolution: "
    f"{video_info.width} x "
    f"{video_info.height}"
)

print(
    f"FPS: "
    f"{video_info.fps}"
)

print(
    f"Frames: "
    f"{video_info.total_frames}"
)


# --------------------------------------------------
# Define PolygonZone
#
# Polygon focused on the main pedestrian-crossing area.
# These coordinates were visually inspected and tested
# using the 1920 x 1080 people_walking.mp4 video.
# --------------------------------------------------

POLYGON = np.array([
    [520, 470],
    [1320, 470],
    [1460, 900],
    [420, 900]
], dtype=np.int32)


# --------------------------------------------------
# Create PolygonZone
# --------------------------------------------------

polygon_zone = sv.PolygonZone(
    polygon=POLYGON
)

polygon_annotator = sv.PolygonZoneAnnotator(
    zone=polygon_zone,
    color=sv.Color.RED,
    thickness=4,
    text_scale=1.2
)


# --------------------------------------------------
# Define LineZone
#
# Vertical line through the central pedestrian flow.
# This orientation produced a confirmed crossing event
# during testing on the crowded pedestrian scene.
# --------------------------------------------------

line_start = sv.Point(
    x=960,
    y=400
)

line_end = sv.Point(
    x=960,
    y=920
)


# --------------------------------------------------
# Create LineZone
# --------------------------------------------------

line_zone = sv.LineZone(
    start=line_start,
    end=line_end
)

line_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_scale=1.2,
    custom_in_text="Crossings In",
    custom_out_text="Crossings Out"
)


# --------------------------------------------------
# Create ByteTrack
#
# The tracker configuration was adjusted for the
# crowded pedestrian scene.
# --------------------------------------------------

tracker = sv.ByteTrack(
    track_activation_threshold=0.15,
    lost_track_buffer=90,
    minimum_matching_threshold=0.70,
    frame_rate=video_info.fps
)


# --------------------------------------------------
# Create Annotators
# --------------------------------------------------

box_annotator = sv.BoxAnnotator(
    thickness=3
)

label_annotator = sv.LabelAnnotator(
    text_scale=0.6,
    text_thickness=2
)


# --------------------------------------------------
# Frame Processing Callback
# --------------------------------------------------

def process_frame(
    frame: np.ndarray,
    frame_index: int
) -> np.ndarray:
    """
    Process one video frame.

    Pipeline:

        Frame
          ↓
        YOLOv8s
          ↓
        Person Detections
          ↓
        Supervision Detections
          ↓
        ByteTrack
          ↓
        Persistent Tracker IDs
          ↓
        PolygonZone
          +
        LineZone
          ↓
        Occupancy + Crossing Analytics
          ↓
        Annotation
          ↓
        Output Frame
    """

    # --------------------------------------------------
    # Person Detection
    # --------------------------------------------------

    result = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        classes=[PERSON_CLASS_ID],
        imgsz=INFERENCE_SIZE,
        verbose=False
    )[0]


    # --------------------------------------------------
    # Convert YOLO Results to Supervision
    # --------------------------------------------------

    detections = sv.Detections.from_ultralytics(
        result
    )


    # --------------------------------------------------
    # Object Tracking
    # --------------------------------------------------

    detections = tracker.update_with_detections(
        detections
    )


    # --------------------------------------------------
    # Trigger PolygonZone
    # --------------------------------------------------

    polygon_zone.trigger(
        detections=detections
    )


    # --------------------------------------------------
    # Trigger LineZone
    # --------------------------------------------------

    line_zone.trigger(
        detections=detections
    )


    # --------------------------------------------------
    # Create Tracker ID Labels
    # --------------------------------------------------

    labels = []

    if (
        len(detections) > 0
        and detections.tracker_id is not None
    ):

        labels = [
            f"Person #{int(tracker_id)}"
            for tracker_id
            in detections.tracker_id
        ]


    # --------------------------------------------------
    # Copy Original Frame
    # --------------------------------------------------

    annotated_frame = frame.copy()


    # --------------------------------------------------
    # Draw Bounding Boxes
    # --------------------------------------------------

    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )


    # --------------------------------------------------
    # Draw Tracker ID Labels
    # --------------------------------------------------

    if len(labels) == len(detections):

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )


    # --------------------------------------------------
    # Draw PolygonZone
    # --------------------------------------------------

    annotated_frame = polygon_annotator.annotate(
        scene=annotated_frame
    )


    # --------------------------------------------------
    # Draw LineZone
    # --------------------------------------------------

    annotated_frame = line_annotator.annotate(
        frame=annotated_frame,
        line_counter=line_zone
    )


    # --------------------------------------------------
    # Additional Analytics Information
    # --------------------------------------------------

    cv2.putText(
        annotated_frame,
        (
            "People in Zone: "
            f"{polygon_zone.current_count}"
        ),
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        annotated_frame,
        f"Frame: {frame_index}",
        (40, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    return annotated_frame


# --------------------------------------------------
# Process Complete Video
# --------------------------------------------------

print(
    "\nStarting final Project 05 processing...\n"
)

sv.process_video(
    source_path=INPUT_VIDEO,
    target_path=OUTPUT_VIDEO,
    callback=process_frame,
    show_progress=True
)


# --------------------------------------------------
# Final Analytics
# --------------------------------------------------

crossings_in = (
    line_zone.in_count
)

crossings_out = (
    line_zone.out_count
)

total_crossings = (
    crossings_in
    +
    crossings_out
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print(
    "\nProject 05 processing completed successfully."
)

print(
    f"Saved: "
    f"{OUTPUT_VIDEO}"
)

print(
    f"Final people in PolygonZone: "
    f"{polygon_zone.current_count}"
)

print(
    f"Crossings In: "
    f"{crossings_in}"
)

print(
    f"Crossings Out: "
    f"{crossings_out}"
)

print(
    f"Total Crossings: "
    f"{total_crossings}"
)


# --------------------------------------------------
# Tested Result
#
# Tested in Google Colab with:
#
# Resolution: 1920 x 1080
# FPS: 50
# Frames: 763
#
# Final people in PolygonZone: 6
# Crossings In: 0
# Crossings Out: 1
# Total Crossings: 1
# --------------------------------------------------
```
