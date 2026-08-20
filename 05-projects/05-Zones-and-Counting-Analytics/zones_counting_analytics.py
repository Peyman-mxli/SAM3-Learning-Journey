"""
zones_counting_analytics.py

Zones and Counting Analytics

This project combines:

- YOLOv8 object detection
- Supervision detections
- ByteTrack object tracking
- Persistent tracker IDs
- PolygonZone occupancy analysis
- LineZone crossing analysis
- Bounding-box and label annotations
- Spatial video analytics

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

MODEL_NAME = "yolov8n.pt"

INPUT_VIDEO = "assets/input/vehicles.mp4"
OUTPUT_VIDEO = "assets/output/zones_counting_analytics.mp4"

CONFIDENCE_THRESHOLD = 0.30

# COCO vehicle classes:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

Path("assets/output").mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Check input video
# --------------------------------------------------

if not Path(INPUT_VIDEO).exists():
    raise FileNotFoundError(
        f"Input video not found: {INPUT_VIDEO}"
    )


# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

print("Loading YOLO model...")

model = YOLO(MODEL_NAME)

print("YOLO model loaded.")


# --------------------------------------------------
# Read video information
# --------------------------------------------------

video_info = sv.VideoInfo.from_video_path(
    INPUT_VIDEO
)

print("\nVideo information:")
print(f"Width: {video_info.width}")
print(f"Height: {video_info.height}")
print(f"FPS: {video_info.fps}")
print(f"Total frames: {video_info.total_frames}")


# --------------------------------------------------
# Create PolygonZone
# --------------------------------------------------

POLYGON_LEFT = np.array(
    [
        [0, video_info.height // 2],
        [
            video_info.width // 2,
            video_info.height // 2
        ],
        [
            video_info.width // 2,
            video_info.height
        ],
        [0, video_info.height]
    ],
    dtype=np.int32
)

polygon_zone = sv.PolygonZone(
    polygon=POLYGON_LEFT
)

polygon_zone_annotator = sv.PolygonZoneAnnotator(
    zone=polygon_zone,
    color=sv.Color.RED,
    thickness=4,
    text_scale=1.5
)


# --------------------------------------------------
# Create LineZone
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

line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_scale=1.5,
    custom_in_text="Crossings Down",
    custom_out_text="Crossings Up"
)


# --------------------------------------------------
# Create tracker
# --------------------------------------------------

tracker = sv.ByteTrack(
    frame_rate=video_info.fps
)


# --------------------------------------------------
# Create annotators
# --------------------------------------------------

box_annotator = sv.BoxAnnotator(
    thickness=3
)

label_annotator = sv.LabelAnnotator(
    text_scale=0.7,
    text_thickness=2
)


# --------------------------------------------------
# Frame callback
# --------------------------------------------------

def process_frame(
    frame: np.ndarray,
    index: int
) -> np.ndarray:
    """
    Process one video frame.

    Pipeline:

    YOLO
      ↓
    sv.Detections
      ↓
    Vehicle Filtering
      ↓
    ByteTrack
      ↓
    PolygonZone
      ↓
    LineZone
      ↓
    Annotations
    """

    # ----------------------------------------------
    # YOLO inference
    # ----------------------------------------------

    result = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )[0]


    # ----------------------------------------------
    # Convert YOLO results to Supervision
    # ----------------------------------------------

    detections = sv.Detections.from_ultralytics(
        result
    )


    # ----------------------------------------------
    # Keep vehicle classes
    # ----------------------------------------------

    if len(detections) > 0:

        vehicle_mask = np.isin(
            detections.class_id,
            VEHICLE_CLASS_IDS
        )

        detections = detections[
            vehicle_mask
        ]


    # ----------------------------------------------
    # Track detections
    # ----------------------------------------------

    detections = tracker.update_with_detections(
        detections
    )


    # ----------------------------------------------
    # Trigger spatial analytics
    # ----------------------------------------------

    polygon_zone.trigger(
        detections=detections
    )

    line_zone.trigger(
        detections=detections
    )


    # ----------------------------------------------
    # Create labels
    # ----------------------------------------------

    labels = []

    for class_id, tracker_id, confidence in zip(
        detections.class_id,
        detections.tracker_id,
        detections.confidence
    ):

        class_name = result.names[
            int(class_id)
        ]

        if tracker_id is None:
            tracker_text = "?"
        else:
            tracker_text = str(
                int(tracker_id)
            )

        labels.append(
            f"{class_name} "
            f"#{tracker_text} "
            f"{confidence:.2f}"
        )


    # ----------------------------------------------
    # Copy original frame
    # ----------------------------------------------

    annotated_frame = frame.copy()


    # ----------------------------------------------
    # Bounding boxes
    # ----------------------------------------------

    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )


    # ----------------------------------------------
    # Tracker labels
    # ----------------------------------------------

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )


    # ----------------------------------------------
    # PolygonZone annotation
    # ----------------------------------------------

    annotated_frame = (
        polygon_zone_annotator.annotate(
            scene=annotated_frame
        )
    )


    # ----------------------------------------------
    # LineZone annotation
    # ----------------------------------------------

    annotated_frame = (
        line_zone_annotator.annotate(
            frame=annotated_frame,
            line_counter=line_zone
        )
    )


    # ----------------------------------------------
    # Frame information
    # ----------------------------------------------

    cv2.putText(
        annotated_frame,
        f"Frame: {index}",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )


    return annotated_frame


# --------------------------------------------------
# Process complete video
# --------------------------------------------------

print("\nStarting video processing...")

sv.process_video(
    source_path=INPUT_VIDEO,
    target_path=OUTPUT_VIDEO,
    callback=process_frame,
    show_progress=True
)


# --------------------------------------------------
# Final analytics
# --------------------------------------------------

crossings_down = line_zone.in_count
crossings_up = line_zone.out_count

total_crossings = (
    crossings_down
    +
    crossings_up
)

print("\nProcessing completed.")
print(f"Saved: {OUTPUT_VIDEO}")

print(
    "Final polygon occupancy:",
    polygon_zone.current_count
)

print(
    "Crossings Down:",
    crossings_down
)

print(
    "Crossings Up:",
    crossings_up
)

print(
    "Total Crossings:",
    total_crossings
)
