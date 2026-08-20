"""
zones_counting_analytics.py

People Zones and Counting Analytics

This project combines:

- YOLOv8 person detection
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

INPUT_VIDEO = "assets/input/people_walking.mp4"
OUTPUT_VIDEO = "assets/output/people_zones_counting.mp4"

CONFIDENCE_THRESHOLD = 0.30

# COCO class 0 = person
PERSON_CLASS_ID = 0


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
# Define PolygonZone
#
# Lower half of the image.
# Coordinates are calculated dynamically so the
# project works with different video resolutions.
# --------------------------------------------------

POLYGON = np.array(
    [
        [0, video_info.height // 2],
        [video_info.width, video_info.height // 2],
        [video_info.width, video_info.height],
        [0, video_info.height],
    ],
    dtype=np.int32
)


# --------------------------------------------------
# Create PolygonZone
# --------------------------------------------------

polygon_zone = sv.PolygonZone(
    polygon=POLYGON
)

polygon_zone_annotator = sv.PolygonZoneAnnotator(
    zone=polygon_zone,
    color=sv.Color.RED,
    thickness=4,
    text_scale=1.5
)


# --------------------------------------------------
# Define horizontal LineZone
#
# The line crosses the middle of the frame.
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

line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_scale=1.5,
    custom_in_text="Crossings Down",
    custom_out_text="Crossings Up"
)


# --------------------------------------------------
# Create ByteTrack
# --------------------------------------------------

tracker = sv.ByteTrack(
    frame_rate=video_info.fps
)


# --------------------------------------------------
# Create object annotators
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

    # ----------------------------------------------
    # YOLO inference
    # ----------------------------------------------

    result = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        classes=[PERSON_CLASS_ID],
        verbose=False
    )[0]


    # ----------------------------------------------
    # Convert YOLO results
    # ----------------------------------------------

    detections = sv.Detections.from_ultralytics(
        result
    )


    # ----------------------------------------------
    # ByteTrack
    # ----------------------------------------------

    detections = tracker.update_with_detections(
        detections
    )


    # ----------------------------------------------
    # Remove detections without confirmed tracker ID
    # ----------------------------------------------

    if (
        len(detections) > 0
        and detections.tracker_id is not None
    ):

        confirmed_mask = (
            detections.tracker_id >= 0
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
    # Create tracker labels
    # ----------------------------------------------

    labels = []

    if (
        len(detections) > 0
        and detections.tracker_id is not None
    ):

        for tracker_id, confidence in zip(
            detections.tracker_id,
            detections.confidence
        ):

            labels.append(
                f"Person #{int(tracker_id)} "
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
    # Tracker ID labels
    # ----------------------------------------------

    if len(labels) == len(detections):

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
    # Additional analytics information
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

    cv2.putText(
        annotated_frame,
        f"People in Zone: {polygon_zone.current_count}",
        (40, 115),
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

print("\nStarting people analytics...")

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


print("\nPeople analytics completed.")

print(
    f"Saved: {OUTPUT_VIDEO}"
)

print(
    f"Final people in polygon: "
    f"{polygon_zone.current_count}"
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
