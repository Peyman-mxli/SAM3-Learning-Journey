"""
03-filter-detections-inside-zone.py

Filter Detections Inside PolygonZone

This example demonstrates how to:

- Detect objects with YOLOv8
- Convert YOLO results to Supervision detections
- Trigger a PolygonZone
- Use the returned Boolean mask
- Keep only detections inside the polygon
- Annotate the filtered detections

Concepts:
- PolygonZone.trigger()
- Boolean masks
- Spatial filtering
- Supervision Detections filtering
- Bounding-box annotation
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

OUTPUT_IMAGE = "filtered_zone_detections.jpg"


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
# Define Polygon
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


# --------------------------------------------------
# Create PolygonZone
# --------------------------------------------------

zone = sv.PolygonZone(
    polygon=POLYGON_LEFT
)


# --------------------------------------------------
# Create Annotators
# --------------------------------------------------

box_annotator = sv.BoxAnnotator()

zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.RED,
    thickness=4
)


# --------------------------------------------------
# Read First Frame
# --------------------------------------------------

capture = cv2.VideoCapture(
    INPUT_VIDEO
)

success, frame = capture.read()

capture.release()

if not success:
    raise RuntimeError(
        "Could not read the first video frame."
    )


# --------------------------------------------------
# Detect Objects
# --------------------------------------------------

results = model(
    frame,
    verbose=False
)[0]


# --------------------------------------------------
# Convert to Supervision Detections
# --------------------------------------------------

detections = sv.Detections.from_ultralytics(
    results
)


# --------------------------------------------------
# Trigger PolygonZone
# --------------------------------------------------

zone_mask = zone.trigger(
    detections=detections
)


# --------------------------------------------------
# Filter Detections
# --------------------------------------------------

detections_inside_zone = detections[
    zone_mask
]


# --------------------------------------------------
# Annotate Filtered Detections
# --------------------------------------------------

annotated_frame = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections_inside_zone
)

annotated_frame = zone_annotator.annotate(
    scene=annotated_frame
)


# --------------------------------------------------
# Save Result
# --------------------------------------------------

cv2.imwrite(
    OUTPUT_IMAGE,
    annotated_frame
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print(
    f"Total detected objects: "
    f"{len(detections)}"
)

print(
    f"Objects inside zone: "
    f"{len(detections_inside_zone)}"
)

print(
    f"zone.current_count: "
    f"{zone.current_count}"
)

print(
    f"Saved: "
    f"{OUTPUT_IMAGE}"
)
