"""
02-polygon-zone-current-count.py

PolygonZone Current Count Example

This example demonstrates how to:

- Detect objects with YOLOv8
- Convert detections to Supervision
- Trigger a PolygonZone
- Read the Boolean zone mask
- Inspect zone.current_count

Concepts:
- YOLOv8 detection
- Supervision Detections
- PolygonZone.trigger()
- Boolean masks
- Current occupancy
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
# Results
# --------------------------------------------------

print(
    f"Total detected objects: "
    f"{len(detections)}"
)

print(
    f"Zone mask: "
    f"{zone_mask}"
)

print(
    f"Objects inside zone: "
    f"{zone_mask.sum()}"
)

print(
    f"zone.current_count: "
    f"{zone.current_count}"
)
