"""
04-tracking-with-polygon-zone.py

Tracking with PolygonZone

This example demonstrates how to:

- Detect objects with YOLOv8
- Track objects with ByteTrack
- Keep only confirmed tracker IDs
- Trigger a PolygonZone
- Filter tracked objects inside the polygon
- Display tracker IDs
- Save an annotated preview frame

Concepts:
- YOLOv8 detection
- ByteTrack tracking
- Persistent tracker IDs
- Confirmed track filtering
- PolygonZone.trigger()
- Spatial filtering
- Tracker ID labels
"""

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO
from trackers import ByteTrackTracker


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "yolov8n.pt"

INPUT_VIDEO = "assets/input/vehicles.mp4"

OUTPUT_IMAGE = "tracking_polygon_zone.jpg"

FRAMES_TO_PROCESS = 20


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
# Create Tracker
# --------------------------------------------------

tracker = ByteTrackTracker()


# --------------------------------------------------
# Create Annotators
# --------------------------------------------------

box_annotator = sv.BoxAnnotator()

label_annotator = sv.LabelAnnotator()

zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.RED,
    thickness=4
)


# --------------------------------------------------
# Open Video
# --------------------------------------------------

capture = cv2.VideoCapture(
    INPUT_VIDEO
)

tracked_frame = None
tracked_detections = None


# --------------------------------------------------
# Process Multiple Frames
# --------------------------------------------------

for frame_index in range(
    FRAMES_TO_PROCESS
):

    success, frame = capture.read()

    if not success:
        break

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

    tracked_frame = frame
    tracked_detections = detections


capture.release()


# --------------------------------------------------
# Validate Tracking Result
# --------------------------------------------------

if tracked_frame is None:
    raise RuntimeError(
        "No video frames were processed."
    )

if tracked_detections is None:
    raise RuntimeError(
        "No tracked detections were produced."
    )


# --------------------------------------------------
# Trigger PolygonZone
# --------------------------------------------------

zone_mask = zone.trigger(
    detections=tracked_detections
)


# --------------------------------------------------
# Filter Tracked Objects Inside Zone
# --------------------------------------------------

detections_inside_zone = tracked_detections[
    zone_mask
]


# --------------------------------------------------
# Create Tracker ID Labels
# --------------------------------------------------

if detections_inside_zone.tracker_id is not None:

    labels = [
        f"ID:{tracker_id}"
        for tracker_id
        in detections_inside_zone.tracker_id
    ]

else:

    labels = []


# --------------------------------------------------
# Annotate Bounding Boxes
# --------------------------------------------------

annotated_frame = box_annotator.annotate(
    scene=tracked_frame.copy(),
    detections=detections_inside_zone
)


# --------------------------------------------------
# Annotate Tracker IDs
# --------------------------------------------------

annotated_frame = label_annotator.annotate(
    scene=annotated_frame,
    detections=detections_inside_zone,
    labels=labels
)


# --------------------------------------------------
# Annotate PolygonZone
# --------------------------------------------------

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
    f"Confirmed tracked objects: "
    f"{len(tracked_detections)}"
)

print(
    f"Tracked objects inside zone: "
    f"{len(detections_inside_zone)}"
)

print(
    f"zone.current_count: "
    f"{zone.current_count}"
)

print(
    "Tracker IDs inside zone:"
)

print(
    detections_inside_zone.tracker_id
)

print(
    f"Saved: "
    f"{OUTPUT_IMAGE}"
)
