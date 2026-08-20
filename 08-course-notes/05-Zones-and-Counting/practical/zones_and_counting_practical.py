
"""
zones_and_counting_practical.py

SAM3 Learning Journey
Session 06 — Zones and Counting

Practical implementation combining:

- YOLOv8 object detection
- Supervision detections and annotations
- ByteTrack object tracking
- PolygonZone occupancy counting
- LineZone directional crossing counting
- OpenCV video processing

Input:
    assets/input/vehicles.mp4

Output:
    assets/output/vehicles_combined.mp4
"""

from pathlib import Path

import numpy as np
import supervision as sv

from ultralytics import YOLO
from trackers import ByteTrackTracker


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "yolov8n.pt"

INPUT_VIDEO = "assets/input/vehicles.mp4"

OUTPUT_VIDEO = "assets/output/vehicles_combined.mp4"


# ============================================================
# Create Required Directories
# ============================================================

Path("assets/input").mkdir(
    parents=True,
    exist_ok=True
)

Path("assets/output").mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Validate Input Video
# ============================================================

if not Path(INPUT_VIDEO).exists():
    raise FileNotFoundError(
        f"Input video not found: {INPUT_VIDEO}"
    )


# ============================================================
# Load YOLO Model
# ============================================================

print(f"Loading model: {MODEL_NAME}")

model = YOLO(
    MODEL_NAME
)


# ============================================================
# Read Video Information
# ============================================================

video_info = sv.VideoInfo.from_video_path(
    INPUT_VIDEO
)

print(
    f"Resolution: "
    f"{video_info.width} x {video_info.height}"
)

print(
    f"FPS: {video_info.fps}"
)

print(
    f"Total frames: {video_info.total_frames}"
)


# ============================================================
# Define PolygonZone
# ============================================================

# Lower-left half of the frame
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


# ============================================================
# Define LineZone
# ============================================================

# Horizontal line across the middle of the video
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


# ============================================================
# Create Tracker
# ============================================================

tracker = ByteTrackTracker()


# ============================================================
# Create Annotators
# ============================================================

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


# ============================================================
# Video Processing Callback
# ============================================================

def callback(
    frame: np.ndarray,
    frame_index: int
) -> np.ndarray:
    """
    Process one frame of the input video.

    Pipeline:

        Frame
          ↓
        YOLO
          ↓
        Supervision Detections
          ↓
        ByteTrack
          ↓
        Confirmed Tracks
          ↓
        PolygonZone + LineZone
          ↓
        Annotation
          ↓
        Output Frame
    """

    # --------------------------------------------------------
    # Object Detection
    # --------------------------------------------------------

    results = model(
        frame,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    # --------------------------------------------------------
    # Object Tracking
    # --------------------------------------------------------

    detections = tracker.update(
        detections
    )

    # --------------------------------------------------------
    # Keep Only Confirmed Tracker IDs
    # --------------------------------------------------------

    if detections.tracker_id is not None:

        confirmed_mask = (
            detections.tracker_id != -1
        )

        detections = detections[
            confirmed_mask
        ]

    # --------------------------------------------------------
    # PolygonZone Occupancy
    # --------------------------------------------------------

    polygon_zone.trigger(
        detections=detections
    )

    # polygon_zone.current_count now contains the number
    # of tracked objects currently inside the polygon.

    # --------------------------------------------------------
    # LineZone Crossings
    # --------------------------------------------------------

    line_zone.trigger(
        detections=detections
    )

    # line_zone.in_count and line_zone.out_count contain
    # accumulated directional crossing counts.

    # --------------------------------------------------------
    # Create Tracker ID Labels
    # --------------------------------------------------------

    if detections.tracker_id is not None:

        labels = [
            f"ID:{tracker_id}"
            for tracker_id
            in detections.tracker_id
        ]

    else:

        labels = []

    # --------------------------------------------------------
    # Draw Bounding Boxes
    # --------------------------------------------------------

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    # --------------------------------------------------------
    # Draw Tracker IDs
    # --------------------------------------------------------

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # --------------------------------------------------------
    # Draw PolygonZone
    # --------------------------------------------------------

    annotated_frame = polygon_annotator.annotate(
        scene=annotated_frame
    )

    # --------------------------------------------------------
    # Draw LineZone
    # --------------------------------------------------------

    annotated_frame = line_annotator.annotate(
        frame=annotated_frame,
        line_counter=line_zone
    )

    return annotated_frame


# ============================================================
# Process Video
# ============================================================

print("\nStarting Zones and Counting processing...\n")

sv.process_video(
    source_path=INPUT_VIDEO,
    target_path=OUTPUT_VIDEO,
    callback=callback,
    show_progress=True
)


# ============================================================
# Final Results
# ============================================================

print("\nProcessing completed successfully.")

print(
    f"Output video: "
    f"{OUTPUT_VIDEO}"
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
