"""
05-basic-line-zone.py

Basic LineZone Example

This example demonstrates how to create and visualize
a virtual counting line using Supervision.

Concepts:
- Supervision Point
- Line start and end coordinates
- LineZone
- LineZoneAnnotator
- Virtual counting boundaries
"""

from pathlib import Path

import cv2
import supervision as sv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_VIDEO = "assets/input/vehicles.mp4"

OUTPUT_IMAGE = "line_zone_preview.jpg"


# --------------------------------------------------
# Validate Input
# --------------------------------------------------

if not Path(INPUT_VIDEO).exists():
    raise FileNotFoundError(
        f"Input video not found: {INPUT_VIDEO}"
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


# --------------------------------------------------
# Define Counting Line
# --------------------------------------------------

# Horizontal line across the middle of the frame

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
# Create LineZone Annotator
# --------------------------------------------------

line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_scale=1.5,
    custom_in_text="Crossings Down",
    custom_out_text="Crossings Up"
)


# --------------------------------------------------
# Read First Video Frame
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
# Draw LineZone
# --------------------------------------------------

annotated_frame = line_zone_annotator.annotate(
    frame=frame.copy(),
    line_counter=line_zone
)


# --------------------------------------------------
# Save Preview
# --------------------------------------------------

cv2.imwrite(
    OUTPUT_IMAGE,
    annotated_frame
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nLineZone created successfully.")

print(
    f"Start: "
    f"({line_start.x}, {line_start.y})"
)

print(
    f"End: "
    f"({line_end.x}, {line_end.y})"
)

print(
    f"Initial in_count: "
    f"{line_zone.in_count}"
)

print(
    f"Initial out_count: "
    f"{line_zone.out_count}"
)

print(
    f"Preview saved to: "
    f"{OUTPUT_IMAGE}"
)
