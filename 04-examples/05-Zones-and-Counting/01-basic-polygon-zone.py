"""
01-basic-polygon-zone.py

Basic PolygonZone Example

This example demonstrates how to create and visualize
a polygonal region using Supervision.

Concepts:
- NumPy polygon coordinates
- Supervision PolygonZone
- PolygonZoneAnnotator
- Video dimensions
"""

from pathlib import Path

import cv2
import numpy as np
import supervision as sv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_VIDEO = "assets/input/vehicles.mp4"


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
# Define Polygon
# --------------------------------------------------

# Lower-left half of the video frame

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
# Create Polygon Annotator
# --------------------------------------------------

zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.RED,
    thickness=4
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
# Draw Polygon
# --------------------------------------------------

annotated_frame = zone_annotator.annotate(
    scene=frame.copy()
)


# --------------------------------------------------
# Save Preview
# --------------------------------------------------

OUTPUT_IMAGE = "polygon_zone_preview.jpg"

cv2.imwrite(
    OUTPUT_IMAGE,
    annotated_frame
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nPolygonZone created successfully.")

print("\nPolygon coordinates:")

print(
    POLYGON_LEFT
)

print(
    f"\nPreview saved to: "
    f"{OUTPUT_IMAGE}"
)
