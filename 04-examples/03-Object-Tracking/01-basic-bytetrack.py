
```python
"""
01-basic-bytetrack.py

Basic Object Tracking with YOLO, Supervision, and ByteTrack

This example demonstrates the core tracking concept from the
Object Tracking lesson:

1. Load a YOLO model
2. Read one frame from a video
3. Detect objects
4. Convert YOLO results to sv.Detections
5. Inspect tracker_id before tracking
6. Apply sv.ByteTrack
7. Inspect tracker_id after tracking
"""

from pathlib import Path
import urllib.request

import cv2
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "yolov8n.pt"

VIDEO_URL = (
    "https://media.roboflow.com/"
    "supervision/video-examples/vehicles.mp4"
)

VIDEO_PATH = "assets/vehicles.mp4"


# --------------------------------------------------
# Create assets directory
# --------------------------------------------------

Path("assets").mkdir(exist_ok=True)


# --------------------------------------------------
# Download sample video if needed
# --------------------------------------------------

if not Path(VIDEO_PATH).exists():

    print("Downloading sample video...")

    urllib.request.urlretrieve(
        VIDEO_URL,
        VIDEO_PATH
    )

    print(
        f"Video saved to: {VIDEO_PATH}"
    )


# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

model = YOLO(MODEL_NAME)


# --------------------------------------------------
# Create ByteTrack tracker
# --------------------------------------------------

tracker = sv.ByteTrack()


# --------------------------------------------------
# Read the first video frame
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame = cap.read()

cap.release()


if not ret or frame is None:

    raise RuntimeError(
        f"Could not read video frame from: "
        f"{VIDEO_PATH}"
    )


# --------------------------------------------------
# Run object detection
# --------------------------------------------------

results = model(
    frame,
    verbose=False
)[0]


# --------------------------------------------------
# Convert YOLO results to Supervision
# --------------------------------------------------

detections = sv.Detections.from_ultralytics(
    results
)


# --------------------------------------------------
# Inspect detections before tracking
# --------------------------------------------------

print("\nBEFORE tracking:")

print(
    f"Detected objects: "
    f"{len(detections)}"
)

print(
    f"tracker_id: "
    f"{detections.tracker_id}"
)


# --------------------------------------------------
# Apply ByteTrack
# --------------------------------------------------

detections = tracker.update_with_detections(
    detections
)


# --------------------------------------------------
# Inspect detections after tracking
# --------------------------------------------------

print("\nAFTER tracking:")

print(
    f"Detected objects: "
    f"{len(detections)}"
)

print(
    f"tracker_id: "
    f"{detections.tracker_id}"
)


# --------------------------------------------------
# Display class and tracker information
# --------------------------------------------------

print("\nTracked objects:")

for class_id, tracker_id in zip(
    detections.class_id,
    detections.tracker_id
):

    class_name = results.names[
        int(class_id)
    ]

    print(
        f"{class_name} "
        f"-> tracker_id {tracker_id}"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print(
    "\nTracking test completed successfully."
)

print(
    "YOLO detected the objects, "
    "and ByteTrack assigned tracker IDs."
)
```
