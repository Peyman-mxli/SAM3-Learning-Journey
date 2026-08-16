from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "yolov8n.pt"

INPUT_IMAGE = (
    "../../05-projects/"
    "03-Detection-Filtering-and-NMS-Pipeline/"
    "assets/input/pedestrian-plaza-detection-test.png"
)

CONFIDENCE_THRESHOLD = 0.50


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(
    INPUT_IMAGE
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {INPUT_IMAGE}"
    )


# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

model = YOLO(
    MODEL_NAME
)


# --------------------------------------------------
# Run detection
# --------------------------------------------------

results = model(
    image
)[0]


# --------------------------------------------------
# Convert to Supervision detections
# --------------------------------------------------

detections = sv.Detections.from_ultralytics(
    results
)


# --------------------------------------------------
# Show original detection count
# --------------------------------------------------

print(
    f"Initial detections: "
    f"{len(detections)}"
)


# --------------------------------------------------
# Confidence filtering
# --------------------------------------------------

high_confidence = detections[
    detections.confidence
    > CONFIDENCE_THRESHOLD
]


# --------------------------------------------------
# Show filtered count
# --------------------------------------------------

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD:.0%}"
)

print(
    f"Detections after filtering: "
    f"{len(high_confidence)}"
)


# --------------------------------------------------
# Print confidence values
# --------------------------------------------------

print()

print(
    "Remaining confidence scores:"
)

for confidence in high_confidence.confidence:
    print(
        f"- {confidence:.1%}"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()

print(
    "Confidence filtering complete."
)
