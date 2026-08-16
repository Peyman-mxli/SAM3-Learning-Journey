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

MIN_AREA = 5000


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
# Inspect bounding-box areas
# --------------------------------------------------

areas = detections.area

print()

print(
    f"Minimum area: "
    f"{areas.min():.0f} px²"
)

print(
    f"Maximum area: "
    f"{areas.max():.0f} px²"
)

print(
    f"Average area: "
    f"{areas.mean():.0f} px²"
)


# --------------------------------------------------
# Size filtering
# --------------------------------------------------

large_detections = detections[
    detections.area
    > MIN_AREA
]


# --------------------------------------------------
# Show filtered result
# --------------------------------------------------

print()

print(
    f"Minimum required area: "
    f"{MIN_AREA} px²"
)

print(
    f"Detections after size filtering: "
    f"{len(large_detections)}"
)


# --------------------------------------------------
# Print remaining detections
# --------------------------------------------------

print()

print(
    "Remaining detections:"
)

for class_id, confidence, area in zip(
    large_detections.class_id,
    large_detections.confidence,
    large_detections.area
):
    print(
        f"- {results.names[int(class_id)]} | "
        f"confidence={confidence:.1%} | "
        f"area={area:.0f} px²"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()

print(
    "Size filtering complete."
)
