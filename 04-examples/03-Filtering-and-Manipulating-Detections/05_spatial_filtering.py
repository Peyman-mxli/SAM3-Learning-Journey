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
# Calculate horizontal center of each box
# --------------------------------------------------

centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2


# --------------------------------------------------
# Calculate image midpoint
# --------------------------------------------------

image_midpoint = (
    image.shape[1] / 2
)

print(
    f"Image width: "
    f"{image.shape[1]} px"
)

print(
    f"Horizontal midpoint: "
    f"{image_midpoint:.0f} px"
)


# --------------------------------------------------
# Create spatial mask
# --------------------------------------------------

right_side_mask = (
    centers_x
    > image_midpoint
)


# --------------------------------------------------
# Filter detections
# --------------------------------------------------

right_side_detections = detections[
    right_side_mask
]


# --------------------------------------------------
# Show filtered result
# --------------------------------------------------

print()

print(
    f"Detections in right half: "
    f"{len(right_side_detections)}"
)


# --------------------------------------------------
# Print remaining detections
# --------------------------------------------------

print()

print(
    "Right-half detections:"
)

for class_id, confidence, center_x in zip(
    right_side_detections.class_id,
    right_side_detections.confidence,
    centers_x[right_side_mask]
):
    print(
        f"- {results.names[int(class_id)]} | "
        f"confidence={confidence:.1%} | "
        f"center_x={center_x:.0f}"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()

print(
    "Spatial filtering complete."
)
