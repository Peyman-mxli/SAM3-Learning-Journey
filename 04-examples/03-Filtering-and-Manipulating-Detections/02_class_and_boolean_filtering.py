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

TARGET_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.60


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
# Show detected classes
# --------------------------------------------------

print(
    f"Initial detections: "
    f"{len(detections)}"
)

print()

print(
    "Detected classes:"
)

for class_id in sorted(
    set(detections.class_id)
):
    count = (
        detections.class_id
        == class_id
    ).sum()

    print(
        f"- Class {class_id} "
        f"({results.names[int(class_id)]}): "
        f"{count}"
    )


# --------------------------------------------------
# Class filtering
# --------------------------------------------------

target_detections = detections[
    detections.class_id
    == TARGET_CLASS_ID
]

print()

print(
    f"Class {TARGET_CLASS_ID} "
    f"({results.names[TARGET_CLASS_ID]}): "
    f"{len(target_detections)} detections"
)


# --------------------------------------------------
# Combine class + confidence conditions
# --------------------------------------------------

filtered_detections = detections[
    (
        detections.class_id
        == TARGET_CLASS_ID
    )
    & (
        detections.confidence
        > CONFIDENCE_THRESHOLD
    )
]


# --------------------------------------------------
# Show combined filtering result
# --------------------------------------------------

print()

print(
    f"{results.names[TARGET_CLASS_ID]} "
    f"detections with confidence > "
    f"{CONFIDENCE_THRESHOLD:.0%}: "
    f"{len(filtered_detections)}"
)


# --------------------------------------------------
# Excluding a class
# --------------------------------------------------

without_target_class = detections[
    detections.class_id
    != TARGET_CLASS_ID
]

print(
    f"Detections excluding "
    f"{results.names[TARGET_CLASS_ID]}: "
    f"{len(without_target_class)}"
)


# --------------------------------------------------
# Print filtered detections
# --------------------------------------------------

print()

print(
    "Filtered detections:"
)

for class_id, confidence in zip(
    filtered_detections.class_id,
    filtered_detections.confidence
):
    print(
        f"- {results.names[int(class_id)]}: "
        f"{confidence:.1%}"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()

print(
    "Class and Boolean filtering complete."
)
