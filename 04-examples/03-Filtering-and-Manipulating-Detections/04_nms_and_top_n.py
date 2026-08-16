from pathlib import Path

import cv2
import numpy as np
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

LOW_CONFIDENCE = 0.30
HIGH_CONFIDENCE = 0.70
NMS_THRESHOLD = 0.50
TOP_N = 3


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
# Run detection with two confidence thresholds
# --------------------------------------------------

results_low = model(
    image,
    conf=LOW_CONFIDENCE
)[0]

results_high = model(
    image,
    conf=HIGH_CONFIDENCE
)[0]


# --------------------------------------------------
# Convert to Supervision detections
# --------------------------------------------------

detections_low = sv.Detections.from_ultralytics(
    results_low
)

detections_high = sv.Detections.from_ultralytics(
    results_high
)


# --------------------------------------------------
# Show original counts
# --------------------------------------------------

print(
    f"Low-confidence detections "
    f"({LOW_CONFIDENCE:.0%}): "
    f"{len(detections_low)}"
)

print(
    f"High-confidence detections "
    f"({HIGH_CONFIDENCE:.0%}): "
    f"{len(detections_high)}"
)


# --------------------------------------------------
# Merge detections
# --------------------------------------------------

merged = sv.Detections.merge([
    detections_low,
    detections_high
])

print()

print(
    f"Detections after merge: "
    f"{len(merged)}"
)


# --------------------------------------------------
# Apply Non-Maximum Suppression
# --------------------------------------------------

after_nms = merged.with_nms(
    threshold=NMS_THRESHOLD
)

print(
    f"Detections after NMS "
    f"(threshold={NMS_THRESHOLD}): "
    f"{len(after_nms)}"
)


# --------------------------------------------------
# Select Top-N detections
# --------------------------------------------------

if (
    len(after_nms) > 0
    and after_nms.confidence is not None
):

    indices_top = np.argsort(
        after_nms.confidence
    )[::-1][:TOP_N]

    top_detections = after_nms[
        indices_top
    ]

else:
    top_detections = after_nms


# --------------------------------------------------
# Show Top-N result
# --------------------------------------------------

print()

print(
    f"Top-{TOP_N} detections:"
)

for class_id, confidence in zip(
    top_detections.class_id,
    top_detections.confidence
):
    print(
        f"- {results_low.names[int(class_id)]}: "
        f"{confidence:.1%}"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()

print(
    "NMS and Top-N selection complete."
)
