"""
04-confidence-threshold.py

Introduction to Supervision — Example 04

Goal:
Compare YOLO detections using the default confidence
threshold and a stricter confidence threshold.
"""

from pathlib import Path
import urllib.request

import cv2
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# 1. Prepare the example image
# --------------------------------------------------

Path("assets").mkdir(exist_ok=True)

image_path = "assets/bus.jpg"

if not Path(image_path).exists():
    urllib.request.urlretrieve(
        "https://ultralytics.com/images/bus.jpg",
        image_path
    )


# --------------------------------------------------
# 2. Load the image
# --------------------------------------------------

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )


# --------------------------------------------------
# 3. Load YOLO
# --------------------------------------------------

model = YOLO("yolov8n.pt")


# --------------------------------------------------
# 4. Run detection with default confidence
# --------------------------------------------------

results_default = model(image)[0]

detections_default = (
    sv.Detections.from_ultralytics(
        results_default
    )
)


# --------------------------------------------------
# 5. Run detection with confidence = 0.8
# --------------------------------------------------

results_strict = model(
    image,
    conf=0.8
)[0]

detections_strict = (
    sv.Detections.from_ultralytics(
        results_strict
    )
)


# --------------------------------------------------
# 6. Compare results
# --------------------------------------------------

print("\n--- Confidence Threshold Comparison ---")

print(
    f"Default confidence: "
    f"{len(detections_default)} objects"
)

print(
    f"Confidence 0.8: "
    f"{len(detections_strict)} objects"
)


# --------------------------------------------------
# 7. Inspect confidence scores
# --------------------------------------------------

print("\nDefault confidence scores:")
print(detections_default.confidence)

print("\nStrict confidence scores:")
print(detections_strict.confidence)


# --------------------------------------------------
# Explanation
# --------------------------------------------------

"""
Lower confidence threshold
        ↓
More detections
        ↓
More uncertain predictions may remain


Higher confidence threshold
        ↓
Fewer detections
        ↓
Only more confident predictions remain


Important:

A higher confidence threshold does NOT automatically
mean that the model is better.

The appropriate threshold depends on the requirements
of the computer vision application.
"""
