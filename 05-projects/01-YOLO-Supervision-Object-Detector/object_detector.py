"""
object_detector.py

YOLO + Supervision Object Detector

This project combines the main concepts from the
Introduction to Supervision session into one reusable
computer vision application.
"""

from pathlib import Path
import json

import cv2
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.50

INPUT_IMAGE = "input/image.jpg"

OUTPUT_IMAGE = "output/annotated_image.jpg"
OUTPUT_JSON = "output/predictions.json"


# --------------------------------------------------
# Create required directories
# --------------------------------------------------

Path("input").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(INPUT_IMAGE)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {INPUT_IMAGE}"
    )


# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

model = YOLO(MODEL_NAME)


# --------------------------------------------------
# Run object detection
# --------------------------------------------------

results = model(
    image,
    conf=CONFIDENCE_THRESHOLD
)[0]


# --------------------------------------------------
# Convert YOLO results to Supervision
# --------------------------------------------------

detections = sv.Detections.from_ultralytics(
    results
)


# --------------------------------------------------
# Create labels
# --------------------------------------------------

labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]


# --------------------------------------------------
# Create annotators
# --------------------------------------------------

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


# --------------------------------------------------
# Annotate image
# --------------------------------------------------

annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)


# --------------------------------------------------
# Save annotated image
# --------------------------------------------------

cv2.imwrite(
    OUTPUT_IMAGE,
    annotated_image
)

print(
    f"Annotated image saved to: "
    f"{OUTPUT_IMAGE}"
)


# --------------------------------------------------
# Convert detections to JSON-compatible structure
# --------------------------------------------------

predictions = []

for i in range(len(detections)):

    x1, y1, x2, y2 = detections.xyxy[i]

    class_id = int(
        detections.class_id[i]
    )

    confidence = float(
        detections.confidence[i]
    )

    prediction = {
        "class_id": class_id,
        "class_name": results.names[class_id],
        "confidence": confidence,
        "bounding_box": {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2)
        }
    }

    predictions.append(prediction)


# --------------------------------------------------
# Create final JSON structure
# --------------------------------------------------

output_data = {
    "model": MODEL_NAME,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "input_image": INPUT_IMAGE,
    "number_of_detections": len(detections),
    "predictions": predictions
}


# --------------------------------------------------
# Save JSON
# --------------------------------------------------

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output_data,
        file,
        indent=4
    )


print(
    f"Prediction data saved to: "
    f"{OUTPUT_JSON}"
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("--- Detection Summary ---")
print(
    f"Model: {MODEL_NAME}"
)
print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD}"
)
print(
    f"Objects detected: "
    f"{len(detections)}"
)

for prediction in predictions:

    print(
        f"- {prediction['class_name']} "
        f"{prediction['confidence']:.1%}"
    )
