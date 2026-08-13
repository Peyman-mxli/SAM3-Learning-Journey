"""
multi_annotator_pipeline.py

YOLO + Supervision Multi-Annotator Visualization Pipeline

This project demonstrates how to:
- Detect objects with YOLOv8
- Convert YOLO results to Supervision Detections
- Apply multiple annotation layers
- Customize colors and thickness
- Add labels with confidence scores
- Save the final annotated image
"""

from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "yolov8n.pt"

INPUT_IMAGE = "input/image.jpg"
OUTPUT_IMAGE = "output/annotated_image.jpg"

CONFIDENCE_THRESHOLD = 0.50


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

box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)

ellipse_annotator = sv.EllipseAnnotator()

dot_annotator = sv.DotAnnotator()

label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)


# --------------------------------------------------
# Start visualization pipeline
# --------------------------------------------------

annotated_image = image.copy()


# --------------------------------------------------
# Layer 1: Bounding boxes
# --------------------------------------------------

annotated_image = box_annotator.annotate(
    scene=annotated_image,
    detections=detections
)


# --------------------------------------------------
# Layer 2: Ellipses
# --------------------------------------------------

annotated_image = ellipse_annotator.annotate(
    scene=annotated_image,
    detections=detections
)


# --------------------------------------------------
# Layer 3: Detection points
# --------------------------------------------------

annotated_image = dot_annotator.annotate(
    scene=annotated_image,
    detections=detections
)


# --------------------------------------------------
# Layer 4: Labels
# --------------------------------------------------

annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)


# --------------------------------------------------
# Save result
# --------------------------------------------------

success = cv2.imwrite(
    OUTPUT_IMAGE,
    annotated_image
)

if not success:
    raise RuntimeError(
        f"Could not save image: {OUTPUT_IMAGE}"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print(
    f"Detected objects: {len(detections)}"
)

print(
    f"Annotated image saved to: {OUTPUT_IMAGE}"
)
