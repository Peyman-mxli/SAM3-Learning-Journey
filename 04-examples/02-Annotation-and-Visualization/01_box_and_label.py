"""
01_box_and_label.py

Basic YOLO + Supervision annotation example.

This example demonstrates how to:
- Detect objects with YOLOv8
- Convert YOLO results to Supervision Detections
- Draw bounding boxes
- Add class names and confidence labels
"""

import cv2
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread("image.jpg")

if image is None:
    raise FileNotFoundError("Could not load image.jpg")


# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

model = YOLO("yolov8n.pt")


# --------------------------------------------------
# Run object detection
# --------------------------------------------------

results = model(image)[0]


# --------------------------------------------------
# Convert results to Supervision Detections
# --------------------------------------------------

detections = sv.Detections.from_ultralytics(results)


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

box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator()


# --------------------------------------------------
# Apply annotation layers
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
# Save result
# --------------------------------------------------

cv2.imwrite(
    "annotated_image.jpg",
    annotated_image
)

print(f"Detected objects: {len(detections)}")
print("Saved: annotated_image.jpg")
