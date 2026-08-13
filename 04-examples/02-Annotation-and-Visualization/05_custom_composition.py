"""
05_custom_composition.py

Create a custom visualization using multiple Supervision annotators.

This example demonstrates:
- EllipseAnnotator
- DotAnnotator
- LabelAnnotator
- Layer composition
"""

import cv2
import matplotlib.pyplot as plt
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
# Run detection
# --------------------------------------------------

results = model(image)[0]

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
# Start with original image
# --------------------------------------------------

scene = image.copy()


# --------------------------------------------------
# Layer 1: EllipseAnnotator
# --------------------------------------------------

ellipse_annotator = sv.EllipseAnnotator()

scene = ellipse_annotator.annotate(
    scene=scene,
    detections=detections
)


# --------------------------------------------------
# Layer 2: DotAnnotator
# --------------------------------------------------

dot_annotator = sv.DotAnnotator()

scene = dot_annotator.annotate(
    scene=scene,
    detections=detections
)


# --------------------------------------------------
# Layer 3: LabelAnnotator
# --------------------------------------------------

label_annotator = sv.LabelAnnotator()

scene = label_annotator.annotate(
    scene=scene,
    detections=detections,
    labels=labels
)


# --------------------------------------------------
# Display final result
# --------------------------------------------------

plt.figure(
    figsize=(12, 7)
)

plt.imshow(
    cv2.cvtColor(
        scene,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")

plt.title(
    "Custom Annotator Composition"
)

plt.show()
