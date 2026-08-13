"""
04_layer_order.py

Demonstrate why annotation layer order matters.

This example compares:
- Box -> Label
- Label -> Box
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
# Create annotators
# --------------------------------------------------

box_ann = sv.BoxAnnotator(
    thickness=3
)

label_ann = sv.LabelAnnotator()


# --------------------------------------------------
# Order A: Box -> Label
# --------------------------------------------------

order_a = box_ann.annotate(
    scene=image.copy(),
    detections=detections
)

order_a = label_ann.annotate(
    scene=order_a,
    detections=detections,
    labels=labels
)


# --------------------------------------------------
# Order B: Label -> Box
# --------------------------------------------------

order_b = label_ann.annotate(
    scene=image.copy(),
    detections=detections,
    labels=labels
)

order_b = box_ann.annotate(
    scene=order_b,
    detections=detections
)


# --------------------------------------------------
# Compare results
# --------------------------------------------------

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

ax1.imshow(
    cv2.cvtColor(
        order_a,
        cv2.COLOR_BGR2RGB
    )
)

ax1.set_title(
    "Box -> Label (Recommended)"
)

ax1.axis("off")


ax2.imshow(
    cv2.cvtColor(
        order_b,
        cv2.COLOR_BGR2RGB
    )
)

ax2.set_title(
    "Label -> Box"
)

ax2.axis("off")


plt.tight_layout()
plt.show()
