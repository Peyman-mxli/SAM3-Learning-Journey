"""
03_customize_visualization.py

Customize Supervision annotations.

This example demonstrates:
- Bounding box colors
- Color palettes
- Bounding box thickness
- Label text scale
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
# Create visualization configurations
# --------------------------------------------------

configs = [
    (
        "Red boxes",
        sv.BoxAnnotator(
            color=sv.Color.RED,
            thickness=3
        ),
        sv.LabelAnnotator(
            text_scale=0.5
        )
    ),
    (
        "Green boxes",
        sv.BoxAnnotator(
            color=sv.Color.GREEN,
            thickness=4
        ),
        sv.LabelAnnotator(
            text_scale=0.7
        )
    ),
    (
        "Color palette",
        sv.BoxAnnotator(
            color=sv.ColorPalette.DEFAULT,
            thickness=5
        ),
        sv.LabelAnnotator(
            text_scale=1.0
        )
    ),
]


# --------------------------------------------------
# Compare configurations
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

for ax, (title, box_ann, label_ann) in zip(
    axes,
    configs
):
    scene = box_ann.annotate(
        scene=image.copy(),
        detections=detections
    )

    scene = label_ann.annotate(
        scene=scene,
        detections=detections,
        labels=labels
    )

    ax.imshow(
        cv2.cvtColor(
            scene,
            cv2.COLOR_BGR2RGB
        )
    )

    ax.set_title(title)
    ax.axis("off")


# --------------------------------------------------
# Display
# --------------------------------------------------

plt.suptitle(
    "Custom Annotation Styles"
)

plt.tight_layout()
plt.show()
