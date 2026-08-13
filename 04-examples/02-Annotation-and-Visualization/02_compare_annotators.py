"""
02_compare_annotators.py

Compare multiple Supervision annotators.

This example demonstrates:
- BoxAnnotator
- RoundBoxAnnotator
- HaloAnnotator
- BlurAnnotator
- BoxCornerAnnotator
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
# Define annotators
# --------------------------------------------------

configs = [
    ("BoxAnnotator", sv.BoxAnnotator()),
    ("RoundBoxAnnotator", sv.RoundBoxAnnotator()),
    ("HaloAnnotator", sv.HaloAnnotator()),
    ("BlurAnnotator", sv.BlurAnnotator()),
    ("BoxCornerAnnotator", sv.BoxCornerAnnotator()),
]


# --------------------------------------------------
# Compare annotators
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10)
)

for ax, (name, annotator) in zip(
    axes.flat,
    configs
):
    scene = annotator.annotate(
        scene=image.copy(),
        detections=detections
    )

    ax.imshow(
        cv2.cvtColor(
            scene,
            cv2.COLOR_BGR2RGB
        )
    )

    ax.set_title(name)
    ax.axis("off")


# Hide unused plot
axes.flat[-1].axis("off")


# --------------------------------------------------
# Display
# --------------------------------------------------

plt.suptitle(
    "Supervision Annotator Comparison"
)

plt.tight_layout()
plt.show()
