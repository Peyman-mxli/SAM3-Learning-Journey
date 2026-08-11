"""
07-custom-image.py

Introduction to Supervision — Example 07

Goal:
Apply the complete YOLO + Supervision pipeline
to a different image.
"""

from pathlib import Path
import urllib.request

import cv2
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# 1. Create assets directory
# --------------------------------------------------

Path("assets").mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Download a different test image
# --------------------------------------------------

image_url = "https://ultralytics.com/images/zidane.jpg"
image_path = "assets/zidane.jpg"

if not Path(image_path).exists():
    urllib.request.urlretrieve(
        image_url,
        image_path
    )


# --------------------------------------------------
# 3. Load the image
# --------------------------------------------------

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )

print(f"Image shape: {image.shape}")


# --------------------------------------------------
# 4. Load YOLO
# --------------------------------------------------

model = YOLO("yolov8n.pt")


# --------------------------------------------------
# 5. Run inference
# --------------------------------------------------

results = model(image)[0]


# --------------------------------------------------
# 6. Convert to Supervision
# --------------------------------------------------

detections = sv.Detections.from_ultralytics(
    results
)


# --------------------------------------------------
# 7. Inspect predictions
# --------------------------------------------------

print(
    f"\nObjects detected: "
    f"{len(detections)}"
)

print("\nBounding boxes:")
print(detections.xyxy)

print("\nConfidence scores:")
print(detections.confidence)

print("\nClass IDs:")
print(detections.class_id)


# --------------------------------------------------
# 8. Display detected classes
# --------------------------------------------------

print("\nDetected classes:")

for class_id in sorted(
    set(detections.class_id)
):
    print(
        f"{class_id}: "
        f"{results.names[class_id]}"
    )


# --------------------------------------------------
# 9. Create annotators
# --------------------------------------------------

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


# --------------------------------------------------
# 10. Create labels
# --------------------------------------------------

labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]


# --------------------------------------------------
# 11. Annotate the image
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
# 12. Display final result
# --------------------------------------------------

annotated_rgb = cv2.cvtColor(
    annotated_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(12, 7))

plt.imshow(annotated_rgb)

plt.title(
    "Custom Image — YOLO + Supervision"
)

plt.axis("off")
plt.show()


# --------------------------------------------------
# Analysis questions
# --------------------------------------------------

"""
After running the example, analyze:

1. What objects were detected?
2. How many objects were detected?
3. Which prediction has the highest confidence?
4. Did YOLO miss any visible objects?
5. Did YOLO detect anything incorrectly?
6. Are any objects partially hidden?
"""


# --------------------------------------------------
# Complete pipeline
# --------------------------------------------------

"""
Custom Image
     ↓
OpenCV
     ↓
YOLO
     ↓
Ultralytics Results
     ↓
sv.Detections
     ↓
Inspect Predictions
     ↓
Create Labels
     ↓
BoxAnnotator
     ↓
LabelAnnotator
     ↓
Annotated Image
     ↓
Evaluate Results
"""
