"""
05-image-annotation.py

Introduction to Supervision — Example 05

Goal:
Use Supervision annotators to draw bounding boxes,
class labels, and confidence scores on an image.
"""

from pathlib import Path
import urllib.request

import cv2
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# 1. Prepare the image
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
# 3. Load YOLO and run inference
# --------------------------------------------------

model = YOLO("yolov8n.pt")

results = model(image)[0]


# --------------------------------------------------
# 4. Convert to Supervision detections
# --------------------------------------------------

detections = sv.Detections.from_ultralytics(
    results
)


# --------------------------------------------------
# 5. Create annotators
# --------------------------------------------------

box_annotator = sv.BoxAnnotator()

label_annotator = sv.LabelAnnotator()


# --------------------------------------------------
# 6. Create labels
# --------------------------------------------------

labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]


# --------------------------------------------------
# 7. Draw bounding boxes
# --------------------------------------------------

annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)


# --------------------------------------------------
# 8. Add labels
# --------------------------------------------------

annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)


# --------------------------------------------------
# 9. Convert BGR to RGB
# --------------------------------------------------

annotated_rgb = cv2.cvtColor(
    annotated_image,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# 10. Display the result
# --------------------------------------------------

plt.figure(figsize=(12, 7))

plt.imshow(annotated_rgb)

plt.title(
    "YOLO + Supervision Annotation"
)

plt.axis("off")

plt.show()


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

"""
Image
  ↓
YOLO
  ↓
sv.Detections
  ↓
BoxAnnotator
  ↓
LabelAnnotator
  ↓
Annotated Image
  ↓
Matplotlib
"""
