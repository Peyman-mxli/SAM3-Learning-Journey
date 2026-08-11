"""
03-supervision-detections.py

Introduction to Supervision — Example 03

Goal:
Convert YOLO predictions into Supervision's
sv.Detections format and inspect the results.
"""

from pathlib import Path
import urllib.request

import cv2
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
# 3. Load YOLO
# --------------------------------------------------

model = YOLO("yolov8n.pt")


# --------------------------------------------------
# 4. Run inference
# --------------------------------------------------

results = model(image)[0]


# --------------------------------------------------
# 5. Convert to sv.Detections
# --------------------------------------------------

detections = sv.Detections.from_ultralytics(
    results
)


# --------------------------------------------------
# 6. Inspect detections
# --------------------------------------------------

print(
    f"Number of detected objects: "
    f"{len(detections)}"
)

print("\nBounding boxes (xyxy):")
print(detections.xyxy)

print("\nConfidence scores:")
print(detections.confidence)

print("\nClass IDs:")
print(detections.class_id)


# --------------------------------------------------
# 7. Translate class IDs
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
# 8. Inspect first detection
# --------------------------------------------------

if len(detections) > 0:
    first = detections[0]

    x1, y1, x2, y2 = first.xyxy[0]

    print("\nFirst detection:")

    print(
        "Class:",
        results.names[first.class_id[0]]
    )

    print(
        f"Confidence: "
        f"{first.confidence[0]:.1%}"
    )

    print(
        f"Position: "
        f"({x1:.0f}, {y1:.0f}) → "
        f"({x2:.0f}, {y2:.0f})"
    )

    print(
        f"Size: "
        f"{x2 - x1:.0f}px × "
        f"{y2 - y1:.0f}px"
    )


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

"""
Image
  ↓
YOLO
  ↓
Ultralytics Results
  ↓
sv.Detections.from_ultralytics()
  ↓
sv.Detections
  ↓
xyxy + confidence + class_id
"""
