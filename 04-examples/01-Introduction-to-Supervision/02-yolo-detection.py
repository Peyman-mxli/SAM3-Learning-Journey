"""
02-yolo-detection.py

Introduction to Supervision — Example 02

Goal:
Load a pretrained YOLOv8 Nano model and perform
object detection on the example image.
"""

from pathlib import Path
import urllib.request

import cv2
from ultralytics import YOLO


# --------------------------------------------------
# 1. Prepare the example image
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
# 3. Load YOLOv8 Nano
# --------------------------------------------------

model = YOLO("yolov8n.pt")

print("YOLOv8 Nano model loaded.")


# --------------------------------------------------
# 4. Run inference
# --------------------------------------------------

results = model(image)[0]


# --------------------------------------------------
# 5. Inspect the result
# --------------------------------------------------

print("\nDetection completed.")
print(results)


# --------------------------------------------------
# 6. Inspect detected classes
# --------------------------------------------------

print("\nAvailable COCO classes:")

for class_id, class_name in results.names.items():
    print(
        f"{class_id}: {class_name}"
    )


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

"""
Image
  ↓
OpenCV
  ↓
YOLOv8 Nano
  ↓
Inference
  ↓
Ultralytics Result

The next example converts this result into
Supervision's sv.Detections format.
"""
