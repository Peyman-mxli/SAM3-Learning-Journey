"""
06-yolo-model-comparison.py

Introduction to Supervision — Example 06

Goal:
Compare YOLOv8 Nano and YOLOv8 Small
using the same image and Supervision pipeline.
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
# 3. Load YOLOv8 Nano
# --------------------------------------------------

model_n = YOLO("yolov8n.pt")

results_n = model_n(image)[0]

detections_n = sv.Detections.from_ultralytics(
    results_n
)


# --------------------------------------------------
# 4. Load YOLOv8 Small
# --------------------------------------------------

model_s = YOLO("yolov8s.pt")

results_s = model_s(image)[0]

detections_s = sv.Detections.from_ultralytics(
    results_s
)


# --------------------------------------------------
# 5. Compare detection counts
# --------------------------------------------------

print("\n--- YOLO Model Comparison ---")

print(
    f"YOLOv8 Nano: "
    f"{len(detections_n)} objects"
)

print(
    f"YOLOv8 Small: "
    f"{len(detections_s)} objects"
)


# --------------------------------------------------
# 6. Compare confidence scores
# --------------------------------------------------

print("\nNano confidence scores:")
print(detections_n.confidence)

print("\nSmall confidence scores:")
print(detections_s.confidence)


# --------------------------------------------------
# 7. Compare detected classes
# --------------------------------------------------

print("\nNano detected classes:")

for class_id in sorted(
    set(detections_n.class_id)
):
    print(
        f"{class_id}: "
        f"{results_n.names[class_id]}"
    )


print("\nSmall detected classes:")

for class_id in sorted(
    set(detections_s.class_id)
):
    print(
        f"{class_id}: "
        f"{results_s.names[class_id]}"
    )


# --------------------------------------------------
# 8. Comparison questions
# --------------------------------------------------

"""
Questions to analyze:

1. Did both models detect the same number of objects?
2. Did they detect the same classes?
3. Which model produced higher confidence scores?
4. Did the Small model detect additional objects?
5. Did either model miss an object?
6. Were smaller or partially hidden objects detected differently?
"""


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

"""
                 IMAGE
                   │
          ┌────────┴────────┐
          ▼                 ▼
   YOLOv8 Nano       YOLOv8 Small
          │                 │
          ▼                 ▼
       Results           Results
          │                 │
          ▼                 ▼
   sv.Detections     sv.Detections
          │                 │
          └────────┬────────┘
                   ▼
                Compare
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      Count    Confidence   Classes
"""
