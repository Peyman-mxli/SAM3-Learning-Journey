"""
Session 06 — Segmentation with SAM 3
Example 01 — YOLO Detection

This example demonstrates the first stage of the segmentation pipeline:

Input Image
    ↓
YOLOv8
    ↓
Object Detections
    ↓
Bounding Boxes

The bounding boxes generated here can later be used as prompts for SAM 3.
"""

from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "bus.jpg"

YOLO_MODEL = "yolov8n.pt"


# ============================================================
# VERIFY INPUT IMAGE
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Input image not found: {IMAGE_PATH}\n"
        "Place 'bus.jpg' inside this example folder."
    )


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise RuntimeError(
        f"OpenCV could not read: {IMAGE_PATH}"
    )


print("============================================")
print("Example 01 — YOLO Detection")
print("============================================")

print(
    f"\nInput image: {IMAGE_PATH.name}"
)

print(
    f"Image shape: {image.shape}"
)


# ============================================================
# LOAD YOLO
# ============================================================

print(
    "\nLoading YOLOv8 model..."
)

model = YOLO(
    YOLO_MODEL
)


# ============================================================
# RUN DETECTION
# ============================================================

print(
    "Running object detection..."
)

results = model(
    image
)[0]


# ============================================================
# CONVERT TO SUPERVISION
# ============================================================

detections = (
    sv.Detections.from_ultralytics(
        results
    )
)


print(
    f"\nDetected objects: "
    f"{len(detections)}"
)


# ============================================================
# PRINT DETECTION INFORMATION
# ============================================================

for index in range(
    len(detections)
):

    bbox = (
        detections.xyxy[index]
    )

    class_id = None
    confidence = None
    class_name = "unknown"

    if detections.class_id is not None:
        class_id = int(
            detections.class_id[index]
        )

        class_name = (
            results.names[class_id]
        )

    if detections.confidence is not None:
        confidence = float(
            detections.confidence[index]
        )


    print(
        f"\nObject {index}"
    )

    print(
        f"Class: {class_name}"
    )

    print(
        f"Class ID: {class_id}"
    )

    print(
        f"Confidence: {confidence}"
    )

    print(
        f"Bounding box: "
        f"{bbox.tolist()}"
    )


# ============================================================
# BOUNDING BOX PROMPTS
# ============================================================

bboxes = (
    detections.xyxy.tolist()
)

print(
    "\nBounding boxes ready "
    "for SAM 3 prompts:"
)

for index, bbox in enumerate(
    bboxes
):
    print(
        f"{index}: {bbox}"
    )


# ============================================================
# FINISHED
# ============================================================

print(
    "\n============================================"
)

print(
    "YOLO detection example completed."
)

print(
    "============================================"
)
