"""
Session 06 — Segmentation with SAM 3
Example 02 — SAM Bounding-Box Segmentation

This example demonstrates how YOLOv8 detections can be used
as bounding-box prompts for SAM 3.

Pipeline:

Input Image
    ↓
YOLOv8
    ↓
Bounding Boxes
    ↓
SAM 3
    ↓
Segmentation Masks
"""

from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO, SAM


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "bus.jpg"

YOLO_MODEL = "yolov8n.pt"

SAM_MODEL = Path(
    "/content/drive/MyDrive/SAM3-Models/sam3.pt"
)


# ============================================================
# VERIFY FILES
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Input image not found: {IMAGE_PATH}\n"
        "Place 'bus.jpg' inside this example folder."
    )

if not SAM_MODEL.exists():
    raise FileNotFoundError(
        f"SAM 3 model not found: {SAM_MODEL}\n\n"
        "If using Google Colab, mount Google Drive first:\n\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')"
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
print("Example 02 — SAM Bounding-Box Segmentation")
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

yolo_model = YOLO(
    YOLO_MODEL
)


# ============================================================
# RUN YOLO DETECTION
# ============================================================

print(
    "Running YOLO detection..."
)

yolo_results = yolo_model(
    image
)[0]

yolo_detections = (
    sv.Detections.from_ultralytics(
        yolo_results
    )
)


print(
    f"YOLO detections: "
    f"{len(yolo_detections)}"
)


if len(yolo_detections) == 0:
    raise RuntimeError(
        "YOLO did not detect any objects."
    )


# ============================================================
# CONVERT BOUNDING BOXES TO PYTHON LIST
# ============================================================

bboxes = (
    yolo_detections.xyxy.tolist()
)


print(
    "\nBounding boxes used "
    "as SAM 3 prompts:"
)

for index, bbox in enumerate(
    bboxes
):
    print(
        f"{index}: {bbox}"
    )


# ============================================================
# LOAD SAM 3
# ============================================================

print(
    "\nLoading SAM 3 model..."
)

sam_model = SAM(
    str(SAM_MODEL)
)

print(
    "SAM 3 model loaded successfully."
)


# ============================================================
# RUN SAM 3 SEGMENTATION
# ============================================================

print(
    "\nGenerating segmentation masks..."
)

sam_results = sam_model(
    image,
    bboxes=bboxes
)[0]


# ============================================================
# CONVERT TO SUPERVISION
# ============================================================

sam_detections = (
    sv.Detections.from_ultralytics(
        sam_results
    )
)


if sam_detections.mask is None:
    raise RuntimeError(
        "SAM 3 did not return segmentation masks."
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    f"\nSAM detections: "
    f"{len(sam_detections)}"
)

print(
    f"SAM masks generated: "
    f"{len(sam_detections.mask)}"
)

print(
    f"Mask array shape: "
    f"{sam_detections.mask.shape}"
)


# ============================================================
# COMPARE YOLO AND SAM COUNTS
# ============================================================

print(
    "\nDetection comparison:"
)

print(
    f"YOLO objects: "
    f"{len(yolo_detections)}"
)

print(
    f"SAM masks: "
    f"{len(sam_detections.mask)}"
)


if len(yolo_detections) == len(
    sam_detections.mask
):
    print(
        "Each YOLO bounding box "
        "produced a SAM 3 mask."
    )
else:
    print(
        "YOLO detection count and "
        "SAM mask count are different."
    )


# ============================================================
# FINISHED
# ============================================================

print(
    "\n============================================"
)

print(
    "SAM bounding-box segmentation "
    "example completed."
)

print(
    "============================================"
)
