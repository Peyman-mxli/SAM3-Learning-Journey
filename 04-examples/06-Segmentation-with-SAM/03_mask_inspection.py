"""
Session 06 — Segmentation with SAM 3
Example 03 — Mask Inspection

This example demonstrates how to inspect a SAM 3 segmentation mask
as a NumPy boolean array.

It focuses on:

- Mask type
- Mask shape
- Unique values
- Object pixel count
- Total pixel count
- Image coverage percentage
"""

from pathlib import Path

import cv2
import numpy as np
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
print("Example 03 — Mask Inspection")
print("============================================")

print(
    f"\nInput image: {IMAGE_PATH.name}"
)

print(
    f"Image shape: {image.shape}"
)


# ============================================================
# YOLO DETECTION
# ============================================================

print(
    "\nRunning YOLOv8 detection..."
)

yolo_model = YOLO(
    YOLO_MODEL
)

yolo_results = yolo_model(
    image
)[0]

yolo_detections = (
    sv.Detections.from_ultralytics(
        yolo_results
    )
)


if len(yolo_detections) == 0:
    raise RuntimeError(
        "YOLO did not detect any objects."
    )


print(
    f"YOLO detections: "
    f"{len(yolo_detections)}"
)


# ============================================================
# PREPARE SAM PROMPTS
# ============================================================

bboxes = (
    yolo_detections.xyxy.tolist()
)


# ============================================================
# SAM 3 SEGMENTATION
# ============================================================

print(
    "\nLoading SAM 3 model..."
)

sam_model = SAM(
    str(SAM_MODEL)
)

print(
    "Generating segmentation masks..."
)

sam_results = sam_model(
    image,
    bboxes=bboxes
)[0]

sam_detections = (
    sv.Detections.from_ultralytics(
        sam_results
    )
)


if sam_detections.mask is None:
    raise RuntimeError(
        "SAM 3 did not return segmentation masks."
    )

if len(sam_detections.mask) == 0:
    raise RuntimeError(
        "SAM 3 returned an empty mask collection."
    )


print(
    f"SAM masks generated: "
    f"{len(sam_detections.mask)}"
)


# ============================================================
# SELECT FIRST MASK
# ============================================================

first_mask = (
    sam_detections.mask[0]
)


# ============================================================
# INSPECT MASK
# ============================================================

print(
    "\nFirst mask information:"
)

print(
    f"Type: {type(first_mask)}"
)

print(
    f"Shape: {first_mask.shape}"
)

print(
    f"Data type: {first_mask.dtype}"
)

print(
    f"Unique values: "
    f"{np.unique(first_mask)}"
)


# ============================================================
# PIXEL COUNTS
# ============================================================

object_pixels = int(
    first_mask.sum()
)

total_pixels = int(
    first_mask.size
)

background_pixels = (
    total_pixels
    - object_pixels
)


print(
    f"\nObject pixels: "
    f"{object_pixels}"
)

print(
    f"Background pixels: "
    f"{background_pixels}"
)

print(
    f"Total pixels: "
    f"{total_pixels}"
)


# ============================================================
# IMAGE COVERAGE
# ============================================================

coverage_fraction = (
    object_pixels
    / total_pixels
)

coverage_percentage = (
    coverage_fraction
    * 100
)


print(
    f"\nImage coverage fraction: "
    f"{coverage_fraction:.4f}"
)

print(
    f"Image coverage percentage: "
    f"{coverage_percentage:.2f}%"
)


# ============================================================
# BOOLEAN MASK EXPLANATION
# ============================================================

print(
    "\nBoolean mask meaning:"
)

print(
    "True  -> pixel belongs to the object"
)

print(
    "False -> pixel belongs to the background"
)


# ============================================================
# MASK ARRAY SHAPE
# ============================================================

print(
    "\nComplete SAM mask array shape:"
)

print(
    sam_detections.mask.shape
)

print(
    "\nShape meaning:"
)

print(
    "(number_of_objects, image_height, image_width)"
)


# ============================================================
# FINISHED
# ============================================================

print(
    "\n============================================"
)

print(
    "Mask inspection example completed."
)

print(
    "============================================"
)
