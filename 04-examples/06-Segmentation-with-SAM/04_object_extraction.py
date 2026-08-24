"""
Session 06 — Segmentation with SAM 3
Example 04 — Object Extraction

This example demonstrates how to use a SAM 3 segmentation mask
to isolate an object from the original image with pixel-level precision.

Pipeline:

Input Image
    ↓
YOLOv8 Detection
    ↓
Bounding Box Prompt
    ↓
SAM 3 Segmentation
    ↓
Boolean Mask
    ↓
Object Extraction
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

OUTPUT_PATH = BASE_DIR / "extracted_object.png"

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
print("Example 04 — Object Extraction")
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
# SELECT FIRST SEGMENTATION MASK
# ============================================================

mask = (
    sam_detections.mask[0]
)


print(
    f"\nSelected mask shape: "
    f"{mask.shape}"
)

print(
    f"Object pixels: "
    f"{int(mask.sum())}"
)


# ============================================================
# EXTRACT OBJECT
# ============================================================

object_image = (
    image.copy()
)

object_image[
    ~mask
] = 0


# ============================================================
# SAVE RESULT
# ============================================================

saved = cv2.imwrite(
    str(OUTPUT_PATH),
    object_image
)

if not saved:
    raise RuntimeError(
        f"Could not save output image: {OUTPUT_PATH}"
    )


print(
    f"\nExtracted object saved to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# EXPLANATION
# ============================================================

print(
    "\nExtraction operation:"
)

print(
    "object_image[~mask] = 0"
)

print(
    "\nMeaning:"
)

print(
    "Pixels outside the SAM 3 mask are set to black."
)

print(
    "Pixels inside the mask remain unchanged."
)


# ============================================================
# FINISHED
# ============================================================

print(
    "\n============================================"
)

print(
    "Object extraction example completed."
)

print(
    "============================================"
)
