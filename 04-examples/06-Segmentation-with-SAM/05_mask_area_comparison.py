"""
Session 06 — Segmentation with SAM 3
Example 05 — Mask Area Comparison

This example compares the area of each SAM 3 segmentation mask
with the area of the corresponding YOLO bounding box.

The goal is to measure how much of each bounding box is actually
occupied by the segmented object.

Formula:

percentage = (mask_area / bounding_box_area) * 100
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
print("Example 05 — Mask Area Comparison")
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
# CHECK COUNTS
# ============================================================

count = min(
    len(yolo_detections),
    len(sam_detections.mask)
)

print(
    f"\nObjects available for comparison: "
    f"{count}"
)


# ============================================================
# COMPARE MASK AND BOX AREAS
# ============================================================

print(
    "\nMask area vs. bounding-box area:"
)

for index in range(
    count
):

    mask = (
        sam_detections.mask[index]
    )

    mask_area = int(
        mask.sum()
    )

    x1, y1, x2, y2 = (
        yolo_detections.xyxy[index]
    )

    box_width = max(
        0,
        x2 - x1
    )

    box_height = max(
        0,
        y2 - y1
    )

    box_area = float(
        box_width
        * box_height
    )


    if box_area > 0:
        percentage = (
            mask_area
            / box_area
        ) * 100
    else:
        percentage = 0.0


    class_name = "unknown"

    if yolo_detections.class_id is not None:
        class_id = int(
            yolo_detections.class_id[index]
        )

        class_name = (
            yolo_results.names[class_id]
        )


    print(
        f"\nObject {index}"
    )

    print(
        f"Class: {class_name}"
    )

    print(
        f"Mask area: "
        f"{mask_area} px"
    )

    print(
        f"Bounding-box area: "
        f"{box_area:.2f} px"
    )

    print(
        f"Mask / box: "
        f"{percentage:.2f}%"
    )


# ============================================================
# EXPLANATION
# ============================================================

print(
    "\nInterpretation:"
)

print(
    "A higher percentage means that more of the bounding box "
    "is occupied by the actual segmented object."
)

print(
    "A lower percentage means that the bounding box "
    "contains more background."
)


# ============================================================
# FINISHED
# ============================================================

print(
    "\n============================================"
)

print(
    "Mask area comparison example completed."
)

print(
    "============================================"
)
