"""
Session 06 — Segmentation with SAM 3
Example 06 — Mask Serialization

This example demonstrates how to convert a SAM 3 segmentation mask
into a JSON-compatible Base64 representation and then reconstruct
the original boolean NumPy mask.

Pipeline:

Boolean Mask
    ↓
Flatten
    ↓
np.packbits
    ↓
Bytes
    ↓
Base64 Encoding
    ↓
JSON-Compatible String
    ↓
Base64 Decoding
    ↓
np.unpackbits
    ↓
Reshape
    ↓
Recovered Boolean Mask
"""

from pathlib import Path
import base64
import json

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO, SAM


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "bus.jpg"

OUTPUT_JSON = BASE_DIR / "mask_serialization_example.json"

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
print("Example 06 — Mask Serialization")
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

original_mask = (
    sam_detections.mask[0]
)

height, width = (
    original_mask.shape
)


print(
    f"\nOriginal mask shape: "
    f"{original_mask.shape}"
)

print(
    f"Original mask dtype: "
    f"{original_mask.dtype}"
)

print(
    f"Object pixels: "
    f"{int(original_mask.sum())}"
)


# ============================================================
# ENCODE MASK
# ============================================================

flattened_mask = (
    original_mask.flatten()
)

packed_mask = np.packbits(
    flattened_mask
)

encoded_mask = (
    base64.b64encode(
        packed_mask.tobytes()
    ).decode(
        "utf-8"
    )
)


print(
    "\nMask encoded successfully."
)

print(
    f"Boolean pixels: "
    f"{flattened_mask.size}"
)

print(
    f"Packed bytes: "
    f"{packed_mask.nbytes}"
)

print(
    f"Base64 characters: "
    f"{len(encoded_mask)}"
)


# ============================================================
# SAVE JSON
# ============================================================

data = {
    "image": IMAGE_PATH.name,
    "mask_shape": [
        height,
        width
    ],
    "mask_b64": encoded_mask,
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        data,
        file,
        indent=4
    )


print(
    f"\nJSON saved to:"
)

print(
    OUTPUT_JSON
)


# ============================================================
# READ JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "r",
    encoding="utf-8"
) as file:

    loaded_data = json.load(
        file
    )


loaded_height, loaded_width = (
    loaded_data[
        "mask_shape"
    ]
)

loaded_encoded_mask = (
    loaded_data[
        "mask_b64"
    ]
)


# ============================================================
# DECODE BASE64
# ============================================================

raw_bytes = (
    base64.b64decode(
        loaded_encoded_mask
    )
)

packed_array = np.frombuffer(
    raw_bytes,
    dtype=np.uint8
)


# ============================================================
# UNPACK BITS
# ============================================================

unpacked_mask = np.unpackbits(
    packed_array
)

pixel_count = (
    loaded_height
    * loaded_width
)

unpacked_mask = (
    unpacked_mask[
        :pixel_count
    ]
)


# ============================================================
# RESTORE ORIGINAL SHAPE
# ============================================================

decoded_mask = (
    unpacked_mask
    .reshape(
        loaded_height,
        loaded_width
    )
    .astype(
        bool
    )
)


print(
    f"\nDecoded mask shape: "
    f"{decoded_mask.shape}"
)

print(
    f"Decoded mask dtype: "
    f"{decoded_mask.dtype}"
)


# ============================================================
# VALIDATE RECOVERED MASK
# ============================================================

masks_match = np.array_equal(
    original_mask,
    decoded_mask
)


print(
    "\nValidation:"
)

print(
    f"Decoded mask matches original: "
    f"{masks_match}"
)


if not masks_match:
    raise RuntimeError(
        "Decoded mask does not match the original mask."
    )


# ============================================================
# FINISHED
# ============================================================

print(
    "\n============================================"
)

print(
    "Mask serialization example completed."
)

print(
    "============================================"
)
