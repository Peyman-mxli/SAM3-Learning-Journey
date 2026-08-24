"""
Session 06 — Segmentation with SAM 3

Practical implementation of a YOLO + SAM 3 segmentation pipeline.

Workflow:
1. Load an input image.
2. Detect objects with YOLO.
3. Convert YOLO detections to Supervision.
4. Use YOLO bounding boxes as SAM 3 prompts.
5. Generate segmentation masks.
6. Convert SAM results to Supervision.
7. Analyze segmentation masks.
8. Extract segmented objects.
9. Compare mask area with bounding-box area.
10. Save segmentation information to JSON.
"""

from pathlib import Path
import base64
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from ultralytics import YOLO, SAM


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "assets" / "input"
OUTPUT_DIR = BASE_DIR / "assets" / "output"

INPUT_IMAGE = INPUT_DIR / "bus.jpg"

YOLO_MODEL = "yolov8n.pt"
SAM_MODEL = "sam3.pt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(str(INPUT_IMAGE))

if image is None:
    raise FileNotFoundError(
        f"Input image not found: {INPUT_IMAGE}\n"
        "Place an image named 'bus.jpg' inside assets/input/."
    )

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"Input image: {INPUT_IMAGE}")
print(f"Image shape: {image.shape}")


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO model...")

yolo_model = YOLO(YOLO_MODEL)


# ============================================================
# YOLO OBJECT DETECTION
# ============================================================

print("Running YOLO object detection...")

yolo_results = yolo_model(image_rgb)[0]

yolo_detections = sv.Detections.from_ultralytics(yolo_results)

print(f"YOLO detections: {len(yolo_detections)}")


if len(yolo_detections) == 0:
    raise RuntimeError(
        "YOLO did not detect any objects in the input image."
    )


# ============================================================
# EXTRACT BOUNDING BOXES
# ============================================================

bboxes = yolo_detections.xyxy.tolist()

print("\nBounding boxes used as SAM prompts:")

for index, bbox in enumerate(bboxes):
    print(f"Object {index}: {bbox}")


# ============================================================
# LOAD SAM 3
# ============================================================

print("\nLoading SAM 3 model...")

sam_model = SAM(SAM_MODEL)


# ============================================================
# GENERATE SEGMENTATION MASKS
# ============================================================

print("Generating segmentation masks...")

sam_results = sam_model(
    image_rgb,
    bboxes=bboxes
)[0]

sam_detections = sv.Detections.from_ultralytics(sam_results)


if sam_detections.mask is None:
    raise RuntimeError(
        "SAM did not return segmentation masks."
    )


print(f"SAM masks generated: {len(sam_detections.mask)}")
print(f"Mask array shape: {sam_detections.mask.shape}")


# ============================================================
# ANALYZE FIRST MASK
# ============================================================

first_mask = sam_detections.mask[0]

print("\nFirst mask information:")
print(f"Type: {type(first_mask)}")
print(f"Shape: {first_mask.shape}")
print(f"Unique values: {np.unique(first_mask)}")
print(f"Object pixels: {first_mask.sum()}")
print(f"Total pixels: {first_mask.size}")

image_fraction = first_mask.sum() / first_mask.size

print(
    f"Image occupied by first segmented object: "
    f"{image_fraction * 100:.2f}%"
)


# ============================================================
# SAVE RAW MASK VISUALIZATION
# ============================================================

raw_mask_path = OUTPUT_DIR / "raw_mask.png"

plt.figure(figsize=(10, 6))
plt.imshow(first_mask, cmap="gray")
plt.title("Raw Segmentation Mask")
plt.axis("off")
plt.tight_layout()
plt.savefig(raw_mask_path, bbox_inches="tight")
plt.close()

print(f"\nRaw mask saved to: {raw_mask_path}")


# ============================================================
# EXTRACT FIRST SEGMENTED OBJECT
# ============================================================

object_crop = image_rgb.copy()

object_crop[~first_mask] = 0

object_output_path = OUTPUT_DIR / "segmented_object.png"

cv2.imwrite(
    str(object_output_path),
    cv2.cvtColor(object_crop, cv2.COLOR_RGB2BGR)
)

print(f"Segmented object saved to: {object_output_path}")


# ============================================================
# MASK AREA VS BOUNDING-BOX AREA
# ============================================================

print("\nMask area vs. bounding-box area:")

area_comparison = []

for index, mask in enumerate(sam_detections.mask):

    mask_area = int(mask.sum())

    x1, y1, x2, y2 = yolo_detections.xyxy[index]

    box_area = float(
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    if box_area > 0:
        percentage = (mask_area / box_area) * 100
    else:
        percentage = 0.0

    result = {
        "object_index": index,
        "mask_area_pixels": mask_area,
        "bounding_box_area_pixels": box_area,
        "mask_to_box_percentage": percentage,
    }

    area_comparison.append(result)

    print(
        f"Object {index}: "
        f"Mask Area = {mask_area} px | "
        f"Box Area = {box_area:.2f} px | "
        f"Mask/Box = {percentage:.2f}%"
    )


# ============================================================
# ENCODE MASKS FOR JSON
# ============================================================

def encode_mask(mask):
    """
    Encode a boolean segmentation mask using NumPy packbits
    and Base64 so that it can be stored inside JSON.
    """

    packed_mask = np.packbits(mask.flatten())

    encoded_mask = base64.b64encode(
        packed_mask.tobytes()
    ).decode("utf-8")

    return encoded_mask


encoded_masks = [
    encode_mask(mask)
    for mask in sam_detections.mask
]


# ============================================================
# PREPARE DETECTION DATA
# ============================================================

class_ids = []

if yolo_detections.class_id is not None:
    class_ids = yolo_detections.class_id.tolist()


confidences = []

if yolo_detections.confidence is not None:
    confidences = yolo_detections.confidence.tolist()


class_names = []

for class_id in class_ids:
    class_names.append(
        yolo_results.names[int(class_id)]
    )


segmentation_data = {
    "input_image": INPUT_IMAGE.name,
    "xyxy": yolo_detections.xyxy.tolist(),
    "confidence": confidences,
    "class_id": class_ids,
    "class_names": class_names,
    "mask_shape": list(first_mask.shape),
    "masks_b64": encoded_masks,
    "area_comparison": area_comparison,
}


# ============================================================
# SAVE JSON
# ============================================================

json_output_path = OUTPUT_DIR / "segmentation_results.json"

with open(
    json_output_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        segmentation_data,
        file,
        indent=4
    )

print(f"\nSegmentation data saved to: {json_output_path}")


# ============================================================
# VERIFY MASK DECODING
# ============================================================

height, width = first_mask.shape

encoded_mask = encoded_masks[0]

raw = np.frombuffer(
    base64.b64decode(encoded_mask),
    dtype=np.uint8
)

decoded_mask = np.unpackbits(raw)[:height * width]

decoded_mask = decoded_mask.reshape(
    height,
    width
).astype(bool)

masks_are_equal = np.array_equal(
    first_mask,
    decoded_mask
)

print(
    f"Decoded mask matches original: "
    f"{masks_are_equal}"
)


# ============================================================
# FINISHED
# ============================================================

print("\nSegmentation practical completed successfully.")

print("\nGenerated outputs:")

for output_file in sorted(OUTPUT_DIR.iterdir()):
    if output_file.is_file():
        print(f"- {output_file.name}")
