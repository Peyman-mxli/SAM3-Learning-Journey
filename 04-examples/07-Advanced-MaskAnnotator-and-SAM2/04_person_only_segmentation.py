from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO, SAM


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "assets" / "input"
OUTPUT_DIR = BASE_DIR / "assets" / "output"

IMAGE_PATH = INPUT_DIR / "bus.jpg"

YOLO_MODEL_NAME = "yolov8n.pt"

SAM_MODEL_PATH = Path(
    "/content/drive/MyDrive/SAM3-Models/sam3.pt"
)

OUTPUT_PATH = OUTPUT_DIR / "04_person_only_segmentation_output.png"

PERSON_CLASS_ID = 0


# ============================================================
# Validate input image
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Input image not found: {IMAGE_PATH}\n\n"
        "Expected repository path:\n"
        "assets/input/bus.jpg"
    )


# ============================================================
# Validate SAM 3 model
# ============================================================

if not SAM_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"SAM 3 model not found: {SAM_MODEL_PATH}\n\n"
        "Expected Google Colab path:\n"
        "/content/drive/MyDrive/SAM3-Models/sam3.pt"
    )


# ============================================================
# Prepare output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load image
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise RuntimeError(
        f"Could not load image: {IMAGE_PATH}"
    )


print("=" * 60)
print("Example 04 — Person-Only Segmentation")
print("=" * 60)

print(
    f"\nInput image: {IMAGE_PATH.name}"
)

print(
    f"Input path: {IMAGE_PATH}"
)

print(
    f"Image shape: {image.shape}"
)


# ============================================================
# Load YOLOv8
# ============================================================

print(
    "\nLoading YOLOv8 model..."
)

yolo_model = YOLO(
    YOLO_MODEL_NAME
)


# ============================================================
# Run YOLO detection
# ============================================================

print(
    "Running YOLOv8 detection..."
)

yolo_results = yolo_model(
    image
)[0]

detections = (
    sv.Detections.from_ultralytics(
        yolo_results
    )
)

print(
    f"Total YOLO detections: "
    f"{len(detections)}"
)

if len(detections) == 0:
    raise RuntimeError(
        "YOLO did not detect any objects."
    )


# ============================================================
# Filter person detections BEFORE SAM
# ============================================================

if detections.class_id is None:
    raise RuntimeError(
        "YOLO detections do not contain class IDs."
    )

person_detections = detections[
    detections.class_id
    == PERSON_CLASS_ID
]

print(
    f"Person detections: "
    f"{len(person_detections)}"
)

if len(person_detections) == 0:
    raise RuntimeError(
        "No persons were detected."
    )


# ============================================================
# Print filtering summary
# ============================================================

print(
    "\nFiltering summary:"
)

print(
    f"All detections: {len(detections)}"
)

print(
    f"Persons kept:   {len(person_detections)}"
)

print(
    f"Objects removed before SAM: "
    f"{len(detections) - len(person_detections)}"
)


# ============================================================
# Load SAM 3
# ============================================================

print(
    "\nLoading SAM 3 model..."
)

sam_model = SAM(
    str(SAM_MODEL_PATH)
)

print(
    "SAM 3 model loaded successfully."
)


# ============================================================
# Use ONLY person boxes as SAM prompts
# ============================================================

person_bounding_boxes = (
    person_detections.xyxy.tolist()
)

print(
    "\nGenerating SAM 3 masks for persons only..."
)

sam_results = sam_model(
    image,
    bboxes=person_bounding_boxes
)[0]

sam_person_detections = (
    sv.Detections.from_ultralytics(
        sam_results
    )
)

if sam_person_detections.mask is None:
    raise RuntimeError(
        "SAM 3 did not return person segmentation masks."
    )

print(
    f"Person masks generated: "
    f"{len(sam_person_detections.mask)}"
)


# ============================================================
# Create annotators
# ============================================================

mask_annotator = sv.MaskAnnotator(
    opacity=0.7
)

box_annotator = sv.BoxAnnotator()


# ============================================================
# Annotate person masks
# ============================================================

annotated_image = (
    mask_annotator.annotate(
        scene=image.copy(),
        detections=sam_person_detections
    )
)


# ============================================================
# Add person bounding boxes
# ============================================================

annotated_image = (
    box_annotator.annotate(
        scene=annotated_image,
        detections=person_detections
    )
)


# ============================================================
# Save output
# ============================================================

success = cv2.imwrite(
    str(OUTPUT_PATH),
    annotated_image
)

if not success:
    raise RuntimeError(
        f"Could not save output image: {OUTPUT_PATH}"
    )


# ============================================================
# Final result
# ============================================================

print(
    "\nPerson-only segmentation saved to:"
)

print(
    OUTPUT_PATH
)

print(
    "\nWorkflow:"
)

print(
    "assets/input/bus.jpg"
)

print(
    "  ↓"
)

print(
    "YOLOv8"
)

print(
    "  ↓"
)

print(
    "All detections"
)

print(
    "  ↓"
)

print(
    "Filter class_id == 0"
)

print(
    "  ↓"
)

print(
    "Person bounding boxes"
)

print(
    "  ↓"
)

print(
    "SAM 3"
)

print(
    "  ↓"
)

print(
    "Person segmentation masks"
)

print(
    "  ↓"
)

print(
    "MaskAnnotator + BoxAnnotator"
)

print(
    "  ↓"
)

print(
    "assets/output/"
    "04_person_only_segmentation_output.png"
)

print(
    "\nImportant concept:"
)

print(
    "Filter BEFORE SAM instead of segmenting "
    "unwanted objects."
)

print(
    "\n" + "=" * 60
)

print(
    "Person-only segmentation example completed."
)

print(
    "=" * 60
)
