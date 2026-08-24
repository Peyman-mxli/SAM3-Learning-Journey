"""
Project 07 — SAM 3 Segmentation Pipeline

Complete multi-object segmentation pipeline using:

- YOLOv8
- SAM 3
- Supervision
- OpenCV
- NumPy
- JSON

Workflow:
1. Load input image.
2. Run YOLOv8 detection.
3. Filter detections by confidence.
4. Convert YOLO bounding boxes into SAM 3 prompts.
5. Generate segmentation masks.
6. Analyze mask geometry.
7. Export individual masks.
8. Extract segmented objects.
9. Create an annotated segmentation visualization.
10. Export structured JSON results.
"""

from pathlib import Path
import json

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO, SAM


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "assets" / "input"
OUTPUT_DIR = BASE_DIR / "assets" / "output"

MASKS_DIR = OUTPUT_DIR / "masks"
EXTRACTED_DIR = OUTPUT_DIR / "extracted_objects"

INPUT_IMAGE = INPUT_DIR / "bus.jpg"

YOLO_MODEL = "yolov8n.pt"

SAM_MODEL = Path(
    "/content/drive/MyDrive/SAM3-Models/sam3.pt"
)

CONFIDENCE_THRESHOLD = 0.25


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MASKS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EXTRACTED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VERIFY INPUT FILE
# ============================================================

if not INPUT_IMAGE.exists():
    raise FileNotFoundError(
        f"Input image not found: {INPUT_IMAGE}\n"
        "Place 'bus.jpg' inside assets/input/."
    )


# ============================================================
# VERIFY SAM MODEL
# ============================================================

if not SAM_MODEL.exists():
    raise FileNotFoundError(
        f"SAM 3 model not found: {SAM_MODEL}\n\n"
        "If running in Google Colab, mount your Google Drive first:\n\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n\n"
        "Expected model location:\n"
        "/content/drive/MyDrive/SAM3-Models/sam3.pt"
    )


# ============================================================
# LOAD INPUT IMAGE
# ============================================================

image = cv2.imread(
    str(INPUT_IMAGE)
)

if image is None:
    raise RuntimeError(
        f"OpenCV could not read input image: {INPUT_IMAGE}"
    )


image_height, image_width = image.shape[:2]

print("============================================")
print("Project 07 — SAM 3 Segmentation Pipeline")
print("============================================")

print(
    f"\nInput image: {INPUT_IMAGE}"
)

print(
    f"Image shape: {image.shape}"
)

print(
    f"Confidence threshold: {CONFIDENCE_THRESHOLD}"
)

print(
    f"SAM 3 model: {SAM_MODEL}"
)


# ============================================================
# LOAD YOLO MODEL
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
    f"\nRaw YOLO detections: "
    f"{len(detections)}"
)


if len(detections) == 0:
    raise RuntimeError(
        "YOLOv8 did not detect any objects."
    )


# ============================================================
# CONFIDENCE FILTERING
# ============================================================

if detections.confidence is None:
    raise RuntimeError(
        "YOLO detections do not contain confidence scores."
    )


confidence_mask = (
    detections.confidence
    >= CONFIDENCE_THRESHOLD
)

filtered_detections = (
    detections[
        confidence_mask
    ]
)


print(
    f"Accepted detections: "
    f"{len(filtered_detections)}"
)


if len(filtered_detections) == 0:
    raise RuntimeError(
        "No detections passed the confidence threshold."
    )


# ============================================================
# PREPARE YOLO METADATA
# ============================================================

class_ids = (
    filtered_detections.class_id
)

confidences = (
    filtered_detections.confidence
)

bounding_boxes = (
    filtered_detections.xyxy
)


class_names = []

for class_id in class_ids:
    class_names.append(
        yolo_results.names[
            int(class_id)
        ]
    )


print(
    "\nAccepted YOLO detections:"
)

for index in range(
    len(filtered_detections)
):

    print(
        f"Object {index}: "
        f"class={class_names[index]} | "
        f"confidence={float(confidences[index]):.4f} | "
        f"bbox={bounding_boxes[index].tolist()}"
    )


# ============================================================
# PREPARE SAM 3 PROMPTS
# ============================================================

sam_bboxes = (
    bounding_boxes.tolist()
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
# GENERATE SEGMENTATION MASKS
# ============================================================

print(
    "\nGenerating SAM 3 segmentation masks..."
)

sam_results = sam_model(
    image,
    bboxes=sam_bboxes
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
        "SAM 3 returned an empty segmentation-mask collection."
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
# ALIGN YOLO DETECTIONS AND SAM MASKS
# ============================================================

object_count = min(
    len(filtered_detections),
    len(sam_detections.mask)
)


print(
    f"Objects available for complete analysis: "
    f"{object_count}"
)


# ============================================================
# PREPARE ANNOTATED IMAGE
# ============================================================

annotated_image = (
    image.copy()
)


# ============================================================
# CREATE SUPERVISION ANNOTATORS
# ============================================================

mask_annotator = (
    sv.MaskAnnotator()
)

box_annotator = (
    sv.BoxAnnotator()
)

label_annotator = (
    sv.LabelAnnotator()
)


# ============================================================
# LIMIT DETECTIONS TO AVAILABLE MASK COUNT
# ============================================================

analysis_detections = (
    filtered_detections[
        np.arange(
            len(filtered_detections)
        ) < object_count
    ]
)


analysis_masks = (
    sam_detections.mask[
        :object_count
    ]
)


analysis_detections.mask = (
    analysis_masks
)


# ============================================================
# CREATE LABELS
# ============================================================

labels = []

for index in range(
    object_count
):

    label = (
        f"{index} | "
        f"{class_names[index]} | "
        f"{float(confidences[index]):.2f}"
    )

    labels.append(
        label
    )


# ============================================================
# ANNOTATE MASKS
# ============================================================

annotated_image = (
    mask_annotator.annotate(
        scene=annotated_image,
        detections=analysis_detections
    )
)


# ============================================================
# ANNOTATE BOXES
# ============================================================

annotated_image = (
    box_annotator.annotate(
        scene=annotated_image,
        detections=analysis_detections
    )
)


# ============================================================
# ANNOTATE LABELS
# ============================================================

annotated_image = (
    label_annotator.annotate(
        scene=annotated_image,
        detections=analysis_detections,
        labels=labels
    )
)


# ============================================================
# SAVE COMPLETE ANNOTATED IMAGE
# ============================================================

annotated_output = (
    OUTPUT_DIR
    / "annotated_segmentation.png"
)


saved = cv2.imwrite(
    str(annotated_output),
    annotated_image
)


if not saved:
    raise RuntimeError(
        f"Could not save annotated image: {annotated_output}"
    )


print(
    f"\nAnnotated segmentation saved to:"
)

print(
    annotated_output
)


# ============================================================
# ANALYZE OBJECTS
# ============================================================

results_objects = []


print(
    "\nObject analysis:"
)


for index in range(
    object_count
):

    mask = (
        analysis_masks[index]
    )

    bbox = (
        bounding_boxes[index]
    )

    x1, y1, x2, y2 = (
        map(
            float,
            bbox
        )
    )


    # --------------------------------------------------------
    # MASK AREA
    # --------------------------------------------------------

    mask_area = int(
        mask.sum()
    )


    # --------------------------------------------------------
    # IMAGE COVERAGE
    # --------------------------------------------------------

    image_pixels = (
        image_height
        * image_width
    )

    image_coverage = (
        mask_area
        / image_pixels
    ) * 100


    # --------------------------------------------------------
    # BOUNDING BOX AREA
    # --------------------------------------------------------

    box_width = max(
        0.0,
        x2 - x1
    )

    box_height = max(
        0.0,
        y2 - y1
    )

    box_area = (
        box_width
        * box_height
    )


    # --------------------------------------------------------
    # MASK / BOX PERCENTAGE
    # --------------------------------------------------------

    if box_area > 0:

        mask_to_box_percentage = (
            mask_area
            / box_area
        ) * 100

    else:

        mask_to_box_percentage = 0.0


    # --------------------------------------------------------
    # SAVE MASK IMAGE
    # --------------------------------------------------------

    mask_image = (
        mask.astype(
            np.uint8
        )
        * 255
    )


    mask_filename = (
        f"object_{index:02d}_mask.png"
    )

    mask_path = (
        MASKS_DIR
        / mask_filename
    )


    cv2.imwrite(
        str(mask_path),
        mask_image
    )


    # --------------------------------------------------------
    # EXTRACT OBJECT
    # --------------------------------------------------------

    extracted_object = (
        image.copy()
    )

    extracted_object[
        ~mask
    ] = 0


    extracted_filename = (
        f"object_{index:02d}.png"
    )

    extracted_path = (
        EXTRACTED_DIR
        / extracted_filename
    )


    cv2.imwrite(
        str(extracted_path),
        extracted_object
    )


    # --------------------------------------------------------
    # OBJECT RESULT
    # --------------------------------------------------------

    object_result = {
        "object_index": index,
        "class_id": int(
            class_ids[index]
        ),
        "class_name": class_names[index],
        "confidence": float(
            confidences[index]
        ),
        "bounding_box": [
            x1,
            y1,
            x2,
            y2
        ],
        "bounding_box_area_pixels": float(
            box_area
        ),
        "mask_area_pixels": mask_area,
        "image_coverage_percentage": float(
            image_coverage
        ),
        "mask_to_box_percentage": float(
            mask_to_box_percentage
        ),
        "mask_file": (
            f"masks/{mask_filename}"
        ),
        "extracted_object_file": (
            f"extracted_objects/{extracted_filename}"
        ),
    }


    results_objects.append(
        object_result
    )


    # --------------------------------------------------------
    # PRINT OBJECT SUMMARY
    # --------------------------------------------------------

    print(
        f"\nObject {index}"
    )

    print(
        f"Class: "
        f"{class_names[index]}"
    )

    print(
        f"Confidence: "
        f"{float(confidences[index]):.4f}"
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
        f"Image coverage: "
        f"{image_coverage:.2f}%"
    )

    print(
        f"Mask / box: "
        f"{mask_to_box_percentage:.2f}%"
    )


# ============================================================
# PREPARE PROJECT SUMMARY
# ============================================================

project_results = {
    "project": (
        "Project 07 — SAM 3 Segmentation Pipeline"
    ),
    "input_image": (
        INPUT_IMAGE.name
    ),
    "image_width": (
        image_width
    ),
    "image_height": (
        image_height
    ),
    "confidence_threshold": (
        CONFIDENCE_THRESHOLD
    ),
    "raw_yolo_detections": (
        len(detections)
    ),
    "accepted_yolo_detections": (
        len(filtered_detections)
    ),
    "sam_masks_generated": (
        len(sam_detections.mask)
    ),
    "objects_analyzed": (
        object_count
    ),
    "annotated_output": (
        annotated_output.name
    ),
    "objects": (
        results_objects
    ),
}


# ============================================================
# SAVE JSON RESULTS
# ============================================================

json_output = (
    OUTPUT_DIR
    / "segmentation_results.json"
)


with open(
    json_output,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        project_results,
        file,
        indent=4
    )


print(
    f"\nStructured results saved to:"
)

print(
    json_output
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n============================================"
)

print(
    "Project 07 completed successfully."
)

print(
    "============================================"
)

print(
    f"\nRaw YOLO detections: "
    f"{len(detections)}"
)

print(
    f"Accepted detections: "
    f"{len(filtered_detections)}"
)

print(
    f"SAM masks generated: "
    f"{len(sam_detections.mask)}"
)

print(
    f"Objects analyzed: "
    f"{object_count}"
)

print(
    "\nGenerated files:"
)

print(
    f"- {annotated_output.name}"
)

print(
    f"- {json_output.name}"
)

print(
    f"- {object_count} mask images"
)

print(
    f"- {object_count} extracted object images"
)
