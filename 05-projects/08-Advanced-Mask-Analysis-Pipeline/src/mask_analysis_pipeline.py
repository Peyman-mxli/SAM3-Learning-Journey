from pathlib import Path
import csv
import json

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO, SAM


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_DIR / "data" / "input"
OUTPUT_DIR = PROJECT_DIR / "data" / "output"

JSON_DIR = PROJECT_DIR / "results" / "json"
CSV_DIR = PROJECT_DIR / "results" / "csv"

YOLO_MODEL_NAME = "yolov8n.pt"

SAM_MODEL_PATH = Path(
    "/content/drive/MyDrive/SAM3-Models/sam3.pt"
)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

MASK_OPACITY = 0.6


# ============================================================
# Directory Preparation
# ============================================================

def prepare_directories():
    """Create output and result directories when required."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CSV_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Model Validation
# ============================================================

def validate_sam_model():
    """Verify that the SAM 3 checkpoint exists."""

    if not SAM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SAM 3 model not found: {SAM_MODEL_PATH}\n\n"
            "Expected Google Colab path:\n"
            "/content/drive/MyDrive/SAM3-Models/sam3.pt"
        )


# ============================================================
# Input Discovery
# ============================================================

def discover_images():
    """Find supported input images."""

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    image_paths = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in: {INPUT_DIR}"
        )

    return image_paths


# ============================================================
# Image Loading
# ============================================================

def load_image(image_path):
    """Load an image using OpenCV."""

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not load image: {image_path}"
        )

    return image


# ============================================================
# Bounding-Box Area
# ============================================================

def calculate_box_area(box):
    """Calculate the area of one XYXY bounding box."""

    x1, y1, x2, y2 = box

    width = max(
        0.0,
        float(x2) - float(x1),
    )

    height = max(
        0.0,
        float(y2) - float(y1),
    )

    return width * height


# ============================================================
# Mask Area
# ============================================================

def calculate_mask_area(mask):
    """Calculate the number of pixels inside a mask."""

    return int(
        np.count_nonzero(mask)
    )


# ============================================================
# Occupancy Ratio
# ============================================================

def calculate_occupancy_ratio(
    mask_area,
    box_area,
):
    """
    Calculate the ratio between segmented pixels
    and bounding-box area.
    """

    if box_area <= 0:
        return 0.0

    return float(
        mask_area / box_area
    )


# ============================================================
# Detection Filtering
# ============================================================

def filter_detections(
    detections,
    target_class_id=None,
):
    """
    Optionally filter detections by class ID.

    When target_class_id is None, all detections
    are preserved.
    """

    if target_class_id is None:
        return detections

    if detections.class_id is None:
        raise RuntimeError(
            "Detections do not contain class IDs."
        )

    return detections[
        detections.class_id
        == target_class_id
    ]


# ============================================================
# Object Analysis
# ============================================================

def build_object_results(
    image_name,
    yolo_detections,
    sam_detections,
    class_names,
):
    """Create structured metrics for each segmented object."""

    if sam_detections.mask is None:
        raise RuntimeError(
            "SAM detections do not contain masks."
        )

    if len(yolo_detections) != len(
        sam_detections.mask
    ):
        raise RuntimeError(
            "Detection and mask counts do not match."
        )

    object_results = []

    for index in range(
        len(yolo_detections)
    ):
        box = yolo_detections.xyxy[index]

        class_id = (
            int(
                yolo_detections.class_id[index]
            )
            if yolo_detections.class_id
            is not None
            else -1
        )

        confidence = (
            float(
                yolo_detections.confidence[index]
            )
            if yolo_detections.confidence
            is not None
            else None
        )

        mask = sam_detections.mask[index]

        box_area = calculate_box_area(
            box
        )

        mask_area = calculate_mask_area(
            mask
        )

        occupancy_ratio = (
            calculate_occupancy_ratio(
                mask_area,
                box_area,
            )
        )

        class_name = class_names.get(
            class_id,
            f"class_{class_id}",
        )

        object_result = {
            "image": image_name,
            "object_id": index + 1,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": (
                round(confidence, 4)
                if confidence is not None
                else None
            ),
            "x1": round(
                float(box[0]),
                2,
            ),
            "y1": round(
                float(box[1]),
                2,
            ),
            "x2": round(
                float(box[2]),
                2,
            ),
            "y2": round(
                float(box[3]),
                2,
            ),
            "box_area": round(
                box_area,
                2,
            ),
            "mask_area": mask_area,
            "occupancy_ratio": round(
                occupancy_ratio,
                4,
            ),
        }

        object_results.append(
            object_result
        )

    return object_results


# ============================================================
# Visualization
# ============================================================

def create_visualization(
    image,
    yolo_detections,
    sam_detections,
):
    """Create mask + bounding-box visualization."""

    mask_annotator = sv.MaskAnnotator(
        opacity=MASK_OPACITY
    )

    box_annotator = sv.BoxAnnotator()

    annotated_image = (
        mask_annotator.annotate(
            scene=image.copy(),
            detections=sam_detections,
        )
    )

    annotated_image = (
        box_annotator.annotate(
            scene=annotated_image,
            detections=yolo_detections,
        )
    )

    return annotated_image


# ============================================================
# Single-Image Analysis
# ============================================================

def analyze_image(
    image_path,
    yolo_model,
    sam_model,
    target_class_id=None,
):
    """
    Run the complete advanced mask-analysis
    pipeline on one image.
    """

    print(
        f"\nProcessing: {image_path.name}"
    )

    image = load_image(
        image_path
    )

    print(
        f"Image shape: {image.shape}"
    )

    # --------------------------------------------------------
    # YOLO Detection
    # --------------------------------------------------------

    print(
        "Running YOLOv8 detection..."
    )

    yolo_result = yolo_model(
        image
    )[0]

    detections = (
        sv.Detections.from_ultralytics(
            yolo_result
        )
    )

    print(
        f"YOLO detections: {len(detections)}"
    )

    if len(detections) == 0:
        raise RuntimeError(
            f"No objects detected in "
            f"{image_path.name}"
        )

    # --------------------------------------------------------
    # Optional Filtering
    # --------------------------------------------------------

    detections = filter_detections(
        detections,
        target_class_id=target_class_id,
    )

    print(
        "Detections after filtering: "
        f"{len(detections)}"
    )

    if len(detections) == 0:
        raise RuntimeError(
            "No detections remain after filtering."
        )

    # --------------------------------------------------------
    # SAM 3 Segmentation
    # --------------------------------------------------------

    bounding_boxes = (
        detections.xyxy.tolist()
    )

    print(
        "Generating SAM 3 masks..."
    )

    sam_result = sam_model(
        image,
        bboxes=bounding_boxes,
    )[0]

    sam_detections = (
        sv.Detections.from_ultralytics(
            sam_result
        )
    )

    if sam_detections.mask is None:
        raise RuntimeError(
            "SAM 3 did not return masks."
        )

    print(
        "SAM masks generated: "
        f"{len(sam_detections.mask)}"
    )

    # --------------------------------------------------------
    # Structured Analysis
    # --------------------------------------------------------

    object_results = build_object_results(
        image_name=image_path.name,
        yolo_detections=detections,
        sam_detections=sam_detections,
        class_names=yolo_result.names,
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    annotated_image = create_visualization(
        image=image,
        yolo_detections=detections,
        sam_detections=sam_detections,
    )

    return (
        annotated_image,
        object_results,
    )


# ============================================================
# Save Annotated Image
# ============================================================

def save_annotated_image(
    image,
    image_path,
):
    """Save the annotated visualization."""

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_analyzed.png"
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise RuntimeError(
            f"Could not save image: {output_path}"
        )

    print(
        f"Annotated image saved: {output_path}"
    )

    return output_path


# ============================================================
# Save JSON
# ============================================================

def save_json_results(
    image_path,
    object_results,
):
    """Save structured analysis as JSON."""

    output_path = (
        JSON_DIR
        / f"{image_path.stem}_analysis.json"
    )

    payload = {
        "image": image_path.name,
        "object_count": len(
            object_results
        ),
        "detections": object_results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print(
        f"JSON results saved: {output_path}"
    )

    return output_path


# ============================================================
# Save CSV
# ============================================================

def save_csv_results(
    image_path,
    object_results,
):
    """Save structured analysis as CSV."""

    output_path = (
        CSV_DIR
        / f"{image_path.stem}_analysis.csv"
    )

    fieldnames = [
        "image",
        "object_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "box_area",
        "mask_area",
        "occupancy_ratio",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            object_results
        )

    print(
        f"CSV results saved: {output_path}"
    )

    return output_path


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "Project 08 — Advanced Mask Analysis Pipeline"
    )

    print(
        "=" * 60
    )

    prepare_directories()

    validate_sam_model()

    image_paths = discover_images()

    print(
        f"\nInput images found: {len(image_paths)}"
    )

    for image_path in image_paths:
        print(
            f"- {image_path.name}"
        )

    # --------------------------------------------------------
    # Load Models Once
    # --------------------------------------------------------

    print(
        "\nLoading YOLOv8..."
    )

    yolo_model = YOLO(
        YOLO_MODEL_NAME
    )

    print(
        "Loading SAM 3..."
    )

    sam_model = SAM(
        str(SAM_MODEL_PATH)
    )

    print(
        "Models loaded successfully."
    )

    # --------------------------------------------------------
    # Process Images
    # --------------------------------------------------------

    total_objects = 0

    for image_path in image_paths:

        (
            annotated_image,
            object_results,
        ) = analyze_image(
            image_path=image_path,
            yolo_model=yolo_model,
            sam_model=sam_model,
        )

        save_annotated_image(
            annotated_image,
            image_path,
        )

        save_json_results(
            image_path,
            object_results,
        )

        save_csv_results(
            image_path,
            object_results,
        )

        total_objects += len(
            object_results
        )

        print(
            f"Objects analyzed: "
            f"{len(object_results)}"
        )

    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "Project 08 processing completed."
    )

    print(
        "=" * 60
    )

    print(
        f"\nImages processed: "
        f"{len(image_paths)}"
    )

    print(
        f"Objects analyzed: "
        f"{total_objects}"
    )

    print(
        "\nWorkflow:"
    )

    print(
        "Input Image "
        "→ YOLOv8 "
        "→ Detection Filtering "
        "→ SAM 3 "
        "→ Mask Analysis "
        "→ MaskAnnotator "
        "→ BoxAnnotator "
        "→ JSON + CSV + Annotated Image"
    )


if __name__ == "__main__":
    main()
