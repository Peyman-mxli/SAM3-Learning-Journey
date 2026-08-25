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

BUS_IMAGE_PATH = INPUT_DIR / "bus.jpg"
ZIDANE_IMAGE_PATH = INPUT_DIR / "zidane.jpg"

YOLO_MODEL_NAME = "yolov8n.pt"

SAM_MODEL_PATH = Path(
    "/content/drive/MyDrive/SAM3-Models/sam3.pt"
)

BUS_OUTPUT_PATH = (
    OUTPUT_DIR
    / "05_reusable_bus_output.png"
)

ZIDANE_OUTPUT_PATH = (
    OUTPUT_DIR
    / "05_reusable_zidane_output.png"
)


# ============================================================
# Validate input images
# ============================================================

for image_path in [
    BUS_IMAGE_PATH,
    ZIDANE_IMAGE_PATH,
]:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}\n\n"
            "Expected repository paths:\n"
            "assets/input/bus.jpg\n"
            "assets/input/zidane.jpg"
        )


# ============================================================
# Validate SAM 3 model
# ============================================================

if not SAM_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"SAM 3 model not found: "
        f"{SAM_MODEL_PATH}\n\n"
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
# Helper — Load Image
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
# Reusable Segmentation Function
# ============================================================

def segment_image(
    image,
    yolo_model,
    sam_model,
    opacity=0.6
):
    """
    Run YOLO detection, SAM 3 segmentation,
    and Supervision visualization on one image.
    """

    # --------------------------------------------------------
    # YOLO detection
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SAM prompts
    # --------------------------------------------------------

    bounding_boxes = (
        yolo_detections.xyxy.tolist()
    )

    sam_results = sam_model(
        image,
        bboxes=bounding_boxes
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

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    mask_annotator = sv.MaskAnnotator(
        opacity=opacity
    )

    box_annotator = sv.BoxAnnotator()

    annotated_image = (
        mask_annotator.annotate(
            scene=image.copy(),
            detections=sam_detections
        )
    )

    annotated_image = (
        box_annotator.annotate(
            scene=annotated_image,
            detections=yolo_detections
        )
    )

    # --------------------------------------------------------
    # Return reusable results
    # --------------------------------------------------------

    return (
        annotated_image,
        yolo_detections,
        sam_detections
    )


# ============================================================
# Helper — Save Image
# ============================================================

def save_image(
    image,
    output_path
):
    """Save an annotated image."""

    success = cv2.imwrite(
        str(output_path),
        image
    )

    if not success:
        raise RuntimeError(
            f"Could not save image: {output_path}"
        )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "Example 05 — Reusable Segmentation Function"
    )

    print(
        "=" * 60
    )

    print(
        f"\nBus input: {BUS_IMAGE_PATH}"
    )

    print(
        f"Zidane input: {ZIDANE_IMAGE_PATH}"
    )

    # --------------------------------------------------------
    # Load models ONCE
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
    # Process bus.jpg
    # --------------------------------------------------------

    print(
        "\nProcessing bus.jpg..."
    )

    bus_image = load_image(
        BUS_IMAGE_PATH
    )

    (
        bus_annotated,
        bus_yolo_detections,
        bus_sam_detections
    ) = segment_image(
        bus_image,
        yolo_model,
        sam_model,
        opacity=0.6
    )

    print(
        f"bus.jpg YOLO detections: "
        f"{len(bus_yolo_detections)}"
    )

    print(
        f"bus.jpg SAM masks: "
        f"{len(bus_sam_detections.mask)}"
    )

    save_image(
        bus_annotated,
        BUS_OUTPUT_PATH
    )

    # --------------------------------------------------------
    # Process zidane.jpg
    # --------------------------------------------------------

    print(
        "\nProcessing zidane.jpg..."
    )

    zidane_image = load_image(
        ZIDANE_IMAGE_PATH
    )

    (
        zidane_annotated,
        zidane_yolo_detections,
        zidane_sam_detections
    ) = segment_image(
        zidane_image,
        yolo_model,
        sam_model,
        opacity=0.6
    )

    print(
        f"zidane.jpg YOLO detections: "
        f"{len(zidane_yolo_detections)}"
    )

    print(
        f"zidane.jpg SAM masks: "
        f"{len(zidane_sam_detections.mask)}"
    )

    save_image(
        zidane_annotated,
        ZIDANE_OUTPUT_PATH
    )

    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------

    print(
        "\nReusable function:"
    )

    print(
        "segment_image("
    )

    print(
        "    image,"
    )

    print(
        "    yolo_model,"
    )

    print(
        "    sam_model,"
    )

    print(
        "    opacity=0.6"
    )

    print(
        ")"
    )

    print(
        "\nThe SAME function processed:"
    )

    print(
        "- assets/input/bus.jpg"
    )

    print(
        "- assets/input/zidane.jpg"
    )

    print(
        "\nGenerated outputs:"
    )

    print(
        "- assets/output/"
        "05_reusable_bus_output.png"
    )

    print(
        "- assets/output/"
        "05_reusable_zidane_output.png"
    )

    print(
        "\nWorkflow:"
    )

    print(
        "assets/input/IMAGE "
        "→ YOLOv8 "
        "→ sv.Detections "
        "→ SAM 3 "
        "→ Masks "
        "→ MaskAnnotator "
        "→ BoxAnnotator "
        "→ assets/output/RESULT"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "Reusable segmentation example completed."
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()
