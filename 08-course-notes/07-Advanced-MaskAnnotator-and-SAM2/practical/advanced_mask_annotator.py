from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLO, SAM


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "assets" / "input"
OUTPUT_DIR = BASE_DIR / "assets" / "output"

BUS_IMAGE = INPUT_DIR / "bus.jpg"
ZIDANE_IMAGE = INPUT_DIR / "zidane.jpg"

SAM_MODEL_PATH = Path(
    "/content/drive/MyDrive/SAM3-Models/sam3.pt"
)

YOLO_MODEL_NAME = "yolov8n.pt"

PERSON_CLASS_ID = 0


# ============================================================
# Helper Functions
# ============================================================

def load_image(image_path):
    """Load an image using OpenCV."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return image


def run_yolo(image, model):
    """Run YOLO detection and convert results to sv.Detections."""

    results = model(image)[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    return detections


def run_sam(image, detections, model):
    """Use detection bounding boxes as SAM prompts."""

    if len(detections) == 0:
        return None

    bounding_boxes = detections.xyxy.tolist()

    results = model(
        image,
        bboxes=bounding_boxes
    )[0]

    sam_detections = sv.Detections.from_ultralytics(
        results
    )

    return sam_detections


def save_image(path, image):
    """Save an image and report its location."""

    success = cv2.imwrite(
        str(path),
        image
    )

    if not success:
        raise RuntimeError(
            f"Could not save image: {path}"
        )

    print(f"Saved: {path}")


# ============================================================
# Visualization Functions
# ============================================================

def create_bbox_vs_mask(
    image,
    yolo_detections,
    sam_detections
):
    """Compare YOLO bounding boxes with SAM segmentation."""

    box_annotator = sv.BoxAnnotator()

    mask_annotator = sv.MaskAnnotator(
        opacity=0.6
    )

    bbox_image = box_annotator.annotate(
        scene=image.copy(),
        detections=yolo_detections
    )

    mask_image = mask_annotator.annotate(
        scene=image.copy(),
        detections=sam_detections
    )

    combined = mask_annotator.annotate(
        scene=image.copy(),
        detections=sam_detections
    )

    combined = box_annotator.annotate(
        scene=combined,
        detections=yolo_detections
    )

    return bbox_image, mask_image, combined


def create_opacity_comparison(
    image,
    sam_detections
):
    """Create a comparison of multiple mask opacity values."""

    opacity_values = [
        0.2,
        0.5,
        0.9
    ]

    annotated_images = []

    for opacity in opacity_values:

        annotator = sv.MaskAnnotator(
            opacity=opacity
        )

        annotated = annotator.annotate(
            scene=image.copy(),
            detections=sam_detections
        )

        annotated_images.append(
            (
                opacity,
                annotated
            )
        )

    return annotated_images


def save_opacity_figure(
    annotated_images,
    output_path
):
    """Save the opacity comparison as one figure."""

    plt.figure(
        figsize=(18, 6)
    )

    for index, (
        opacity,
        image
    ) in enumerate(
        annotated_images,
        start=1
    ):

        plt.subplot(
            1,
            len(annotated_images),
            index
        )

        plt.imshow(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
        )

        plt.title(
            f"Opacity = {opacity}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Main Practical
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "Session 07 — Advanced MaskAnnotator and SAM2"
    )

    print(
        "=" * 60
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Validate assets
    # --------------------------------------------------------

    if not BUS_IMAGE.exists():
        raise FileNotFoundError(
            f"Missing input image: {BUS_IMAGE}"
        )

    if not ZIDANE_IMAGE.exists():
        raise FileNotFoundError(
            f"Missing input image: {ZIDANE_IMAGE}"
        )

    if not SAM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing SAM 3 model: {SAM_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\nLoading YOLOv8...")

    yolo_model = YOLO(
        YOLO_MODEL_NAME
    )

    print("Loading SAM 3...")

    sam_model = SAM(
        str(SAM_MODEL_PATH)
    )

    print("Models loaded successfully.")

    # --------------------------------------------------------
    # Experiment 1 — Main image
    # --------------------------------------------------------

    print(
        "\nExperiment 1 — YOLO + SAM segmentation"
    )

    bus_image = load_image(
        BUS_IMAGE
    )

    print(
        f"Input image: {BUS_IMAGE.name}"
    )

    print(
        f"Image shape: {bus_image.shape}"
    )

    yolo_detections = run_yolo(
        bus_image,
        yolo_model
    )

    print(
        f"YOLO detections: {len(yolo_detections)}"
    )

    sam_detections = run_sam(
        bus_image,
        yolo_detections,
        sam_model
    )

    if sam_detections is None:
        raise RuntimeError(
            "No detections available for SAM."
        )

    print(
        f"SAM masks: {len(sam_detections)}"
    )

    # --------------------------------------------------------
    # Experiment 2 — Bounding boxes vs masks
    # --------------------------------------------------------

    print(
        "\nExperiment 2 — Bounding boxes vs masks"
    )

    (
        bbox_image,
        mask_image,
        combined_image
    ) = create_bbox_vs_mask(
        bus_image,
        yolo_detections,
        sam_detections
    )

    save_image(
        OUTPUT_DIR / "bounding_boxes.png",
        bbox_image
    )

    save_image(
        OUTPUT_DIR / "segmentation_masks.png",
        mask_image
    )

    save_image(
        OUTPUT_DIR / "bbox_vs_mask.png",
        combined_image
    )

    # --------------------------------------------------------
    # Experiment 3 — Opacity comparison
    # --------------------------------------------------------

    print(
        "\nExperiment 3 — Mask opacity comparison"
    )

    opacity_images = create_opacity_comparison(
        bus_image,
        sam_detections
    )

    save_opacity_figure(
        opacity_images,
        OUTPUT_DIR / "opacity_comparison.png"
    )

    # --------------------------------------------------------
    # Experiment 4 — Filter persons before SAM
    # --------------------------------------------------------

    print(
        "\nExperiment 4 — Person-only segmentation"
    )

    person_detections = yolo_detections[
        yolo_detections.class_id
        == PERSON_CLASS_ID
    ]

    print(
        f"Person detections: {len(person_detections)}"
    )

    if len(person_detections) > 0:

        person_sam_detections = run_sam(
            bus_image,
            person_detections,
            sam_model
        )

        person_mask_annotator = sv.MaskAnnotator(
            opacity=0.6
        )

        person_box_annotator = sv.BoxAnnotator()

        person_output = (
            person_mask_annotator.annotate(
                scene=bus_image.copy(),
                detections=person_sam_detections
            )
        )

        person_output = (
            person_box_annotator.annotate(
                scene=person_output,
                detections=person_detections
            )
        )

        save_image(
            OUTPUT_DIR
            / "person_only_segmentation.png",
            person_output
        )

    else:

        print(
            "No persons detected."
        )

    # --------------------------------------------------------
    # Experiment 5 — Reuse pipeline
    # --------------------------------------------------------

    print(
        "\nExperiment 5 — Second image"
    )

    zidane_image = load_image(
        ZIDANE_IMAGE
    )

    print(
        f"Input image: {ZIDANE_IMAGE.name}"
    )

    second_yolo_detections = run_yolo(
        zidane_image,
        yolo_model
    )

    print(
        "YOLO detections on second image:",
        len(second_yolo_detections)
    )

    second_sam_detections = run_sam(
        zidane_image,
        second_yolo_detections,
        sam_model
    )

    if second_sam_detections is not None:

        second_mask_annotator = (
            sv.MaskAnnotator(
                opacity=0.6
            )
        )

        second_box_annotator = (
            sv.BoxAnnotator()
        )

        second_output = (
            second_mask_annotator.annotate(
                scene=zidane_image.copy(),
                detections=second_sam_detections
            )
        )

        second_output = (
            second_box_annotator.annotate(
                scene=second_output,
                detections=second_yolo_detections
            )
        )

        save_image(
            OUTPUT_DIR
            / "second_image_segmentation.png",
            second_output
        )

    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Session 07 practical completed."
    )

    print(
        "=" * 60
    )

    print(
        "\nConcepts demonstrated:"
    )

    print(
        "- YOLO object detection"
    )

    print(
        "- SAM 3 segmentation"
    )

    print(
        "- MaskAnnotator"
    )

    print(
        "- Bounding-box + mask visualization"
    )

    print(
        "- Mask opacity comparison"
    )

    print(
        "- Filtering before SAM"
    )

    print(
        "- Person-only segmentation"
    )

    print(
        "- Reusable segmentation pipeline"
    )

    print(
        "\nSAM2 concept:"
    )

    print(
        "Static masks can be extended toward "
        "temporal mask propagation in video."
    )


if __name__ == "__main__":
    main()
