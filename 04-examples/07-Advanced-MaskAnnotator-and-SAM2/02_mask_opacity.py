from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLO, SAM


# ============================================================
# Configuration
# ============================================================

IMAGE_URL = "https://ultralytics.com/images/bus.jpg"

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "bus.jpg"

YOLO_MODEL_NAME = "yolov8n.pt"

SAM_MODEL_PATH = Path(
    "/content/drive/MyDrive/SAM3-Models/sam3.pt"
)

OUTPUT_PATH = BASE_DIR / "02_mask_opacity_output.png"

OPACITY_VALUES = [
    0.2,
    0.5,
    0.9
]


# ============================================================
# Download input image if needed
# ============================================================

if not IMAGE_PATH.exists():
    import urllib.request

    print("Downloading bus.jpg...")

    urllib.request.urlretrieve(
        IMAGE_URL,
        IMAGE_PATH
    )

    print(
        f"Downloaded: {IMAGE_PATH}"
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
print("Example 02 — Mask Opacity")
print("=" * 60)

print(
    f"\nInput image: {IMAGE_PATH.name}"
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

yolo_detections = (
    sv.Detections.from_ultralytics(
        yolo_results
    )
)


print(
    f"YOLO detections: "
    f"{len(yolo_detections)}"
)


if len(yolo_detections) == 0:
    raise RuntimeError(
        "YOLO did not detect any objects."
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
# Generate SAM 3 masks
# ============================================================

bounding_boxes = (
    yolo_detections.xyxy.tolist()
)


print(
    "\nGenerating SAM 3 segmentation masks..."
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


print(
    f"SAM masks generated: "
    f"{len(sam_detections.mask)}"
)


# ============================================================
# Create opacity visualizations
# ============================================================

annotated_images = []


for opacity in OPACITY_VALUES:

    print(
        f"Creating mask visualization "
        f"with opacity={opacity}"
    )

    mask_annotator = sv.MaskAnnotator(
        opacity=opacity
    )

    annotated_image = (
        mask_annotator.annotate(
            scene=image.copy(),
            detections=sam_detections
        )
    )

    annotated_images.append(
        (
            opacity,
            annotated_image
        )
    )


# ============================================================
# Save comparison figure
# ============================================================

plt.figure(
    figsize=(18, 6)
)


for index, (
    opacity,
    annotated_image
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
            annotated_image,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"Mask Opacity = {opacity}"
    )

    plt.axis("off")


plt.suptitle(
    "SAM 3 MaskAnnotator Opacity Comparison",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Final result
# ============================================================

print(
    "\nOpacity comparison saved to:"
)

print(
    OUTPUT_PATH
)

print(
    "\nOpacity values tested:"
)

for opacity in OPACITY_VALUES:
    print(
        f"- {opacity}"
    )


print(
    "\nInterpretation:"
)

print(
    "0.2 → original image more visible"
)

print(
    "0.5 → balanced visualization"
)

print(
    "0.9 → segmentation mask more visible"
)


print(
    "\n" + "=" * 60
)

print(
    "Mask opacity example completed."
)

print(
    "=" * 60
)
