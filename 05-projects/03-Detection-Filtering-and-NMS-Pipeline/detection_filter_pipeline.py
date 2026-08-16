```python
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "yolov8n.pt"

INPUT_IMAGE = "assets/input/image.jpg"
OUTPUT_IMAGE = "assets/output/filtered_detections.jpg"

CONFIDENCE_THRESHOLD = 0.50
MIN_AREA = 5000
NMS_THRESHOLD = 0.50
TOP_N = 5

# COCO class 0 = person
TARGET_CLASS_ID = 0

# If True, keep only detections whose center is
# located in the right half of the image
FILTER_RIGHT_HALF = True


# --------------------------------------------------
# Create required directories
# --------------------------------------------------

Path("assets/input").mkdir(
    parents=True,
    exist_ok=True
)

Path("assets/output").mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(
    INPUT_IMAGE
)

if image is None:
    raise FileNotFoundError(
        f"Could not load input image: {INPUT_IMAGE}"
    )


# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

model = YOLO(
    MODEL_NAME
)


# --------------------------------------------------
# Run object detection
# --------------------------------------------------

results = model(
    image
)[0]


# --------------------------------------------------
# Convert YOLO results to Supervision detections
# --------------------------------------------------

detections = sv.Detections.from_ultralytics(
    results
)

print(
    f"Initial detections: {len(detections)}"
)


# --------------------------------------------------
# Confidence filtering
# --------------------------------------------------

detections = detections[
    detections.confidence
    > CONFIDENCE_THRESHOLD
]

print(
    f"After confidence filtering: "
    f"{len(detections)}"
)


# --------------------------------------------------
# Class filtering
# --------------------------------------------------

detections = detections[
    detections.class_id
    == TARGET_CLASS_ID
]

print(
    f"After class filtering: "
    f"{len(detections)}"
)


# --------------------------------------------------
# Size filtering
# --------------------------------------------------

detections = detections[
    detections.area
    > MIN_AREA
]

print(
    f"After size filtering: "
    f"{len(detections)}"
)


# --------------------------------------------------
# Non-Maximum Suppression
# --------------------------------------------------

detections = detections.with_nms(
    threshold=NMS_THRESHOLD
)

print(
    f"After NMS: "
    f"{len(detections)}"
)


# --------------------------------------------------
# Top-N confidence selection
# --------------------------------------------------

if (
    len(detections) > 0
    and detections.confidence is not None
):

    indices = np.argsort(
        detections.confidence
    )[::-1][:TOP_N]

    detections = detections[
        indices
    ]


print(
    f"After Top-{TOP_N} selection: "
    f"{len(detections)}"
)


# --------------------------------------------------
# Spatial filtering
# --------------------------------------------------

if (
    FILTER_RIGHT_HALF
    and len(detections) > 0
):

    centers_x = (
        detections.xyxy[:, 0]
        + detections.xyxy[:, 2]
    ) / 2

    image_midpoint = (
        image.shape[1] / 2
    )

    detections = detections[
        centers_x > image_midpoint
    ]


print(
    f"After spatial filtering: "
    f"{len(detections)}"
)


# --------------------------------------------------
# Create labels
# --------------------------------------------------

labels = []

if (
    len(detections) > 0
    and detections.class_id is not None
    and detections.confidence is not None
):

    labels = [
        (
            f"{results.names[class_id]} "
            f"{confidence:.1%}"
        )
        for class_id, confidence in zip(
            detections.class_id,
            detections.confidence
        )
    ]


# --------------------------------------------------
# Create annotators
# --------------------------------------------------

box_annotator = sv.BoxAnnotator()

label_annotator = sv.LabelAnnotator()


# --------------------------------------------------
# Annotate image
# --------------------------------------------------

annotated_image = image.copy()

annotated_image = box_annotator.annotate(
    scene=annotated_image,
    detections=detections
)

annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)


# --------------------------------------------------
# Draw image midpoint
# --------------------------------------------------

midpoint_x = int(
    image.shape[1] / 2
)

cv2.line(
    annotated_image,
    (midpoint_x, 0),
    (
        midpoint_x,
        image.shape[0]
    ),
    (255, 255, 255),
    2
)


# --------------------------------------------------
# Save output image
# --------------------------------------------------

success = cv2.imwrite(
    OUTPUT_IMAGE,
    annotated_image
)

if not success:
    raise RuntimeError(
        f"Could not save output image: "
        f"{OUTPUT_IMAGE}"
    )


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print()
print(
    "Detection Filtering Pipeline Complete"
)

print(
    f"Input image: {INPUT_IMAGE}"
)

print(
    f"Output image: {OUTPUT_IMAGE}"
)

print(
    f"Final detections: "
    f"{len(detections)}"
)
