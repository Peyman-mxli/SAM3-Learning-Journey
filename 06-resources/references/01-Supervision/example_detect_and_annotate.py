"""Run YOLO inference and annotate the detections with Supervision."""

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import supervision as sv
from ultralytics import YOLO


IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
CONFIDENCE_THRESHOLD = 0.50
ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "assets" / "input" / "bus.jpg"
OUTPUT_PATH = ROOT / "assets" / "output" / "bus_annotated.jpg"


def prepare_directories() -> None:
    """Create the input and output directories when they do not exist."""
    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def download_image() -> None:
    """Download the sample only when it is not already available locally."""
    if not INPUT_PATH.exists():
        print(f"Downloading sample image to {INPUT_PATH}")
        urlretrieve(IMAGE_URL, INPUT_PATH)


def build_labels(detections: sv.Detections) -> list[str]:
    """Create one readable class-and-confidence label per detection."""
    class_names = detections.data.get("class_name")
    labels: list[str] = []

    for index, confidence in enumerate(detections.confidence):
        if class_names is not None:
            class_name = str(class_names[index])
        else:
            class_name = f"class_{int(detections.class_id[index])}"
        labels.append(f"{class_name} {confidence:.2f}")

    return labels


def main() -> None:
    """Execute detection, conversion, filtering, annotation, and export."""
    prepare_directories()
    download_image()

    image = cv2.imread(str(INPUT_PATH))
    if image is None:
        raise FileNotFoundError(f"OpenCV could not read: {INPUT_PATH}")

    model = YOLO("yolov8n.pt")
    result = model(image, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence >= CONFIDENCE_THRESHOLD]
    labels = build_labels(detections)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.55, text_thickness=1)

    annotated = box_annotator.annotate(
        scene=image.copy(),
        detections=detections,
    )
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels,
    )

    if not cv2.imwrite(str(OUTPUT_PATH), annotated):
        raise RuntimeError(f"OpenCV could not write: {OUTPUT_PATH}")

    print(f"Supervision version: {sv.__version__}")
    print(f"Detections retained: {len(detections)}")
    print(f"Saved annotated image: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
