"""Create a four-panel gallery of Supervision detection annotators."""

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import supervision as sv
from ultralytics import YOLO


IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
CONFIDENCE_THRESHOLD = 0.50
ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "assets" / "input" / "bus.jpg"
OUTPUT_PATH = ROOT / "assets" / "output" / "annotator_gallery.jpg"


def prepare_input() -> None:
    """Create directories and download the sample image when necessary."""
    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not INPUT_PATH.exists():
        urlretrieve(IMAGE_URL, INPUT_PATH)


def create_labels(detections: sv.Detections) -> list[str]:
    """Create labels from the final, filtered detections."""
    names = detections.data.get("class_name")
    labels: list[str] = []
    for index, confidence in enumerate(detections.confidence):
        name = str(names[index]) if names is not None else str(detections.class_id[index])
        labels.append(f"{name} {confidence:.2f}")
    return labels


def add_title(image, title: str):
    """Add a readable panel title to an annotated image."""
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 52), (20, 20, 20), -1)
    cv2.putText(
        result,
        title,
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def annotate_panel(image, detections, labels, shape_annotator, title: str):
    """Apply a shape annotator followed by labels."""
    label_annotator = sv.LabelAnnotator(text_scale=0.50, text_thickness=1)
    panel = shape_annotator.annotate(image.copy(), detections)
    panel = label_annotator.annotate(panel, detections, labels)
    return add_title(panel, title)


def main() -> None:
    """Run inference and save a four-style annotator comparison."""
    prepare_input()
    image = cv2.imread(str(INPUT_PATH))
    if image is None:
        raise FileNotFoundError(f"OpenCV could not read: {INPUT_PATH}")

    result = YOLO("yolov8n.pt")(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence >= CONFIDENCE_THRESHOLD]
    labels = create_labels(detections)

    panels = [
        annotate_panel(image, detections, labels, sv.BoxAnnotator(), "Boxes + Labels"),
        annotate_panel(
            image,
            detections,
            labels,
            sv.BoxCornerAnnotator(thickness=4, corner_length=30),
            "Corners + Labels",
        ),
        annotate_panel(image, detections, labels, sv.EllipseAnnotator(), "Ellipses + Labels"),
        annotate_panel(
            image,
            detections,
            labels,
            sv.DotAnnotator(position=sv.Position.BOTTOM_CENTER, radius=8),
            "Bottom-Center Dots",
        ),
    ]

    target_width = 640
    resized = [
        cv2.resize(panel, (target_width, int(panel.shape[0] * target_width / panel.shape[1])))
        for panel in panels
    ]
    gallery = cv2.vconcat([
        cv2.hconcat(resized[:2]),
        cv2.hconcat(resized[2:]),
    ])

    if not cv2.imwrite(str(OUTPUT_PATH), gallery):
        raise RuntimeError(f"OpenCV could not write: {OUTPUT_PATH}")

    print(f"Supervision version: {sv.__version__}")
    print(f"Detections retained: {len(detections)}")
    print(f"Saved annotator gallery: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
