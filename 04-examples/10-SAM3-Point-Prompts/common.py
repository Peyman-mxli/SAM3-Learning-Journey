"""Shared utilities for the SAM 3 point-prompt examples."""

import os
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import SAM, YOLO


BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "assets/input/bus.jpg"
OUTPUT_DIR = BASE_DIR / "assets/output"
DEFAULT_MODEL_PATH = Path("/content/drive/MyDrive/SAM3-Models/sam3.pt")


def load_image():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")
    return image


def load_sam() -> SAM:
    model_path = Path(os.getenv("SAM3_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.exists():
        raise FileNotFoundError(
            f"SAM 3 checkpoint not found: {model_path}. "
            "Set SAM3_MODEL_PATH to its location."
        )
    return SAM(str(model_path))


def yolo_detections(image):
    return sv.Detections.from_ultralytics(YOLO("yolov8n.pt")(image, verbose=False)[0])


def box_center(box) -> list[int]:
    x1, y1, x2, y2 = box
    return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


def point_detections(sam: SAM, image, points, labels):
    result = sam.predict(
        source=image,
        points=points,
        labels=labels,
        verbose=False,
    )[0]
    detections = sv.Detections.from_ultralytics(result)
    if detections.mask is None:
        raise RuntimeError("SAM 3 returned no segmentation masks.")
    return detections


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save output: {path}")
