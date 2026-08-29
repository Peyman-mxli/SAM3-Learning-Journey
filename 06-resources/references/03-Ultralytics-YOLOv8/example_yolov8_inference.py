"""Run YOLOv8 detection and export visual plus structured results."""

import json
from pathlib import Path
from urllib.request import urlretrieve

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "assets" / "input" / "bus.jpg"
OUTPUT_IMAGE = ROOT / "assets" / "output" / "bus_yolov8.jpg"
OUTPUT_JSON = ROOT / "assets" / "output" / "bus_yolov8_detections.json"
IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


def main() -> None:
    INPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        urlretrieve(IMAGE_URL, INPUT)

    image = cv2.imread(str(INPUT))
    if image is None:
        raise FileNotFoundError(f"Could not read {INPUT}")

    result = YOLO("yolov8n.pt").predict(image, conf=0.50, imgsz=640, verbose=False)[0]
    records = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        records.append({
            "xyxy": [round(float(value), 2) for value in box.xyxy[0].cpu().tolist()],
            "confidence": round(float(box.conf.item()), 4),
            "class_id": class_id,
            "class_name": result.names[class_id],
        })

    if not cv2.imwrite(str(OUTPUT_IMAGE), result.plot()):
        raise RuntimeError(f"Could not write {OUTPUT_IMAGE}")
    OUTPUT_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(f"Detections: {len(records)}")
    print(f"Image: {OUTPUT_IMAGE}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
