"""Artifact exporters for agent and segmentation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def write_detections_csv(path: str | Path, detections: list[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["detection_id", "label", "confidence", "x1", "y1", "x2", "y2", "pixel_area", "mask_path"]
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in detections:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            writer.writerow({
                "detection_id": item["detection_id"], "label": item["label"],
                "confidence": item["confidence"], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "pixel_area": item["pixel_area"], "mask_path": item["mask_path"],
            })
    return destination
