from __future__ import annotations

import csv
import json
from pathlib import Path


def export_run(payload: dict, json_path: str, csv_path: str) -> None:
    jp, cp = Path(json_path), Path(csv_path)
    jp.parent.mkdir(parents=True, exist_ok=True); cp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = ["detection_id", "label", "confidence", "x1", "y1", "x2", "y2", "pixel_area", "mask_path", "backend"]
    with cp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for d in payload["segmentation"]["detections"]:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            writer.writerow({**{k: d[k] for k in ("detection_id", "label", "confidence", "pixel_area", "mask_path")}, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "backend": payload["segmentation"]["backend"]})
