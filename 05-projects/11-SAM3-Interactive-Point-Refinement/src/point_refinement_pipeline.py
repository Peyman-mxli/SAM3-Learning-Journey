"""Run an analytical SAM 3 point-prompt refinement pipeline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import SAM, YOLO


DEFAULT_SAM_PATH = Path("/content/drive/MyDrive/SAM3-Models/sam3.pt")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict) -> None:
    if config["max_objects"] < 1:
        raise ValueError("max_objects must be positive")
    if len(config["negative_point_offset"]) != 2:
        raise ValueError("negative_point_offset must contain [x, y]")
    if not 0 <= config["mask_opacity"] <= 1:
        raise ValueError("mask_opacity must be between 0 and 1")


def load_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read input image: {path}")
    return image


def box_center(box) -> list[int]:
    x1, y1, x2, y2 = box
    return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


def predict_points(sam: SAM, image, points: list[list[int]], labels: list[int]) -> sv.Detections:
    result = sam.predict(source=image, points=points, labels=labels, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    if detections.mask is None or len(detections) == 0:
        raise RuntimeError(f"SAM returned no mask for points: {points}")
    return detections


def mask_metrics(detections: sv.Detections) -> tuple[int, float]:
    area = int(detections.mask[0].sum())
    confidence = (
        float(detections.confidence[0])
        if detections.confidence is not None
        else 0.0
    )
    return area, confidence


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_gallery(path: Path, image, records: list[dict], opacity: float) -> None:
    figure, axes = plt.subplots(1, len(records), figsize=(6 * len(records), 5))
    if len(records) == 1:
        axes = [axes]
    annotator = sv.MaskAnnotator(opacity=opacity)
    for axis, record in zip(axes, records):
        scene = annotator.annotate(image.copy(), record["detections"])
        cv2.circle(scene, record["point"], 9, (0, 0, 255), -1)
        axis.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        axis.set_title(
            f"Object {record['object_index']} | class {record['class_id']}\n"
            f"area={record['mask_area_pixels']:,} px² | conf={record['confidence']:.3f}"
        )
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_refinement(path: Path, image, positive, refined, positive_point, negative_point, opacity: float) -> None:
    annotator = sv.MaskAnnotator(opacity=opacity)
    scenes = [
        annotator.annotate(image.copy(), positive),
        annotator.annotate(image.copy(), refined),
    ]
    cv2.circle(scenes[0], positive_point, 9, (0, 255, 0), -1)
    cv2.circle(scenes[1], positive_point, 9, (0, 255, 0), -1)
    cv2.circle(scenes[1], negative_point, 9, (0, 0, 255), -1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, scene, title in zip(axes, scenes, ["Positive point", "Positive + negative refinement"]):
        axis.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--config", type=Path, default=root / "config/pipeline.json")
    parser.add_argument("--input", type=Path, default=root / "data/input/bus.jpg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    validate_config(config)
    image = load_image(args.input)
    model_path = Path(os.getenv("SAM3_MODEL_PATH", str(DEFAULT_SAM_PATH)))
    if not model_path.exists():
        raise FileNotFoundError(f"SAM 3 checkpoint not found: {model_path}")

    yolo = sv.Detections.from_ultralytics(
        YOLO(config["yolo_model"])(image, verbose=False)[0]
    )
    object_count = min(config["max_objects"], len(yolo))
    if object_count == 0:
        raise RuntimeError("YOLO discovered no objects")

    sam = SAM(str(model_path))
    records = []
    for index in range(object_count):
        point = box_center(yolo.xyxy[index])
        detections = predict_points(sam, image, [point], [1])
        area, confidence = mask_metrics(detections)
        records.append(
            {
                "object_index": index,
                "class_id": int(yolo.class_id[index]),
                "point_x": point[0],
                "point_y": point[1],
                "mask_area_pixels": area,
                "confidence": confidence,
                "point": point,
                "detections": detections,
            }
        )

    first = records[0]
    dx, dy = config["negative_point_offset"]
    negative_point = [max(0, first["point_x"] + dx), max(0, first["point_y"] + dy)]
    refined = predict_points(sam, image, [first["point"], negative_point], [1, 0])
    refined_area, refined_confidence = mask_metrics(refined)

    output_dir = args.project_root / "data/output"
    save_gallery(output_dir / "bus_point_prompt_gallery.png", image, records, config["mask_opacity"])
    save_refinement(
        output_dir / "bus_point_refinement_comparison.png",
        image,
        first["detections"],
        refined,
        first["point"],
        negative_point,
        config["mask_opacity"],
    )

    object_rows = [
        {key: value for key, value in record.items() if key not in {"point", "detections"}}
        for record in records
    ]
    refinement = {
        "object_index": first["object_index"],
        "positive_point_x": first["point_x"],
        "positive_point_y": first["point_y"],
        "negative_point_x": negative_point[0],
        "negative_point_y": negative_point[1],
        "positive_area_pixels": first["mask_area_pixels"],
        "refined_area_pixels": refined_area,
        "area_change_pixels": refined_area - first["mask_area_pixels"],
        "positive_confidence": first["confidence"],
        "refined_confidence": refined_confidence,
    }
    write_csv(args.project_root / "results/csv/object_point_metrics.csv", object_rows)
    write_csv(args.project_root / "results/csv/refinement_summary.csv", [refinement])

    report = {
        "project": "11-SAM3-Interactive-Point-Refinement",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": args.input.name,
        "configuration": config,
        "objects_analyzed": object_count,
        "objects": object_rows,
        "refinement": refinement,
    }
    report_path = args.project_root / "results/json/point_prompt_analysis.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Project 11 — SAM3 Interactive Point Refinement")
    print(f"Objects analyzed: {object_count}")
    for row in object_rows:
        print(
            f"Object {row['object_index']}: class={row['class_id']}, "
            f"point=({row['point_x']}, {row['point_y']}), "
            f"area={row['mask_area_pixels']:,} px², confidence={row['confidence']:.3f}"
        )
    print(f"Refined area change: {refinement['area_change_pixels']:,} px²")
    print("Project 11 point-refinement pipeline completed.")


if __name__ == "__main__":
    main()
