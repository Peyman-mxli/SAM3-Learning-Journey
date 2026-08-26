from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "data" / "input"
OUTPUT_DIR = PROJECT_DIR / "data" / "output"
JSON_DIR = PROJECT_DIR / "results" / "json"
CSV_DIR = PROJECT_DIR / "results" / "csv"
CONFIG_PATH = PROJECT_DIR / "config" / "prompts.json"
DEFAULT_MODEL = Path("/content/drive/MyDrive/SAM3-Models/sam3.pt")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 semantic prompt analytics pipeline")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def prepare_directories():
    for directory in (OUTPUT_DIR, JSON_DIR, CSV_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_config(path: Path):
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not config.get("prompts"):
        raise ValueError("Configuration must contain at least one prompt.")
    return config


def discover_images():
    images = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No supported images found in {INPUT_DIR}")
    return images


def build_predictor(model_path: Path, model_confidence: float):
    if not model_path.exists():
        raise FileNotFoundError(f"SAM 3 checkpoint not found: {model_path}")
    overrides = {
        "conf": model_confidence,
        "task": "segment",
        "mode": "predict",
        "model": str(model_path),
    }
    if torch.cuda.is_available():
        overrides["half"] = True
    return SAM3SemanticPredictor(overrides=overrides)


def analyze_prompt(predictor, image, image_name, prompt, confidence_min, area_min):
    result = predictor(text=[prompt])[0]
    detections = sv.Detections.from_ultralytics(result)
    if detections.mask is None:
        raise RuntimeError(f"No masks returned for prompt: {prompt}")

    records = []
    for index, mask in enumerate(detections.mask):
        confidence = float(detections.confidence[index])
        mask_area = int(np.count_nonzero(mask))
        box = detections.xyxy[index]
        records.append({
            "image": image_name,
            "prompt": prompt,
            "object_id": index + 1,
            "confidence": round(confidence, 4),
            "mask_area": mask_area,
            "x1": round(float(box[0]), 2),
            "y1": round(float(box[1]), 2),
            "x2": round(float(box[2]), 2),
            "y2": round(float(box[3]), 2),
            "reliable": confidence >= confidence_min and mask_area >= area_min,
        })
    return detections, records


def create_filtered_visualization(image, detections, records):
    keep = np.array([record["reliable"] for record in records], dtype=bool)
    filtered = detections[keep]
    labels = [f'{record["prompt"]} {record["confidence"]:.2f}' for record in records if record["reliable"]]
    scene = sv.MaskAnnotator(opacity=0.6).annotate(scene=image.copy(), detections=filtered)
    return sv.LabelAnnotator().annotate(scene=scene, detections=filtered, labels=labels)


def save_comparison(image, prompt_results, output_path):
    columns = 2
    rows = int(np.ceil(len(prompt_results) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 7 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (prompt, detections) in zip(axes, prompt_results.items()):
        scene = sv.MaskAnnotator(opacity=0.6).annotate(scene=image.copy(), detections=detections)
        ax.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        ax.set_title(f'"{prompt}" — {len(detections)} objects')
        ax.axis("off")
    for ax in axes[len(prompt_results):]:
        ax.axis("off")
    plt.suptitle("SAM3 Semantic Prompt Analytics", fontsize=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_json(image_name, records, summaries):
    path = JSON_DIR / f"{Path(image_name).stem}_semantic_analysis.json"
    payload = {"image": image_name, "prompt_summary": summaries, "detections": records}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_image(image_path, predictor, config):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not load: {image_path}")
    predictor.set_image(image)

    all_records, summaries, prompt_results = [], [], {}
    for prompt in config["prompts"]:
        detections, records = analyze_prompt(
            predictor, image, image_path.name, prompt,
            config["confidence_min"], config["area_min"],
        )
        prompt_results[prompt] = detections
        all_records.extend(records)
        summaries.append({
            "image": image_path.name,
            "prompt": prompt,
            "objects": len(records),
            "reliable_objects": sum(record["reliable"] for record in records),
        })

        if prompt == config.get("filtered_prompt"):
            filtered_scene = create_filtered_visualization(image, detections, records)
            cv2.imwrite(str(OUTPUT_DIR / f"{image_path.stem}_filtered_{prompt}.png"), filtered_scene)

    save_comparison(image, prompt_results, OUTPUT_DIR / f"{image_path.stem}_prompt_comparison.png")
    write_json(image_path.name, all_records, summaries)
    write_csv(
        CSV_DIR / f"{image_path.stem}_detections.csv", all_records,
        ["image", "prompt", "object_id", "confidence", "mask_area", "x1", "y1", "x2", "y2", "reliable"],
    )
    write_csv(
        CSV_DIR / f"{image_path.stem}_prompt_summary.csv", summaries,
        ["image", "prompt", "objects", "reliable_objects"],
    )
    return len(all_records), len(summaries)


def main():
    args = parse_args()
    prepare_directories()
    config = load_config(args.config)
    predictor = build_predictor(args.model, config["model_confidence"])
    images = discover_images()
    total_records = 0
    for image_path in images:
        records, prompt_count = process_image(image_path, predictor, config)
        total_records += records
        print(f"{image_path.name}: {prompt_count} prompts, {records} object records")
    print(f"Images processed: {len(images)}")
    print(f"Object records generated: {total_records}")
    print("Project 09 semantic prompt analytics completed.")


if __name__ == "__main__":
    main()
