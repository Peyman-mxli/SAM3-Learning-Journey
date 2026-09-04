"""Class 13 — Native SAM 3 with Hugging Face

Execution-ready practical runner for Google Colab or another GPU environment.

Requirements:
- Approved access to https://huggingface.co/facebook/sam3
- An authenticated Hugging Face session
- Python packages: transformers, torch, supervision, ultralytics,
  opencv-python, numpy, matplotlib, Pillow, huggingface_hub

This script never stores a Hugging Face token in the repository.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
OUTPUTS = ROOT / "outputs"
ASSETS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

BUS_URL = "https://ultralytics.com/images/bus.jpg"
BUS_PATH = ASSETS / "bus.jpg"


def download_asset(url: str, destination: Path) -> None:
    if not destination.exists():
        print(f"Downloading {url} -> {destination}")
        urllib.request.urlretrieve(url, destination)


def sam3_to_detections(results: dict) -> sv.Detections:
    masks = results["masks"].detach().cpu().numpy().astype(bool)
    xyxy = results["boxes"].detach().cpu().numpy()
    scores = results["scores"].detach().cpu().numpy()
    return sv.Detections(xyxy=xyxy, mask=masks, confidence=scores)


def annotate(image_bgr: np.ndarray, detections: sv.Detections, labels: bool = True) -> np.ndarray:
    mask_annotator = sv.MaskAnnotator(opacity=0.6, color_lookup=sv.ColorLookup.INDEX)
    scene = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    if labels and detections.confidence is not None:
        label_annotator = sv.LabelAnnotator(text_scale=0.5, color_lookup=sv.ColorLookup.INDEX)
        text = [f"{score:.2f}" for score in detections.confidence]
        scene = label_annotator.annotate(scene=scene, detections=detections, labels=text)
    return scene


def save_image(path: Path, image_bgr: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise RuntimeError(f"Could not save {path}")
    print(f"Saved: {path}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device != "cuda":
        print("WARNING: CUDA is not available. SAM 3 may be very slow or exceed system memory on CPU.")

    download_asset(BUS_URL, BUS_PATH)
    image_pil = Image.open(BUS_PATH).convert("RGB")
    image_bgr = cv2.imread(str(BUS_PATH))
    if image_bgr is None:
        raise RuntimeError(f"Could not read {BUS_PATH}")

    print("Loading facebook/sam3 from Hugging Face...")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    model.eval()

    parameter_count_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {parameter_count_m:.0f} M parameters")

    # Experiment 1 — text prompt: person
    inputs = processor(images=image_pil, text="person", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=[image_pil.size[::-1]],
    )[0]
    detections = sam3_to_detections(results)
    person_scene = annotate(image_bgr, detections)
    save_image(OUTPUTS / "person_text_prompt.jpg", person_scene)

    # Experiment 2 — multiple semantic concepts
    inputs_multi = processor(
        images=image_pil,
        text=[["person", "vehicle"]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs_multi = model(**inputs_multi)

    results_multi = processor.post_process_instance_segmentation(
        outputs_multi,
        threshold=0.3,
        mask_threshold=0.3,
        target_sizes=[image_pil.size[::-1]],
    )[0]
    det_multi = sam3_to_detections(results_multi)
    multi_scene = annotate(image_bgr, det_multi, labels=False)
    save_image(OUTPUTS / "multi_concept_prompt.jpg", multi_scene)

    # Experiment 3 — YOLO bounding boxes -> SAM 3
    yolo_model = YOLO("yolov8n.pt")
    yolo_result = yolo_model(image_bgr, verbose=False)[0]
    yolo_det = sv.Detections.from_ultralytics(yolo_result)
    boxes_list = yolo_det.xyxy.tolist()

    if not boxes_list:
        raise RuntimeError("YOLO returned no boxes; bounding-box experiment cannot continue.")

    inputs_bbox = processor(
        images=image_pil,
        input_boxes=[boxes_list],
        input_boxes_labels=[[1] * len(boxes_list)],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs_bbox = model(**inputs_bbox)

    results_bbox = processor.post_process_instance_segmentation(
        outputs_bbox,
        target_sizes=[image_pil.size[::-1]],
    )[0]
    det_bbox = sam3_to_detections(results_bbox)
    bbox_scene = annotate(image_bgr, det_bbox, labels=False)
    save_image(OUTPUTS / "yolo_bbox_prompt.jpg", bbox_scene)

    # Side-by-side text vs bbox comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].imshow(cv2.cvtColor(person_scene, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Text prompt: "person"')
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(bbox_scene, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Bounding-box prompt (YOLO)")
    axes[1].axis("off")
    fig.tight_layout()
    comparison_path = OUTPUTS / "text_vs_bbox_comparison.jpg"
    fig.savefig(comparison_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {comparison_path}")

    # Experiment 4 — confidence threshold comparison
    thresholds = [0.2, 0.5, 0.8]
    threshold_counts: dict[str, int] = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, threshold in zip(axes, thresholds):
        result_threshold = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=[image_pil.size[::-1]],
        )[0]
        det_threshold = sam3_to_detections(result_threshold)
        threshold_counts[str(threshold)] = len(det_threshold)
        scene = annotate(image_bgr, det_threshold, labels=False)
        ax.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        ax.set_title(f"threshold={threshold} ({len(det_threshold)} objects)")
        ax.axis("off")

    fig.tight_layout()
    threshold_path = OUTPUTS / "threshold_comparison.jpg"
    fig.savefig(threshold_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {threshold_path}")

    summary = {
        "model": "facebook/sam3",
        "device": device,
        "model_parameters_millions": round(parameter_count_m, 2),
        "image": "assets/bus.jpg",
        "text_prompt": "person",
        "text_prompt_detections": len(detections),
        "multi_concept_prompt": ["person", "vehicle"],
        "multi_concept_detections": len(det_multi),
        "yolo_detections": len(yolo_det),
        "bbox_prompt_detections": len(det_bbox),
        "threshold_detection_counts": threshold_counts,
        "outputs": [
            "outputs/person_text_prompt.jpg",
            "outputs/multi_concept_prompt.jpg",
            "outputs/yolo_bbox_prompt.jpg",
            "outputs/text_vs_bbox_comparison.jpg",
            "outputs/threshold_comparison.jpg",
        ],
    }

    summary_path = OUTPUTS / "execution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
