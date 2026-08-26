"""Session 08 practical: semantic segmentation with SAM 3 text prompts.

Run in Google Colab after mounting Drive. The SAM 3 checkpoint is expected at
/content/drive/MyDrive/SAM3-Models/sam3.pt unless --model is supplied.
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
from ultralytics.models.sam import SAM3SemanticPredictor


def build_predictor(model_path: str, confidence: float = 0.25):
    overrides = {"conf": confidence, "task": "segment", "mode": "predict", "model": model_path}
    if torch.cuda.is_available():
        overrides["half"] = True
    return SAM3SemanticPredictor(overrides=overrides)


def predict(predictor, image, prompt: str):
    predictor.set_image(image)
    result = predictor(text=[prompt])[0]
    return sv.Detections.from_ultralytics(result)


def annotate(image, detections, labels=None):
    scene = sv.MaskAnnotator(opacity=0.6).annotate(image.copy(), detections)
    if labels:
        scene = sv.LabelAnnotator().annotate(scene, detections, labels=labels)
    return scene


def save_prompt_comparison(predictor, image, output_path: Path):
    prompts = ["vehicle", "bus", "person", "wheel"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 14))
    counts = {}
    for ax, prompt in zip(axes.ravel(), prompts):
        detections = predict(predictor, image, prompt)
        counts[prompt] = len(detections)
        scene = annotate(image, detections)
        ax.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        ax.set_title(f'"{prompt}" — {len(detections)} objects')
        ax.axis("off")
    plt.suptitle("SAM3 Text-Prompt Segmentation", fontsize=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return counts


def save_filtered_people(predictor, image, output_path: Path):
    detections = predict(predictor, image, "person")
    areas = np.array([int(mask.sum()) for mask in detections.mask])
    keep = (detections.confidence >= 0.50) & (areas >= 1_000)
    filtered = detections[keep]
    labels = [f"person {confidence:.2f}" for confidence in filtered.confidence]
    cv2.imwrite(str(output_path), annotate(image, filtered, labels))
    return len(detections), len(filtered)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", default="/content/drive/MyDrive/SAM3-Models/sam3.pt")
    parser.add_argument("--image", default="assets/input/bus.jpg")
    parser.add_argument("--output", default="assets/output")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load input image: {args.image}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"SAM 3 checkpoint not found: {args.model}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictor = build_predictor(args.model)

    original, filtered = save_filtered_people(
        predictor, image, output / "sam3_person_text_prompt_filtered.jpg"
    )
    counts = save_prompt_comparison(
        predictor, image, output / "sam3_text_prompts_comparison.jpg"
    )

    print(f"Person detections: {original}; filtered: {filtered}")
    print("Prompt counts:", counts)
    print("Session 08 practical completed.")


if __name__ == "__main__":
    main()
