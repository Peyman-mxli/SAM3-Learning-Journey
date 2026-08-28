"""Discover people with text, then refine one instance with a point."""

import os

import cv2
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics.models.sam import SAM3SemanticPredictor

from common import DEFAULT_MODEL_PATH, OUTPUT_DIR, box_center, load_image, load_sam, point_detections


def main() -> None:
    image = load_image()
    model_path = os.getenv("SAM3_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    predictor = SAM3SemanticPredictor(overrides={"conf": 0.25, "task": "segment", "mode": "predict", "model": model_path})
    predictor.set_image(image)
    discovered = sv.Detections.from_ultralytics(predictor(text=["person"])[0])
    if len(discovered) == 0:
        raise RuntimeError("The text prompt found no people.")
    center = box_center(discovered.xyxy[0])
    refined = point_detections(load_sam(), image, [center], [1])

    scenes = [
        sv.MaskAnnotator(opacity=0.4).annotate(image.copy(), discovered),
        sv.MaskAnnotator(opacity=0.7).annotate(image.copy(), refined),
    ]
    cv2.circle(scenes[1], center, 10, (0, 0, 255), -1)
    figure, axes = plt.subplots(1, 2, figsize=(18, 7))
    for axis, scene, title in zip(axes, scenes, [f'Text "person": {len(discovered)}', "Point-refined instance"]):
        axis.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    output = OUTPUT_DIR / "05_text_to_point_refinement_output.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Text discovered {len(discovered)} people; refined center: {center}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
