"""Compare text, point, and YOLO bounding-box prompts on one image."""

import os

import cv2
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics.models.sam import SAM3SemanticPredictor

from common import DEFAULT_MODEL_PATH, OUTPUT_DIR, box_center, load_image, load_sam, point_detections, yolo_detections


def main() -> None:
    image = load_image()
    yolo = yolo_detections(image)
    if len(yolo) == 0:
        raise RuntimeError("YOLO found no objects.")
    model_path = os.getenv("SAM3_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    predictor = SAM3SemanticPredictor(overrides={"conf": 0.25, "task": "segment", "mode": "predict", "model": model_path})
    predictor.set_image(image)
    text = sv.Detections.from_ultralytics(predictor(text=["person"])[0])

    sam = load_sam()
    point = point_detections(sam, image, [box_center(yolo.xyxy[0])], [1])
    people = yolo[yolo.class_id == 0]
    if len(people) == 0:
        raise RuntimeError("YOLO found no person boxes.")
    bbox = sv.Detections.from_ultralytics(sam(image, bboxes=people.xyxy.tolist(), verbose=False)[0])

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, detections, title in zip(axes, [text, point, bbox], ["Text", "Point", "YOLO boxes"]):
        scene = sv.MaskAnnotator(opacity=0.6).annotate(image.copy(), detections)
        axis.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{title} — {len(detections)} object(s)")
        axis.axis("off")
    output = OUTPUT_DIR / "04_prompt_type_comparison_output.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
