"""Refine an object mask using one positive and one negative point."""

import cv2
import matplotlib.pyplot as plt
import supervision as sv

from common import OUTPUT_DIR, box_center, load_image, load_sam, point_detections, yolo_detections


def main() -> None:
    image = load_image()
    yolo = yolo_detections(image)
    if len(yolo) == 0:
        raise RuntimeError("YOLO found no objects.")
    positive = box_center(yolo.xyxy[0])
    negative = [max(0, positive[0] - 60), max(0, positive[1] - 80)]
    sam = load_sam()
    positive_only = point_detections(sam, image, [positive], [1])
    refined = point_detections(sam, image, [positive, negative], [1, 0])

    scenes = [
        sv.MaskAnnotator(opacity=0.6).annotate(image.copy(), positive_only),
        sv.MaskAnnotator(opacity=0.6).annotate(image.copy(), refined),
    ]
    cv2.circle(scenes[0], positive, 8, (0, 255, 0), -1)
    cv2.circle(scenes[1], positive, 8, (0, 255, 0), -1)
    cv2.circle(scenes[1], negative, 8, (0, 0, 255), -1)

    figure, axes = plt.subplots(1, 2, figsize=(18, 7))
    for axis, scene, title in zip(axes, scenes, ["Positive point", "Positive + negative"]):
        axis.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    output = OUTPUT_DIR / "03_positive_negative_refinement_output.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Positive point: {positive}; negative point: {negative}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
