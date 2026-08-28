"""Segment the first three YOLO detections using their centers as points."""

import cv2
import matplotlib.pyplot as plt
import supervision as sv

from common import OUTPUT_DIR, box_center, load_image, load_sam, point_detections, yolo_detections


def main() -> None:
    image = load_image()
    yolo = yolo_detections(image)
    object_count = min(3, len(yolo))
    if object_count == 0:
        raise RuntimeError("YOLO found no objects.")
    sam = load_sam()
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    for index, axis in enumerate(axes):
        if index >= object_count:
            axis.axis("off")
            continue
        center = box_center(yolo.xyxy[index])
        detections = point_detections(sam, image, [center], [1])
        scene = sv.MaskAnnotator(opacity=0.6).annotate(image.copy(), detections)
        cv2.circle(scene, center, 8, (0, 0, 255), -1)
        axis.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        axis.set_title(f"Object {index} — class {int(yolo.class_id[index])}")
        axis.axis("off")

    output = OUTPUT_DIR / "06_three_object_point_challenge_output.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Segmented objects: {object_count}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
