"""Compare one, two, and three positive points inside one detected object."""

import cv2
import matplotlib.pyplot as plt
import supervision as sv

from common import OUTPUT_DIR, box_center, load_image, load_sam, point_detections, yolo_detections


def main() -> None:
    image = load_image()
    yolo = yolo_detections(image)
    if len(yolo) == 0:
        raise RuntimeError("YOLO found no objects.")
    x1, y1, x2, y2 = yolo.xyxy[0]
    center = box_center(yolo.xyxy[0])
    candidates = [
        center,
        [int(x1 + (x2 - x1) * 0.25), int(y1 + (y2 - y1) * 0.50)],
        [int(x1 + (x2 - x1) * 0.75), int(y1 + (y2 - y1) * 0.50)],
    ]
    sam = load_sam()
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for count, axis in enumerate(axes, start=1):
        points = candidates[:count]
        detections = point_detections(sam, image, points, [1] * count)
        scene = sv.MaskAnnotator(opacity=0.6).annotate(image.copy(), detections)
        for point in points:
            cv2.circle(scene, point, 8, (0, 0, 255), -1)
        axis.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{count} positive point(s)")
        axis.axis("off")
    output = OUTPUT_DIR / "02_multiple_positive_points_output.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
