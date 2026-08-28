"""Segment one object using a positive point at its YOLO box center."""

import cv2
import supervision as sv

from common import OUTPUT_DIR, box_center, load_image, load_sam, point_detections, save_image, yolo_detections


def main() -> None:
    image = load_image()
    yolo = yolo_detections(image)
    if len(yolo) == 0:
        raise RuntimeError("YOLO found no objects from which to select a point.")

    point = box_center(yolo.xyxy[0])
    detections = point_detections(load_sam(), image, [point], [1])
    scene = sv.MaskAnnotator(opacity=0.6).annotate(image.copy(), detections)
    cv2.circle(scene, point, 10, (0, 0, 255), -1)
    output = OUTPUT_DIR / "01_basic_positive_point_output.png"
    save_image(output, scene)

    print(f"Positive point: {point}")
    print(f"Masks generated: {len(detections)}")
    for index, mask in enumerate(detections.mask):
        confidence = detections.confidence[index] if detections.confidence is not None else 0.0
        print(f"Mask {index}: area={int(mask.sum()):,} px², confidence={confidence:.3f}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
