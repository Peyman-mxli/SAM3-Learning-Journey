"""Four-point football-pitch homography demonstrated in Session 11.

Select four pitch corners interactively or use the included default points,
then transform the perspective view into a normalized top-down field.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_POINTS = np.array(
    [[240, 90], [1040, 90], [1220, 650], [60, 650]], dtype=np.float32
)
DEMO_BOXES = np.array(
    [[300, 235, 370, 390], [685, 175, 755, 330], [880, 390, 955, 565]],
    dtype=np.float32,
)


def create_demo_pitch(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create an original perspective football-pitch image."""
    image = np.full((height, width, 3), (46, 118, 38), dtype=np.uint8)
    for x in range(0, width, 80):
        color = (49, 126, 41) if (x // 80) % 2 == 0 else (42, 110, 35)
        cv2.rectangle(image, (x, 0), (x + 80, height), color, -1)

    pitch = DEFAULT_POINTS.astype(np.int32)
    cv2.polylines(image, [pitch], True, (245, 245, 245), 6, cv2.LINE_AA)
    top_mid = ((pitch[0] + pitch[1]) // 2).tolist()
    bottom_mid = ((pitch[3] + pitch[2]) // 2).tolist()
    cv2.line(image, top_mid, bottom_mid, (245, 245, 245), 4, cv2.LINE_AA)
    center = tuple(((pitch[0] + pitch[1] + pitch[2] + pitch[3]) // 4).tolist())
    cv2.circle(image, center, 75, (245, 245, 245), 4, cv2.LINE_AA)
    cv2.circle(image, center, 6, (245, 245, 245), -1, cv2.LINE_AA)
    robot_colors = ((40, 190, 255), (255, 120, 45), (195, 80, 230))
    for box, color in zip(DEMO_BOXES.astype(np.int32), robot_colors):
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        cv2.circle(image, ((x1 + x2) // 2, y1 - 18), 20, color, -1, cv2.LINE_AA)
    cv2.circle(image, (805, 470), 14, (0, 0, 230), -1, cv2.LINE_AA)
    return image


def detect_objects(
    image: np.ndarray,
    model_name: str,
    class_id: int,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run YOLO and return filtered boxes, confidence values, and labels."""
    result = YOLO(model_name)(image, conf=confidence, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    keep = classes == class_id
    filtered_boxes = boxes[keep].astype(np.float32)
    filtered_confidences = confidences[keep].astype(np.float32)
    labels = [result.names[int(value)] for value in classes[keep]]
    return filtered_boxes, filtered_confidences, labels


def bottom_center_points(boxes: np.ndarray) -> np.ndarray:
    """Convert XYXY boxes into ground-contact points at the feet."""
    if len(boxes) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack(
        ((boxes[:, 0] + boxes[:, 2]) * 0.5, boxes[:, 3])
    ).astype(np.float32)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply homography H to an arbitrary set of 2D points."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
    labels: list[str],
    anchors: np.ndarray | None = None,
) -> np.ndarray:
    """Draw boxes, IDs, confidence, and optional foot-anchor points."""
    annotated = image.copy()
    for index, box in enumerate(boxes.astype(np.int32)):
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 215, 255), 3)
        confidence = float(confidences[index]) if index < len(confidences) else 1.0
        label = labels[index] if index < len(labels) else "object"
        cv2.putText(
            annotated,
            f"ID {index + 1} {label} {confidence:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if anchors is not None:
            point = tuple(np.rint(anchors[index]).astype(int).tolist())
            cv2.circle(annotated, point, 10, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(annotated, point, 15, (255, 255, 255), 3, cv2.LINE_AA)
    return annotated


def create_minimap(
    width: int,
    height: int,
    transformed_points: np.ndarray,
) -> np.ndarray:
    """Draw a clean top-down football minimap with transformed positions."""
    minimap = np.full((height, width, 3), (44, 128, 52), dtype=np.uint8)
    cv2.rectangle(minimap, (5, 5), (width - 6, height - 6), (255, 255, 255), 4)
    cv2.line(minimap, (width // 2, 5), (width // 2, height - 6), (255, 255, 255), 3)
    cv2.circle(minimap, (width // 2, height // 2), min(width, height) // 8, (255, 255, 255), 3)
    cv2.rectangle(minimap, (5, height // 3), (width // 6, 2 * height // 3), (255, 255, 255), 3)
    cv2.rectangle(minimap, (5 * width // 6, height // 3), (width - 6, 2 * height // 3), (255, 255, 255), 3)
    colors = ((0, 215, 255), (255, 120, 45), (195, 80, 230), (50, 220, 80))
    for index, point in enumerate(transformed_points):
        x = int(np.clip(round(float(point[0])), 0, width - 1))
        y = int(np.clip(round(float(point[1])), 0, height - 1))
        color = colors[index % len(colors)]
        cv2.circle(minimap, (x, y), 13, color, -1, cv2.LINE_AA)
        cv2.circle(minimap, (x, y), 17, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(minimap, str(index + 1), (x + 19, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return minimap


def save_coordinate_evidence(
    output_dir: Path,
    matrix: np.ndarray,
    image_points: np.ndarray,
    field_points: np.ndarray,
) -> None:
    """Save H and original/transformed coordinates as JSON and CSV."""
    (output_dir / "homography_matrix.json").write_text(
        json.dumps(
            {
                "matrix_h": matrix.tolist(),
                "source_points": DEFAULT_POINTS.tolist(),
                "image_anchor_points": image_points.tolist(),
                "field_points": field_points.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "transformed_coordinates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(("object_id", "image_x", "image_y", "field_x", "field_y"))
        for index, (source, target) in enumerate(zip(image_points, field_points), 1):
            writer.writerow(
                (
                    index,
                    f"{source[0]:.2f}",
                    f"{source[1]:.2f}",
                    f"{target[0]:.2f}",
                    f"{target[1]:.2f}",
                )
            )


def select_four_points(image: np.ndarray) -> np.ndarray:
    """Collect four points in order: TL, TR, BR, BL."""
    selected: list[tuple[int, int]] = []
    canvas = image.copy()

    def on_click(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(selected) >= 4:
            return
        selected.append((x, y))
        cv2.circle(canvas, (x, y), 10, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(len(selected)),
            (x + 14, y - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    window = "Select TL, TR, BR, BL — press Enter after four points"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_click)
    while len(selected) < 4:
        cv2.imshow(window, canvas)
        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Point selection cancelled.")
    cv2.imshow(window, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.asarray(selected, dtype=np.float32)


def draw_selected_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Draw the four ordered points and connect the selected quadrilateral."""
    annotated = image.copy()
    integer_points = points.astype(np.int32)
    cv2.polylines(
        annotated, [integer_points], True, (0, 215, 255), 4, cv2.LINE_AA
    )
    names = ("1 · TL", "2 · TR", "3 · BR", "4 · BL")
    for point, name in zip(integer_points, names):
        x, y = point.tolist()
        cv2.circle(annotated, (x, y), 12, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, (x, y), 16, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(
            annotated,
            name,
            (x + 20, y - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def transform_pitch(
    image: np.ndarray,
    source_points: np.ndarray,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the selected quadrilateral to a top-down rectangle."""
    target_points = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source_points, target_points)
    transformed = cv2.warpPerspective(
        image, matrix, (output_width, output_height)
    )
    return transformed, matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select four football-pitch points and apply homography."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/output"))
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=1050)
    parser.add_argument("--height", type=int, default=680)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        image = cv2.imread(str(args.input))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.input}")
    else:
        image = create_demo_pitch()

    points = select_four_points(image) if args.interactive else DEFAULT_POINTS.copy()
    annotated = draw_selected_points(image, points)
    transformed, matrix = transform_pitch(
        image, points, args.width, args.height
    )

    if args.detect:
        boxes, confidences, labels = detect_objects(
            image, args.yolo_model, args.class_id, args.confidence
        )
    else:
        boxes = DEMO_BOXES.copy()
        confidences = np.array([0.98, 0.96, 0.94], dtype=np.float32)
        labels = ["robot", "robot", "robot"]

    anchors = bottom_center_points(boxes)
    field_points = transform_points(anchors, matrix)
    detected = draw_detections(image, boxes, confidences, labels)
    anchored = draw_detections(image, boxes, confidences, labels, anchors)
    minimap = create_minimap(args.width, args.height, field_points)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "01_original_field.png"), image)
    cv2.imwrite(str(args.output_dir / "02_four_selected_points.png"), annotated)
    cv2.imwrite(str(args.output_dir / "03_top_down_field.png"), transformed)
    cv2.imwrite(str(args.output_dir / "04_detected_players.png"), detected)
    cv2.imwrite(str(args.output_dir / "05_player_anchor_points.png"), anchored)
    cv2.imwrite(str(args.output_dir / "06_football_minimap.png"), minimap)
    save_coordinate_evidence(args.output_dir, matrix, anchors, field_points)
    print(f"Saved football-pitch homography outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
