"""Four-point football-pitch homography demonstrated in Session 11.

Select four pitch corners interactively or use the included default points,
then transform the perspective view into a normalized top-down field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_POINTS = np.array(
    [[240, 90], [1040, 90], [1220, 650], [60, 650]], dtype=np.float32
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
    return image


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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "football_pitch_four_points.png"), annotated)
    cv2.imwrite(str(args.output_dir / "football_pitch_top_down.png"), transformed)
    (args.output_dir / "homography_matrix.json").write_text(
        json.dumps(
            {
                "source_points": points.tolist(),
                "matrix": matrix.tolist(),
                "target_size": [args.width, args.height],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved football-pitch homography outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
