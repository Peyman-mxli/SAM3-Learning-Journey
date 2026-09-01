from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Extract a deterministic Project 13 ground-truth review set."
    )
    parser.add_argument("video")
    parser.add_argument("--output-dir", default="evaluation/ground_truth_frames")
    parser.add_argument("--every", type=int, default=3,
                        help="Extract every Nth frame. Default 3 => 25 frames from the 75-frame test video.")
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    if args.every < 1:
        raise ValueError("--every must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    manifest_path = output_dir / "manifest.csv"
    rows = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index >= args.start and (frame_index - args.start) % args.every == 0:
            filename = f"frame_{frame_index:04d}.jpg"
            path = output_dir / filename
            cv2.imwrite(str(path), frame)
            rows.append({
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / fps if fps else "",
                "image_file": filename,
                "review_status": "pending",
            })

        frame_index += 1

    cap.release()

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_index", "timestamp_seconds", "image_file", "review_status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    gt_template = output_dir / "ground_truth_template.csv"
    with gt_template.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index", "class_name", "x1", "y1", "x2", "y2",
            "track_id", "reviewed", "notes"
        ])

    print(f"Video frames reported: {total}")
    print(f"FPS: {fps}")
    print(f"Extracted review frames: {len(rows)}")
    print(f"Manifest: {manifest_path}")
    print(f"Ground-truth template: {gt_template}")


if __name__ == "__main__":
    main()
