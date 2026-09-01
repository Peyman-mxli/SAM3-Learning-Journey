"""Create deterministic Project 13 validation videos from one recorded input.

The official laboratory proposal requires 2–5 short recorded-video tests under
varied conditions. This utility creates four controlled variants without
changing the scene content:

1. baseline copy
2. low-light
3. partial occlusion
4. motion blur
5. reduced-scale reconstruction

The variants are intended for reproducible robustness testing. They are not
ground truth and must not be described as natural-world recordings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np


def _write_variant(source: Path, destination: Path, transform) -> dict:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {destination}")

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(transform(frame, frames))
        frames += 1

    cap.release()
    writer.release()

    return {
        "file": destination.name,
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
    }


def low_light(frame, _):
    return cv2.convertScaleAbs(frame, alpha=0.45, beta=-15)


def partial_occlusion(frame, frame_index):
    out = frame.copy()
    h, w = out.shape[:2]
    box_w = max(40, int(w * 0.22))
    box_h = max(40, int(h * 0.45))
    x = int((frame_index * 7) % max(1, w - box_w))
    y = int(h * 0.28)
    cv2.rectangle(out, (x, y), (x + box_w, min(h - 1, y + box_h)), (0, 0, 0), -1)
    return out


def motion_blur(frame, _):
    kernel_size = 11
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(frame, -1, kernel)


def reduced_scale(frame, _):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output-dir", default="data/validation_videos")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline = out / "validation_01_baseline.mp4"
    shutil.copyfile(source, baseline)

    variants = [
        ("validation_02_low_light.mp4", low_light),
        ("validation_03_partial_occlusion.mp4", partial_occlusion),
        ("validation_04_motion_blur.mp4", motion_blur),
        ("validation_05_reduced_scale.mp4", reduced_scale),
    ]

    print(f"[OK] {baseline}")
    for filename, transform in variants:
        info = _write_variant(source, out / filename, transform)
        print(f"[OK] {info}")

    print("\nCreated 5 reproducible validation videos.")
    print("These are controlled perturbations, not new natural-world scenes.")


if __name__ == "__main__":
    main()
