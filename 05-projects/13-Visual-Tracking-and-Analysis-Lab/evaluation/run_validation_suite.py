"""Run the Project 13 tracking pipeline across all validation videos.

This utility executes each controlled validation video as a separate SQLite
session and exports its observations and tracker summary. It preserves the
condition label in the session notes so downstream comparison is reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.export_results import export_session
from src.pipeline import process_video


CONDITIONS = {
    "validation_01_baseline.mp4": "baseline",
    "validation_02_low_light.mp4": "low_light",
    "validation_03_partial_occlusion.mp4": "partial_occlusion",
    "validation_04_motion_blur.mp4": "motion_blur",
    "validation_05_reduced_scale.mp4": "reduced_scale",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/validation_videos")
    parser.add_argument("--db", default="data/project13_validation.sqlite3")
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--sam-prompt", default="person")
    parser.add_argument("--sam-every", type=int, default=10)
    parser.add_argument("--sam-device", default="cuda")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_map = []

    for filename, condition in CONDITIONS.items():
        source = video_dir / filename
        if not source.exists():
            raise FileNotFoundError(
                f"Missing {source}. Run evaluation/create_condition_variants.py first."
            )

        sid = process_video(
            source=str(source),
            database_path=args.db,
            model_name=args.model,
            confidence_threshold=args.confidence,
            notes=f"Project13 validation condition={condition}",
            sam_checkpoint=args.sam_checkpoint,
            sam_prompt=args.sam_prompt,
            sam_every=args.sam_every,
            sam_device=args.sam_device,
        )
        export_session(args.db, sid, str(output_dir))
        session_map.append((condition, sid, filename))
        print(f"[DONE] {condition}: {sid}")

    manifest = output_dir / "validation_sessions.csv"
    manifest.write_text(
        "condition,session_id,source_video\n"
        + "".join(f"{c},{sid},{name}\n" for c, sid, name in session_map),
        encoding="utf-8",
    )
    print(f"[OK] Saved {manifest}")


if __name__ == "__main__":
    main()
