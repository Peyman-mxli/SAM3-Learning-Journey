from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "frame_index", "class_name", "x1", "y1", "x2", "y2",
    "track_id", "reviewed", "notes"
]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(
        description="Create review candidates from Project 13 predictions. "
                    "These rows are NOT ground truth until a human reviews them."
    )
    parser.add_argument("observations_csv")
    parser.add_argument("manifest_csv")
    parser.add_argument("--output", default="evaluation/ground_truth_frames/review_candidates.csv")
    args = parser.parse_args()

    observations = read_csv(args.observations_csv)
    manifest = read_csv(args.manifest_csv)
    selected = {int(r["frame_index"]) for r in manifest}

    rows = []
    for r in observations:
        if int(float(r["frame_index"])) not in selected:
            continue
        rows.append({
            "frame_index": int(float(r["frame_index"])),
            "class_name": r["class_name"],
            "x1": r["x1"],
            "y1": r["y1"],
            "x2": r["x2"],
            "y2": r["y2"],
            "track_id": r.get("tracker_id", ""),
            "reviewed": "0",
            "notes": "MODEL CANDIDATE — verify, correct, add misses, remove false positives",
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Candidate rows: {len(rows)}")
    print(f"Output: {out}")
    print("IMPORTANT: This file is not ground truth until every row/frame is human-reviewed.")


if __name__ == "__main__":
    main()
