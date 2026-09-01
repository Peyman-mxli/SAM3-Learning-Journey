"""Validate and finalize reviewed temporal tracking annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED = {
    "frame_index",
    "review_track_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "reviewed",
    "keep",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation/review_packet/tracking_review.csv",
    )
    parser.add_argument(
        "--output",
        default="evaluation/tracking_ground_truth.csv",
    )
    parser.add_argument("--require-all-reviewed", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    reviewed = df["reviewed"].fillna(False).astype(bool)
    keep = df["keep"].fillna(False).astype(bool)

    if args.require_all_reviewed and not bool(reviewed.all()):
        remaining = int((~reviewed).sum())
        raise RuntimeError(f"{remaining} candidate rows have not been reviewed.")

    accepted = df[reviewed & keep].copy()

    if accepted.empty:
        raise RuntimeError("There are no reviewed/accepted annotations.")

    if accepted["review_track_id"].isna().any():
        raise RuntimeError("Some accepted annotations have no review_track_id.")

    invalid_boxes = (
        (accepted["x2"] <= accepted["x1"])
        | (accepted["y2"] <= accepted["y1"])
    )
    if invalid_boxes.any():
        raise RuntimeError("One or more accepted boxes are invalid.")

    gt = accepted[
        [
            "frame_index",
            "review_track_id",
            "class_name",
            "x1",
            "y1",
            "x2",
            "y2",
            "notes",
        ]
    ].rename(columns={"review_track_id": "track_id"})
    gt["reviewed"] = True

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    gt.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Reviewed ground-truth rows: {len(gt)}")
    print(f"Unique temporal identities: {gt.track_id.nunique()}")


if __name__ == "__main__":
    main()
