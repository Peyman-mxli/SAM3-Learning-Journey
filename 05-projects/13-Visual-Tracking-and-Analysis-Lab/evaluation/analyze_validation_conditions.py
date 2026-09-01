"""Analyze Project 13 controlled validation sessions.

This is a robustness/sensitivity analysis, not ground-truth tracking accuracy.
It compares observed system behavior under deterministic perturbations.

Outputs:
  results/validation/condition_summary.csv
  results/validation/condition_deltas.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_observations(path: Path) -> dict:
    df = pd.read_csv(path)

    if df.empty:
        return {
            "total_observations": 0,
            "unique_trackers": 0,
            "average_confidence": 0.0,
            "average_track_length": 0.0,
            "sam_mask_observations": 0,
        }

    valid_tracks = df.dropna(subset=["tracker_id"]).copy()

    track_lengths = (
        valid_tracks.groupby("tracker_id").size()
        if not valid_tracks.empty
        else pd.Series(dtype=float)
    )

    confidence = pd.to_numeric(df["confidence"], errors="coerce")
    mask_area = (
        pd.to_numeric(df["mask_area"], errors="coerce")
        if "mask_area" in df.columns
        else pd.Series(dtype=float)
    )

    return {
        "total_observations": int(len(df)),
        "unique_trackers": int(valid_tracks["tracker_id"].nunique()),
        "average_confidence": float(confidence.mean()),
        "average_track_length": (
            float(track_lengths.mean()) if not track_lengths.empty else 0.0
        ),
        "sam_mask_observations": int(mask_area.notna().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/validation")
    args = parser.parse_args()

    root = Path(args.results_dir)
    sessions = pd.read_csv(root / "validation_sessions.csv")

    rows = []

    for _, row in sessions.iterrows():
        observations_file = root / f"{row.session_id}_observations.csv"

        if not observations_file.exists():
            raise FileNotFoundError(observations_file)

        metrics = summarize_observations(observations_file)

        rows.append(
            {
                "condition": row.condition,
                "session_id": row.session_id,
                **metrics,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(root / "condition_summary.csv", index=False)

    baseline = summary.loc[summary["condition"] == "baseline"]

    if baseline.empty:
        raise RuntimeError("Baseline condition is missing.")

    b = baseline.iloc[0]
    deltas = summary.copy()

    for column in [
        "total_observations",
        "unique_trackers",
        "average_confidence",
        "average_track_length",
        "sam_mask_observations",
    ]:
        deltas[f"{column}_delta_vs_baseline"] = (
            deltas[column] - b[column]
        )

    deltas.to_csv(root / "condition_deltas.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print(
        "Important: these are robustness indicators, "
        "not ground-truth tracking accuracy metrics."
    )


if __name__ == "__main__":
    main()
