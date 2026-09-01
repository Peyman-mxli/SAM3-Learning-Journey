"""Analyze Project 13 validation sessions from exported tracker summaries.

This is a robustness/sensitivity analysis, not ground-truth accuracy. It
compares observed tracker behavior under deterministic perturbations.

Outputs:
  results/validation/condition_summary.csv
  results/validation/condition_deltas.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_tracker_file(path: Path) -> dict:
    df = pd.read_csv(path)
    return {
        "total_observations": int(df["observations"].sum()) if not df.empty else 0,
        "unique_trackers": int(df["tracker_id"].nunique()) if not df.empty else 0,
        "average_confidence": float(df["average_confidence"].mean()) if not df.empty else 0.0,
        "average_track_length": float(df["observations"].mean()) if not df.empty else 0.0,
        "sam_mask_tracks": int(df["average_mask_area"].notna().sum()) if "average_mask_area" in df else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/validation")
    args = parser.parse_args()

    root = Path(args.results_dir)
    sessions = pd.read_csv(root / "validation_sessions.csv")

    rows = []
    for _, row in sessions.iterrows():
        tracker_file = root / f"{row.session_id}_tracker_summary.csv"
        metrics = summarize_tracker_file(tracker_file)
        rows.append({"condition": row.condition, "session_id": row.session_id, **metrics})

    summary = pd.DataFrame(rows)
    summary.to_csv(root / "condition_summary.csv", index=False)

    baseline = summary.loc[summary["condition"] == "baseline"]
    if baseline.empty:
        raise RuntimeError("Baseline condition is missing.")

    b = baseline.iloc[0]
    deltas = summary.copy()
    for col in [
        "total_observations",
        "unique_trackers",
        "average_confidence",
        "average_track_length",
        "sam_mask_tracks",
    ]:
        deltas[f"{col}_delta_vs_baseline"] = deltas[col] - b[col]

    deltas.to_csv(root / "condition_deltas.csv", index=False)

    print(summary.to_string(index=False))
    print("\nImportant: these are robustness indicators, not ground-truth accuracy metrics.")


if __name__ == "__main__":
    main()
