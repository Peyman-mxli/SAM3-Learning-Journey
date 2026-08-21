"""
Performance Analysis
====================

Visual Tracking and Analysis System

This module analyzes the existing completed Project 06 reports and generates:

    reports/performance_summary.csv
    reports/performance_chart.png

The script intentionally uses the already-generated CSV reports instead of
requiring the original SQLite database or rerunning YOLO, ByteTrack, or SAM 3.

Required existing reports:

    reports/tracker_summary.csv
    reports/trajectory_summary.csv

Optional:
    You can provide the actual total processing duration with:

        python analytics/performance_analysis.py --processing-seconds 42.5

When processing time is provided, the script also calculates:

    - Effective processing FPS
    - Average processing time per frame
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "reports"

TRACKER_SUMMARY_CSV = REPORTS_DIR / "tracker_summary.csv"
TRAJECTORY_SUMMARY_CSV = REPORTS_DIR / "trajectory_summary.csv"

PERFORMANCE_CSV = REPORTS_DIR / "performance_summary.csv"
PERFORMANCE_CHART = REPORTS_DIR / "performance_chart.png"


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_required_files() -> None:
    """
    Confirm that the completed analytics CSV reports exist.
    """

    required_files = [
        TRACKER_SUMMARY_CSV,
        TRAJECTORY_SUMMARY_CSV,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:

        missing_names = "\n".join(
            f"- {path.relative_to(PROJECT_ROOT)}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "\nRequired analytics files were not found:\n"
            f"{missing_names}\n"
        )


# ============================================================
# DATA LOADING
# ============================================================

def load_reports() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the existing tracker and trajectory reports.
    """

    tracker_df = pd.read_csv(
        TRACKER_SUMMARY_CSV
    )

    trajectory_df = pd.read_csv(
        TRAJECTORY_SUMMARY_CSV
    )

    if tracker_df.empty:
        raise RuntimeError(
            "tracker_summary.csv is empty."
        )

    if trajectory_df.empty:
        raise RuntimeError(
            "trajectory_summary.csv is empty."
        )

    print(
        f"[INFO] Loaded tracker report: "
        f"{len(tracker_df)} tracker records"
    )

    print(
        f"[INFO] Loaded trajectory report: "
        f"{len(trajectory_df)} trajectory records"
    )

    return tracker_df, trajectory_df


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_columns(
    tracker_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
) -> None:
    """
    Validate the columns required for performance analysis.
    """

    required_tracker_columns = {
        "tracker_id",
        "first_frame",
        "last_frame",
        "observations",
        "duration_seconds",
        "average_confidence",
    }

    required_trajectory_columns = {
        "tracker_id",
        "movement_distance_pixels",
        "average_movement_pixels",
    }

    missing_tracker_columns = (
        required_tracker_columns
        - set(tracker_df.columns)
    )

    missing_trajectory_columns = (
        required_trajectory_columns
        - set(trajectory_df.columns)
    )

    if missing_tracker_columns:

        raise RuntimeError(
            "tracker_summary.csv is missing columns: "
            + ", ".join(
                sorted(missing_tracker_columns)
            )
        )

    if missing_trajectory_columns:

        raise RuntimeError(
            "trajectory_summary.csv is missing columns: "
            + ", ".join(
                sorted(missing_trajectory_columns)
            )
        )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    tracker_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    processing_seconds: float | None = None,
) -> dict[str, float | int | str]:
    """
    Calculate system-level metrics from the completed reports.
    """

    # --------------------------------------------------------
    # Core tracking metrics
    # --------------------------------------------------------

    total_frames = int(
        tracker_df["last_frame"].max()
    )

    total_observations = int(
        tracker_df["observations"].sum()
    )

    unique_tracker_ids = int(
        tracker_df["tracker_id"].nunique()
    )

    average_observations_per_frame = (
        total_observations / total_frames
        if total_frames > 0
        else 0
    )

    average_observations_per_tracker = float(
        tracker_df["observations"].mean()
    )

    minimum_tracker_observations = int(
        tracker_df["observations"].min()
    )

    maximum_tracker_observations = int(
        tracker_df["observations"].max()
    )

    # --------------------------------------------------------
    # Confidence metrics
    # --------------------------------------------------------

    average_confidence = float(
        tracker_df["average_confidence"].mean()
    )

    minimum_average_confidence = float(
        tracker_df["average_confidence"].min()
    )

    maximum_average_confidence = float(
        tracker_df["average_confidence"].max()
    )

    # --------------------------------------------------------
    # Tracker duration metrics
    # --------------------------------------------------------

    average_tracker_duration = float(
        tracker_df["duration_seconds"].mean()
    )

    minimum_tracker_duration = float(
        tracker_df["duration_seconds"].min()
    )

    maximum_tracker_duration = float(
        tracker_df["duration_seconds"].max()
    )

    # --------------------------------------------------------
    # Movement metrics
    # --------------------------------------------------------

    total_movement_distance = float(
        trajectory_df[
            "movement_distance_pixels"
        ].sum()
    )

    average_movement_distance = float(
        trajectory_df[
            "movement_distance_pixels"
        ].mean()
    )

    maximum_movement_distance = float(
        trajectory_df[
            "movement_distance_pixels"
        ].max()
    )

    average_step_movement = float(
        trajectory_df[
            "average_movement_pixels"
        ].mean()
    )

    # --------------------------------------------------------
    # Processing-performance metrics
    # --------------------------------------------------------

    total_processing_time: float | str = "N/A"
    average_processing_time: float | str = "N/A"
    effective_processing_fps: float | str = "N/A"

    if processing_seconds is not None:

        if processing_seconds <= 0:
            raise ValueError(
                "Processing time must be greater than zero."
            )

        total_processing_time = float(
            processing_seconds
        )

        average_processing_time = (
            processing_seconds / total_frames
            if total_frames > 0
            else 0
        )

        effective_processing_fps = (
            total_frames / processing_seconds
            if processing_seconds > 0
            else 0
        )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    metrics = {
        "Total Processed Frames":
            total_frames,

        "Total Observations":
            total_observations,

        "Unique Tracker IDs":
            unique_tracker_ids,

        "Average Observations per Frame":
            average_observations_per_frame,

        "Average Observations per Tracker":
            average_observations_per_tracker,

        "Minimum Tracker Observations":
            minimum_tracker_observations,

        "Maximum Tracker Observations":
            maximum_tracker_observations,

        "Average Confidence":
            average_confidence,

        "Minimum Average Confidence":
            minimum_average_confidence,

        "Maximum Average Confidence":
            maximum_average_confidence,

        "Average Tracker Duration (seconds)":
            average_tracker_duration,

        "Minimum Tracker Duration (seconds)":
            minimum_tracker_duration,

        "Maximum Tracker Duration (seconds)":
            maximum_tracker_duration,

        "Total Movement Distance (pixels)":
            total_movement_distance,

        "Average Movement Distance per Tracker (pixels)":
            average_movement_distance,

        "Maximum Movement Distance (pixels)":
            maximum_movement_distance,

        "Average Step Movement (pixels)":
            average_step_movement,

        "Total Processing Time (seconds)":
            total_processing_time,

        "Average Processing Time per Frame (seconds)":
            average_processing_time,

        "Effective Processing FPS":
            effective_processing_fps,
    }

    return metrics


# ============================================================
# CSV OUTPUT
# ============================================================

def save_performance_csv(
    metrics: dict[str, float | int | str],
) -> None:
    """
    Save system-level metrics to performance_summary.csv.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for metric_name, value in metrics.items():

        if isinstance(value, float):
            value = round(
                value,
                4,
            )

        rows.append(
            {
                "metric": metric_name,
                "value": value,
            }
        )

    performance_df = pd.DataFrame(
        rows
    )

    performance_df.to_csv(
        PERFORMANCE_CSV,
        index=False,
    )

    print(
        f"[SUCCESS] Saved: "
        f"{PERFORMANCE_CSV.relative_to(PROJECT_ROOT)}"
    )


# ============================================================
# PERFORMANCE CHART
# ============================================================

def save_performance_chart(
    metrics: dict[str, float | int | str],
) -> None:
    """
    Generate a visual summary of key system metrics.
    """

    chart_labels = [
        "Frames",
        "Observations",
        "Tracker IDs",
    ]

    chart_values = [
        metrics["Total Processed Frames"],
        metrics["Total Observations"],
        metrics["Unique Tracker IDs"],
    ]

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        chart_labels,
        chart_values,
    )

    plt.title(
        "Visual Tracking and Analysis System\n"
        "Performance Summary"
    )

    plt.xlabel(
        "Metric"
    )

    plt.ylabel(
        "Count"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    for bar, value in zip(
        bars,
        chart_values,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    plt.savefig(
        PERFORMANCE_CHART,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[SUCCESS] Saved: "
        f"{PERFORMANCE_CHART.relative_to(PROJECT_ROOT)}"
    )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

def print_metrics(
    metrics: dict[str, float | int | str],
) -> None:
    """
    Print calculated performance metrics.
    """

    print()
    print("=" * 72)
    print(
        "VISUAL TRACKING AND ANALYSIS SYSTEM"
    )
    print(
        "PERFORMANCE SUMMARY"
    )
    print("=" * 72)

    for metric_name, value in metrics.items():

        if isinstance(value, float):

            formatted_value = (
                f"{value:.4f}"
            )

        else:

            formatted_value = value

        print(
            f"{metric_name:<56} "
            f"{formatted_value}"
        )

    print("=" * 72)
    print()


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse optional command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate system-level performance analytics "
            "from the completed Project 06 reports."
        )
    )

    parser.add_argument(
        "--processing-seconds",
        type=float,
        default=None,
        help=(
            "Actual total processing duration in seconds. "
            "If provided, effective FPS and average "
            "processing time per frame will be calculated."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Execute the complete performance-analysis pipeline.
    """

    args = parse_arguments()

    print()
    print(
        "[INFO] Starting performance analysis..."
    )

    validate_required_files()

    tracker_df, trajectory_df = (
        load_reports()
    )

    validate_columns(
        tracker_df,
        trajectory_df,
    )

    metrics = calculate_metrics(
        tracker_df=tracker_df,
        trajectory_df=trajectory_df,
        processing_seconds=args.processing_seconds,
    )

    print_metrics(
        metrics
    )

    save_performance_csv(
        metrics
    )

    save_performance_chart(
        metrics
    )

    print()
    print(
        "[SUCCESS] Performance analysis completed."
    )
    print()


if __name__ == "__main__":
    main()
