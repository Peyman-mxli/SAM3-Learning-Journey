"""
Session Comparison
==================

Visual Tracking and Analysis System

This module reads the preserved historical session registry from:

    data/session_history.csv

and generates session-level comparison outputs for Project 06.

Current outputs:

    reports/session_comparison_summary.csv
    reports/session_comparison_chart.png

The first session uses the verified Project 06 results already produced by
the tracking and analytics pipeline.

The module is designed to support additional future sessions without
rerunning or modifying previous verified results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

SESSION_HISTORY_CSV = DATA_DIR / "session_history.csv"

SESSION_COMPARISON_CSV = (
    REPORTS_DIR / "session_comparison_summary.csv"
)

SESSION_COMPARISON_CHART = (
    REPORTS_DIR / "session_comparison_chart.png"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "session_id",
    "session_name",
    "source_media",
    "processed_date",
    "processed_frames",
    "total_observations",
    "unique_tracker_ids",
    "average_confidence",
    "average_tracker_duration_seconds",
    "total_movement_distance_pixels",
    "status",
}


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_session_history() -> None:
    """
    Confirm that the session-history registry exists.
    """

    if not SESSION_HISTORY_CSV.exists():

        raise FileNotFoundError(
            "\nSession history file was not found:\n"
            f"{SESSION_HISTORY_CSV.relative_to(PROJECT_ROOT)}"
        )


def validate_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Confirm that all required session-history columns exist.
    """

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "session_history.csv is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )


# ============================================================
# DATA LOADING
# ============================================================

def load_session_history() -> pd.DataFrame:
    """
    Load the historical session registry.
    """

    validate_session_history()

    dataframe = pd.read_csv(
        SESSION_HISTORY_CSV
    )

    if dataframe.empty:

        raise RuntimeError(
            "session_history.csv contains no sessions."
        )

    validate_columns(
        dataframe
    )

    numeric_columns = [
        "processed_frames",
        "total_observations",
        "unique_tracker_ids",
        "average_confidence",
        "average_tracker_duration_seconds",
        "total_movement_distance_pixels",
    ]

    for column in numeric_columns:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe["processed_date"] = pd.to_datetime(
        dataframe["processed_date"],
        errors="coerce",
    )

    print(
        f"[INFO] Loaded {len(dataframe)} session(s)."
    )

    return dataframe


# ============================================================
# SESSION SUMMARY
# ============================================================

def build_session_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the exportable session comparison table.
    """

    summary = dataframe.copy()

    summary["observations_per_frame"] = (
        summary["total_observations"]
        / summary["processed_frames"]
    )

    summary["observations_per_tracker"] = (
        summary["total_observations"]
        / summary["unique_tracker_ids"]
    )

    summary["processed_date"] = (
        summary["processed_date"]
        .dt.strftime("%Y-%m-%d")
    )

    ordered_columns = [
        "session_id",
        "session_name",
        "source_media",
        "processed_date",
        "processed_frames",
        "total_observations",
        "unique_tracker_ids",
        "observations_per_frame",
        "observations_per_tracker",
        "average_confidence",
        "average_tracker_duration_seconds",
        "total_movement_distance_pixels",
        "status",
    ]

    return summary[
        ordered_columns
    ]


# ============================================================
# CSV OUTPUT
# ============================================================

def save_comparison_csv(
    summary: pd.DataFrame,
) -> None:
    """
    Save the session-level comparison table.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_df = summary.copy()

    decimal_columns = [
        "observations_per_frame",
        "observations_per_tracker",
        "average_confidence",
        "average_tracker_duration_seconds",
        "total_movement_distance_pixels",
    ]

    for column in decimal_columns:

        export_df[column] = (
            export_df[column]
            .round(4)
        )

    export_df.to_csv(
        SESSION_COMPARISON_CSV,
        index=False,
    )

    print(
        "[SUCCESS] Saved: "
        f"{SESSION_COMPARISON_CSV.relative_to(PROJECT_ROOT)}"
    )


# ============================================================
# VISUAL COMPARISON
# ============================================================

def save_comparison_chart(
    summary: pd.DataFrame,
) -> None:
    """
    Create a session comparison chart.

    The chart compares observation counts across sessions.

    When only one verified session exists, the chart still provides
    a baseline that future sessions can be compared against.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = summary[
        "session_id"
    ].astype(str)

    values = summary[
        "total_observations"
    ]

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        labels,
        values,
    )

    plt.title(
        "Visual Tracking and Analysis System\n"
        "Session Observation Comparison"
    )

    plt.xlabel(
        "Session"
    )

    plt.ylabel(
        "Total Observations"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    plt.savefig(
        SESSION_COMPARISON_CHART,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "[SUCCESS] Saved: "
        f"{SESSION_COMPARISON_CHART.relative_to(PROJECT_ROOT)}"
    )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

def print_session_summary(
    summary: pd.DataFrame,
) -> None:
    """
    Print all registered sessions to the terminal.
    """

    print()
    print("=" * 88)
    print(
        "VISUAL TRACKING AND ANALYSIS SYSTEM"
    )
    print(
        "SESSION HISTORY AND COMPARISON"
    )
    print("=" * 88)

    for _, row in summary.iterrows():

        print()
        print(
            f"Session ID: "
            f"{row['session_id']}"
        )

        print(
            f"Session Name: "
            f"{row['session_name']}"
        )

        print(
            f"Source Media: "
            f"{row['source_media']}"
        )

        print(
            f"Processed Date: "
            f"{row['processed_date']}"
        )

        print(
            f"Processed Frames: "
            f"{int(row['processed_frames'])}"
        )

        print(
            f"Total Observations: "
            f"{int(row['total_observations'])}"
        )

        print(
            f"Unique Tracker IDs: "
            f"{int(row['unique_tracker_ids'])}"
        )

        print(
            "Observations per Frame: "
            f"{row['observations_per_frame']:.4f}"
        )

        print(
            "Observations per Tracker: "
            f"{row['observations_per_tracker']:.4f}"
        )

        print(
            "Average Confidence: "
            f"{row['average_confidence']:.4f}"
        )

        print(
            "Average Tracker Duration: "
            f"{row['average_tracker_duration_seconds']:.4f} s"
        )

        print(
            "Total Movement Distance: "
            f"{row['total_movement_distance_pixels']:.2f} px"
        )

        print(
            f"Status: "
            f"{row['status']}"
        )

    print()
    print("=" * 88)
    print()


# ============================================================
# MULTI-SESSION COMPARISON
# ============================================================

def print_multi_session_comparison(
    summary: pd.DataFrame,
) -> None:
    """
    Compare registered sessions when more than one exists.

    With only one session, the function explicitly records that
    the current data represents the historical baseline.
    """

    if len(summary) == 1:

        session = summary.iloc[0]

        print(
            "[INFO] One verified session is currently registered."
        )

        print(
            "[INFO] This session is the historical comparison baseline:"
        )

        print(
            f"       {session['session_id']} - "
            f"{session['session_name']}"
        )

        print(
            "[INFO] Add future verified sessions to "
            "data/session_history.csv to enable direct comparison."
        )

        return

    baseline = summary.iloc[0]
    latest = summary.iloc[-1]

    observation_change = (
        latest["total_observations"]
        - baseline["total_observations"]
    )

    tracker_change = (
        latest["unique_tracker_ids"]
        - baseline["unique_tracker_ids"]
    )

    confidence_change = (
        latest["average_confidence"]
        - baseline["average_confidence"]
    )

    movement_change = (
        latest["total_movement_distance_pixels"]
        - baseline["total_movement_distance_pixels"]
    )

    print()
    print("[INFO] Baseline vs latest session comparison")
    print()

    print(
        f"Baseline: {baseline['session_id']}"
    )

    print(
        f"Latest:   {latest['session_id']}"
    )

    print(
        f"Observation change: "
        f"{observation_change:+.0f}"
    )

    print(
        f"Tracker-ID change: "
        f"{tracker_change:+.0f}"
    )

    print(
        f"Average-confidence change: "
        f"{confidence_change:+.4f}"
    )

    print(
        f"Movement-distance change: "
        f"{movement_change:+.2f} px"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Execute the session-history analysis pipeline.
    """

    print()
    print(
        "[INFO] Starting session history analysis..."
    )

    session_history = load_session_history()

    session_summary = build_session_summary(
        session_history
    )

    print_session_summary(
        session_summary
    )

    print_multi_session_comparison(
        session_summary
    )

    save_comparison_csv(
        session_summary
    )

    save_comparison_chart(
        session_summary
    )

    print()
    print(
        "[SUCCESS] Session history analysis completed."
    )
    print()


if __name__ == "__main__":
    main()
