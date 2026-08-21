"""
Performance Analysis
====================

Visual Tracking and Analysis System

This module analyzes persisted tracking observations and generates:

    reports/performance_summary.csv
    reports/performance_chart.png

The script automatically searches the project for a SQLite database,
detects the tracking-observation table, calculates system-level analytics,
and generates a performance summary.

Optional:
    You can provide the actual total video-processing time with:

        python analytics/performance_analysis.py --processing-seconds 42.5

This allows the script to calculate actual processing FPS and average
processing time per frame without rerunning YOLO, ByteTrack, or SAM 3.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

PERFORMANCE_CSV = REPORTS_DIR / "performance_summary.csv"
PERFORMANCE_CHART = REPORTS_DIR / "performance_chart.png"


# ============================================================
# COLUMN NAME CANDIDATES
# ============================================================

FRAME_COLUMNS = [
    "frame_number",
    "frame_id",
    "frame",
    "frame_index",
]

TRACKER_COLUMNS = [
    "tracker_id",
    "track_id",
    "tracking_id",
]

CONFIDENCE_COLUMNS = [
    "confidence",
    "conf",
    "score",
]

CLASS_NAME_COLUMNS = [
    "class_name",
    "label",
    "class",
]

CLASS_ID_COLUMNS = [
    "class_id",
    "category_id",
]


# ============================================================
# DATABASE DISCOVERY
# ============================================================

def find_sqlite_database() -> Path:
    """
    Search the project directory for an existing SQLite database.
    """

    patterns = [
        "*.db",
        "*.sqlite",
        "*.sqlite3",
    ]

    databases = []

    for pattern in patterns:
        databases.extend(PROJECT_ROOT.rglob(pattern))

    databases = [
        path
        for path in databases
        if path.is_file()
    ]

    if not databases:
        raise FileNotFoundError(
            "\nNo SQLite database was found inside the project.\n"
            "Expected a .db, .sqlite, or .sqlite3 file."
        )

    databases.sort()

    database = databases[0]

    print(f"[INFO] SQLite database found: {database.relative_to(PROJECT_ROOT)}")

    return database


# ============================================================
# DATABASE INSPECTION
# ============================================================

def get_tables(connection: sqlite3.Connection) -> list[str]:
    """
    Return all user-created SQLite tables.
    """

    query = """
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
      AND name NOT LIKE 'sqlite_%'
    ORDER BY name;
    """

    rows = connection.execute(query).fetchall()

    return [row[0] for row in rows]


def get_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> list[str]:
    """
    Return column names for a SQLite table.
    """

    safe_table = table_name.replace('"', '""')

    rows = connection.execute(
        f'PRAGMA table_info("{safe_table}")'
    ).fetchall()

    return [row[1] for row in rows]


def find_matching_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    """
    Find the first matching column from a list of candidates.
    """

    lowercase_map = {
        column.lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lowercase_map:
            return lowercase_map[candidate.lower()]

    return None


def find_tracking_table(
    connection: sqlite3.Connection,
) -> tuple[str, dict[str, str | None]]:
    """
    Detect the table containing tracking observations.

    The table must contain at least:
        - a frame column
        - a tracker-ID column
    """

    tables = get_tables(connection)

    if not tables:
        raise RuntimeError(
            "The SQLite database contains no user tables."
        )

    for table in tables:

        columns = get_columns(connection, table)

        frame_column = find_matching_column(
            columns,
            FRAME_COLUMNS,
        )

        tracker_column = find_matching_column(
            columns,
            TRACKER_COLUMNS,
        )

        confidence_column = find_matching_column(
            columns,
            CONFIDENCE_COLUMNS,
        )

        class_name_column = find_matching_column(
            columns,
            CLASS_NAME_COLUMNS,
        )

        class_id_column = find_matching_column(
            columns,
            CLASS_ID_COLUMNS,
        )

        if frame_column and tracker_column:

            detected_columns = {
                "frame": frame_column,
                "tracker": tracker_column,
                "confidence": confidence_column,
                "class_name": class_name_column,
                "class_id": class_id_column,
            }

            print(f"[INFO] Tracking table detected: {table}")
            print(f"[INFO] Frame column: {frame_column}")
            print(f"[INFO] Tracker column: {tracker_column}")

            if confidence_column:
                print(
                    f"[INFO] Confidence column: "
                    f"{confidence_column}"
                )

            if class_name_column:
                print(
                    f"[INFO] Class-name column: "
                    f"{class_name_column}"
                )

            return table, detected_columns

    raise RuntimeError(
        "\nCould not automatically identify the tracking table.\n"
        "A tracking table must contain frame and tracker-ID columns."
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_tracking_data(
    connection: sqlite3.Connection,
    table_name: str,
) -> pd.DataFrame:
    """
    Load the complete tracking-observation table into Pandas.
    """

    safe_table = table_name.replace('"', '""')

    query = f'SELECT * FROM "{safe_table}"'

    dataframe = pd.read_sql_query(
        query,
        connection,
    )

    if dataframe.empty:
        raise RuntimeError(
            "The tracking table exists but contains no observations."
        )

    print(
        f"[INFO] Loaded {len(dataframe)} tracking observations."
    )

    return dataframe


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    dataframe: pd.DataFrame,
    columns: dict[str, str | None],
    processing_seconds: float | None = None,
) -> dict[str, float | int | str]:
    """
    Calculate system-level analytics from tracking observations.
    """

    frame_column = columns["frame"]
    tracker_column = columns["tracker"]
    confidence_column = columns["confidence"]

    # --------------------------------------------------------
    # Core tracking statistics
    # --------------------------------------------------------

    total_observations = int(len(dataframe))

    total_frames = int(
        dataframe[frame_column].nunique()
    )

    valid_trackers = dataframe[
        dataframe[tracker_column].notna()
    ]

    unique_trackers = int(
        valid_trackers[tracker_column].nunique()
    )

    average_observations_per_frame = (
        total_observations / total_frames
        if total_frames > 0
        else 0
    )

    # --------------------------------------------------------
    # Confidence statistics
    # --------------------------------------------------------

    average_confidence = None
    minimum_confidence = None
    maximum_confidence = None

    if confidence_column:

        confidence_values = pd.to_numeric(
            dataframe[confidence_column],
            errors="coerce",
        ).dropna()

        if not confidence_values.empty:

            average_confidence = float(
                confidence_values.mean()
            )

            minimum_confidence = float(
                confidence_values.min()
            )

            maximum_confidence = float(
                confidence_values.max()
            )

    # --------------------------------------------------------
    # Tracker persistence
    # --------------------------------------------------------

    tracker_observation_counts = (
        valid_trackers
        .groupby(tracker_column)
        .size()
    )

    if not tracker_observation_counts.empty:

        average_observations_per_tracker = float(
            tracker_observation_counts.mean()
        )

        maximum_tracker_observations = int(
            tracker_observation_counts.max()
        )

        minimum_tracker_observations = int(
            tracker_observation_counts.min()
        )

    else:

        average_observations_per_tracker = 0
        maximum_tracker_observations = 0
        minimum_tracker_observations = 0

    # --------------------------------------------------------
    # Actual processing-performance statistics
    # --------------------------------------------------------

    processing_fps = None
    average_processing_time = None

    if processing_seconds is not None:

        if processing_seconds <= 0:
            raise ValueError(
                "Processing time must be greater than zero."
            )

        processing_fps = (
            total_frames / processing_seconds
        )

        average_processing_time = (
            processing_seconds / total_frames
            if total_frames > 0
            else 0
        )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    metrics = {
        "Total Processed Frames": total_frames,
        "Total Observations": total_observations,
        "Unique Tracker IDs": unique_trackers,

        "Average Observations per Frame":
            average_observations_per_frame,

        "Average Observations per Tracker":
            average_observations_per_tracker,

        "Minimum Tracker Observations":
            minimum_tracker_observations,

        "Maximum Tracker Observations":
            maximum_tracker_observations,

        "Average Confidence":
            average_confidence
            if average_confidence is not None
            else "N/A",

        "Minimum Confidence":
            minimum_confidence
            if minimum_confidence is not None
            else "N/A",

        "Maximum Confidence":
            maximum_confidence
            if maximum_confidence is not None
            else "N/A",

        "Total Processing Time (seconds)":
            processing_seconds
            if processing_seconds is not None
            else "N/A",

        "Average Processing Time per Frame (seconds)":
            average_processing_time
            if average_processing_time is not None
            else "N/A",

        "Effective Processing FPS":
            processing_fps
            if processing_fps is not None
            else "N/A",
    }

    return metrics


# ============================================================
# CSV REPORT
# ============================================================

def save_performance_csv(
    metrics: dict[str, float | int | str],
) -> None:
    """
    Save metrics to reports/performance_summary.csv.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for metric_name, value in metrics.items():

        if isinstance(value, float):
            value = round(value, 4)

        rows.append(
            {
                "metric": metric_name,
                "value": value,
            }
        )

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        PERFORMANCE_CSV,
        index=False,
    )

    print(
        f"[SUCCESS] Performance summary saved: "
        f"{PERFORMANCE_CSV.relative_to(PROJECT_ROOT)}"
    )


# ============================================================
# PERFORMANCE CHART
# ============================================================

def save_performance_chart(
    metrics: dict[str, float | int | str],
) -> None:
    """
    Generate a compact visual summary of the primary system metrics.
    """

    chart_metrics = {
        "Frames": metrics["Total Processed Frames"],
        "Observations": metrics["Total Observations"],
        "Tracker IDs": metrics["Unique Tracker IDs"],
    }

    labels = list(chart_metrics.keys())
    values = list(chart_metrics.values())

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        labels,
        values,
    )

    plt.title(
        "Visual Tracking System - Performance Summary"
    )

    plt.ylabel(
        "Count"
    )

    plt.xlabel(
        "Metric"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
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
        f"[SUCCESS] Performance chart saved: "
        f"{PERFORMANCE_CHART.relative_to(PROJECT_ROOT)}"
    )


# ============================================================
# TERMINAL SUMMARY
# ============================================================

def print_metrics(
    metrics: dict[str, float | int | str],
) -> None:
    """
    Print the calculated metrics to the terminal.
    """

    print()
    print("=" * 64)
    print("VISUAL TRACKING AND ANALYSIS SYSTEM")
    print("PERFORMANCE SUMMARY")
    print("=" * 64)

    for name, value in metrics.items():

        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = value

        print(
            f"{name:<48} {formatted_value}"
        )

    print("=" * 64)
    print()


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Read optional command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate performance analytics for the "
            "Visual Tracking and Analysis System."
        )
    )

    parser.add_argument(
        "--processing-seconds",
        type=float,
        default=None,
        help=(
            "Actual total processing duration in seconds. "
            "When provided, processing FPS and average "
            "processing time per frame are calculated."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Execute the performance-analysis pipeline.
    """

    args = parse_arguments()

    print()
    print(
        "[INFO] Starting performance analysis..."
    )

    database_path = find_sqlite_database()

    with sqlite3.connect(database_path) as connection:

        table_name, detected_columns = (
            find_tracking_table(connection)
        )

        tracking_data = load_tracking_data(
            connection,
            table_name,
        )

    metrics = calculate_metrics(
        dataframe=tracking_data,
        columns=detected_columns,
        processing_seconds=args.processing_seconds,
    )

    print_metrics(metrics)

    save_performance_csv(metrics)

    save_performance_chart(metrics)

    print()
    print(
        "[SUCCESS] Performance analysis completed."
    )
    print()


if __name__ == "__main__":
    main()
