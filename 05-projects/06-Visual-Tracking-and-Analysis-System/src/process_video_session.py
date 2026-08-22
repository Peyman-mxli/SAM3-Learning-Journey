"""
process_video_session.py

Visual Tracking and Analysis System
Project 06 — SAM3 Learning Journey

Reusable recorded-video processing pipeline for verified historical sessions.

This script:

1. Loads an input video.
2. Runs YOLO object detection.
3. Runs ByteTrack multi-object tracking.
4. Runs SAM 3 text-prompt segmentation.
5. Preserves tracker state across sequential frames.
6. Records structured tracker observations.
7. Generates an annotated temporary MP4.
8. Converts the result to browser-compatible H.264 / yuv420p.
9. Generates tracker and trajectory summaries.
10. Registers the completed session in data/session_history.csv.

The script reuses the existing verified project modules:

    src/pipeline.py
    src/detector.py
    src/tracker.py
    src/segmenter.py
    src/visualization.py

Example:

    python src/process_video_session.py \
        --input assets/input/tracking_test_02.mp4 \
        --output assets/output/sam3_tracking_output_02.mp4 \
        --checkpoint /path/to/sam3.pt \
        --session-id session_002 \
        --session-name "Busy Street Video Run" \
        --prompt person \
        --max-frames 75
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.pipeline import VisualAnalysisPipeline


# ============================================================
# PROJECT PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "output"

SESSION_HISTORY_CSV = DATA_DIR / "session_history.csv"


# ============================================================
# SESSION HISTORY COLUMNS
# ============================================================

SESSION_HISTORY_COLUMNS = [
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
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Process a recorded video as a verified historical "
            "tracking and SAM 3 analysis session."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video path.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Final H.264 output video path.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local SAM 3 checkpoint path.",
    )

    parser.add_argument(
        "--session-id",
        required=True,
        help="Unique session identifier, for example session_002.",
    )

    parser.add_argument(
        "--session-name",
        required=True,
        help="Human-readable session name.",
    )

    parser.add_argument(
        "--prompt",
        default="person",
        help="SAM 3 text prompt. Default: person.",
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO model. Default: yolov8n.pt.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="YOLO confidence threshold. Default: 0.50.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=75,
        help=(
            "Maximum number of frames to process. "
            "Default: 75 for comparison with session_001."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="SAM 3 inference device. Default: cuda.",
    )

    return parser.parse_args()


# ============================================================
# VIDEO METADATA
# ============================================================

def open_video(
    video_path: Path,
) -> tuple[cv2.VideoCapture, float, int, int, int]:
    """
    Open the video and return its metadata.
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open input video: {video_path}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if fps <= 0:
        fps = 30.0

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "Invalid video resolution."
        )

    return (
        capture,
        fps,
        total_frames,
        width,
        height,
    )


# ============================================================
# TEMPORARY VIDEO WRITER
# ============================================================

def create_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """
    Create an OpenCV temporary MP4 writer.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create video writer: {output_path}"
        )

    return writer


# ============================================================
# TRACKING OBSERVATION EXTRACTION
# ============================================================

def extract_observations(
    tracked_detections,
    frame_number: int,
    timestamp_seconds: float,
    class_names: dict | list,
) -> list[dict]:
    """
    Convert Supervision tracked detections into structured observations.
    """

    observations = []

    if tracked_detections is None:
        return observations

    xyxy = getattr(
        tracked_detections,
        "xyxy",
        None,
    )

    tracker_ids = getattr(
        tracked_detections,
        "tracker_id",
        None,
    )

    class_ids = getattr(
        tracked_detections,
        "class_id",
        None,
    )

    confidences = getattr(
        tracked_detections,
        "confidence",
        None,
    )

    if xyxy is None:
        return observations

    detection_count = len(
        xyxy
    )

    for index in range(
        detection_count
    ):

        tracker_id = None

        if (
            tracker_ids is not None
            and index < len(tracker_ids)
        ):
            raw_tracker_id = tracker_ids[
                index
            ]

            if raw_tracker_id is not None:
                tracker_id = int(
                    raw_tracker_id
                )

        if tracker_id is None:
            continue

        class_id = None

        if (
            class_ids is not None
            and index < len(class_ids)
        ):
            raw_class_id = class_ids[
                index
            ]

            if raw_class_id is not None:
                class_id = int(
                    raw_class_id
                )

        class_name = "unknown"

        if class_id is not None:

            if isinstance(
                class_names,
                dict,
            ):
                class_name = str(
                    class_names.get(
                        class_id,
                        class_id,
                    )
                )

            elif (
                isinstance(
                    class_names,
                    list,
                )
                and 0 <= class_id < len(
                    class_names
                )
            ):
                class_name = str(
                    class_names[
                        class_id
                    ]
                )

        confidence = np.nan

        if (
            confidences is not None
            and index < len(confidences)
        ):
            raw_confidence = confidences[
                index
            ]

            if raw_confidence is not None:
                confidence = float(
                    raw_confidence
                )

        x1, y1, x2, y2 = map(
            float,
            xyxy[index],
        )

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        observations.append(
            {
                "frame_number":
                    int(
                        frame_number
                    ),

                "timestamp_seconds":
                    float(
                        timestamp_seconds
                    ),

                "tracker_id":
                    tracker_id,

                "class_id":
                    class_id,

                "class_name":
                    class_name,

                "confidence":
                    confidence,

                "x1":
                    x1,

                "y1":
                    y1,

                "x2":
                    x2,

                "y2":
                    y2,

                "center_x":
                    center_x,

                "center_y":
                    center_y,
            }
        )

    return observations


# ============================================================
# TRACKER SUMMARY
# ============================================================

def build_tracker_summary(
    observations_df: pd.DataFrame,
    fps: float,
) -> pd.DataFrame:
    """
    Generate tracker-level temporal statistics.
    """

    columns = [
        "tracker_id",
        "class_name",
        "first_frame",
        "last_frame",
        "observations",
        "duration_seconds",
        "average_confidence",
    ]

    if observations_df.empty:
        return pd.DataFrame(
            columns=columns
        )

    rows = []

    grouped = observations_df.groupby(
        "tracker_id",
        sort=True,
    )

    for tracker_id, group in grouped:

        first_frame = int(
            group["frame_number"].min()
        )

        last_frame = int(
            group["frame_number"].max()
        )

        observation_count = int(
            len(group)
        )

        duration_seconds = (
            observation_count / fps
            if fps > 0
            else 0
        )

        class_modes = group[
            "class_name"
        ].mode()

        if not class_modes.empty:
            class_name = str(
                class_modes.iloc[0]
            )
        else:
            class_name = "unknown"

        confidence_values = pd.to_numeric(
            group["confidence"],
            errors="coerce",
        ).dropna()

        if confidence_values.empty:
            average_confidence = np.nan
        else:
            average_confidence = float(
                confidence_values.mean()
            )

        rows.append(
            {
                "tracker_id":
                    int(
                        tracker_id
                    ),

                "class_name":
                    class_name,

                "first_frame":
                    first_frame,

                "last_frame":
                    last_frame,

                "observations":
                    observation_count,

                "duration_seconds":
                    round(
                        duration_seconds,
                        2,
                    ),

                "average_confidence":
                    round(
                        average_confidence,
                        4,
                    )
                    if not math.isnan(
                        average_confidence
                    )
                    else np.nan,
            }
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


# ============================================================
# TRAJECTORY SUMMARY
# ============================================================

def build_trajectory_summary(
    observations_df: pd.DataFrame,
    fps: float,
) -> pd.DataFrame:
    """
    Generate tracker-level movement statistics.
    """

    columns = [
        "tracker_id",
        "first_frame",
        "last_frame",
        "frames_observed",
        "duration_seconds",
        "movement_distance_pixels",
        "average_movement_pixels",
    ]

    if observations_df.empty:
        return pd.DataFrame(
            columns=columns
        )

    rows = []

    grouped = observations_df.groupby(
        "tracker_id",
        sort=True,
    )

    for tracker_id, group in grouped:

        group = group.sort_values(
            "frame_number"
        )

        centers = group[
            [
                "center_x",
                "center_y",
            ]
        ].to_numpy(
            dtype=float
        )

        total_distance = 0.0

        movement_steps = []

        if len(centers) > 1:

            differences = np.diff(
                centers,
                axis=0,
            )

            distances = np.sqrt(
                np.sum(
                    differences ** 2,
                    axis=1,
                )
            )

            movement_steps = distances.tolist()

            total_distance = float(
                distances.sum()
            )

        if movement_steps:
            average_movement = float(
                np.mean(
                    movement_steps
                )
            )
        else:
            average_movement = 0.0

        frames_observed = int(
            len(group)
        )

        duration_seconds = (
            frames_observed / fps
            if fps > 0
            else 0
        )

        rows.append(
            {
                "tracker_id":
                    int(
                        tracker_id
                    ),

                "first_frame":
                    int(
                        group[
                            "frame_number"
                        ].min()
                    ),

                "last_frame":
                    int(
                        group[
                            "frame_number"
                        ].max()
                    ),

                "frames_observed":
                    frames_observed,

                "duration_seconds":
                    round(
                        duration_seconds,
                        2,
                    ),

                "movement_distance_pixels":
                    round(
                        total_distance,
                        2,
                    ),

                "average_movement_pixels":
                    round(
                        average_movement,
                        2,
                    ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


# ============================================================
# H.264 CONVERSION
# ============================================================

def convert_to_h264(
    temporary_video: Path,
    final_video: Path,
) -> None:
    """
    Convert the OpenCV MP4 into browser-compatible H.264 / yuv420p.
    """

    if shutil.which(
        "ffmpeg"
    ) is None:
        raise RuntimeError(
            "FFmpeg is required but was not found."
        )

    final_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(
            temporary_video
        ),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(
            final_video
        ),
    ]

    print()
    print(
        "[INFO] Converting output to H.264..."
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        print(
            result.stderr
        )

        raise RuntimeError(
            "FFmpeg H.264 conversion failed."
        )

    print(
        "[SUCCESS] H.264 conversion completed."
    )


# ============================================================
# SESSION REGISTRATION
# ============================================================

def register_session(
    session_id: str,
    session_name: str,
    source_media: str,
    processed_frames: int,
    tracker_summary: pd.DataFrame,
    trajectory_summary: pd.DataFrame,
) -> None:
    """
    Add or update the completed session in session_history.csv.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_observations = int(
        tracker_summary[
            "observations"
        ].sum()
    )

    unique_tracker_ids = int(
        tracker_summary[
            "tracker_id"
        ].nunique()
    )

    confidence_values = pd.to_numeric(
        tracker_summary[
            "average_confidence"
        ],
        errors="coerce",
    ).dropna()

    if confidence_values.empty:
        average_confidence = np.nan
    else:
        average_confidence = float(
            confidence_values.mean()
        )

    duration_values = pd.to_numeric(
        tracker_summary[
            "duration_seconds"
        ],
        errors="coerce",
    ).dropna()

    if duration_values.empty:
        average_duration = 0.0
    else:
        average_duration = float(
            duration_values.mean()
        )

    movement_values = pd.to_numeric(
        trajectory_summary[
            "movement_distance_pixels"
        ],
        errors="coerce",
    ).fillna(
        0
    )

    total_movement = float(
        movement_values.sum()
    )

    session_row = {
        "session_id":
            session_id,

        "session_name":
            session_name,

        "source_media":
            source_media,

        "processed_date":
            pd.Timestamp.now().strftime(
                "%Y-%m-%d"
            ),

        "processed_frames":
            int(
                processed_frames
            ),

        "total_observations":
            total_observations,

        "unique_tracker_ids":
            unique_tracker_ids,

        "average_confidence":
            round(
                average_confidence,
                4,
            )
            if not math.isnan(
                average_confidence
            )
            else np.nan,

        "average_tracker_duration_seconds":
            round(
                average_duration,
                4,
            ),

        "total_movement_distance_pixels":
            round(
                total_movement,
                2,
            ),

        "status":
            "verified",
    }

    if SESSION_HISTORY_CSV.exists():

        history_df = pd.read_csv(
            SESSION_HISTORY_CSV
        )

    else:

        history_df = pd.DataFrame(
            columns=SESSION_HISTORY_COLUMNS
        )

    if (
        not history_df.empty
        and "session_id"
        in history_df.columns
        and session_id
        in history_df[
            "session_id"
        ].astype(
            str
        ).tolist()
    ):

        history_df = history_df[
            history_df[
                "session_id"
            ].astype(
                str
            )
            != str(
                session_id
            )
        ]

    history_df = pd.concat(
        [
            history_df,
            pd.DataFrame(
                [
                    session_row
                ]
            ),
        ],
        ignore_index=True,
    )

    history_df = history_df[
        SESSION_HISTORY_COLUMNS
    ]

    history_df.to_csv(
        SESSION_HISTORY_CSV,
        index=False,
    )

    print(
        "[SUCCESS] Session registered: "
        f"{SESSION_HISTORY_CSV.relative_to(PROJECT_ROOT)}"
    )


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(
    args: argparse.Namespace,
) -> None:
    """
    Run the complete recorded-video processing workflow.
    """

    input_path = Path(
        args.input
    )

    if not input_path.is_absolute():
        input_path = (
            PROJECT_ROOT
            / input_path
        )

    final_output = Path(
        args.output
    )

    if not final_output.is_absolute():
        final_output = (
            PROJECT_ROOT
            / final_output
        )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM 3 checkpoint not found: {checkpoint_path}"
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = (
        final_output.parent
        / (
            final_output.stem
            + "_temporary.mp4"
        )
    )

    observation_csv = (
        DATA_DIR
        / f"{args.session_id}_observations.csv"
    )

    tracker_summary_csv = (
        REPORTS_DIR
        / f"tracker_summary_{args.session_id}.csv"
    )

    trajectory_summary_csv = (
        REPORTS_DIR
        / f"trajectory_summary_{args.session_id}.csv"
    )

    print()
    print("=" * 76)
    print(
        "VISUAL TRACKING AND ANALYSIS SYSTEM"
    )
    print(
        f"PROCESSING {args.session_id.upper()}"
    )
    print("=" * 76)

    print(
        f"Input: {input_path}"
    )

    print(
        f"Output: {final_output}"
    )

    print(
        f"SAM 3 prompt: {args.prompt}"
    )

    print(
        f"Maximum frames: {args.max_frames}"
    )

    print()

    capture, fps, source_frame_count, width, height = (
        open_video(
            input_path
        )
    )

    print(
        f"[INFO] Source FPS: {fps:.2f}"
    )

    print(
        f"[INFO] Source frames: {source_frame_count}"
    )

    print(
        f"[INFO] Resolution: {width}x{height}"
    )

    frame_limit = min(
        args.max_frames,
        source_frame_count,
    )

    writer = create_writer(
        temporary_output,
        fps,
        width,
        height,
    )

    print()
    print(
        "[INFO] Loading YOLO + ByteTrack + SAM 3..."
    )

    pipeline = VisualAnalysisPipeline(
        sam3_checkpoint_path=str(
            checkpoint_path
        ),
        model_name=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    pipeline.reset_tracker()

    class_names = (
        pipeline.detector.get_class_names()
    )

    all_observations = []

    processed_frames = 0

    start_time = time.perf_counter()

    try:

        while (
            processed_frames
            < frame_limit
        ):

            success, frame = capture.read()

            if not success:
                break

            frame_number = (
                processed_frames + 1
            )

            timestamp_seconds = (
                processed_frames / fps
            )

            result = pipeline.process_image(
                image_bgr=frame,
                segmentation_prompt=args.prompt,
            )

            annotated_frame = result[
                "annotated_image"
            ]

            if (
                annotated_frame.shape[1]
                != width
                or annotated_frame.shape[0]
                != height
            ):

                annotated_frame = cv2.resize(
                    annotated_frame,
                    (
                        width,
                        height,
                    ),
                )

            writer.write(
                annotated_frame
            )

            frame_observations = (
                extract_observations(
                    tracked_detections=result[
                        "tracked_detections"
                    ],
                    frame_number=frame_number,
                    timestamp_seconds=timestamp_seconds,
                    class_names=class_names,
                )
            )

            all_observations.extend(
                frame_observations
            )

            processed_frames += 1

            print(
                f"\r[PROCESSING] "
                f"Frame {processed_frames}/{frame_limit} | "
                f"Observations: {len(all_observations)}",
                end="",
                flush=True,
            )

    finally:

        capture.release()

        writer.release()

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    print()
    print()

    print(
        f"[INFO] Processed frames: "
        f"{processed_frames}"
    )

    print(
        f"[INFO] Total observations: "
        f"{len(all_observations)}"
    )

    print(
        f"[INFO] Processing time: "
        f"{elapsed_seconds:.2f} seconds"
    )

    if elapsed_seconds > 0:

        effective_fps = (
            processed_frames
            / elapsed_seconds
        )

        print(
            f"[INFO] Effective processing FPS: "
            f"{effective_fps:.4f}"
        )

    observations_df = pd.DataFrame(
        all_observations
    )

    if observations_df.empty:

        raise RuntimeError(
            "No tracked observations were produced."
        )

    observations_df.to_csv(
        observation_csv,
        index=False,
    )

    print(
        "[SUCCESS] Observations saved: "
        f"{observation_csv.relative_to(PROJECT_ROOT)}"
    )

    tracker_summary = build_tracker_summary(
        observations_df,
        fps,
    )

    tracker_summary.to_csv(
        tracker_summary_csv,
        index=False,
    )

    print(
        "[SUCCESS] Tracker summary saved: "
        f"{tracker_summary_csv.relative_to(PROJECT_ROOT)}"
    )

    trajectory_summary = (
        build_trajectory_summary(
            observations_df,
            fps,
        )
    )

    trajectory_summary.to_csv(
        trajectory_summary_csv,
        index=False,
    )

    print(
        "[SUCCESS] Trajectory summary saved: "
        f"{trajectory_summary_csv.relative_to(PROJECT_ROOT)}"
    )

    convert_to_h264(
        temporary_output,
        final_output,
    )

    if temporary_output.exists():
        temporary_output.unlink()

    register_session(
        session_id=args.session_id,
        session_name=args.session_name,
        source_media=input_path.name,
        processed_frames=processed_frames,
        tracker_summary=tracker_summary,
        trajectory_summary=trajectory_summary,
    )

    print()
    print("=" * 76)
    print(
        f"{args.session_id.upper()} COMPLETE"
    )
    print("=" * 76)

    print(
        f"Processed frames: {processed_frames}"
    )

    print(
        f"Recorded observations: "
        f"{len(observations_df)}"
    )

    print(
        f"Unique tracker IDs: "
        f"{tracker_summary['tracker_id'].nunique()}"
    )

    print(
        "Final H.264 output:"
    )

    print(
        final_output
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Main application entry point.
    """

    args = parse_arguments()

    process_video(
        args
    )


if __name__ == "__main__":
    main()
