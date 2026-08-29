"""Reusable SAM 3 video-segmentation pipelines for Session 11.

Pipeline A combines YOLO, ByteTrack, and SAM 3. Pipeline B uses
SAM3VideoSemanticPredictor with natural-language prompts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
from trackers import ByteTrackTracker
from ultralytics import SAM, YOLO
from ultralytics.models.sam import SAM3VideoSemanticPredictor


DEFAULT_VIDEO_URL = (
    "https://media.roboflow.com/supervision/video-examples/vehicles.mp4"
)
DEFAULT_PROMPTS = ("car", "bus", "truck")


def patch_numpy_cross() -> None:
    """Keep Supervision 2D geometry compatible with NumPy 2.x."""
    original_cross = np.cross

    def cross_patched(
        a: np.ndarray,
        b: np.ndarray,
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: int | None = None,
    ) -> np.ndarray:
        array_a = np.asarray(a)
        array_b = np.asarray(b)
        if array_a.shape[-1] == 2 and array_b.shape[-1] == 2:
            return array_a[..., 0] * array_b[..., 1] - array_a[..., 1] * array_b[..., 0]
        return original_cross(array_a, array_b, axisa, axisb, axisc, axis)

    np.cross = cross_patched


def ensure_input_video(video_path: Path, download_url: str) -> Path:
    """Return an existing input video or download the lesson sample."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if not video_path.exists():
        print(f"Downloading sample video to {video_path}...")
        urlretrieve(download_url, video_path)
    return video_path


def validate_video(video_path: Path) -> sv.VideoInfo:
    """Read and validate basic video metadata."""
    info = sv.VideoInfo.from_video_path(str(video_path))
    if info.width <= 0 or info.height <= 0 or info.fps <= 0:
        raise ValueError(f"Invalid video metadata: {video_path}")
    print(
        f"Input: {video_path} | {info.width}x{info.height} | "
        f"{info.fps:.2f} FPS | {info.total_frames} frames"
    )
    return info


def transfer_attributes(
    sam_detections: sv.Detections,
    yolo_detections: sv.Detections,
) -> sv.Detections:
    """Transfer tracker, class, and confidence data when detections align."""
    if len(sam_detections) != len(yolo_detections):
        return sam_detections
    sam_detections.tracker_id = yolo_detections.tracker_id
    sam_detections.class_id = yolo_detections.class_id
    sam_detections.confidence = yolo_detections.confidence
    return sam_detections


class DetectorGuidedPipeline:
    """YOLO + ByteTrack + SAM 3 video pipeline and lesson experiments."""

    def __init__(self, sam_model_path: Path, yolo_model_name: str) -> None:
        if not sam_model_path.exists():
            raise FileNotFoundError(
                f"SAM 3 checkpoint not found: {sam_model_path}. "
                "Pass its location with --sam-model."
            )
        self.yolo = YOLO(yolo_model_name)
        self.sam = SAM(str(sam_model_path))
        self.mask_annotator = sv.MaskAnnotator(opacity=0.6)
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

    def detect_and_segment(
        self,
        frame: np.ndarray,
        tracker: ByteTrackTracker,
    ) -> tuple[sv.Detections, object]:
        """Detect, track, segment, and transfer aligned metadata."""
        yolo_result = self.yolo(frame, verbose=False)[0]
        yolo_detections = sv.Detections.from_ultralytics(yolo_result)
        yolo_detections = tracker.update(yolo_detections)
        if len(yolo_detections) == 0:
            return sv.Detections.empty(), yolo_result

        sam_result = self.sam(
            frame,
            bboxes=yolo_detections.xyxy.tolist(),
            verbose=False,
        )[0]
        sam_detections = sv.Detections.from_ultralytics(sam_result)
        return transfer_attributes(sam_detections, yolo_detections), yolo_result

    def run_full(self, source: Path, target: Path) -> None:
        """Create the full mask, label, and movement-trace video."""
        tracker = ByteTrackTracker()

        def callback(frame: np.ndarray, _: int) -> np.ndarray:
            detections, yolo_result = self.detect_and_segment(frame, tracker)
            if len(detections) == 0:
                return frame.copy()
            labels = [
                f"ID:{tracker_id} {yolo_result.names[class_id]}"
                for tracker_id, class_id in zip(
                    detections.tracker_id,
                    detections.class_id,
                )
            ]
            annotated = self.mask_annotator.annotate(frame.copy(), detections)
            annotated = self.label_annotator.annotate(annotated, detections, labels)
            return self.trace_annotator.annotate(annotated, detections)

        process_video(source, target, callback)

    def run_zone(self, source: Path, target: Path, info: sv.VideoInfo) -> None:
        """Segment only objects inside the lower-left polygon zone."""
        polygon = np.array(
            [
                [0, info.height // 2],
                [info.width // 2, info.height // 2],
                [info.width // 2, info.height],
                [0, info.height],
            ]
        )
        zone = sv.PolygonZone(polygon=polygon)
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=sv.Color.RED,
            thickness=3,
        )
        tracker = ByteTrackTracker()

        def callback(frame: np.ndarray, _: int) -> np.ndarray:
            yolo_result = self.yolo(frame, verbose=False)[0]
            detections = tracker.update(
                sv.Detections.from_ultralytics(yolo_result)
            )
            detections = detections[zone.trigger(detections=detections)]
            annotated = frame.copy()
            if len(detections):
                sam_result = self.sam(
                    frame,
                    bboxes=detections.xyxy.tolist(),
                    verbose=False,
                )[0]
                sam_detections = transfer_attributes(
                    sv.Detections.from_ultralytics(sam_result), detections
                )
                annotated = self.mask_annotator.annotate(
                    annotated, sam_detections
                )
            return zone_annotator.annotate(annotated)

        process_video(source, target, callback)

    def run_opacity(self, source: Path, target: Path) -> None:
        """Use detection confidence as per-object mask opacity."""
        tracker = ByteTrackTracker()

        def callback(frame: np.ndarray, _: int) -> np.ndarray:
            detections, _ = self.detect_and_segment(frame, tracker)
            annotated = frame.copy()
            for index in range(len(detections)):
                detection = detections[index]
                opacity = (
                    float(detection.confidence[0])
                    if detection.confidence is not None
                    else 0.5
                )
                opacity = max(0.0, min(1.0, opacity))
                annotated = sv.MaskAnnotator(opacity=opacity).annotate(
                    annotated, detection
                )
            return annotated

        process_video(source, target, callback)

    def run_area_analysis(
        self,
        source: Path,
        target: Path,
        chart_path: Path,
        json_path: Path,
    ) -> None:
        """Track mask area by object ID and save video, chart, and JSON."""
        tracker = ByteTrackTracker()
        areas: dict[int, list[dict[str, int]]] = defaultdict(list)

        def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
            detections, _ = self.detect_and_segment(frame, tracker)
            if detections.mask is not None and detections.tracker_id is not None:
                for index, tracker_id in enumerate(detections.tracker_id):
                    areas[int(tracker_id)].append(
                        {
                            "frame": frame_index,
                            "area_px": int(detections.mask[index].sum()),
                        }
                    )
            return self.mask_annotator.annotate(frame.copy(), detections)

        process_video(source, target, callback)
        save_area_outputs(areas, chart_path, json_path)


def process_video(source: Path, target: Path, callback: object) -> None:
    """Run a Supervision callback and ensure its output directory exists."""
    target.parent.mkdir(parents=True, exist_ok=True)
    sv.process_video(
        source_path=str(source),
        target_path=str(target),
        callback=callback,
        show_progress=True,
    )
    print(f"Saved: {target}")


def save_area_outputs(
    areas: dict[int, list[dict[str, int]]],
    chart_path: Path,
    json_path: Path,
) -> None:
    """Save temporal mask areas as machine-readable data and a chart."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(areas, indent=2), encoding="utf-8")

    selected_ids = sorted(areas, key=lambda key: len(areas[key]), reverse=True)[:3]
    plt.figure(figsize=(12, 4))
    for tracker_id in selected_ids:
        observations = areas[tracker_id]
        plt.plot(
            [item["frame"] for item in observations],
            [item["area_px"] for item in observations],
            label=f"ID {tracker_id}",
        )
    plt.xlabel("Frame")
    plt.ylabel("Mask area (px²)")
    plt.title("SAM 3 mask area through time")
    if selected_ids:
        plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close()
    print(f"Saved: {chart_path}")
    print(f"Saved: {json_path}")


def run_semantic_pipeline(
    source: Path,
    target: Path,
    model_path: Path,
    prompts: Iterable[str],
    confidence: float,
) -> None:
    """Run SAM 3 semantic video segmentation from text prompts."""
    overrides: dict[str, object] = {
        "conf": confidence,
        "task": "segment",
        "mode": "predict",
        "model": str(model_path),
    }
    if torch.cuda.is_available():
        overrides["half"] = True

    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    results = predictor(
        source=str(source),
        text=list(prompts),
        stream=True,
    )
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {target}")

    annotator = sv.MaskAnnotator(opacity=0.6)
    try:
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            writer.write(
                annotator.annotate(result.orig_img.copy(), detections)
            )
    finally:
        writer.release()
    print(f"Saved: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Session 11 SAM 3 video-segmentation pipelines."
    )
    parser.add_argument(
        "--mode",
        choices=("full", "zone", "opacity", "areas", "semantic", "all"),
        default="full",
    )
    parser.add_argument("--sam-model", type=Path, required=True)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/input/vehicles.mp4"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/output"),
    )
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_numpy_cross()
    source = ensure_input_video(args.input, args.video_url)
    info = validate_video(source)
    output = args.output_dir

    detector_modes = {"full", "zone", "opacity", "areas", "all"}
    pipeline = None
    if args.mode in detector_modes:
        pipeline = DetectorGuidedPipeline(args.sam_model, args.yolo_model)

    if args.mode in {"full", "all"}:
        pipeline.run_full(source, output / "vehicles_sam.mp4")
    if args.mode in {"zone", "all"}:
        pipeline.run_zone(source, output / "vehicles_sam_zone.mp4", info)
    if args.mode in {"opacity", "all"}:
        pipeline.run_opacity(source, output / "vehicles_sam_opacity.mp4")
    if args.mode in {"areas", "all"}:
        pipeline.run_area_analysis(
            source,
            output / "vehicles_sam_areas.mp4",
            output / "mask_area_chart.png",
            output / "mask_areas.json",
        )
    if args.mode in {"semantic", "all"}:
        run_semantic_pipeline(
            source,
            output / "vehicles_text_prompts.mp4",
            args.sam_model,
            args.prompts,
            args.confidence,
        )


if __name__ == "__main__":
    main()
