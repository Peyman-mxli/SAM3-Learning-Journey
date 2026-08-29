"""Detect and track road vehicles with YOLOv8 and ByteTrackTracker."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO


VEHICLE_CLASS_IDS = [2, 3, 5, 7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/output"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--preview-frame", type=int, default=250)
    return parser.parse_args()


def resize_frame(frame, target_width: int):
    if frame.shape[1] <= target_width:
        return frame
    scale = target_width / frame.shape[1]
    return cv2.resize(
        frame,
        (target_width, int(round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_video = args.output_dir / "vehicles_bytetrack.mp4"
    output_preview = args.output_dir / "vehicles_bytetrack_preview.jpg"
    output_json = args.output_dir / "tracking_summary.json"
    output_csv = args.output_dir / "tracking_observations.csv"

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.input}")

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_width = min(source_width, args.width)
    output_height = int(round(source_height * output_width / source_width))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {output_video}")

    model = YOLO(args.model)
    tracker = ByteTrackTracker(
        frame_rate=fps,
        track_activation_threshold=args.confidence,
        minimum_consecutive_frames=2,
    )
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color_lookup=sv.ColorLookup.TRACK,
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.45,
        text_thickness=1,
        color_lookup=sv.ColorLookup.TRACK,
    )
    trace_annotator = sv.TraceAnnotator(
        position=sv.Position.BOTTOM_CENTER,
        trace_length=50,
        thickness=2,
        color_lookup=sv.ColorLookup.TRACK,
    )

    tracks = defaultdict(lambda: {
        "first_frame": None,
        "last_frame": None,
        "observations": 0,
        "max_confidence": 0.0,
        "class_names": set(),
    })
    observations = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = resize_frame(frame, args.width)
            result = model.predict(
                frame,
                conf=args.confidence,
                classes=VEHICLE_CLASS_IDS,
                imgsz=640,
                verbose=False,
            )[0]
            detections = sv.Detections.from_ultralytics(result)
            tracked = tracker.update(detections)
            tracked = tracked[tracked.tracker_id >= 0]

            names = tracked.data.get("class_name", [])
            labels = []
            for index, tracker_id in enumerate(tracked.tracker_id):
                tracker_id = int(tracker_id)
                confidence = float(tracked.confidence[index])
                class_name = str(names[index]) if len(names) else str(tracked.class_id[index])
                x1, y1, x2, y2 = [float(v) for v in tracked.xyxy[index]]
                labels.append(f"#{tracker_id} {class_name} {confidence:.2f}")

                record = tracks[tracker_id]
                if record["first_frame"] is None:
                    record["first_frame"] = frame_index
                record["last_frame"] = frame_index
                record["observations"] += 1
                record["max_confidence"] = max(record["max_confidence"], confidence)
                record["class_names"].add(class_name)
                observations.append({
                    "frame": frame_index,
                    "time_seconds": round(frame_index / fps, 3),
                    "tracker_id": tracker_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                })

            annotated = trace_annotator.annotate(frame.copy(), tracked)
            annotated = box_annotator.annotate(annotated, tracked)
            annotated = label_annotator.annotate(annotated, tracked, labels)
            cv2.putText(
                annotated,
                f"Frame {frame_index + 1}/{source_frames} | Active tracks: {len(tracked)}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)
            if frame_index == args.preview_frame:
                cv2.imwrite(str(output_preview), annotated)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    summary_tracks = {}
    for tracker_id, record in sorted(tracks.items()):
        summary_tracks[str(tracker_id)] = {
            "first_frame": record["first_frame"],
            "last_frame": record["last_frame"],
            "observations": record["observations"],
            "max_confidence": round(record["max_confidence"], 4),
            "class_names": sorted(record["class_names"]),
        }

    summary = {
        "input": {
            "filename": args.input.name,
            "source_width": source_width,
            "source_height": source_height,
            "fps": fps,
            "declared_frames": source_frames,
        },
        "processing": {
            "model": args.model,
            "confidence_threshold": args.confidence,
            "vehicle_class_ids": VEHICLE_CLASS_IDS,
            "output_width": output_width,
            "output_height": output_height,
            "processed_frames": frame_index,
        },
        "results": {
            "unique_tracker_ids": len(summary_tracks),
            "total_tracked_observations": len(observations),
            "tracks": summary_tracks,
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = list(observations[0]) if observations else [
        "frame", "time_seconds", "tracker_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(observations)

    print(f"Processed frames: {frame_index}")
    print(f"Unique tracker IDs: {len(summary_tracks)}")
    print(f"Tracked observations: {len(observations)}")
    print(f"Video: {output_video}")
    print(f"Preview: {output_preview}")
    print(f"Summary: {output_json}")
    print(f"Observations: {output_csv}")


if __name__ == "__main__":
    main()
