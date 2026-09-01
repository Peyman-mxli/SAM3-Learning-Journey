from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid

import cv2
import supervision as sv
from ultralytics import YOLO

from database import connect, create_session, insert_observations


def process_video(
    source: str,
    database_path: str,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.35,
    notes: str = "",
) -> str:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    model = YOLO(model_name)
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    session_id = f"lab13_{uuid.uuid4().hex[:10]}"
    created_at = datetime.now(timezone.utc).isoformat()

    conn = connect(database_path)
    create_session(
        conn,
        session_id=session_id,
        source_path=str(source_path),
        source_type="video",
        created_at=created_at,
        notes=notes,
    )

    frame_index = 0
    pending_rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        timestamp_seconds = frame_index / fps

        result = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        if len(detections) == 0:
            continue

        detections = detections[detections.confidence >= confidence_threshold]
        tracked = tracker.update_with_detections(detections)

        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            class_id = int(tracked.class_id[i]) if tracked.class_id is not None else None
            confidence = float(tracked.confidence[i]) if tracked.confidence is not None else None
            tracker_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
            class_name = result.names.get(class_id, str(class_id)) if class_id is not None else None

            center_x = float((x1 + x2) / 2)
            center_y = float((y1 + y2) / 2)

            # SAM 3 segmentation is intentionally an explicit adapter boundary.
            # Set mask_area after the project-specific SAM 3 integration returns
            # a validated mask for this tracked detection.
            mask_area = None

            pending_rows.append(
                (
                    session_id,
                    frame_index,
                    timestamp_seconds,
                    tracker_id,
                    class_id,
                    class_name,
                    confidence,
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    center_x,
                    center_y,
                    mask_area,
                    "",
                )
            )

        if len(pending_rows) >= 500:
            insert_observations(conn, pending_rows)
            pending_rows.clear()

    if pending_rows:
        insert_observations(conn, pending_rows)

    cap.release()
    conn.close()
    return session_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 13 recorded-video tracking pipeline")
    parser.add_argument("source", help="Path to recorded video")
    parser.add_argument("--db", default="data/project13.sqlite3")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sid = process_video(
        source=args.source,
        database_path=args.db,
        model_name=args.model,
        confidence_threshold=args.confidence,
        notes=args.notes,
    )
    print(f"Completed session: {sid}")
