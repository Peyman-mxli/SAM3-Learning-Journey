from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from database import connect, create_session, insert_observations
from metrics import box_iou


def _sam_mask_areas_for_tracks(
    frame_bgr,
    tracked,
    segmenter,
    prompt: str,
    min_score: float,
    match_iou: float,
):
    """
    Run text-prompt SAM 3 segmentation for a frame and associate returned
    masks to tracked detections by bounding-box IoU.

    Returns one mask-area value per tracked detection. Unmatched detections
    receive None rather than fabricated segmentation values.
    """
    if segmenter is None or len(tracked) == 0:
        return [None] * len(tracked)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    sam = segmenter.segment_text(frame_rgb, prompt)

    if sam.boxes.size == 0 or sam.masks.size == 0:
        return [None] * len(tracked)

    boxes = np.asarray(sam.boxes).reshape(-1, 4)
    masks = np.asarray(sam.masks)
    scores = np.asarray(sam.scores).reshape(-1) if sam.scores.size else np.ones(len(boxes))

    areas = [None] * len(tracked)

    for track_index, track_box in enumerate(tracked.xyxy):
        best_index = None
        best_iou = 0.0

        for sam_index, sam_box in enumerate(boxes):
            if sam_index >= len(scores) or float(scores[sam_index]) < min_score:
                continue
            iou = box_iou(track_box, sam_box)
            if iou > best_iou:
                best_iou = iou
                best_index = sam_index

        if best_index is not None and best_iou >= match_iou and best_index < len(masks):
            areas[track_index] = float(np.count_nonzero(masks[best_index]))

    return areas


def process_video(
    source: str,
    database_path: str,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.35,
    notes: str = "",
    sam_checkpoint: str | None = None,
    sam_prompt: str = "person",
    sam_every: int = 10,
    sam_min_score: float = 0.25,
    sam_match_iou: float = 0.25,
    sam_device: str = "cuda",
) -> str:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    model = YOLO(model_name)
    tracker = sv.ByteTrack()

    segmenter = None
    if sam_checkpoint:
        from segmenter import SAM3TextSegmenter
        segmenter = SAM3TextSegmenter(
            checkpoint_path=sam_checkpoint,
            device=sam_device,
        )

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
        if len(detections) == 0:
            continue

        tracked = tracker.update_with_detections(detections)

        should_segment = (
            segmenter is not None
            and sam_every > 0
            and frame_index % sam_every == 0
        )
        mask_areas = (
            _sam_mask_areas_for_tracks(
                frame_bgr=frame,
                tracked=tracked,
                segmenter=segmenter,
                prompt=sam_prompt,
                min_score=sam_min_score,
                match_iou=sam_match_iou,
            )
            if should_segment
            else [None] * len(tracked)
        )

        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            class_id = int(tracked.class_id[i]) if tracked.class_id is not None else None
            confidence = float(tracked.confidence[i]) if tracked.confidence is not None else None
            tracker_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
            class_name = result.names.get(class_id, str(class_id)) if class_id is not None else None

            center_x = float((x1 + x2) / 2)
            center_y = float((y1 + y2) / 2)
            mask_area = mask_areas[i]

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
                    (
                        f"SAM3 prompt={sam_prompt}"
                        if mask_area is not None
                        else ""
                    ),
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
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--sam-prompt", default="person")
    parser.add_argument("--sam-every", type=int, default=10)
    parser.add_argument("--sam-min-score", type=float, default=0.25)
    parser.add_argument("--sam-match-iou", type=float, default=0.25)
    parser.add_argument("--sam-device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sid = process_video(
        source=args.source,
        database_path=args.db,
        model_name=args.model,
        confidence_threshold=args.confidence,
        notes=args.notes,
        sam_checkpoint=args.sam_checkpoint,
        sam_prompt=args.sam_prompt,
        sam_every=args.sam_every,
        sam_min_score=args.sam_min_score,
        sam_match_iou=args.sam_match_iou,
        sam_device=args.sam_device,
    )
    print(f"Completed session: {sid}")
