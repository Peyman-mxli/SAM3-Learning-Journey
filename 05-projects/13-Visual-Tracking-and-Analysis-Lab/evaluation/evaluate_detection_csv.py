from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.metrics import box_iou, f1_score, precision, recall


REQUIRED_COLUMNS = {
    "frame_index",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
}


def greedy_match(predictions: pd.DataFrame, ground_truth: pd.DataFrame, iou_threshold: float):
    tp = fp = fn = 0
    matched_ious = []

    frame_ids = sorted(set(predictions.frame_index) | set(ground_truth.frame_index))

    for frame_id in frame_ids:
        pred_frame = predictions[predictions.frame_index == frame_id].reset_index(drop=True)
        gt_frame = ground_truth[ground_truth.frame_index == frame_id].reset_index(drop=True)
        used_gt = set()

        for _, pred in pred_frame.iterrows():
            best_index = None
            best_iou = 0.0

            for gt_index, gt in gt_frame.iterrows():
                if gt_index in used_gt:
                    continue
                if str(pred.class_name) != str(gt.class_name):
                    continue

                iou = box_iou(
                    [pred.x1, pred.y1, pred.x2, pred.y2],
                    [gt.x1, gt.y1, gt.x2, gt.y2],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_index = gt_index

            if best_index is not None and best_iou >= iou_threshold:
                tp += 1
                used_gt.add(best_index)
                matched_ious.append(best_iou)
            else:
                fp += 1

        fn += len(gt_frame) - len(used_gt)

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision(tp, fp),
        "recall": recall(tp, fn),
        "f1_score": f1_score(tp, fp, fn),
        "mean_matched_iou": (
            sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        ),
    }


def validate_columns(df: pd.DataFrame, label: str):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv")
    parser.add_argument("ground_truth_csv")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--output", default="results/detection_evaluation.json")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_csv)
    ground_truth = pd.read_csv(args.ground_truth_csv)
    validate_columns(predictions, "predictions")
    validate_columns(ground_truth, "ground truth")

    summary = greedy_match(predictions, ground_truth, args.iou)
    summary["iou_threshold"] = args.iou

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
