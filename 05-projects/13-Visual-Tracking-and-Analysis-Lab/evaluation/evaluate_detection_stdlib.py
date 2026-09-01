from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED = {"frame_index", "class_name", "x1", "y1", "x2", "y2"}


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    missing = REQUIRED - set(rows[0])
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return rows


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, (a["x1"], a["y1"], a["x2"], a["y2"]))
    bx1, by1, bx2, by2 = map(float, (b["x1"], b["y1"], b["x2"], b["y2"]))
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def score(preds, gts, threshold):
    frames = sorted({int(float(r["frame_index"])) for r in preds + gts})
    tp = fp = fn = 0
    matched_ious = []
    per_class = {}

    for frame in frames:
        pframe = [r for r in preds if int(float(r["frame_index"])) == frame]
        gframe = [r for r in gts if int(float(r["frame_index"])) == frame]
        used = set()

        for p in pframe:
            cls = p["class_name"]
            per_class.setdefault(cls, {"tp":0,"fp":0,"fn":0})
            best_i, best = None, 0.0
            for i, g in enumerate(gframe):
                if i in used or g["class_name"] != cls:
                    continue
                iou = box_iou(p, g)
                if iou > best:
                    best_i, best = i, iou
            if best_i is not None and best >= threshold:
                tp += 1
                per_class[cls]["tp"] += 1
                used.add(best_i)
                matched_ious.append(best)
            else:
                fp += 1
                per_class[cls]["fp"] += 1

        for i, g in enumerate(gframe):
            if i not in used:
                fn += 1
                cls = g["class_name"]
                per_class.setdefault(cls, {"tp":0,"fp":0,"fn":0})
                per_class[cls]["fn"] += 1

    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0

    for cls, m in per_class.items():
        p = m["tp"]/(m["tp"]+m["fp"]) if m["tp"]+m["fp"] else 0.0
        r = m["tp"]/(m["tp"]+m["fn"]) if m["tp"]+m["fn"] else 0.0
        m["precision"] = p
        m["recall"] = r
        m["f1"] = 2*p*r/(p+r) if p+r else 0.0

    return {
        "iou_threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_matched_iou": sum(matched_ious)/len(matched_ious) if matched_ious else 0.0,
        "per_class": per_class,
        "confusion_matrix_detection": {
            "rows": ["ground_truth_object", "background"],
            "columns": ["predicted_object", "background"],
            "values": [[tp, fn], [fp, 0]]
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions_csv")
    ap.add_argument("ground_truth_csv")
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--output", default="results/detection_evaluation.json")
    args = ap.parse_args()

    gt = load(args.ground_truth_csv)
    if any(str(r.get("reviewed","")).strip().lower() not in {"1","true","yes"} for r in gt):
        raise ValueError("Ground-truth CSV contains unreviewed rows. Refusing to score.")

    result = score(load(args.predictions_csv), gt, args.iou)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
