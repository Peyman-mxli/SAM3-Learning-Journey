"""Evaluate predicted segmentation masks against human-reviewed masks.

Manifest CSV columns:
    sample_id,predicted_mask,ground_truth_mask

Masks must be single-channel images or images whose non-zero pixels represent
foreground. The evaluator computes per-sample and aggregate mask IoU/Dice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def metrics(pred_path: Path, gt_path: Path):
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if pred is None or gt is None:
        raise FileNotFoundError(f"Missing mask: {pred_path} or {gt_path}")
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shapes differ: {pred.shape} vs {gt.shape}")

    p = pred > 0
    g = gt > 0
    intersection = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    p_area = int(p.sum())
    g_area = int(g.sum())

    iou = 1.0 if union == 0 else intersection / union
    dice = 1.0 if p_area + g_area == 0 else (2 * intersection) / (p_area + g_area)
    return intersection, union, p_area, g_area, iou, dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", default="results/segmentation_evaluation.csv")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    df = pd.read_csv(manifest_path)
    required = {"sample_id", "predicted_mask", "ground_truth_mask"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    rows = []
    base = manifest_path.parent
    for _, row in df.iterrows():
        vals = metrics(base / row.predicted_mask, base / row.ground_truth_mask)
        rows.append({
            "sample_id": row.sample_id,
            "intersection_pixels": vals[0],
            "union_pixels": vals[1],
            "predicted_area_pixels": vals[2],
            "ground_truth_area_pixels": vals[3],
            "mask_iou": vals[4],
            "dice": vals[5],
        })

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"\nMean mask IoU: {out.mask_iou.mean():.4f}")
    print(f"Mean Dice: {out.dice.mean():.4f}")


if __name__ == "__main__":
    main()
