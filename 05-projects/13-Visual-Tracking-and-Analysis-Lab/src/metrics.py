from __future__ import annotations

import numpy as np


def box_iou(box_a, box_b) -> float:
    """Intersection over Union for [x1, y1, x2, y2] boxes."""
    x_left = max(float(box_a[0]), float(box_b[0]))
    y_top = max(float(box_a[1]), float(box_b[1]))
    x_right = min(float(box_a[2]), float(box_b[2]))
    y_bottom = min(float(box_a[3]), float(box_b[3]))

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (float(box_a[2]) - float(box_a[0]))) * max(
        0.0, (float(box_a[3]) - float(box_a[1]))
    )
    area_b = max(0.0, (float(box_b[2]) - float(box_b[0]))) * max(
        0.0, (float(box_b[3]) - float(box_b[1]))
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def precision(tp: int, fp: int) -> float:
    denominator = tp + fp
    return tp / denominator if denominator else 0.0


def recall(tp: int, fn: int) -> float:
    denominator = tp + fn
    return tp / denominator if denominator else 0.0


def f1_score(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def mask_iou(pred_mask, gt_mask) -> float:
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shapes differ: {pred.shape} != {gt.shape}")
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / union) if union else 1.0


def dice_score(pred_mask, gt_mask) -> float:
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shapes differ: {pred.shape} != {gt.shape}")
    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()
    return float((2 * intersection) / denominator) if denominator else 1.0
