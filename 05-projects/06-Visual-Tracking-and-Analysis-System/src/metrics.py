"""
metrics.py

Evaluation metrics module for the Visual Tracking and Analysis System.

This module provides reusable functions for evaluating detection
and segmentation results.
"""


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Bounding boxes must use the format:

    [x1, y1, x2, y2]
    """

    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (
        (x_right - x_left) *
        (y_bottom - y_top)
    )

    box_a_area = (
        (box_a[2] - box_a[0]) *
        (box_a[3] - box_a[1])
    )

    box_b_area = (
        (box_b[2] - box_b[0]) *
        (box_b[3] - box_b[1])
    )

    union_area = (
        box_a_area +
        box_b_area -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def calculate_precision(true_positives, false_positives):
    """
    Calculate precision.
    """

    denominator = true_positives + false_positives

    if denominator == 0:
        return 0.0

    return true_positives / denominator


def calculate_recall(true_positives, false_negatives):
    """
    Calculate recall.
    """

    denominator = true_positives + false_negatives

    if denominator == 0:
        return 0.0

    return true_positives / denominator


def calculate_dice(intersection, predicted_area, ground_truth_area):
    """
    Calculate the Dice coefficient.

    This metric can be used to evaluate segmentation overlap.
    """

    denominator = predicted_area + ground_truth_area

    if denominator == 0:
        return 0.0

    return (2 * intersection) / denominator
