"""
evaluate_ground_truth.py

Ground-truth evaluation for Project 06 —
Visual Tracking and Analysis System.

This script evaluates SAM 3 person segmentation predictions
against the manually annotated COCO Segmentation ground truth.

Required evaluation outputs:

- IoU
- Dice coefficient
- Precision
- Recall
- False positives
- False negatives / omissions
- Binary confusion matrix values

The evaluator uses the existing Project 06 pipeline and does not
train or modify any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


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
from src.metrics import (
    calculate_precision,
    calculate_recall,
    calculate_dice,
)


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_GROUND_TRUTH_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "ground_truth"
    / "roboflow_export"
)

DEFAULT_COCO_PATH = (
    DEFAULT_GROUND_TRUTH_DIR
    / "_annotations.coco.json"
)

DEFAULT_RESULTS_CSV = (
    PROJECT_ROOT
    / "evaluation"
    / "evaluation_metrics.csv"
)

DEFAULT_SUMMARY_JSON = (
    PROJECT_ROOT
    / "evaluation"
    / "evaluation_summary.json"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SAM 3 person segmentation against "
            "manual COCO ground-truth annotations."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the local SAM 3 checkpoint.",
    )

    parser.add_argument(
        "--ground-truth-dir",
        default=str(DEFAULT_GROUND_TRUTH_DIR),
        help="Directory containing COCO images and annotations.",
    )

    parser.add_argument(
        "--coco",
        default=str(DEFAULT_COCO_PATH),
        help="Path to _annotations.coco.json.",
    )

    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_RESULTS_CSV),
        help="Per-image evaluation CSV output.",
    )

    parser.add_argument(
        "--summary-json",
        default=str(DEFAULT_SUMMARY_JSON),
        help="Evaluation summary JSON output.",
    )

    parser.add_argument(
        "--prompt",
        default="person",
        help="SAM 3 text prompt. Default: person.",
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model used by the existing Project 06 pipeline.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="YOLO confidence threshold.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="SAM 3 device. Default: cuda.",
    )

    return parser.parse_args()


# ============================================================
# COCO LOADING
# ============================================================

def load_coco(
    coco_path: Path,
) -> dict:
    """
    Load the COCO ground-truth JSON file.
    """

    if not coco_path.exists():
        raise FileNotFoundError(
            f"COCO annotation file not found: {coco_path}"
        )

    with coco_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def get_person_category_id(
    coco: dict,
) -> int:
    """
    Return the category ID for the person class.
    """

    for category in coco.get(
        "categories",
        []
    ):
        if (
            str(
                category.get(
                    "name",
                    ""
                )
            ).strip().lower()
            == "person"
        ):
            return int(
                category["id"]
            )

    raise RuntimeError(
        "The COCO dataset does not contain a 'person' category."
    )


# ============================================================
# COCO POLYGON → BINARY MASK
# ============================================================

def polygon_to_mask(
    segmentation,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert COCO polygon segmentation into one binary mask.
    """

    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    if not segmentation:
        return mask

    for polygon in segmentation:

        if not polygon:
            continue

        coordinates = np.asarray(
            polygon,
            dtype=np.float32,
        ).reshape(
            -1,
            2,
        )

        coordinates = np.round(
            coordinates
        ).astype(
            np.int32
        )

        cv2.fillPoly(
            mask,
            [
                coordinates
            ],
            1,
        )

    return mask.astype(
        bool
    )


def build_ground_truth_masks(
    coco: dict,
    image_id: int,
    person_category_id: int,
    width: int,
    height: int,
) -> list[np.ndarray]:
    """
    Build all person ground-truth masks for one image.
    """

    masks = []

    for annotation in coco.get(
        "annotations",
        []
    ):

        if int(
            annotation.get(
                "image_id",
                -1
            )
        ) != image_id:
            continue

        if int(
            annotation.get(
                "category_id",
                -1
            )
        ) != person_category_id:
            continue

        segmentation = annotation.get(
            "segmentation"
        )

        if not isinstance(
            segmentation,
            list
        ):
            continue

        mask = polygon_to_mask(
            segmentation=segmentation,
            width=width,
            height=height,
        )

        if np.any(
            mask
        ):
            masks.append(
                mask
            )

    return masks


# ============================================================
# SAM 3 OUTPUT → BINARY MASKS
# ============================================================

def extract_predicted_masks(
    segmentation_output,
    width: int,
    height: int,
) -> list[np.ndarray]:
    """
    Extract binary masks from SAM 3 segmentation output.
    """

    if segmentation_output is None:
        return []

    masks = segmentation_output.get(
        "masks"
    )

    if masks is None:
        return []

    predicted_masks = []

    for mask in masks:

        mask_np = (
            mask
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        mask_np = np.squeeze(
            mask_np
        )

        if mask_np.ndim != 2:
            continue

        if (
            mask_np.shape[1] != width
            or mask_np.shape[0] != height
        ):
            mask_np = cv2.resize(
                mask_np,
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        binary_mask = (
            mask_np > 0.5
        )

        if np.any(
            binary_mask
        ):
            predicted_masks.append(
                binary_mask
            )

    return predicted_masks


# ============================================================
# MASK METRICS
# ============================================================

def calculate_mask_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """
    Calculate segmentation-mask IoU.
    """

    intersection = np.logical_and(
        mask_a,
        mask_b,
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b,
    ).sum()

    if union == 0:
        return 0.0

    return float(
        intersection / union
    )


def calculate_mask_dice(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """
    Calculate Dice coefficient using the existing
    Project 06 calculate_dice function.
    """

    intersection = int(
        np.logical_and(
            mask_a,
            mask_b,
        ).sum()
    )

    predicted_area = int(
        mask_a.sum()
    )

    ground_truth_area = int(
        mask_b.sum()
    )

    return float(
        calculate_dice(
            intersection=intersection,
            predicted_area=predicted_area,
            ground_truth_area=ground_truth_area,
        )
    )


# ============================================================
# INSTANCE MATCHING
# ============================================================

def match_instances(
    predicted_masks: list[np.ndarray],
    ground_truth_masks: list[np.ndarray],
    iou_threshold: float = 0.50,
) -> tuple[list[dict], int, int, int]:
    """
    Match predicted masks to ground-truth masks.

    A prediction can match only one ground-truth instance.

    Matching uses the highest available IoU first.
    """

    candidate_matches = []

    for prediction_index, predicted_mask in enumerate(
        predicted_masks
    ):

        for ground_truth_index, ground_truth_mask in enumerate(
            ground_truth_masks
        ):

            iou = calculate_mask_iou(
                predicted_mask,
                ground_truth_mask,
            )

            candidate_matches.append(
                {
                    "prediction_index":
                        prediction_index,

                    "ground_truth_index":
                        ground_truth_index,

                    "iou":
                        iou,
                }
            )

    candidate_matches.sort(
        key=lambda item: item["iou"],
        reverse=True,
    )

    matched_predictions = set()
    matched_ground_truth = set()

    matches = []

    for candidate in candidate_matches:

        if candidate["iou"] < iou_threshold:
            continue

        prediction_index = candidate[
            "prediction_index"
        ]

        ground_truth_index = candidate[
            "ground_truth_index"
        ]

        if prediction_index in matched_predictions:
            continue

        if ground_truth_index in matched_ground_truth:
            continue

        matched_predictions.add(
            prediction_index
        )

        matched_ground_truth.add(
            ground_truth_index
        )

        dice = calculate_mask_dice(
            predicted_masks[
                prediction_index
            ],
            ground_truth_masks[
                ground_truth_index
            ],
        )

        matches.append(
            {
                "prediction_index":
                    prediction_index,

                "ground_truth_index":
                    ground_truth_index,

                "iou":
                    float(
                        candidate["iou"]
                    ),

                "dice":
                    float(
                        dice
                    ),
            }
        )

    true_positives = len(
        matches
    )

    false_positives = (
        len(
            predicted_masks
        )
        - true_positives
    )

    false_negatives = (
        len(
            ground_truth_masks
        )
        - true_positives
    )

    return (
        matches,
        true_positives,
        false_positives,
        false_negatives,
    )


# ============================================================
# PIXEL CONFUSION MATRIX
# ============================================================

def union_masks(
    masks: list[np.ndarray],
    height: int,
    width: int,
) -> np.ndarray:
    """
    Combine instance masks into one binary semantic mask.
    """

    combined = np.zeros(
        (
            height,
            width,
        ),
        dtype=bool,
    )

    for mask in masks:
        combined = np.logical_or(
            combined,
            mask,
        )

    return combined


def calculate_pixel_confusion(
    predicted_union: np.ndarray,
    ground_truth_union: np.ndarray,
) -> dict:
    """
    Calculate binary pixel-level confusion-matrix values.
    """

    true_positive = int(
        np.logical_and(
            predicted_union,
            ground_truth_union,
        ).sum()
    )

    false_positive = int(
        np.logical_and(
            predicted_union,
            np.logical_not(
                ground_truth_union
            ),
        ).sum()
    )

    false_negative = int(
        np.logical_and(
            np.logical_not(
                predicted_union
            ),
            ground_truth_union,
        ).sum()
    )

    true_negative = int(
        np.logical_and(
            np.logical_not(
                predicted_union
            ),
            np.logical_not(
                ground_truth_union
            ),
        ).sum()
    )

    return {
        "tp_pixels":
            true_positive,

        "fp_pixels":
            false_positive,

        "fn_pixels":
            false_negative,

        "tn_pixels":
            true_negative,
    }


# ============================================================
# SINGLE IMAGE EVALUATION
# ============================================================

def evaluate_image(
    pipeline: VisualAnalysisPipeline,
    image_path: Path,
    image_record: dict,
    coco: dict,
    person_category_id: int,
    prompt: str,
) -> dict:
    """
    Evaluate one image.
    """

    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[
        :2
    ]

    ground_truth_masks = (
        build_ground_truth_masks(
            coco=coco,
            image_id=int(
                image_record["id"]
            ),
            person_category_id=person_category_id,
            width=width,
            height=height,
        )
    )

    pipeline.reset_tracker()

    result = pipeline.process_image(
        image_bgr=image,
        segmentation_prompt=prompt,
    )

    predicted_masks = (
        extract_predicted_masks(
            segmentation_output=result[
                "segmentation"
            ],
            width=width,
            height=height,
        )
    )

    (
        matches,
        true_positives,
        false_positives,
        false_negatives,
    ) = match_instances(
        predicted_masks=predicted_masks,
        ground_truth_masks=ground_truth_masks,
        iou_threshold=0.50,
    )

    precision = calculate_precision(
        true_positives=true_positives,
        false_positives=false_positives,
    )

    recall = calculate_recall(
        true_positives=true_positives,
        false_negatives=false_negatives,
    )

    if matches:

        average_iou = float(
            np.mean(
                [
                    match["iou"]
                    for match in matches
                ]
            )
        )

        average_dice = float(
            np.mean(
                [
                    match["dice"]
                    for match in matches
                ]
            )
        )

    else:

        average_iou = 0.0
        average_dice = 0.0

    predicted_union = union_masks(
        predicted_masks,
        height,
        width,
    )

    ground_truth_union = union_masks(
        ground_truth_masks,
        height,
        width,
    )

    pixel_confusion = (
        calculate_pixel_confusion(
            predicted_union,
            ground_truth_union,
        )
    )

    return {
        "image_id":
            int(
                image_record["id"]
            ),

        "file_name":
            image_record["file_name"],

        "ground_truth_instances":
            len(
                ground_truth_masks
            ),

        "predicted_instances":
            len(
                predicted_masks
            ),

        "true_positives":
            true_positives,

        "false_positives":
            false_positives,

        "false_negatives":
            false_negatives,

        "precision":
            round(
                precision,
                4,
            ),

        "recall":
            round(
                recall,
                4,
            ),

        "average_iou":
            round(
                average_iou,
                4,
            ),

        "average_dice":
            round(
                average_dice,
                4,
            ),

        "tp_pixels":
            pixel_confusion[
                "tp_pixels"
            ],

        "fp_pixels":
            pixel_confusion[
                "fp_pixels"
            ],

        "fn_pixels":
            pixel_confusion[
                "fn_pixels"
            ],

        "tn_pixels":
            pixel_confusion[
                "tn_pixels"
            ],
    }


# ============================================================
# CSV EXPORT
# ============================================================

def save_csv(
    results: list[dict],
    output_path: Path,
) -> None:
    """
    Save per-image evaluation results.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not results:
        return

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    results: list[dict],
) -> dict:
    """
    Build the complete evaluation summary.
    """

    total_ground_truth = sum(
        result[
            "ground_truth_instances"
        ]
        for result in results
    )

    total_predictions = sum(
        result[
            "predicted_instances"
        ]
        for result in results
    )

    total_true_positives = sum(
        result[
            "true_positives"
        ]
        for result in results
    )

    total_false_positives = sum(
        result[
            "false_positives"
        ]
        for result in results
    )

    total_false_negatives = sum(
        result[
            "false_negatives"
        ]
        for result in results
    )

    precision = calculate_precision(
        total_true_positives,
        total_false_positives,
    )

    recall = calculate_recall(
        total_true_positives,
        total_false_negatives,
    )

    average_iou = float(
        np.mean(
            [
                result[
                    "average_iou"
                ]
                for result in results
            ]
        )
    )

    average_dice = float(
        np.mean(
            [
                result[
                    "average_dice"
                ]
                for result in results
            ]
        )
    )

    total_tp_pixels = sum(
        result[
            "tp_pixels"
        ]
        for result in results
    )

    total_fp_pixels = sum(
        result[
            "fp_pixels"
        ]
        for result in results
    )

    total_fn_pixels = sum(
        result[
            "fn_pixels"
        ]
        for result in results
    )

    total_tn_pixels = sum(
        result[
            "tn_pixels"
        ]
        for result in results
    )

    return {
        "evaluated_images":
            len(
                results
            ),

        "ground_truth_instances":
            total_ground_truth,

        "predicted_instances":
            total_predictions,

        "true_positives":
            total_true_positives,

        "false_positives":
            total_false_positives,

        "false_negatives_omissions":
            total_false_negatives,

        "precision":
            round(
                precision,
                4,
            ),

        "recall":
            round(
                recall,
                4,
            ),

        "average_iou":
            round(
                average_iou,
                4,
            ),

        "average_dice":
            round(
                average_dice,
                4,
            ),

        "confusion_matrix": {
            "true_positive_pixels":
                total_tp_pixels,

            "false_positive_pixels":
                total_fp_pixels,

            "false_negative_pixels":
                total_fn_pixels,

            "true_negative_pixels":
                total_tn_pixels,
        },
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation(
    args: argparse.Namespace,
) -> None:
    """
    Execute the complete ground-truth evaluation.
    """

    ground_truth_dir = Path(
        args.ground_truth_dir
    )

    coco_path = Path(
        args.coco
    )

    output_csv = Path(
        args.output_csv
    )

    summary_json = Path(
        args.summary_json
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM 3 checkpoint not found: {checkpoint_path}"
        )

    coco = load_coco(
        coco_path
    )

    person_category_id = (
        get_person_category_id(
            coco
        )
    )

    images = coco.get(
        "images",
        []
    )

    print()
    print("=" * 80)
    print(
        "PROJECT 06 — GROUND-TRUTH EVALUATION"
    )
    print("=" * 80)

    print(
        f"Images: {len(images)}"
    )

    print(
        f"Ground truth: {coco_path}"
    )

    print(
        f"Prompt: {args.prompt}"
    )

    print()

    print(
        "[INFO] Loading Project 06 pipeline..."
    )

    pipeline = VisualAnalysisPipeline(
        sam3_checkpoint_path=str(
            checkpoint_path
        ),
        model_name=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    results = []

    for index, image_record in enumerate(
        images,
        start=1,
    ):

        image_path = (
            ground_truth_dir
            / image_record[
                "file_name"
            ]
        )

        print(
            f"[{index:02d}/{len(images):02d}] "
            f"{image_record['file_name']}"
        )

        result = evaluate_image(
            pipeline=pipeline,
            image_path=image_path,
            image_record=image_record,
            coco=coco,
            person_category_id=person_category_id,
            prompt=args.prompt,
        )

        results.append(
            result
        )

        print(
            "    "
            f"GT={result['ground_truth_instances']} | "
            f"Pred={result['predicted_instances']} | "
            f"TP={result['true_positives']} | "
            f"FP={result['false_positives']} | "
            f"FN={result['false_negatives']} | "
            f"IoU={result['average_iou']:.4f} | "
            f"Dice={result['average_dice']:.4f}"
        )

    save_csv(
        results,
        output_csv,
    )

    summary = build_summary(
        results
    )

    summary_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print()
    print("=" * 80)
    print(
        "EVALUATION SUMMARY"
    )
    print("=" * 80)

    print(
        f"Evaluated Images: "
        f"{summary['evaluated_images']}"
    )

    print(
        f"Ground-Truth Instances: "
        f"{summary['ground_truth_instances']}"
    )

    print(
        f"Predicted Instances: "
        f"{summary['predicted_instances']}"
    )

    print(
        f"True Positives: "
        f"{summary['true_positives']}"
    )

    print(
        f"False Positives: "
        f"{summary['false_positives']}"
    )

    print(
        f"False Negatives / Omissions: "
        f"{summary['false_negatives_omissions']}"
    )

    print(
        f"Precision: "
        f"{summary['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{summary['recall']:.4f}"
    )

    print(
        f"Average IoU: "
        f"{summary['average_iou']:.4f}"
    )

    print(
        f"Average Dice: "
        f"{summary['average_dice']:.4f}"
    )

    print()

    print(
        "[SUCCESS] Per-image metrics:"
    )

    print(
        output_csv
    )

    print()

    print(
        "[SUCCESS] Evaluation summary:"
    )

    print(
        summary_json
    )

    print()


def main() -> None:
    """
    Command-line entry point.
    """

    args = parse_arguments()

    run_evaluation(
        args
    )


if __name__ == "__main__":
    main()
