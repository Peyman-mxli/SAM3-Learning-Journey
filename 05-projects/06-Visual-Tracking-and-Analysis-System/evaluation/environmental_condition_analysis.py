"""Environmental condition evaluation for Project 06.

This module answers the project's guiding research question:

    How do variations in lighting and occlusion affect object-tracking accuracy
    in recorded videos?

It consumes a condition-labeled CSV and produces reproducible summaries for
lighting, occlusion, and their interaction. It never invents measurements:
only rows containing observed counts are analyzed.

Required input:
    evaluation/environmental_conditions.csv

Outputs:
    reports/environmental_condition_results.csv
    reports/lighting_condition_summary.csv
    reports/occlusion_condition_summary.csv
    reports/environmental_condition_chart.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_CSV = EVALUATION_DIR / "environmental_conditions.csv"
RESULTS_CSV = REPORTS_DIR / "environmental_condition_results.csv"
LIGHTING_CSV = REPORTS_DIR / "lighting_condition_summary.csv"
OCCLUSION_CSV = REPORTS_DIR / "occlusion_condition_summary.csv"
CHART_PNG = REPORTS_DIR / "environmental_condition_chart.png"

REQUIRED_COLUMNS = {
    "case_id",
    "session_id",
    "lighting_condition",
    "occlusion_condition",
    "true_positives",
    "false_positives",
    "false_negatives",
    "id_switches",
    "tracked_id_opportunities",
    "average_confidence",
}


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_CSV.relative_to(PROJECT_ROOT)}. "
            "Copy environmental_conditions_template.csv to "
            "environmental_conditions.csv and record real observations."
        )

    df = pd.read_csv(INPUT_CSV)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if df.empty:
        raise ValueError(
            "environmental_conditions.csv contains no observations. "
            "Add measured cases before running the analysis."
        )

    numeric = [
        "true_positives",
        "false_positives",
        "false_negatives",
        "id_switches",
        "tracked_id_opportunities",
        "average_confidence",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df


def add_case_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["precision"] = result.apply(
        lambda r: safe_divide(r.true_positives, r.true_positives + r.false_positives),
        axis=1,
    )
    result["recall"] = result.apply(
        lambda r: safe_divide(r.true_positives, r.true_positives + r.false_negatives),
        axis=1,
    )
    result["f1"] = result.apply(
        lambda r: safe_divide(
            2 * r.precision * r.recall,
            r.precision + r.recall,
        ),
        axis=1,
    )
    result["id_switch_rate"] = result.apply(
        lambda r: safe_divide(r.id_switches, r.tracked_id_opportunities),
        axis=1,
    )
    return result


def grouped_summary(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for name, group in df.groupby(group_column, dropna=False):
        tp = float(group["true_positives"].sum())
        fp = float(group["false_positives"].sum())
        fn = float(group["false_negatives"].sum())
        switches = float(group["id_switches"].sum())
        opportunities = float(group["tracked_id_opportunities"].sum())

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        rows.append(
            {
                group_column: name,
                "cases": int(len(group)),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "id_switches": int(switches),
                "tracked_id_opportunities": int(opportunities),
                "id_switch_rate": safe_divide(switches, opportunities),
                "average_confidence": float(group["average_confidence"].mean()),
            }
        )
    return pd.DataFrame(rows)


def save_chart(lighting: pd.DataFrame, occlusion: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(lighting["lighting_condition"].astype(str), lighting["f1"])
    axes[0].set_title("Tracking F1 by Lighting Condition")
    axes[0].set_xlabel("Lighting")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 1)

    axes[1].bar(occlusion["occlusion_condition"].astype(str), occlusion["f1"])
    axes[1].set_title("Tracking F1 by Occlusion Condition")
    axes[1].set_xlabel("Occlusion")
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0, 1)

    fig.suptitle("Project 06 — Environmental Condition Evaluation")
    fig.tight_layout()
    fig.savefig(CHART_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = add_case_metrics(load_data())
    lighting = grouped_summary(df, "lighting_condition")
    occlusion = grouped_summary(df, "occlusion_condition")

    df.round(4).to_csv(RESULTS_CSV, index=False)
    lighting.round(4).to_csv(LIGHTING_CSV, index=False)
    occlusion.round(4).to_csv(OCCLUSION_CSV, index=False)
    save_chart(lighting, occlusion)

    print("Environmental-condition evaluation completed.")
    print(f"Cases evaluated: {len(df)}")
    print(f"Saved: {RESULTS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {LIGHTING_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {OCCLUSION_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {CHART_PNG.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
