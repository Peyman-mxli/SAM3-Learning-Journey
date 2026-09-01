from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.database import connect


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "project13.sqlite3"
RESULTS_DIR = PROJECT_ROOT / "results"

RUN_SUMMARY = RESULTS_DIR / "lab13_0d46777217_run_summary.json"
DETECTION_EVAL = RESULTS_DIR / "detection_evaluation.json"

st.set_page_config(
    page_title="Project 13 — Visual Tracking Lab",
    layout="wide",
)

st.title("Project 13 — Visual Tracking and Analysis Laboratory")
st.caption(
    "Portfolio dashboard for verified execution evidence, human-reviewed "
    "detection evaluation, and historical tracking sessions."
)

tab_verified, tab_database, tab_validation = st.tabs(
    ["Verified evidence", "Session database", "Validation suite"]
)


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


with tab_verified:
    st.subheader("Definitive persistent run")

    run = load_json(RUN_SUMMARY)
    if run:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames", run.get("frames", "N/A"))
        c2.metric("Observations", run.get("total_observations", "N/A"))
        c3.metric("Unique tracker IDs", run.get("unique_tracker_ids", "N/A"))
        c4.metric(
            "Average confidence",
            f"{run.get('average_confidence', 0):.4f}",
        )

        c5, c6, c7 = st.columns(3)
        c5.metric("SAM 3 mask observations", run.get("sam_mask_observations", "N/A"))
        c6.metric(
            "Average SAM mask area",
            f"{run.get('average_sam_mask_area', 0):.2f} px",
        )
        c7.metric("Pipeline", run.get("pipeline", "N/A"))

        st.json(run)
    else:
        st.info("Verified run summary is not present in results/.")

    st.subheader("Human-reviewed detection evaluation")
    evaluation = load_json(DETECTION_EVAL)

    if evaluation:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Precision", f"{evaluation.get('precision', 0):.4f}")
        e2.metric("Recall", f"{evaluation.get('recall', 0):.4f}")
        e3.metric("F1", f"{evaluation.get('f1_score', 0):.4f}")
        e4.metric(
            "Mean matched IoU",
            f"{evaluation.get('mean_matched_iou', 0):.4f}",
        )

        counts = pd.DataFrame(
            {
                "Metric": ["True positives", "False positives", "False negatives"],
                "Count": [
                    evaluation.get("true_positives", 0),
                    evaluation.get("false_positives", 0),
                    evaluation.get("false_negatives", 0),
                ],
            }
        )
        st.bar_chart(counts.set_index("Metric"))

        per_class = evaluation.get("per_class", {})
        if per_class:
            rows = []
            for class_name, values in per_class.items():
                rows.append({"class": class_name, **values})
            st.subheader("Per-class metrics")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        matrix = evaluation.get("confusion_matrix_detection")
        if matrix:
            st.subheader("Detection confusion matrix")
            matrix_df = pd.DataFrame(
                matrix["values"],
                index=matrix["rows"],
                columns=matrix["columns"],
            )
            st.dataframe(matrix_df, use_container_width=True)
    else:
        st.info("Detection evaluation JSON is not present in results/.")


with tab_database:
    st.subheader("Historical session explorer")

    if not DB_PATH.exists():
        st.info(
            "The persistent SQLite database is not committed to GitHub. "
            "Run src/pipeline.py locally/Colab or copy the persistent database "
            "from Google Drive to data/project13.sqlite3 to enable this tab."
        )
    else:
        conn = connect(DB_PATH)

        sessions = pd.read_sql_query(
            """
            SELECT
                s.session_id,
                s.source_path,
                s.created_at,
                s.notes,
                COUNT(o.observation_id) AS observations,
                COUNT(DISTINCT o.tracker_id) AS unique_trackers,
                AVG(o.confidence) AS average_confidence
            FROM sessions s
            LEFT JOIN observations o ON o.session_id = s.session_id
            GROUP BY s.session_id, s.source_path, s.created_at, s.notes
            ORDER BY s.created_at DESC
            """,
            conn,
        )

        st.dataframe(sessions, use_container_width=True)

        if not sessions.empty:
            selected = st.selectbox(
                "Select session",
                sessions["session_id"].tolist(),
            )

            observations = pd.read_sql_query(
                """
                SELECT *
                FROM observations
                WHERE session_id = ?
                ORDER BY frame_index
                """,
                conn,
                params=(selected,),
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Observations", len(observations))
            c2.metric(
                "Unique trackers",
                observations["tracker_id"].dropna().nunique()
                if not observations.empty
                else 0,
            )
            c3.metric(
                "Average confidence",
                f"{observations['confidence'].mean():.3f}"
                if not observations.empty
                else "N/A",
            )

            st.dataframe(observations, use_container_width=True)

            if not observations.empty:
                tracker_counts = (
                    observations.dropna(subset=["tracker_id"])
                    .groupby("tracker_id")
                    .size()
                    .rename("observations")
                    .reset_index()
                )
                st.subheader("Tracker persistence")
                st.bar_chart(tracker_counts.set_index("tracker_id"))

                st.subheader("Class distribution")
                st.bar_chart(observations["class_name"].value_counts())

        conn.close()


with tab_validation:
    st.subheader("Controlled robustness validation")

    st.markdown(
        """
Project 13 includes a five-video controlled validation suite:

1. baseline
2. low light
3. partial occlusion
4. motion blur
5. reduced scale

Run:

\`\`\`bash
python evaluation/create_condition_variants.py data/input/tracking_test_01.mp4
python evaluation/run_validation_suite.py
python evaluation/analyze_validation_conditions.py
\`\`\`

The resulting condition summaries are **robustness indicators**. They are not
ground-truth tracking accuracy unless human-reviewed temporal identities are
also supplied.
"""
    )

    condition_summary = RESULTS_DIR / "validation" / "condition_summary.csv"
    condition_deltas = RESULTS_DIR / "validation" / "condition_deltas.csv"

    if condition_summary.exists():
        st.subheader("Condition summary")
        st.dataframe(pd.read_csv(condition_summary), use_container_width=True)

    if condition_deltas.exists():
        st.subheader("Condition deltas vs baseline")
        st.dataframe(pd.read_csv(condition_deltas), use_container_width=True)

    if not condition_summary.exists() and not condition_deltas.exists():
        st.info(
            "Validation-result CSV files will appear here after the controlled "
            "validation suite is executed."
        )
