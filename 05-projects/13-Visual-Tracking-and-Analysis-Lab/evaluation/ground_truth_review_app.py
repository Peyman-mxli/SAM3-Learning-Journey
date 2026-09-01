"""Streamlit ground-truth review interface for Project 13.

Run:
    streamlit run evaluation/ground_truth_review_app.py

The reviewer can:
- inspect each exported frame,
- accept/reject candidate detections,
- correct temporal identity IDs,
- correct bounding boxes numerically,
- mark rows reviewed,
- export a clean tracking_ground_truth.csv.

Segmentation masks are kept as a separate evidence track because drawing and
reviewing pixel-accurate polygons requires a dedicated mask annotation tool.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "evaluation" / "review_packet"
TRACKING_REVIEW = PACKET / "tracking_review.csv"
OUTPUT_GT = ROOT / "evaluation" / "tracking_ground_truth.csv"

st.set_page_config(page_title="Project 13 Ground-Truth Review", layout="wide")
st.title("Project 13 — Tracking Ground-Truth Review")
st.warning(
    "Candidate IDs and boxes are model suggestions. They become reference "
    "ground truth only after a reviewer checks and marks them reviewed."
)

if not TRACKING_REVIEW.exists():
    st.error(
        "Review packet not found. Run generate_ground_truth_review_packet.py first."
    )
    st.stop()

df = pd.read_csv(TRACKING_REVIEW)
frames = sorted(df.frame_index.unique().tolist())

selected = st.selectbox("Frame", frames)
frame_path = PACKET / "frames" / f"frame_{int(selected):04d}.jpg"
if frame_path.exists():
    st.image(str(frame_path), caption=f"Frame {selected}", use_container_width=True)

subset = df[df.frame_index == selected].copy()

st.subheader("Review candidate detections")
edited = st.data_editor(
    subset[
        [
            "candidate_track_id",
            "review_track_id",
            "class_name",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "keep",
            "reviewed",
            "notes",
        ]
    ],
    use_container_width=True,
    num_rows="dynamic",
    key=f"editor_{selected}",
)

if st.button("Save this frame review"):
    keep_index = df.index[df.frame_index == selected]
    replacement = edited.copy()
    replacement.insert(0, "frame_index", selected)

    if len(replacement) != len(keep_index):
        df = df.drop(index=keep_index)
        df = pd.concat([df, replacement], ignore_index=True)
    else:
        for col in replacement.columns:
            df.loc[keep_index, col] = replacement[col].to_numpy()

    df.to_csv(TRACKING_REVIEW, index=False)
    st.success("Frame review saved.")

st.divider()

reviewed_count = int(df.reviewed.fillna(False).astype(bool).sum())
st.metric("Reviewed rows", f"{reviewed_count}/{len(df)}")

if st.button("Export reviewed tracking ground truth"):
    current = pd.read_csv(TRACKING_REVIEW)
    accepted = current[
        current["reviewed"].fillna(False).astype(bool)
        & current["keep"].fillna(False).astype(bool)
    ].copy()

    if accepted.empty:
        st.error("No reviewed/accepted rows are available.")
    else:
        gt = accepted[
            [
                "frame_index",
                "review_track_id",
                "class_name",
                "x1",
                "y1",
                "x2",
                "y2",
                "notes",
            ]
        ].rename(columns={"review_track_id": "track_id"})
        gt["reviewed"] = True
        gt.to_csv(OUTPUT_GT, index=False)
        st.success(f"Saved {OUTPUT_GT.relative_to(ROOT)}")
