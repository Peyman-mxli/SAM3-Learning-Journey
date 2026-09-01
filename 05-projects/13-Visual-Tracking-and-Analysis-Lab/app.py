from pathlib import Path

import pandas as pd
import streamlit as st

from src.database import connect

DB_PATH = Path("data/project13.sqlite3")

st.set_page_config(page_title="Project 13 — Visual Tracking Lab", layout="wide")
st.title("Project 13 — Visual Tracking and Analysis Laboratory")
st.caption("Historical tracking explorer for the SAM 3 computer-vision laboratory project.")

if not DB_PATH.exists():
    st.info(
        "No Project 13 database exists yet. Run src/pipeline.py with a recorded video "
        "to create the first session."
    )
    st.stop()

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

st.subheader("Session history")
st.dataframe(sessions, use_container_width=True)

if sessions.empty:
    conn.close()
    st.stop()

selected = st.selectbox("Select session", sessions["session_id"].tolist())

observations = pd.read_sql_query(
    "SELECT * FROM observations WHERE session_id = ? ORDER BY frame_index",
    conn,
    params=(selected,),
)

c1, c2, c3 = st.columns(3)
c1.metric("Observations", len(observations))
c2.metric(
    "Unique trackers",
    observations["tracker_id"].dropna().nunique() if not observations.empty else 0,
)
c3.metric(
    "Average confidence",
    f"{observations['confidence'].mean():.3f}" if not observations.empty else "N/A",
)

st.subheader("Observations")
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

    class_counts = observations["class_name"].value_counts()
    st.subheader("Class distribution")
    st.bar_chart(class_counts)

conn.close()
