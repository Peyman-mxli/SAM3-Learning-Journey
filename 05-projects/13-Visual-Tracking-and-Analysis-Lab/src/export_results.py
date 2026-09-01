from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from database import connect


def export_session(db_path: str, session_id: str, output_dir: str) -> None:
    conn = connect(db_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    observations = pd.read_sql_query(
        "SELECT * FROM observations WHERE session_id = ? ORDER BY frame_index",
        conn,
        params=(session_id,),
    )
    observations.to_csv(out / f"{session_id}_observations.csv", index=False)

    tracker_summary = pd.read_sql_query(
        """
        SELECT
            tracker_id,
            class_name,
            COUNT(*) AS observations,
            AVG(confidence) AS average_confidence,
            MIN(frame_index) AS first_frame,
            MAX(frame_index) AS last_frame,
            AVG(mask_area) AS average_mask_area
        FROM observations
        WHERE session_id = ?
        GROUP BY tracker_id, class_name
        ORDER BY observations DESC
        """,
        conn,
        params=(session_id,),
    )
    tracker_summary.to_csv(out / f"{session_id}_tracker_summary.csv", index=False)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    parser.add_argument("--db", default="data/project13.sqlite3")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    export_session(args.db, args.session_id, args.output_dir)
