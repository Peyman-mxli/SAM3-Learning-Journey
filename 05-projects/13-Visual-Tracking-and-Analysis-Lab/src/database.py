from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    timestamp_seconds REAL NOT NULL,
    tracker_id INTEGER,
    class_id INTEGER,
    class_name TEXT,
    confidence REAL,
    x1 REAL,
    y1 REAL,
    x2 REAL,
    y2 REAL,
    center_x REAL,
    center_y REAL,
    mask_area REAL,
    notes TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def create_session(
    conn: sqlite3.Connection,
    session_id: str,
    source_path: str,
    source_type: str,
    created_at: str,
    notes: str = "",
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO sessions
        (session_id, source_path, source_type, created_at, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, source_path, source_type, created_at, notes),
    )
    conn.commit()


def insert_observations(
    conn: sqlite3.Connection,
    rows: Iterable[tuple],
) -> None:
    conn.executemany(
        """
        INSERT INTO observations (
            session_id, frame_index, timestamp_seconds, tracker_id,
            class_id, class_name, confidence,
            x1, y1, x2, y2, center_x, center_y, mask_area, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def session_summary(conn: sqlite3.Connection):
    return conn.execute(
        """
        SELECT
            s.session_id,
            s.source_path,
            s.created_at,
            COUNT(o.observation_id) AS observations,
            COUNT(DISTINCT o.tracker_id) AS unique_trackers,
            AVG(o.confidence) AS average_confidence
        FROM sessions s
        LEFT JOIN observations o ON o.session_id = s.session_id
        GROUP BY s.session_id, s.source_path, s.created_at
        ORDER BY s.created_at DESC
        """
    ).fetchall()
