"""SQLite database models for storing meme analysis results."""

import os
import sqlite3
from datetime import datetime, timezone


def get_connection(db_path):
    """Return a sqlite3 connection with row-factory enabled."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path):
    """Create the database tables if they do not exist."""
    conn = get_connection(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meme_analysis (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            filename      TEXT    NOT NULL,
            extracted_text TEXT,
            is_harmful    INTEGER NOT NULL DEFAULT 0,
            harm_score    REAL    NOT NULL DEFAULT 0.0,
            categories    TEXT,
            justification TEXT,
            image_features TEXT,
            created_at    TEXT    NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_analysis(db_path, *, filename, extracted_text, is_harmful,
                  harm_score, categories, justification, image_features):
    """Persist a single meme analysis record and return its id."""
    conn = get_connection(db_path)
    cur = conn.execute(
        """
        INSERT INTO meme_analysis
            (filename, extracted_text, is_harmful, harm_score,
             categories, justification, image_features, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            extracted_text,
            int(is_harmful),
            harm_score,
            categories,
            justification,
            image_features,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    record_id = cur.lastrowid
    conn.close()
    return record_id


def get_analysis(db_path, record_id):
    """Fetch a single analysis by id."""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT * FROM meme_analysis WHERE id = ?", (record_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_analyses(db_path, limit=50):
    """Return the most recent analyses."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM meme_analysis ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
