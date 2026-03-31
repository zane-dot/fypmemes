"""SQLite database models for storing meme analysis results."""

import os
import sqlite3
from datetime import datetime, timezone


def get_connection(db_path):
    """Return a sqlite3 connection with row-factory enabled."""
    dirname = os.path.dirname(db_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path):
    """Create the database tables if they do not exist."""
    conn = get_connection(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            username       TEXT    NOT NULL UNIQUE,
            password_hash  TEXT    NOT NULL,
            created_at     TEXT    NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meme_analysis (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER,
            filename      TEXT    NOT NULL,
            extracted_text TEXT,
            is_harmful    INTEGER NOT NULL DEFAULT 0,
            harm_score    REAL    NOT NULL DEFAULT 0.0,
            categories    TEXT,
            justification TEXT,
            image_features TEXT,
            analysis_method TEXT,
            pro_rationale TEXT,
            con_rationale TEXT,
            judge_reasoning TEXT,
            judge_side TEXT,
            created_at    TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    # Backward-compatible migration for existing DB files.
    existing_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(meme_analysis)").fetchall()
    }
    if "analysis_method" not in existing_cols:
        conn.execute("ALTER TABLE meme_analysis ADD COLUMN analysis_method TEXT")
    if "pro_rationale" not in existing_cols:
        conn.execute("ALTER TABLE meme_analysis ADD COLUMN pro_rationale TEXT")
    if "con_rationale" not in existing_cols:
        conn.execute("ALTER TABLE meme_analysis ADD COLUMN con_rationale TEXT")
    if "judge_reasoning" not in existing_cols:
        conn.execute("ALTER TABLE meme_analysis ADD COLUMN judge_reasoning TEXT")
    if "judge_side" not in existing_cols:
        conn.execute("ALTER TABLE meme_analysis ADD COLUMN judge_side TEXT")
    if "user_id" not in existing_cols:
        conn.execute("ALTER TABLE meme_analysis ADD COLUMN user_id INTEGER")
    conn.commit()
    conn.close()


def create_user(db_path, *, username, password_hash):
    """Create a user account and return its id."""
    conn = get_connection(db_path)
    cur = conn.execute(
        """
        INSERT INTO users (username, password_hash, created_at)
        VALUES (?, ?, ?)
        """,
        (username, password_hash, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def get_user_by_username(db_path, username):
    """Fetch user row by username."""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(db_path, user_id):
    """Fetch user row by id."""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_analysis(db_path, *, filename, extracted_text, is_harmful,
                  harm_score, categories, justification, image_features,
                  user_id=None,
                  analysis_method=None, pro_rationale=None,
                  con_rationale=None, judge_reasoning=None,
                  judge_side=None):
    """Persist a single meme analysis record and return its id."""
    conn = get_connection(db_path)
    cur = conn.execute(
        """
        INSERT INTO meme_analysis
            (user_id, filename, extracted_text, is_harmful, harm_score,
             categories, justification, image_features, analysis_method,
             pro_rationale, con_rationale, judge_reasoning, judge_side,
             created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            filename,
            extracted_text,
            int(is_harmful),
            harm_score,
            categories,
            justification,
            image_features,
            analysis_method,
            pro_rationale,
            con_rationale,
            judge_reasoning,
            judge_side,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    record_id = cur.lastrowid
    conn.close()
    return record_id


def get_analysis(db_path, record_id, user_id=None):
    """Fetch a single analysis by id."""
    conn = get_connection(db_path)
    if user_id is None:
        row = conn.execute(
            "SELECT * FROM meme_analysis WHERE id = ?", (record_id,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM meme_analysis WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_analyses(
    db_path,
    limit=50,
    user_id=None,
    created_date=None,
    start_date=None,
    end_date=None,
    is_harmful=None,
):
    """Return the most recent analyses with optional filters."""
    conn = get_connection(db_path)

    conditions = []
    params = []

    if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)
    if created_date:
        conditions.append("date(created_at) = ?")
        params.append(created_date)
    else:
        if start_date:
            conditions.append("date(created_at) >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date(created_at) <= ?")
            params.append(end_date)
    if is_harmful is not None:
        conditions.append("is_harmful = ?")
        params.append(int(bool(is_harmful)))

    query = "SELECT * FROM meme_analysis"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_analysis_dates(db_path, user_id=None):
    """Return distinct analysis dates (YYYY-MM-DD), newest first."""
    conn = get_connection(db_path)
    if user_id is None:
        rows = conn.execute(
            """
            SELECT DISTINCT date(created_at) AS analysis_date
            FROM meme_analysis
            ORDER BY analysis_date DESC
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT DISTINCT date(created_at) AS analysis_date
            FROM meme_analysis
            WHERE user_id = ?
            ORDER BY analysis_date DESC
            """,
            (user_id,),
        ).fetchall()
    conn.close()
    return [r["analysis_date"] for r in rows if r["analysis_date"]]
