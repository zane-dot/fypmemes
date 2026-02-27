"""Tests for the database module."""

import os
import tempfile

import pytest

from models.database import (
    get_all_analyses,
    get_analysis,
    init_db,
    save_analysis,
)


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


def test_init_db_creates_file(db_path):
    assert os.path.exists(db_path)


def test_save_and_get(db_path):
    rid = save_analysis(
        db_path,
        filename="test.png",
        extracted_text="hello",
        is_harmful=True,
        harm_score=0.85,
        categories='["Hate Speech"]',
        justification="Bad meme.",
        image_features='{"width": 100}',
    )
    row = get_analysis(db_path, rid)
    assert row is not None
    assert row["filename"] == "test.png"
    assert row["is_harmful"] == 1
    assert row["harm_score"] == 0.85


def test_get_nonexistent_returns_none(db_path):
    assert get_analysis(db_path, 9999) is None


def test_get_all_analyses(db_path):
    for i in range(5):
        save_analysis(
            db_path,
            filename=f"meme{i}.png",
            extracted_text="",
            is_harmful=False,
            harm_score=0.0,
            categories="[]",
            justification="Safe.",
            image_features="{}",
        )
    rows = get_all_analyses(db_path, limit=3)
    assert len(rows) == 3
    # Most recent first
    assert rows[0]["filename"] == "meme4.png"
