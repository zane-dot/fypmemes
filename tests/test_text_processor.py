"""Tests for the text processor module."""

import os

import pytest

from processors.text_processor import analyse_text, _match_keywords, _match_patterns

KEYWORDS_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "harmful_keywords.json"
)


def test_empty_text_returns_safe():
    result = analyse_text("", KEYWORDS_PATH)
    assert result["is_harmful"] is False
    assert result["overall_score"] == 0.0
    assert result["matched_categories"] == []


def test_none_text_returns_safe():
    result = analyse_text(None, KEYWORDS_PATH)
    assert result["is_harmful"] is False


def test_benign_text_returns_safe():
    result = analyse_text("I love cute puppies and sunshine!", KEYWORDS_PATH)
    assert result["is_harmful"] is False


def test_hate_speech_detected():
    result = analyse_text("all immigrants are criminals", KEYWORDS_PATH)
    assert result["is_harmful"] is True
    labels = [m["label"] for m in result["matched_categories"]]
    assert "Hate Speech" in labels


def test_violence_detected():
    result = analyse_text("I will kill you", KEYWORDS_PATH)
    assert result["is_harmful"] is True
    labels = [m["label"] for m in result["matched_categories"]]
    assert "Violence / Threats" in labels


def test_cyberbullying_detected():
    result = analyse_text("you are ugly and pathetic", KEYWORDS_PATH)
    assert result["is_harmful"] is True
    labels = [m["label"] for m in result["matched_categories"]]
    assert "Cyberbullying / Harassment" in labels


def test_multiple_categories():
    result = analyse_text("kill yourself you ugly loser", KEYWORDS_PATH)
    assert result["is_harmful"] is True
    assert len(result["matched_categories"]) >= 2


def test_match_keywords_basic():
    assert _match_keywords("i hate this ugly thing", ["ugly", "stupid"]) == ["ugly"]


def test_match_keywords_case_insensitive():
    assert _match_keywords("UGLY stuff", ["ugly"]) == ["ugly"]


def test_match_patterns_basic():
    hits = _match_patterns(
        "all muslims are terrorists",
        [r"all \w+ are (criminals|terrorists|rapists)"],
    )
    assert len(hits) == 1


def test_match_patterns_no_match():
    hits = _match_patterns("have a nice day", [r"kill all \w+"])
    assert hits == []
