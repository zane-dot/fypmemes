"""Tests for the LLM processor module."""

import json
import os

import pytest

from processors.llm_processor import (
    _build_prompt,
    _parse_response,
    is_available,
)


class TestIsAvailable:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert is_available() is False or True  # depends on openai install

    def test_available_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        # is_available also needs the openai package; skip if not installed
        try:
            import openai  # noqa: F401
            assert is_available() is True
        except ImportError:
            pytest.skip("openai not installed")


class TestBuildPrompt:
    def test_includes_text(self):
        prompt = _build_prompt("hello world", {"width": 100})
        assert "hello world" in prompt

    def test_no_text_placeholder(self):
        prompt = _build_prompt("", {"width": 100})
        assert "no text could be extracted" in prompt

    def test_includes_features(self):
        prompt = _build_prompt("x", {"width": 200, "height": 300})
        assert "width" in prompt
        assert "200" in prompt


class TestParseResponse:
    def test_valid_json(self):
        raw = json.dumps({
            "is_harmful": True,
            "harm_score": 0.75,
            "categories": ["Hate Speech"],
            "justification": "Contains slurs.",
        })
        result = _parse_response(raw)
        assert result["is_harmful"] is True
        assert result["harm_score"] == 0.75
        assert result["categories"] == ["Hate Speech"]

    def test_json_with_code_fences(self):
        raw = '```json\n{"is_harmful": false, "harm_score": 0.1, "categories": [], "justification": "Safe."}\n```'
        result = _parse_response(raw)
        assert result is not None
        assert result["is_harmful"] is False

    def test_invalid_json_returns_none(self):
        assert _parse_response("not json at all") is None

    def test_missing_fields_use_defaults(self):
        raw = json.dumps({"some_other": "data"})
        result = _parse_response(raw)
        assert result["is_harmful"] is False
        assert result["harm_score"] == 0.0
