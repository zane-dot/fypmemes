"""Tests for the LLM processor module."""

import json
import os

import pytest

from processors.llm_processor import (
    _build_prompt,
    _parse_response,
    is_available,
    is_vision_available,
)


class TestIsAvailable:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Without an API key, is_available should return False
        assert is_available() is False

    def test_available_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        # is_available also needs the openai package; skip if not installed
        try:
            import openai  # noqa: F401
            assert is_available() is True
        except ImportError:
            pytest.skip("openai not installed")


class TestIsVisionAvailable:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)
        assert is_vision_available() is False

    def test_unavailable_without_vision_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)
        assert is_vision_available() is False

    def test_unavailable_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_VISION_MODEL", "gpt-4o")
        assert is_vision_available() is False

    def test_available_with_key_and_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_VISION_MODEL", "gpt-4o")
        try:
            import openai  # noqa: F401
            assert is_vision_available() is True
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

    def test_ocr_failure_with_text_region_warns_caution(self):
        """When OCR fails but has_text_region is True, prompt warns to be cautious."""
        prompt = _build_prompt("", {"width": 100, "has_text_region": True})
        assert "OCR failed" in prompt
        assert "text overlay region" in prompt
        # Should NOT use the generic placeholder when text region is detected
        assert "no text could be extracted" not in prompt

    def test_no_text_no_text_region_uses_plain_placeholder(self):
        """When OCR fails and no text region, use the generic placeholder."""
        prompt = _build_prompt("", {"width": 100, "has_text_region": False})
        assert "no text could be extracted" in prompt
        assert "OCR failed" not in prompt


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


class TestAnalyseMemeWithVision:
    """Tests for analyse_meme_with_vision."""

    def test_returns_none_when_vision_unavailable(self, tmp_path, monkeypatch):
        """Returns None when no vision model is configured."""
        from processors.llm_processor import analyse_meme_with_vision
        monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)
        path = tmp_path / "img.png"
        from PIL import Image
        Image.new("RGB", (100, 100)).save(str(path))
        result = analyse_meme_with_vision(str(path), {"width": 100})
        assert result is None

    def test_returns_none_for_unreadable_file(self, monkeypatch):
        """Returns None when image file cannot be read."""
        from processors.llm_processor import analyse_meme_with_vision
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_VISION_MODEL", "gpt-4o")
        result = analyse_meme_with_vision("/nonexistent/image.png", {})
        assert result is None

    def test_calls_vision_api_with_image(self, tmp_path, monkeypatch):
        """analyse_meme_with_vision sends the image to the vision model."""
        import json
        from unittest.mock import MagicMock, patch
        from PIL import Image
        from processors import llm_processor

        path = tmp_path / "meme.png"
        Image.new("RGB", (200, 200), color=(200, 180, 160)).save(str(path))

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_VISION_MODEL", "gpt-4o")

        fake_response_content = json.dumps({
            "is_harmful": True,
            "harm_score": 0.8,
            "categories": ["Hate Speech"],
            "justification": "Vision model detected harmful text.",
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            fake_response_content
        )

        with patch.object(llm_processor, "OpenAI", return_value=mock_client):
            result = llm_processor.analyse_meme_with_vision(
                str(path), {"width": 200, "height": 200}
            )

        assert result is not None
        assert result["is_harmful"] is True
        assert result["harm_score"] == 0.8
        assert "Hate Speech" in result["categories"]

        # Verify the call included an image_url content part
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[-1]
        content_parts = user_msg["content"]
        image_parts = [p for p in content_parts if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/")

    def test_returns_none_on_api_failure(self, tmp_path, monkeypatch):
        """Returns None when the vision API call raises an exception."""
        from unittest.mock import MagicMock, patch
        from PIL import Image
        from processors import llm_processor

        path = tmp_path / "meme.png"
        Image.new("RGB", (200, 200)).save(str(path))

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_VISION_MODEL", "gpt-4o")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        with patch.object(llm_processor, "OpenAI", return_value=mock_client):
            result = llm_processor.analyse_meme_with_vision(str(path), {})

        assert result is None

    def test_explicit_model_bypasses_vision_model_check(self, tmp_path, monkeypatch):
        """When model is passed explicitly, OPENAI_VISION_MODEL need not be set."""
        import json
        from unittest.mock import MagicMock, patch
        from PIL import Image
        from processors import llm_processor

        path = tmp_path / "meme.png"
        Image.new("RGB", (200, 200), color=(220, 200, 180)).save(str(path))

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)

        fake_response_content = json.dumps({
            "is_harmful": False,
            "harm_score": 0.1,
            "categories": [],
            "justification": "Safe meme.",
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            fake_response_content
        )

        with patch.object(llm_processor, "OpenAI", return_value=mock_client):
            result = llm_processor.analyse_meme_with_vision(
                str(path), {"width": 200}, model="gpt-4o",
            )

        assert result is not None
        assert result["is_harmful"] is False

    def test_explicit_model_returns_none_without_api_key(self, tmp_path, monkeypatch):
        """With explicit model but no API key, returns None gracefully."""
        from PIL import Image
        from processors.llm_processor import analyse_meme_with_vision

        path = tmp_path / "img.png"
        Image.new("RGB", (100, 100)).save(str(path))

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)

        result = analyse_meme_with_vision(str(path), {}, model="gpt-4o")
        assert result is None
