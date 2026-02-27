"""Tests for the classifier module."""

import json

from models.classifier import classify, _keyword_classify, _build_justification


def _safe_text_result():
    return {
        "matched_categories": [],
        "overall_score": 0.0,
        "is_harmful": False,
    }


def _harmful_text_result():
    return {
        "matched_categories": [
            {
                "category": "hate_speech",
                "label": "Hate Speech",
                "description": "Promotes hatred.",
                "severity": 0.9,
                "keyword_matches": ["slur"],
                "pattern_matches": [],
            }
        ],
        "overall_score": 0.9,
        "is_harmful": True,
    }


def _basic_features():
    return {
        "width": 400,
        "height": 400,
        "format": "PNG",
        "mode": "RGB",
        "has_text_region": False,
        "dominant_colors": ["#ff0000"],
        "brightness": 120.0,
        "contrast": 30.0,
        "color_variance": 20.0,
    }


class TestKeywordClassify:
    def test_safe_meme(self):
        result = _keyword_classify(_safe_text_result(), _basic_features())
        assert result["is_harmful"] is False
        assert result["harm_score"] == 0.0
        assert result["analysis_method"] == "keyword"

    def test_harmful_meme(self):
        result = _keyword_classify(_harmful_text_result(), _basic_features())
        assert result["is_harmful"] is True
        assert result["harm_score"] >= 0.4
        cats = json.loads(result["categories"])
        assert "Hate Speech" in cats

    def test_image_boost_on_text_region(self):
        features = _basic_features()
        features["has_text_region"] = True
        result = _keyword_classify(_safe_text_result(), features)
        assert result["harm_score"] == 0.05  # boost only

    def test_justification_present(self):
        result = _keyword_classify(_harmful_text_result(), _basic_features())
        assert "HARMFUL" in result["justification"]


class TestClassifyFallback:
    """classify() should fall back to keywords when LLM is unavailable."""

    def test_fallback_safe(self, monkeypatch):
        monkeypatch.setattr(
            "models.classifier.llm_available", lambda: False
        )
        result = classify(_safe_text_result(), _basic_features())
        assert result["is_harmful"] is False
        assert result["analysis_method"] == "keyword"

    def test_fallback_harmful(self, monkeypatch):
        monkeypatch.setattr(
            "models.classifier.llm_available", lambda: False
        )
        result = classify(_harmful_text_result(), _basic_features())
        assert result["is_harmful"] is True
        assert result["analysis_method"] == "keyword"


class TestClassifyWithLLM:
    """classify() should use LLM result when available."""

    def test_llm_result_used(self, monkeypatch):
        monkeypatch.setattr(
            "models.classifier.llm_available", lambda: True
        )
        fake_llm = {
            "is_harmful": True,
            "harm_score": 0.85,
            "categories": ["Hate Speech"],
            "justification": "LLM says harmful.",
        }
        monkeypatch.setattr(
            "models.classifier.analyse_meme",
            lambda *a, **kw: fake_llm,
        )
        result = classify(_safe_text_result(), _basic_features(), extracted_text="test")
        assert result["is_harmful"] is True
        assert result["analysis_method"] == "llm"
        assert "LLM says harmful" in result["justification"]

    def test_llm_none_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            "models.classifier.llm_available", lambda: True
        )
        monkeypatch.setattr(
            "models.classifier.analyse_meme",
            lambda *a, **kw: None,
        )
        result = classify(_safe_text_result(), _basic_features())
        assert result["analysis_method"] == "keyword"


class TestClassifyWithVision:
    """classify() should prefer vision analysis when image_path is provided."""

    def test_vision_result_used_when_available(self, monkeypatch, tmp_path):
        from PIL import Image
        path = tmp_path / "meme.png"
        Image.new("RGB", (200, 200)).save(str(path))

        monkeypatch.setattr("models.classifier.llm_vision_available", lambda: True)
        fake_vision = {
            "is_harmful": True,
            "harm_score": 0.9,
            "categories": ["Violence / Threats"],
            "justification": "Vision model detected harmful content.",
        }
        monkeypatch.setattr(
            "models.classifier.analyse_meme_with_vision",
            lambda *a, **kw: fake_vision,
        )
        result = classify(
            _safe_text_result(), _basic_features(),
            extracted_text="", image_path=str(path),
        )
        assert result["is_harmful"] is True
        assert result["analysis_method"] == "llm"
        assert "Vision model" in result["justification"]

    def test_vision_none_falls_back_to_text_llm(self, monkeypatch, tmp_path):
        from PIL import Image
        path = tmp_path / "meme.png"
        Image.new("RGB", (200, 200)).save(str(path))

        monkeypatch.setattr("models.classifier.llm_vision_available", lambda: True)
        monkeypatch.setattr(
            "models.classifier.analyse_meme_with_vision",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr("models.classifier.llm_available", lambda: True)
        fake_llm = {
            "is_harmful": False,
            "harm_score": 0.1,
            "categories": [],
            "justification": "Text LLM says safe.",
        }
        monkeypatch.setattr(
            "models.classifier.analyse_meme",
            lambda *a, **kw: fake_llm,
        )
        result = classify(
            _safe_text_result(), _basic_features(),
            extracted_text="some text", image_path=str(path),
        )
        assert result["analysis_method"] == "llm"
        assert "Text LLM" in result["justification"]

    def test_no_image_path_skips_vision(self, monkeypatch):
        """classify() skips vision analysis when image_path is None."""
        vision_called = []

        def _fake_vision(*a, **kw):
            vision_called.append(1)
            return {"is_harmful": False, "harm_score": 0.0,
                    "categories": [], "justification": "x"}

        monkeypatch.setattr("models.classifier.llm_vision_available", lambda: True)
        monkeypatch.setattr("models.classifier.analyse_meme_with_vision", _fake_vision)
        monkeypatch.setattr("models.classifier.llm_available", lambda: False)
        classify(_safe_text_result(), _basic_features())
        assert vision_called == [], "Vision should not be called without image_path"

    def test_ocr_failure_vision_fallback_with_text_region(self, monkeypatch, tmp_path):
        """When OCR returns empty and text region detected, vision fallback is tried."""
        import os
        from PIL import Image
        path = tmp_path / "meme.png"
        Image.new("RGB", (200, 200), color=(230, 210, 190)).save(str(path))

        features = _basic_features()
        features["has_text_region"] = True

        monkeypatch.setattr("models.classifier.llm_vision_available", lambda: False)
        monkeypatch.setattr("models.classifier.llm_available", lambda: True)
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

        vision_calls = []
        fake_vision = {
            "is_harmful": True,
            "harm_score": 0.8,
            "categories": ["Hate Speech"],
            "justification": "Vision fallback detected harmful text in the image.",
        }

        def _capture_vision(image_path, image_features, *, model=None, **kw):
            vision_calls.append(model)
            return fake_vision

        monkeypatch.setattr("models.classifier.analyse_meme_with_vision", _capture_vision)

        result = classify(
            _safe_text_result(), features,
            extracted_text="", image_path=str(path),
        )

        assert result["is_harmful"] is True
        assert result["analysis_method"] == "llm"
        assert len(vision_calls) == 1
        assert vision_calls[0] == "gpt-4o"

    def test_ocr_failure_vision_fallback_skipped_when_text_extracted(self, monkeypatch, tmp_path):
        """Vision fallback is NOT attempted when OCR did extract text."""
        from PIL import Image
        path = tmp_path / "meme.png"
        Image.new("RGB", (200, 200)).save(str(path))

        features = _basic_features()
        features["has_text_region"] = True

        monkeypatch.setattr("models.classifier.llm_vision_available", lambda: False)
        monkeypatch.setattr("models.classifier.llm_available", lambda: True)
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

        vision_calls = []

        def _capture_vision(*a, **kw):
            vision_calls.append(1)
            return None

        monkeypatch.setattr("models.classifier.analyse_meme_with_vision", _capture_vision)

        fake_llm = {"is_harmful": False, "harm_score": 0.1, "categories": [], "justification": "ok"}
        monkeypatch.setattr("models.classifier.analyse_meme", lambda *a, **kw: fake_llm)

        classify(_safe_text_result(), features, extracted_text="some text", image_path=str(path))

        # Vision fallback should NOT be called because extracted_text is non-empty
        assert vision_calls == []

    def test_ocr_failure_vision_fallback_skipped_without_text_region(self, monkeypatch, tmp_path):
        """Vision fallback is NOT attempted when no text region is detected."""
        from PIL import Image
        path = tmp_path / "meme.png"
        Image.new("RGB", (200, 200)).save(str(path))

        features = _basic_features()
        features["has_text_region"] = False

        monkeypatch.setattr("models.classifier.llm_vision_available", lambda: False)
        monkeypatch.setattr("models.classifier.llm_available", lambda: True)
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

        vision_calls = []

        def _capture_vision(*a, **kw):
            vision_calls.append(1)
            return None

        monkeypatch.setattr("models.classifier.analyse_meme_with_vision", _capture_vision)

        fake_llm = {"is_harmful": False, "harm_score": 0.0, "categories": [], "justification": "ok"}
        monkeypatch.setattr("models.classifier.analyse_meme", lambda *a, **kw: fake_llm)

        classify(_safe_text_result(), features, extracted_text="", image_path=str(path))

        assert vision_calls == []


class TestBuildJustification:
    def test_safe_justification(self):
        j = _build_justification(False, 0.0, [], _basic_features())
        assert "NOT harmful" in j

    def test_harmful_justification_has_category(self):
        cats = _harmful_text_result()["matched_categories"]
        j = _build_justification(True, 0.9, cats, _basic_features())
        assert "HARMFUL" in j
        assert "Hate Speech" in j
