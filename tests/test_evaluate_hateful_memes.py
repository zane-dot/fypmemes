"""Tests for the evaluation script utilities."""

import json
import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Import helpers from the evaluation script
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from evaluate_hateful_memes import _compute_metrics, _classify_sample


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_all_correct(self):
        labels = [1, 0, 1, 0]
        preds =  [1, 0, 1, 0]
        m = _compute_metrics(labels, preds)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["tp"] == 2
        assert m["tn"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_all_wrong(self):
        labels = [1, 0, 1, 0]
        preds =  [0, 1, 0, 1]
        m = _compute_metrics(labels, preds)
        assert m["accuracy"] == 0.0
        assert m["tp"] == 0
        assert m["tn"] == 0
        assert m["fp"] == 2
        assert m["fn"] == 2

    def test_all_predicted_positive(self):
        labels = [1, 0, 1, 0]
        preds =  [1, 1, 1, 1]
        m = _compute_metrics(labels, preds)
        assert m["precision"] == 0.5   # 2 TP / (2 TP + 2 FP)
        assert m["recall"] == 1.0      # 2 TP / (2 TP + 0 FN)

    def test_all_predicted_negative(self):
        labels = [1, 0, 1, 0]
        preds =  [0, 0, 0, 0]
        m = _compute_metrics(labels, preds)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_total_count(self):
        labels = [1, 0, 1]
        preds =  [1, 0, 0]
        m = _compute_metrics(labels, preds)
        assert m["total"] == 3

    def test_empty_lists(self):
        m = _compute_metrics([], [])
        assert m["accuracy"] == 0.0
        assert m["total"] == 0

    def test_f1_formula(self):
        # precision=0.5, recall=1.0 → F1 = 2*0.5*1/(0.5+1) = 2/3
        labels = [1, 1, 0, 0]
        preds  = [1, 1, 1, 1]  # TP=2, FP=2, FN=0, TN=0
        m = _compute_metrics(labels, preds)
        expected_f1 = round(2 * 0.5 * 1.0 / (0.5 + 1.0), 4)
        assert abs(m["f1"] - expected_f1) < 1e-4


# ---------------------------------------------------------------------------
# _classify_sample (pipeline integration – no LLM, uses keyword fallback)
# ---------------------------------------------------------------------------

class TestClassifySample:
    def test_returns_required_keys(self, tmp_path):
        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (200, 200), color=(200, 200, 200)).save(str(img_path))

        keywords_path = str(_REPO_ROOT / "data" / "harmful_keywords.json")
        sample = {
            "id": 1,
            "label": 0,
            "text": "just a harmless meme",
            "image_path": str(img_path),
        }
        result = _classify_sample(sample, keywords_path)

        assert "id" in result
        assert "true_label" in result
        assert "predicted_label" in result
        assert "harm_score" in result
        assert "analysis_method" in result
        assert result["true_label"] == 0
        assert result["predicted_label"] in (0, 1)

    def test_pipeline_error_returns_safe_default(self, tmp_path):
        """Pipeline error on a non-image file should not crash and return 0."""
        bad_file = tmp_path / "bad.png"
        bad_file.write_text("not an image")

        keywords_path = str(_REPO_ROOT / "data" / "harmful_keywords.json")
        sample = {
            "id": 999,
            "label": 1,
            "text": "something",
            "image_path": str(bad_file),
        }
        result = _classify_sample(sample, keywords_path)
        # Should not raise; falls back gracefully
        assert result["id"] == 999
        assert result["true_label"] == 1
        assert result["predicted_label"] in (0, 1)
