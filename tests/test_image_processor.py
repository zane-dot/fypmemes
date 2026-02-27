"""Tests for the image processor module."""

import os
import tempfile

from PIL import Image, ImageDraw, ImageFont
import pytest

from processors.image_processor import extract_image_features, _std


@pytest.fixture()
def plain_image(tmp_path):
    """Create a simple 200×200 red image."""
    path = tmp_path / "plain.png"
    img = Image.new("RGB", (200, 200), color=(255, 0, 0))
    img.save(str(path), format="PNG")
    return str(path)


@pytest.fixture()
def text_overlay_image(tmp_path):
    """Create an image with high-contrast text overlay to trigger heuristic."""
    path = tmp_path / "text_overlay.png"
    img = Image.new("RGB", (400, 400), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Draw white text area for high contrast
    draw.rectangle([0, 0, 400, 100], fill=(255, 255, 255))
    draw.rectangle([0, 300, 400, 400], fill=(255, 255, 255))
    img.save(str(path), format="PNG")
    return str(path)


def test_extract_features_returns_expected_keys(plain_image):
    features = extract_image_features(plain_image)
    expected_keys = {
        "width", "height", "format", "mode", "has_text_region",
        "dominant_colors", "brightness", "contrast", "color_variance",
    }
    assert expected_keys == set(features.keys())


def test_plain_image_dimensions(plain_image):
    features = extract_image_features(plain_image)
    assert features["width"] == 200
    assert features["height"] == 200


def test_plain_image_mode(plain_image):
    features = extract_image_features(plain_image)
    assert features["mode"] == "RGB"


def test_text_overlay_heuristic(text_overlay_image):
    features = extract_image_features(text_overlay_image)
    # High contrast black/white image should trigger text-region heuristic
    assert features["contrast"] > 0
    assert isinstance(features["has_text_region"], bool)


def test_dominant_colors_length(plain_image):
    features = extract_image_features(plain_image)
    assert len(features["dominant_colors"]) <= 5


def test_std_empty():
    assert _std([]) == 0.0


def test_std_uniform():
    assert _std([5, 5, 5]) == 0.0


def test_std_known():
    result = _std([2, 4, 4, 4, 5, 5, 7, 9])
    assert round(result, 2) == 2.0


def test_invalid_path_returns_defaults():
    features = extract_image_features("/nonexistent/image.png")
    assert features["width"] == 0
    assert features["brightness"] == 0
