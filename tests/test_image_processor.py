"""Tests for the image processor module."""

import os
import tempfile
from unittest import mock

from PIL import Image, ImageDraw, ImageFont
import pytest

from processors.image_processor import extract_image_features, extract_text, _std


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


# ---------------------------------------------------------------------- #
# extract_text tests
# ---------------------------------------------------------------------- #

def test_extract_text_returns_string(tmp_path):
    """extract_text always returns a string, never None."""
    path = tmp_path / "blank.png"
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(str(path))
    result = extract_text(str(path))
    assert isinstance(result, str)


def test_extract_text_invalid_path_returns_empty():
    """extract_text returns empty string for non-existent images."""
    result = extract_text("/nonexistent/image.png")
    assert result == ""


def test_extract_text_uses_easyocr_when_available(tmp_path):
    """When EasyOCR is available it is used as primary backend."""
    path = tmp_path / "img.png"
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(str(path))

    import processors.image_processor as ip
    mock_reader = mock.MagicMock()
    # First call (paragraph=True) returns results; second call should not run.
    mock_reader.readtext.side_effect = [["Hello", "world"]]

    with mock.patch.object(ip, "_HAS_EASYOCR", True), \
         mock.patch.object(ip, "_get_easyocr_reader", return_value=mock_reader), \
         mock.patch.object(ip, "_extract_text_via_vision", return_value=None):
        result = extract_text(str(path))

    assert result == "Hello world"
    # readtext is called with a numpy array and paragraph=True first.
    first_call_kwargs = mock_reader.readtext.call_args_list[0].kwargs
    assert first_call_kwargs.get("paragraph") is True
    assert first_call_kwargs.get("text_threshold") == 0.5


def test_extract_text_easyocr_chinese(tmp_path):
    """EasyOCR backend correctly joins Chinese text results."""
    path = tmp_path / "zh.png"
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(str(path))

    import processors.image_processor as ip
    mock_reader = mock.MagicMock()
    mock_reader.readtext.side_effect = [["你好", "世界"]]

    with mock.patch.object(ip, "_HAS_EASYOCR", True), \
         mock.patch.object(ip, "_get_easyocr_reader", return_value=mock_reader), \
         mock.patch.object(ip, "_extract_text_via_vision", return_value=None):
        result = extract_text(str(path))

    assert result == "你好 世界"


def test_extract_text_falls_back_to_pytesseract_when_easyocr_unavailable(tmp_path):
    """When EasyOCR is absent, pytesseract is used with multilingual config."""
    path = tmp_path / "img.png"
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(str(path))

    import processors.image_processor as ip

    with mock.patch.object(ip, "_HAS_EASYOCR", False), \
         mock.patch.object(ip, "_HAS_TESSERACT", True), \
         mock.patch.object(ip, "_extract_text_via_vision", return_value=None), \
         mock.patch("processors.image_processor.pytesseract") as mock_ts:
        mock_ts.image_to_string.return_value = "test text"
        result = extract_text(str(path))

    assert result == "test text"
    # Verify multilingual lang string was requested
    assert mock_ts.image_to_string.call_args is not None
    call_kwargs = mock_ts.image_to_string.call_args.kwargs
    assert call_kwargs.get("lang") == "eng+chi_sim+chi_tra"


def test_extract_text_returns_empty_when_no_backend(tmp_path):
    """Returns empty string when neither OCR backend is available."""
    path = tmp_path / "img.png"
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(str(path))

    import processors.image_processor as ip

    with mock.patch.object(ip, "_HAS_EASYOCR", False), \
         mock.patch.object(ip, "_HAS_TESSERACT", False), \
         mock.patch.object(ip, "_extract_text_via_vision", return_value=None):
        result = extract_text(str(path))

    assert result == ""


def test_extract_text_vision_api_used_first(tmp_path):
    """Vision API is tried before EasyOCR when the API key is configured."""
    path = tmp_path / "img.png"
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(str(path))

    import processors.image_processor as ip

    with mock.patch.object(ip, "_extract_text_via_vision", return_value="vision text") as mock_vision:
        result = extract_text(str(path))

    assert result == "vision text"
    mock_vision.assert_called_once_with(str(path))


def test_extract_text_vision_api_fallback_to_easyocr(tmp_path):
    """When vision API returns None, EasyOCR is used as fallback."""
    path = tmp_path / "img.png"
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(str(path))

    import processors.image_processor as ip
    mock_reader = mock.MagicMock()
    mock_reader.readtext.side_effect = [["easyocr text"]]

    with mock.patch.object(ip, "_extract_text_via_vision", return_value=None), \
         mock.patch.object(ip, "_HAS_EASYOCR", True), \
         mock.patch.object(ip, "_get_easyocr_reader", return_value=mock_reader):
        result = extract_text(str(path))

    assert result == "easyocr text"


def test_preprocess_for_ocr_upscales_small_images(tmp_path):
    """Small images are upscaled before being fed to OCR."""
    import numpy as np
    import processors.image_processor as ip

    path = tmp_path / "small.png"
    Image.new("RGB", (200, 200), color=(128, 128, 128)).save(str(path))

    arr = ip._preprocess_for_ocr(str(path))
    # Should be upscaled to at least _OCR_MIN_DIM in both dimensions
    assert arr.shape[0] >= ip._OCR_MIN_DIM
    assert arr.shape[1] >= ip._OCR_MIN_DIM


def test_extract_text_via_vision_returns_none_without_api_key(tmp_path):
    """_extract_text_via_vision returns None when no OPENAI_API_KEY is set."""
    import os
    import processors.image_processor as ip

    path = tmp_path / "img.png"
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(str(path))

    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with mock.patch.dict(os.environ, env, clear=True):
        result = ip._extract_text_via_vision(str(path))

    assert result is None

