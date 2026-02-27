"""Image processing module for meme analysis.

Handles image feature extraction and OCR text extraction from meme images.
"""

import logging

from PIL import Image

logger = logging.getLogger(__name__)

# EasyOCR is the preferred OCR backend – it supports both English and Chinese
# (simplified) without requiring a separate Tesseract installation.
try:
    import easyocr as _easyocr

    _HAS_EASYOCR = True
except ImportError:
    _HAS_EASYOCR = False

# Tesseract / pytesseract is kept as a fallback for environments where EasyOCR
# is not available.  The system still works (without OCR) when neither backend
# is installed.
try:
    import pytesseract

    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False

# Module-level EasyOCR reader cache – initialised on first use so that the
# heavy model download/load only happens when OCR is actually needed.
_easyocr_reader = None


def _get_easyocr_reader():
    """Return (and lazily initialise) the shared EasyOCR reader.

    The reader is configured for English (``en``) and Simplified Chinese
    (``ch_sim``), which covers the vast majority of text found in memes.
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("Initialising EasyOCR reader for languages: en, ch_sim")
        _easyocr_reader = _easyocr.Reader(["en", "ch_sim"], gpu=False)
    return _easyocr_reader


def extract_text(image_path):
    """Extract text from a meme image using OCR.

    Tries EasyOCR first (supports English **and** Chinese).  Falls back to
    *pytesseract* (English-only) when EasyOCR is not installed.  Returns an
    empty string when no OCR backend is available.
    """
    # --- EasyOCR (preferred) -------------------------------------------
    if _HAS_EASYOCR:
        try:
            reader = _get_easyocr_reader()
            results = reader.readtext(image_path, detail=0)
            return " ".join(results).strip()
        except Exception:
            logger.exception("EasyOCR failed for %s", image_path)
            # Fall through to the pytesseract fallback below.

    # --- pytesseract fallback ------------------------------------------
    if _HAS_TESSERACT:
        try:
            img = Image.open(image_path)
            # Request both English and Chinese (simplified + traditional).
            # Tesseract will silently skip language packs that are not
            # installed, so this is safe even in minimal environments.
            text = pytesseract.image_to_string(img, lang="eng+chi_sim+chi_tra")
            return text.strip()
        except Exception:
            logger.exception("pytesseract OCR failed for %s", image_path)
            return ""

    logger.warning("No OCR backend available – skipping text extraction")
    return ""


def extract_image_features(image_path):
    """Analyse an image and return a dict of visual features.

    Features include:
    * ``width`` / ``height`` – pixel dimensions
    * ``format`` – image format (PNG, JPEG, …)
    * ``mode`` – colour mode (RGB, L, …)
    * ``has_text_region`` – heuristic flag that guesses whether the image
      contains a text overlay (based on brightness variance)
    * ``dominant_colors`` – the 5 most common colours (as hex strings)
    * ``brightness`` – mean brightness 0-255
    * ``contrast`` – standard-deviation of brightness
    * ``color_variance`` – spread of colour values
    """
    try:
        img = Image.open(image_path)
        width, height = img.size

        # Convert to RGB for analysis
        rgb = img.convert("RGB")
        pixels = list(rgb.getdata())

        # Brightness & contrast -------------------------------------------
        brightness_vals = [(r * 299 + g * 587 + b * 114) / 1000
                          for r, g, b in pixels]
        mean_brightness = sum(brightness_vals) / len(brightness_vals) if brightness_vals else 0
        variance = (sum((b - mean_brightness) ** 2 for b in brightness_vals)
                    / len(brightness_vals)) if brightness_vals else 0
        contrast = variance ** 0.5

        # Dominant colours (quick quantise) --------------------------------
        quantised = rgb.quantize(colors=5, method=Image.Quantize.MEDIANCUT)
        palette = quantised.getpalette()[:15]  # 5 colours × 3 channels
        dominant = []
        for i in range(0, len(palette), 3):
            dominant.append("#{:02x}{:02x}{:02x}".format(
                palette[i], palette[i + 1], palette[i + 2]))

        # Colour variance --------------------------------------------------
        r_vals = [p[0] for p in pixels]
        g_vals = [p[1] for p in pixels]
        b_vals = [p[2] for p in pixels]
        color_variance = (
            _std(r_vals) + _std(g_vals) + _std(b_vals)
        ) / 3

        # Text-region heuristic --------------------------------------------
        # High-contrast, wide variation often indicates text overlays
        has_text_region = contrast > 60 and color_variance > 40

        return {
            "width": width,
            "height": height,
            "format": img.format or "unknown",
            "mode": img.mode,
            "has_text_region": has_text_region,
            "dominant_colors": dominant,
            "brightness": round(mean_brightness, 2),
            "contrast": round(contrast, 2),
            "color_variance": round(color_variance, 2),
        }
    except Exception:
        logger.exception("Image feature extraction failed for %s", image_path)
        return {
            "width": 0, "height": 0, "format": "unknown", "mode": "unknown",
            "has_text_region": False, "dominant_colors": [],
            "brightness": 0, "contrast": 0, "color_variance": 0,
        }


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _std(values):
    """Return the population standard deviation of *values*."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
