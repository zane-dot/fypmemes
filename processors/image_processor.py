"""Image processing module for meme analysis.

Handles image feature extraction and OCR text extraction from meme images.
"""

import base64
import logging
import os

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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

try:
    from openai import OpenAI as _OpenAI

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

# Module-level EasyOCR reader cache – initialised on first use so that the
# heavy model download/load only happens when OCR is actually needed.
_easyocr_reader = None

# Minimum image dimension (pixels) for reliable OCR.  Images smaller than
# this are upscaled before being fed to OCR engines.
_OCR_MIN_DIM = 800

# Contrast enhancement factor applied during preprocessing.
_OCR_CONTRAST_FACTOR = 1.5

# MIME type map for base64-encoded image data URIs sent to the vision API.
_IMAGE_MIME_TYPES = {
    "jpg": "jpeg", "jpeg": "jpeg", "png": "png",
    "gif": "gif", "webp": "webp",
}


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


def _preprocess_for_ocr(image_path):
    """Return a preprocessed numpy array of the image ready for EasyOCR.

    Steps applied:
    1. Upscale if either dimension is below ``_OCR_MIN_DIM`` (small images
       confuse OCR engines).
    2. Enhance contrast so text stands out from busy backgrounds.
    3. Apply a mild sharpening pass to improve character edge clarity.
    """
    img = Image.open(image_path).convert("RGB")

    # 1. Upscale small images -----------------------------------------------
    w, h = img.size
    if w < _OCR_MIN_DIM or h < _OCR_MIN_DIM:
        scale = max(_OCR_MIN_DIM / w, _OCR_MIN_DIM / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # 2. Contrast enhancement -----------------------------------------------
    img = ImageEnhance.Contrast(img).enhance(_OCR_CONTRAST_FACTOR)

    # 3. Sharpening ---------------------------------------------------------
    img = img.filter(ImageFilter.SHARPEN)

    return np.array(img)


def _preprocess_for_ocr_high_contrast(image_path):
    """Return a high-contrast enhanced numpy array for EasyOCR.

    Applies aggressive contrast enhancement (factor 3.0) with double sharpening
    to improve text extraction from low-contrast meme images (e.g. text on
    neutral or light-coloured backgrounds such as beige, white, or brown).
    """
    img = Image.open(image_path).convert("RGB")

    w, h = img.size
    if w < _OCR_MIN_DIM or h < _OCR_MIN_DIM:
        scale = max(_OCR_MIN_DIM / w, _OCR_MIN_DIM / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img = ImageEnhance.Contrast(img).enhance(3.0)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SHARPEN)

    return np.array(img)


def _preprocess_for_ocr_binarised(image_path):
    """Return a binarised (grayscale + median threshold) numpy array.

    Meme text is often rendered in a high-contrast style (e.g. white Impact
    font with a dark outline) that is well-suited to binarisation.  This
    variant is used as a second-pass attempt when the standard preprocessing
    fails to extract any text.
    """
    img = Image.open(image_path).convert("RGB")

    # Upscale as before
    w, h = img.size
    if w < _OCR_MIN_DIM or h < _OCR_MIN_DIM:
        scale = max(_OCR_MIN_DIM / w, _OCR_MIN_DIM / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Convert to grayscale, enhance contrast, then apply a global median
    # threshold to produce a clean black-and-white image.  The median is a
    # robust centre-point for meme images where the background and foreground
    # pixel populations are roughly balanced.
    gray = img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = gray.filter(ImageFilter.SHARPEN)
    arr = np.array(gray)
    # Simple global threshold at the median value
    threshold = int(np.median(arr))
    binary = (arr > threshold).astype(np.uint8) * 255
    # Return as RGB array (EasyOCR accepts both grayscale and RGB)
    return np.stack([binary, binary, binary], axis=-1)


def _preprocess_for_ocr_equalized(image_path):
    """Return a histogram-equalised grayscale numpy array for EasyOCR.

    Histogram equalisation redistributes pixel intensities to span the full
    0–255 range.  This dramatically improves contrast for memes where the text
    and background share a similar average brightness, which is common on
    coloured backgrounds (pink, beige, pastel, etc.) where a standard
    luminance-based preprocessing leaves text nearly invisible.
    """
    img = Image.open(image_path).convert("L")

    w, h = img.size
    if w < _OCR_MIN_DIM or h < _OCR_MIN_DIM:
        scale = max(_OCR_MIN_DIM / w, _OCR_MIN_DIM / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img = ImageOps.equalize(img)
    arr = np.array(img)
    return np.stack([arr, arr, arr], axis=-1)


def _preprocess_for_ocr_saturation(image_path):
    """Return a per-pixel colour-saturation map as a numpy array for EasyOCR.

    The saturation map measures the spread across RGB channels at each pixel
    (max – min).  Pixels where one channel dominates look bright; uniform-grey
    pixels look dark.  For memes with coloured text on a differently-coloured
    background (e.g. red text on a pink background, yellow text on a
    brown/beige background) the text edges become highly visible in this
    representation even though standard greyscale conversion loses the contrast.
    """
    img = Image.open(image_path).convert("RGB")

    w, h = img.size
    if w < _OCR_MIN_DIM or h < _OCR_MIN_DIM:
        scale = max(_OCR_MIN_DIM / w, _OCR_MIN_DIM / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)
    # Per-pixel saturation: range of RGB values at each location
    sat = arr.max(axis=-1) - arr.min(axis=-1)
    if sat.max() > 0:
        sat = (sat / sat.max() * 255).astype(np.uint8)
    else:
        sat = sat.astype(np.uint8)
    return np.stack([sat, sat, sat], axis=-1)


def _extract_text_via_vision(image_path):
    """Extract text from *image_path* using an OpenAI-compatible vision model.

    The model is selected from the ``OPENAI_VISION_MODEL`` environment variable
    (falls back to ``OPENAI_MODEL``, then ``deepseek-chat``).  Set
    ``OPENAI_VISION_MODEL`` to a vision-capable model such as ``gpt-4o`` or
    ``deepseek-vl2`` to use this path.

    Returns ``None`` when the API is not configured or the call fails, so the
    caller can fall through to the next OCR backend.
    """
    if not _HAS_OPENAI:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        with open(image_path, "rb") as fh:
            img_bytes = fh.read()
    except OSError:
        logger.warning("Vision OCR: cannot read file %s", image_path)
        return None

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = _IMAGE_MIME_TYPES.get(ext, "jpeg")

    # Only proceed when a vision-capable model is explicitly configured.
    # Falling back to a text-only model (e.g. deepseek-chat) would cause the
    # API call to fail because it cannot process image inputs.
    resolved_model = os.environ.get("OPENAI_VISION_MODEL")
    if not resolved_model:
        return None

    resolved_base = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")

    client_kwargs = {"api_key": api_key}
    if resolved_base:
        client_kwargs["base_url"] = resolved_base

    client = _OpenAI(**client_kwargs)

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all text visible in this meme image. "
                            "Return only the extracted text with no extra "
                            "commentary. If no text is present, return an "
                            "empty string."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{mime};base64,{img_b64}",
                        },
                    },
                ],
            }],
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.warning("Vision API OCR failed for %s", image_path, exc_info=True)
        return None


def extract_text(image_path):
    """Extract text from a meme image using OCR.

    Pipeline (most-to-least accurate):
    1. **Vision API** – sends the image to an OpenAI-compatible vision model
       (e.g. ``gpt-4o`` or ``deepseek-vl2``).  Skipped when
       ``OPENAI_API_KEY`` is not set or the API call fails.
    2. **EasyOCR** with up to seven preprocessing passes, each targeting a
       different class of meme imagery:

       * Pass 1 – paragraph mode with standard preprocessing.
       * Pass 2 – flat (non-paragraph) mode with standard preprocessing.
       * Pass 3 – binarised image (robust for Impact-style white-on-dark text).
       * Pass 4 – high-contrast (×3) image (low-contrast backgrounds).
       * Pass 5 – inverted binarised image (light text on light backgrounds).
       * Pass 6 – histogram-equalised greyscale (coloured/pastel backgrounds
         where text and background share similar average brightness).
       * Pass 7 – colour-saturation map (coloured text on a differently-
         coloured background, e.g. red text on a pink background).

       Supports English and Simplified Chinese without an external binary.
    3. **pytesseract** – Tesseract-based fallback for environments where
       EasyOCR is not available.

    Returns an empty string when no OCR backend is available or all backends
    fail.
    """
    # 1. Vision API (most accurate for memes) --------------------------------
    vision_text = _extract_text_via_vision(image_path)
    if vision_text is not None:
        logger.debug("Vision API OCR succeeded for %s", image_path)
        return vision_text

    # 2. EasyOCR with preprocessing -----------------------------------------
    if _HAS_EASYOCR:
        try:
            reader = _get_easyocr_reader()
            img_array = _preprocess_for_ocr(image_path)
            results = reader.readtext(
                img_array,
                detail=0,
                paragraph=True,
                text_threshold=0.3,
                low_text=0.3,
                adjust_contrast=0.5,
                min_size=2,
            )
            text = " ".join(results).strip()
            if text:
                return text
            # If paragraph mode returned nothing, retry without paragraph
            # grouping (sometimes paragraph=True misses isolated words).
            results_flat = reader.readtext(
                img_array,
                detail=0,
                paragraph=False,
                text_threshold=0.3,
                low_text=0.3,
                adjust_contrast=0.5,
                min_size=2,
            )
            text = " ".join(results_flat).strip()
            if text:
                return text
            # Third EasyOCR attempt: binarised preprocessing is more robust
            # for meme-style text (e.g. white Impact font on a photo).
            img_bin = _preprocess_for_ocr_binarised(image_path)
            results_bin = reader.readtext(
                img_bin,
                detail=0,
                paragraph=False,
                text_threshold=0.3,
                low_text=0.3,
                min_size=2,
            )
            text = " ".join(results_bin).strip()
            if text:
                return text
            # Fourth attempt: high-contrast preprocessing helps with text on
            # busy or low-contrast backgrounds (e.g. beige/brown meme images).
            img_hc = _preprocess_for_ocr_high_contrast(image_path)
            results_hc = reader.readtext(
                img_hc,
                detail=0,
                paragraph=False,
                text_threshold=0.3,
                low_text=0.3,
                min_size=2,
            )
            text = " ".join(results_hc).strip()
            if text:
                return text
            # Fifth attempt: inverted binarised image handles light-coloured
            # text on light backgrounds that the standard binarisation inverts.
            results_inv = reader.readtext(
                255 - img_bin,
                detail=0,
                paragraph=False,
                text_threshold=0.3,
                low_text=0.3,
                min_size=2,
            )
            text = " ".join(results_inv).strip()
            if text:
                return text
            # Sixth attempt: histogram-equalised image helps with low-contrast
            # memes (pink/reddish/pastel backgrounds) where text and background
            # share a similar average brightness.
            img_eq = _preprocess_for_ocr_equalized(image_path)
            results_eq = reader.readtext(
                img_eq,
                detail=0,
                paragraph=False,
                text_threshold=0.3,
                low_text=0.3,
                min_size=2,
            )
            text = " ".join(results_eq).strip()
            if text:
                return text
            # Seventh attempt: colour-saturation map highlights text whose
            # colour differs from the background (e.g. red text on pink) even
            # when the luminance contrast is too low for other passes.
            img_sat = _preprocess_for_ocr_saturation(image_path)
            results_sat = reader.readtext(
                img_sat,
                detail=0,
                paragraph=False,
                text_threshold=0.3,
                low_text=0.3,
                min_size=2,
            )
            text = " ".join(results_sat).strip()
            if text:
                return text
        except Exception:
            logger.exception("EasyOCR failed for %s", image_path)
            # Fall through to the pytesseract fallback below.

    # 3. pytesseract fallback -----------------------------------------------
    # Used when EasyOCR is unavailable OR when all EasyOCR passes return empty.
    if _HAS_TESSERACT:
        try:
            img = Image.open(image_path)
            # Apply the same preprocessing for consistency.
            w, h = img.size
            if w < _OCR_MIN_DIM or h < _OCR_MIN_DIM:
                scale = max(_OCR_MIN_DIM / w, _OCR_MIN_DIM / h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            img = ImageEnhance.Contrast(img).enhance(_OCR_CONTRAST_FACTOR)
            # Request both English and Chinese (simplified + traditional).
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
        # High-contrast, wide variation often indicates text overlays.
        # The contrast threshold is set to 35 (rather than 60) so that memes
        # with moderate brightness contrast on coloured backgrounds (pink,
        # reddish, pastel, etc.) are also correctly flagged.  A white-text-on-
        # pink-background meme typically yields contrast ≈ 50–60 and
        # color_variance ≈ 45–55, which was missed by the previous threshold.
        has_text_region = contrast > 35 and color_variance > 40

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
