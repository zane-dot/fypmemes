"""Harmfulness classifier for memes.

Uses an LLM (when available) as the **primary** analysis engine.  Falls
back to the keyword / pattern-based classifier when no LLM API key is
configured.

The final output always contains:
* ``is_harmful`` – boolean
* ``harm_score`` – 0.0 … 1.0
* ``categories`` – JSON-encoded list of category labels
* ``justification`` – human-readable explanation
* ``image_features`` – JSON-encoded dict of visual features
* ``analysis_method`` – ``"llm"`` or ``"keyword"``
"""

import json
import logging

from processors.llm_processor import (
    analyse_meme,
    analyse_meme_with_vision,
    is_available as llm_available,
    is_vision_available as llm_vision_available,
)

logger = logging.getLogger(__name__)


def classify(text_result, image_features, extracted_text="", image_path=None):
    """Classify a meme and generate a justification.

    Parameters
    ----------
    text_result : dict
        Output of :func:`processors.text_processor.analyse_text`.
    image_features : dict
        Output of :func:`processors.image_processor.extract_image_features`.
    extracted_text : str
        Raw OCR text (passed through to the LLM for richer analysis).
    image_path : str | None
        Absolute path to the original meme image.  When provided and a
        vision-capable model is configured, the image is sent directly to the
        vision LLM for harm analysis.  This is more reliable than OCR-then-
        analyse for memes where text extraction is unreliable (e.g. neutral
        colour palettes, low-contrast text).

    Returns
    -------
    dict
    """
    # ---- Try vision-based analysis first (most accurate for memes) -----
    if image_path and llm_vision_available():
        vision_result = analyse_meme_with_vision(image_path, image_features)
        if vision_result is not None:
            return _format_llm_result(vision_result, image_features)
        logger.warning("Vision analysis returned None – falling back to text LLM")

    # ---- Try text-only LLM analysis ------------------------------------
    if llm_available():
        llm_result = analyse_meme(extracted_text, image_features)
        if llm_result is not None:
            return _format_llm_result(llm_result, image_features)
        logger.warning("LLM analysis returned None – falling back to keywords")

    # ---- Keyword / pattern fallback ------------------------------------
    return _keyword_classify(text_result, image_features)


# ------------------------------------------------------------------ #
# LLM result formatting
# ------------------------------------------------------------------ #

def _format_llm_result(llm_result, image_features):
    categories = llm_result.get("categories", [])
    return {
        "is_harmful": llm_result["is_harmful"],
        "harm_score": round(llm_result["harm_score"], 4),
        "categories": json.dumps(categories),
        "justification": llm_result["justification"],
        "image_features": json.dumps(image_features),
        "analysis_method": "llm",
    }


# ------------------------------------------------------------------ #
# Keyword / pattern fallback
# ------------------------------------------------------------------ #

def _keyword_classify(text_result, image_features):
    text_score = text_result.get("overall_score", 0.0)
    matched = text_result.get("matched_categories", [])

    # Image-only heuristic boost
    image_boost = 0.0
    if image_features.get("has_text_region") and not matched:
        image_boost = 0.05

    harm_score = min(1.0, text_score + image_boost)
    is_harmful = harm_score >= 0.4

    justification = _build_justification(is_harmful, harm_score, matched,
                                         image_features)
    category_labels = [m["label"] for m in matched]

    return {
        "is_harmful": is_harmful,
        "harm_score": round(harm_score, 4),
        "categories": json.dumps(category_labels),
        "justification": justification,
        "image_features": json.dumps(image_features),
        "analysis_method": "keyword",
    }


def _build_justification(is_harmful, score, matched_categories,
                         image_features):
    """Generate a human-readable justification string."""
    parts = []

    if not is_harmful:
        parts.append(
            "This meme has been classified as NOT harmful "
            f"(score: {score:.2f}/1.00)."
        )
        if image_features.get("has_text_region"):
            parts.append(
                "A text overlay region was detected in the image, but no "
                "harmful textual content was identified."
            )
        else:
            parts.append("No harmful content was detected in the meme.")
        return " ".join(parts)

    parts.append(
        f"This meme has been classified as HARMFUL (score: {score:.2f}/1.00)."
    )

    for cat in matched_categories:
        label = cat["label"]
        desc = cat["description"]
        kw = cat.get("keyword_matches", [])
        pat = cat.get("pattern_matches", [])

        parts.append(f"\n• Category: {label}")
        parts.append(f"  {desc}")
        if kw:
            parts.append(f"  Matched keywords: {', '.join(kw)}")
        if pat:
            parts.append(f"  Matched patterns: {len(pat)} pattern(s)")

    if image_features.get("has_text_region"):
        parts.append(
            "\nAdditionally, image analysis detected a prominent text "
            "overlay region, which is consistent with meme-format imagery "
            "that embeds harmful messages."
        )

    return "\n".join(parts)
