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
import os

from processors.llm_processor import (
    analyse_meme,
    analyse_meme_with_vision,
    is_explainhm_available,
    is_available as llm_available,
    is_vision_available as llm_vision_available,
    run_explainhm_pipeline,
)
from models.small_model import is_small_model_available, predict_with_small_model

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
    # ---- ExplainHM debate + judge + small model (primary) --------------
    if _explainhm_enabled() and is_explainhm_available():
        explainhm_result = run_explainhm_pipeline(
            extracted_text,
            image_features,
            image_path=image_path,
        )
        if explainhm_result is not None:
            final_result = _format_llm_result(explainhm_result, image_features)
            final_result["pro_rationale"] = explainhm_result.get("pro_rationale", "")
            final_result["con_rationale"] = explainhm_result.get("con_rationale", "")
            final_result["judge_reasoning"] = explainhm_result.get("judge_reasoning", "")
            final_result["judge_side"] = explainhm_result.get("judge_side", "")
            final_result["analysis_method"] = "explainhm_judge"

            if is_small_model_available():
                small_result = predict_with_small_model({
                    "extracted_text": extracted_text,
                    "pro_rationale": final_result["pro_rationale"],
                    "con_rationale": final_result["con_rationale"],
                    "judge_reasoning": final_result["judge_reasoning"],
                    "judge_harm_score": final_result["harm_score"],
                    "keyword_score": text_result.get("overall_score", 0.0),
                    "has_text_region": 1.0 if image_features.get("has_text_region") else 0.0,
                    "brightness": float(image_features.get("brightness", 0.0) or 0.0),
                    "contrast": float(image_features.get("contrast", 0.0) or 0.0),
                    "color_variance": float(image_features.get("color_variance", 0.0) or 0.0),
                })
                if small_result is not None:
                    final_score = round(
                        (0.65 * small_result["harm_score"]) +
                        (0.35 * final_result["harm_score"]),
                        4,
                    )
                    final_result["harm_score"] = final_score
                    final_result["is_harmful"] = final_score >= 0.5
                    final_result["analysis_method"] = "explainhm_small_model"
                    final_result["justification"] = (
                        final_result["justification"]
                        + "\n\nSmall-model refinement score: "
                        + f"{small_result['harm_score']:.2f}."
                    )
            return _ensure_debate_fields(final_result, extracted_text)

    # ---- Try vision-based analysis first (legacy path) -----------------
    if image_path and llm_vision_available():
        vision_result = analyse_meme_with_vision(image_path, image_features)
        if vision_result is not None:
            return _ensure_debate_fields(
                _format_llm_result(vision_result, image_features),
                extracted_text,
            )
        logger.warning("Vision analysis returned None – falling back to text LLM")

    # ---- OCR-failure vision fallback ------------------------------------
    # When all OCR passes returned nothing but a text region was detected,
    # attempt vision analysis using OPENAI_MODEL (e.g. gpt-4o) even if
    # OPENAI_VISION_MODEL is not explicitly configured.  This mirrors the
    # approach used by other harmful-meme detection platforms that send the
    # image directly to a multimodal model.  The call fails gracefully and
    # falls through to text-only LLM if the model does not support images.
    if (
        image_path
        and not extracted_text
        and image_features.get("has_text_region")
        and llm_available()
        and not llm_vision_available()
    ):
        fallback_model = os.environ.get("OPENAI_MODEL")
        if fallback_model:
            logger.info(
                "OCR returned empty with text region detected – attempting "
                "vision fallback with model: %s", fallback_model,
            )
            vision_result = analyse_meme_with_vision(
                image_path, image_features, model=fallback_model,
            )
            if vision_result is not None:
                return _ensure_debate_fields(
                    _format_llm_result(vision_result, image_features),
                    extracted_text,
                )

    # ---- Try text-only LLM analysis ------------------------------------
    if llm_available():
        llm_result = analyse_meme(extracted_text, image_features)
        if llm_result is not None:
            return _ensure_debate_fields(
                _format_llm_result(llm_result, image_features),
                extracted_text,
            )
        logger.warning("LLM analysis returned None – falling back to keywords")

    # ---- Keyword / pattern fallback ------------------------------------
    return _ensure_debate_fields(
        _keyword_classify(text_result, image_features),
        extracted_text,
    )


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


def _explainhm_enabled():
    return os.environ.get("EXPLAINHM_ENABLED", "1") != "0"


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


def _ensure_debate_fields(result, extracted_text=""):
    """Ensure positive/negative explanations are always present for UI display."""
    if result.get("pro_rationale") and result.get("con_rationale"):
        if not result.get("judge_reasoning"):
            result["judge_reasoning"] = result.get("justification", "")
        if not result.get("judge_side"):
            result["judge_side"] = "harmful" if result.get("is_harmful") else "benign"
        return result

    justification = (result.get("justification") or "").strip()
    is_harmful = bool(result.get("is_harmful"))
    snippet = (extracted_text or "").strip()
    if len(snippet) > 220:
        snippet = snippet[:220] + "..."
    quoted = f' Observed text: "{snippet}".' if snippet else ""

    if not result.get("pro_rationale"):
        if is_harmful:
            result["pro_rationale"] = (
                "Benign-side interpretation: The meme could be read as satire, "
                "opinion, or ambiguous humor, and available evidence alone does "
                "not conclusively prove a targeted hateful intent."
                + quoted
            )
        else:
            result["pro_rationale"] = (
                justification
                or "Benign-side interpretation: No explicit hate target or "
                "malicious intent is strongly supported by current evidence."
            )

    if not result.get("con_rationale"):
        if is_harmful:
            result["con_rationale"] = (
                justification
                or "Harmful-side interpretation: The content includes cues that "
                "may promote hate, discrimination, or harmful misinformation."
            )
        else:
            result["con_rationale"] = (
                "Harmful-side interpretation: Some phrases or visual cues may "
                "appear risky, but the evidence is weak and remains below the "
                "harmful threshold."
                + quoted
            )

    if not result.get("judge_reasoning"):
        result["judge_reasoning"] = justification or "Final decision based on combined evidence and risk score."

    if not result.get("judge_side"):
        result["judge_side"] = "harmful" if is_harmful else "benign"

    return result


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


