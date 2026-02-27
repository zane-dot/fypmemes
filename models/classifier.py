"""Harmfulness classifier for memes.

Combines image features and text-analysis results to produce a final
classification together with a human-readable justification.
"""

import json


def classify(text_result, image_features):
    """Classify a meme and generate a justification.

    Parameters
    ----------
    text_result : dict
        Output of :func:`processors.text_processor.analyse_text`.
    image_features : dict
        Output of :func:`processors.image_processor.extract_image_features`.

    Returns
    -------
    dict with keys:
        * ``is_harmful`` (bool)
        * ``harm_score`` (float 0-1)
        * ``categories`` (JSON string – list of matched category labels)
        * ``justification`` (str – human-readable explanation)
        * ``image_features`` (JSON string)
    """
    text_score = text_result.get("overall_score", 0.0)
    matched = text_result.get("matched_categories", [])

    # Image-only heuristic boost (e.g. high-contrast text overlay detected)
    image_boost = 0.0
    if image_features.get("has_text_region") and not matched:
        image_boost = 0.05  # slight bump when text region detected

    harm_score = min(1.0, text_score + image_boost)
    is_harmful = harm_score >= 0.4

    # Build justification --------------------------------------------------
    justification = _build_justification(is_harmful, harm_score, matched,
                                         image_features)

    category_labels = [m["label"] for m in matched]

    return {
        "is_harmful": is_harmful,
        "harm_score": round(harm_score, 4),
        "categories": json.dumps(category_labels),
        "justification": justification,
        "image_features": json.dumps(image_features),
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
