"""Text processing module for meme content analysis.

Provides harmful-content detection by matching extracted meme text against
a curated database of keywords and regex patterns.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def load_keywords(keywords_path):
    """Load the harmful keywords database from *keywords_path*.

    Returns the ``categories`` dict (category-name → info).
    """
    with open(keywords_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("categories", {})


def analyse_text(text, keywords_path):
    """Analyse *text* for harmful content.

    Returns a dict with:
    * ``matched_categories`` – list of dicts per matched category
    * ``overall_score`` – aggregate harm score (0.0 – 1.0)
    * ``is_harmful`` – boolean flag (score ≥ 0.4)
    """
    if not text or not text.strip():
        return {
            "matched_categories": [],
            "overall_score": 0.0,
            "is_harmful": False,
        }

    categories = load_keywords(keywords_path)
    normalised = text.lower()

    matched = []

    for cat_key, cat_info in categories.items():
        kw_hits = _match_keywords(normalised, cat_info.get("keywords", []))
        pat_hits = _match_patterns(normalised, cat_info.get("patterns", []))

        if kw_hits or pat_hits:
            matched.append({
                "category": cat_key,
                "label": cat_info["label"],
                "description": cat_info["description"],
                "severity": cat_info.get("severity", 0.5),
                "keyword_matches": kw_hits,
                "pattern_matches": pat_hits,
            })

    if not matched:
        return {
            "matched_categories": [],
            "overall_score": 0.0,
            "is_harmful": False,
        }

    overall = max(m["severity"] for m in matched)
    return {
        "matched_categories": matched,
        "overall_score": round(overall, 4),
        "is_harmful": overall >= 0.4,
    }


# ---------------------------------------------------------------------- #
# Internal helpers
# ---------------------------------------------------------------------- #

def _match_keywords(text, keywords):
    """Return the subset of *keywords* found in *text*."""
    hits = []
    for kw in keywords:
        if kw.lower() in text:
            hits.append(kw)
    return hits


def _match_patterns(text, patterns):
    """Return matching regex *patterns* found in *text*."""
    hits = []
    for pat in patterns:
        try:
            if re.search(pat, text, re.IGNORECASE):
                hits.append(pat)
        except re.error:
            logger.warning("Invalid regex pattern: %s", pat)
    return hits
