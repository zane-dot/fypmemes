"""LLM-based meme content analysis.

Uses an OpenAI-compatible API to analyse the extracted text and image
features of a meme.  The LLM provides:

* A harmfulness classification (harmful / not harmful)
* A harm score (0.0 – 1.0)
* Detected harmful categories
* A human-readable justification explaining the reasoning

When no API key is configured the module returns ``None`` so the caller
can fall back to the keyword-based classifier.
"""

import base64
import json
import logging
import os

logger = logging.getLogger(__name__)

# MIME type map for base64-encoded image data URIs sent to the vision API.
_IMAGE_MIME_TYPES = {
    "jpg": "jpeg", "jpeg": "jpeg", "png": "png",
    "gif": "gif", "webp": "webp",
}

try:
    from openai import OpenAI

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def is_available():
    """Return True when the LLM backend is usable."""
    return _HAS_OPENAI and bool(os.environ.get("OPENAI_API_KEY"))


def is_vision_available():
    """Return True when a vision-capable model is configured.

    Requires both ``OPENAI_API_KEY`` and ``OPENAI_VISION_MODEL`` to be set.
    """
    return (
        _HAS_OPENAI
        and bool(os.environ.get("OPENAI_API_KEY"))
        and bool(os.environ.get("OPENAI_VISION_MODEL"))
    )


def analyse_meme(extracted_text, image_features, *, model=None, base_url=None):
    """Analyse meme content using an LLM.

    Parameters
    ----------
    extracted_text : str
        Text extracted from the meme via OCR.
    image_features : dict
        Visual features produced by :func:`processors.image_processor.extract_image_features`.
    model : str | None
        Override the model name (defaults to ``OPENAI_MODEL`` env-var or
        ``gpt-4o-mini``).
    base_url : str | None
        Override the API base URL (defaults to ``OPENAI_BASE_URL`` env-var).

    Returns
    -------
    dict | None
        ``None`` when the LLM backend is unavailable, otherwise a dict with:
        ``is_harmful``, ``harm_score``, ``categories``, ``justification``.
    """
    if not is_available():
        logger.info("LLM backend unavailable – skipping LLM analysis")
        return None

    api_key = os.environ["OPENAI_API_KEY"]
    resolved_base = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    resolved_model = model or os.environ.get("OPENAI_MODEL", "deepseek-chat")

    client_kwargs = {"api_key": api_key}
    if resolved_base:
        client_kwargs["base_url"] = resolved_base

    client = OpenAI(**client_kwargs)

    prompt = _build_prompt(extracted_text, image_features)

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        return _parse_response(content)
    except Exception:
        logger.exception("LLM analysis failed")
        return None


def analyse_meme_with_vision(image_path, image_features, *, model=None, base_url=None):
    """Analyse meme content by sending the image directly to a vision LLM.

    This approach is used by many harmful-meme detection platforms because it
    bypasses the OCR step entirely: the vision model can read text embedded in
    the image and reason about both the visual content and that text in one
    pass.  It is particularly effective for memes with low-contrast text on
    neutral backgrounds (e.g. white, beige, or brown) that confuse traditional
    OCR pipelines.

    Parameters
    ----------
    image_path : str
        Absolute path to the meme image file.
    image_features : dict
        Visual features produced by
        :func:`processors.image_processor.extract_image_features`.
    model : str | None
        Override the vision model name (defaults to ``OPENAI_VISION_MODEL``).
    base_url : str | None
        Override the API base URL (defaults to ``OPENAI_BASE_URL``).

    Returns
    -------
    dict | None
        ``None`` when the vision backend is unavailable or the call fails,
        otherwise a dict with: ``is_harmful``, ``harm_score``, ``categories``,
        ``justification``.
    """
    if not is_vision_available():
        logger.info("Vision LLM backend unavailable – skipping vision analysis")
        return None

    try:
        with open(image_path, "rb") as fh:
            img_bytes = fh.read()
    except OSError:
        logger.warning("Vision analysis: cannot read file %s", image_path)
        return None

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = _IMAGE_MIME_TYPES.get(ext, "jpeg")

    api_key = os.environ["OPENAI_API_KEY"]
    resolved_model = model or os.environ.get("OPENAI_VISION_MODEL")
    resolved_base = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")

    client_kwargs = {"api_key": api_key}
    if resolved_base:
        client_kwargs["base_url"] = resolved_base

    client = OpenAI(**client_kwargs)

    features_text = "\n".join(f"- {k}: {v}" for k, v in image_features.items())

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyse the following meme image for harmful content. "
                                "Read all text visible in the image and consider both "
                                "the visual content and any embedded text.\n\n"
                                "## Image features\n"
                                f"{features_text}\n\n"
                                "Please classify this meme and provide your detailed "
                                "justification."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime};base64,{img_b64}",
                            },
                        },
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        return _parse_response(content)
    except Exception:
        logger.exception("Vision-based meme analysis failed for %s", image_path)
        return None


# ------------------------------------------------------------------ #
# Prompt engineering
# ------------------------------------------------------------------ #

_SYSTEM_PROMPT = """\
You are an expert content-moderation AI. Your task is to analyse memes and \
determine whether they contain harmful content.

A meme consists of a picture combined with concise text. You will receive:
1. Text extracted from the meme image (via OCR).
2. Visual features of the image (size, colours, contrast, etc.).

Based on this information you MUST return a JSON object with EXACTLY these keys:
{
  "is_harmful": true/false,
  "harm_score": <float 0.0-1.0>,
  "categories": [<list of category strings>],
  "justification": "<detailed multi-sentence explanation>"
}

Possible categories (use only those that apply):
- "Hate Speech"
- "Violence / Threats"
- "Cyberbullying / Harassment"
- "Sexual / Explicit Content"
- "Self-Harm / Suicide"
- "Misinformation"
- "Other Harmful Content"

Rules:
- Be precise and fair. Not every meme with strong language is harmful.
- Consider context, sarcasm, and satire.
- Always provide a thorough justification regardless of the classification.
- Return ONLY valid JSON, no markdown fences, no extra text.
"""


def _build_prompt(extracted_text, image_features):
    """Build the user message sent to the LLM."""
    parts = ["Analyse the following meme content:\n"]

    if extracted_text and extracted_text.strip():
        parts.append(f"## Extracted text\n```\n{extracted_text}\n```\n")
    else:
        parts.append("## Extracted text\n(no text could be extracted)\n")

    parts.append("## Image features")
    for key, value in image_features.items():
        parts.append(f"- {key}: {value}")

    parts.append(
        "\nPlease classify this meme and provide your detailed justification."
    )
    return "\n".join(parts)


# ------------------------------------------------------------------ #
# Response parsing
# ------------------------------------------------------------------ #

def _parse_response(raw):
    """Parse the LLM JSON response into a normalised dict."""
    # Strip markdown code fences if the model wraps the output
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n", 1)
        cleaned = lines[1] if len(lines) > 1 else ""
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON: %s", raw[:200])
        return None

    return {
        "is_harmful": bool(data.get("is_harmful", False)),
        "harm_score": float(data.get("harm_score", 0.0)),
        "categories": data.get("categories", []),
        "justification": data.get("justification", ""),
    }
