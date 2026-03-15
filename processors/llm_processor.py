"""LLM-based meme content analysis and ExplainHM-style debate pipeline."""

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


DEEPSEEK_BASE_URL_DEFAULT = "https://api.deepseek.com"
ALIYUN_BASE_URL_DEFAULT = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def is_available():
    """Return True when the LLM backend is usable."""
    return _HAS_OPENAI and bool(os.environ.get("OPENAI_API_KEY"))


def is_vision_available():
    """Return True when a vision-capable model is configured.

    Requires a vision-capable model and either a dedicated vision API key
    or the default OPENAI_API_KEY.
    """
    vision_key = os.environ.get("OPENAI_VISION_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return (
        _HAS_OPENAI
        and bool(vision_key)
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
    resolved_base = base_url or os.environ.get("OPENAI_BASE_URL", DEEPSEEK_BASE_URL_DEFAULT)
    resolved_model = model or os.environ.get("OPENAI_MODEL", "deepseek-chat")

    client_kwargs = {"api_key": api_key}
    if resolved_base:
        client_kwargs["base_url"] = resolved_base

    client = OpenAI(**client_kwargs)

    prompt = _build_prompt(extracted_text, image_features)
    return _call_and_parse_classification(
        client,
        model=resolved_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
        log_prefix="LLM analysis",
    )


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
        When an explicit *model* is provided the function only requires
        ``OPENAI_API_KEY`` to be set; ``OPENAI_VISION_MODEL`` is not needed.
        This allows callers to attempt vision analysis using ``OPENAI_MODEL``
        when ``OPENAI_VISION_MODEL`` is not configured – the call fails
        gracefully if the model does not support image inputs.
    base_url : str | None
        Override the API base URL (defaults to ``OPENAI_BASE_URL``).

    Returns
    -------
    dict | None
        ``None`` when the vision backend is unavailable or the call fails,
        otherwise a dict with: ``is_harmful``, ``harm_score``, ``categories``,
        ``justification``.
    """
    vision_key = os.environ.get("OPENAI_VISION_API_KEY") or os.environ.get("OPENAI_API_KEY")
    vision_base = base_url or os.environ.get("OPENAI_VISION_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL", DEEPSEEK_BASE_URL_DEFAULT
    )

    # When an explicit model is provided we only need OPENAI_API_KEY.
    # Without an explicit model, require OPENAI_VISION_MODEL to be set so that
    # we never silently fall back to a text-only model as the vision backend.
    needs_key_only = model is not None
    if needs_key_only:
        if not (_HAS_OPENAI and bool(vision_key)):
            logger.info("LLM backend unavailable – skipping vision analysis")
            return None
    elif not is_vision_available():
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

    api_key = vision_key
    resolved_model = model or os.environ.get("OPENAI_VISION_MODEL")
    resolved_base = vision_base

    client_kwargs = {"api_key": api_key}
    if resolved_base:
        client_kwargs["base_url"] = resolved_base

    client = OpenAI(**client_kwargs)

    features_text = "\n".join(f"- {k}: {v}" for k, v in image_features.items())

    return _call_and_parse_classification(
        client,
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
        log_prefix=f"Vision-based meme analysis for {image_path}",
    )


def is_explainhm_available():
    """Return True when both debate sides can run.

    - Positive/benign debater: DeepSeek (`OPENAI_*`)
    - Negative/harmful debater: Aliyun (`OPENAI_VISION_*`)
    """
    has_deepseek = _HAS_OPENAI and bool(os.environ.get("OPENAI_API_KEY")) and bool(
        os.environ.get("OPENAI_MODEL", "deepseek-chat")
    )
    has_aliyun = _HAS_OPENAI and bool(os.environ.get("OPENAI_VISION_API_KEY")) and bool(
        os.environ.get("OPENAI_VISION_MODEL", "qwen-vl-plus")
    )
    return has_deepseek and has_aliyun


def run_explainhm_pipeline(extracted_text, image_features, *, image_path=None):
    """Run ExplainHM-like three-act pipeline: debate -> judge -> result."""
    if not is_explainhm_available():
        return None

    benign_argument = _generate_debate_argument(
        side="benign",
        extracted_text=extracted_text,
        image_features=image_features,
        image_path=image_path,
    )
    harmful_argument = _generate_debate_argument(
        side="harmful",
        extracted_text=extracted_text,
        image_features=image_features,
        image_path=image_path,
    )
    if benign_argument is None or harmful_argument is None:
        return None

    judged = _judge_debate(
        extracted_text=extracted_text,
        image_features=image_features,
        image_path=image_path,
        benign_argument=benign_argument,
        harmful_argument=harmful_argument,
    )
    if judged is None:
        return None

    return {
        "is_harmful": judged["is_harmful"],
        "harm_score": judged["harm_score"],
        "categories": judged.get("categories", []),
        "justification": judged.get("justification", ""),
        "pro_rationale": benign_argument.get("rationale", ""),
        "con_rationale": harmful_argument.get("rationale", ""),
        "judge_reasoning": judged.get("judge_reasoning", ""),
        "judge_side": judged.get("selected_side", ""),
        "debate": {
            "benign": benign_argument,
            "harmful": harmful_argument,
        },
    }


def _generate_debate_argument(side, extracted_text, image_features, image_path):
    is_benign = side == "benign"
    if is_benign:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", DEEPSEEK_BASE_URL_DEFAULT)
        model = os.environ.get("OPENAI_MODEL", "deepseek-chat")
        system_prompt = _DEBATER_BENIGN_SYSTEM_PROMPT
    else:
        api_key = os.environ.get("OPENAI_VISION_API_KEY")
        base_url = os.environ.get("OPENAI_VISION_BASE_URL", ALIYUN_BASE_URL_DEFAULT)
        model = os.environ.get("OPENAI_VISION_MODEL", "qwen-vl-plus")
        system_prompt = _DEBATER_HARMFUL_SYSTEM_PROMPT

    if not (_HAS_OPENAI and api_key and model):
        return None

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    user_content = _build_debate_user_content(
        extracted_text=extracted_text,
        image_features=image_features,
        image_path=image_path,
        include_image=(not is_benign and bool(image_path)),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        payload = _parse_json_block(response.choices[0].message.content)
        if payload is None:
            return None
        return {
            "stance": side,
            "confidence": float(payload.get("confidence", 0.5)),
            "rationale": str(payload.get("rationale", "")).strip(),
            "evidence": payload.get("evidence", []),
        }
    except Exception:
        logger.exception("ExplainHM debate generation failed for side=%s", side)
        return None


def _judge_debate(extracted_text, image_features, image_path,
                  benign_argument, harmful_argument):
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "deepseek-chat")
    base_url = os.environ.get("OPENAI_BASE_URL", DEEPSEEK_BASE_URL_DEFAULT)
    if not (_HAS_OPENAI and api_key and model):
        return None

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    judge_payload = {
        "extracted_text": extracted_text or "",
        "image_features": image_features,
        "benign_argument": benign_argument,
        "harmful_argument": harmful_argument,
    }
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(judge_payload, ensure_ascii=False)},
    ]
    if image_path:
        image_part = _build_inline_image_part(image_path)
        if image_part is not None:
            messages[1] = {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(judge_payload, ensure_ascii=False)},
                    image_part,
                ],
            }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1200,
        )
        payload = _parse_json_block(response.choices[0].message.content)
        if payload is None:
            return None
        return {
            "is_harmful": bool(payload.get("is_harmful", False)),
            "harm_score": float(payload.get("harm_score", 0.0)),
            "categories": payload.get("categories", []),
            "justification": str(payload.get("justification", "")).strip(),
            "selected_side": str(payload.get("selected_side", "")).strip(),
            "judge_reasoning": str(payload.get("judge_reasoning", "")).strip(),
        }
    except Exception:
        logger.exception("ExplainHM judge failed")
        return None


def _build_debate_user_content(extracted_text, image_features, image_path,
                               include_image):
    payload = {
        "task": "analyse meme and produce rationale",
        "extracted_text": extracted_text or "",
        "image_features": image_features,
    }
    if include_image and image_path:
        image_part = _build_inline_image_part(image_path)
        if image_part is not None:
            return [
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                image_part,
            ]
    return json.dumps(payload, ensure_ascii=False)


def _build_inline_image_part(image_path):
    try:
        with open(image_path, "rb") as fh:
            img_bytes = fh.read()
    except OSError:
        logger.warning("Cannot read image for inline content: %s", image_path)
        return None

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = _IMAGE_MIME_TYPES.get(ext, "jpeg")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{mime};base64,{img_b64}"},
    }


def _call_and_parse_classification(client, *, model, messages,
                                   temperature, max_tokens, log_prefix):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return _parse_response(content)
    except Exception:
        logger.exception("%s failed", log_prefix)
        return None


def _parse_json_block(raw):
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n", 1)
        cleaned = lines[1] if len(lines) > 1 else ""
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Model returned invalid JSON block: %s", cleaned[:180])
        return None


# ------------------------------------------------------------------ #
# Prompt engineering
# ------------------------------------------------------------------ #

_SYSTEM_PROMPT = """\
You are an expert content-moderation AI specialising in hateful meme detection. \
Your task is to analyse memes and determine whether they contain harmful content.

A meme consists of a picture combined with concise text. You will receive:
1. Text extracted from the meme image (via OCR).
2. Visual features of the image (size, colours, contrast, etc.).

## Key analysis principles

**Combination effect**: Hateful memes derive much of their harm from the *combination* \
of image and text. A caption that appears benign in isolation (e.g. "look how \
they live") can become hateful when attached to an image of a specific ethnic group. \
Always consider whether the text + image combination targets a protected group.

**Protected characteristics**: Pay careful attention to content that targets people \
based on race, ethnicity, religion, gender, sexual orientation, disability, \
nationality, or immigration status.

**Coded language and dog whistles**: Hateful content often uses indirect references, \
numeric codes (e.g. "14 words" – a white-supremacist slogan, "88" – code for "Heil Hitler"), \
or coded phrases to signal hate while maintaining plausible deniability. Common examples include:
- Dehumanising comparisons of ethnic groups to animals, vermin, or diseases
- Tropes about Jewish people controlling media, banks, or governments
- Framing of immigrants or minorities as inherently criminal or dangerous
- "Great Replacement" or similar demographic-threat narratives
- Anti-LGBTQ+ framing of sexual orientation as deviant or dangerous

**Satire vs. genuine hate**: Satire that *criticises* hateful viewpoints is not \
itself harmful. However, satire that *expresses* or *normalises* hateful views is \
harmful even when framed as a joke. When in doubt, lean toward classifying as \
harmful.

**Benign confounders**: Some memes use the same image as a hateful meme but with \
different captions. Evaluate the actual text present, not assumed context.

Based on the provided information you MUST return a JSON object with EXACTLY these keys:
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

Harm score guidance:
- 0.0–0.2  : Clearly benign content
- 0.2–0.4  : Mildly concerning but unlikely to cause harm
- 0.4–0.6  : Moderately harmful – offensive or demeaning to a group
- 0.6–0.8  : Clearly harmful – promotes hatred, discrimination, or violence
- 0.8–1.0  : Severely harmful – explicit hate speech, incitement, or graphic content

Rules:
- Be precise and fair. Not every meme with strong language is harmful.
- Consider context, sarcasm, and satire, but remember that "it's just a joke" \
  does not excuse genuine hate directed at protected groups.
- Always provide a thorough justification regardless of the classification.
- If content targets a protected group in a dehumanising or threatening way, \
  classify as harmful even if the language appears indirect.
- Return ONLY valid JSON, no markdown fences, no extra text.
"""


_DEBATER_BENIGN_SYSTEM_PROMPT = """\
You are the benign-side debater in harmful meme analysis.

Goal: argue that the meme is NOT harmful, using evidence from image/text/context.

Return ONLY JSON with keys:
{
    "confidence": <float 0.0-1.0>,
    "rationale": "<2-5 sentence argument for non-harmful interpretation>",
    "evidence": ["<short evidence item>", "..."]
}
"""


_DEBATER_HARMFUL_SYSTEM_PROMPT = """\
You are the harmful-side debater in harmful meme analysis.

Goal: argue that the meme IS harmful, using evidence from image/text/context,
including implicit hate, discrimination, dehumanization, misinformation, or
violent framing if present.

Return ONLY JSON with keys:
{
    "confidence": <float 0.0-1.0>,
    "rationale": "<2-5 sentence argument for harmful interpretation>",
    "evidence": ["<short evidence item>", "..."]
}
"""


_JUDGE_SYSTEM_PROMPT = """\
You are the debate judge for harmful meme detection.

Input includes:
- extracted_text
- image_features
- benign_argument
- harmful_argument

Choose which side is more convincing and output a final decision.

Return ONLY JSON with EXACT keys:
{
    "is_harmful": true/false,
    "harm_score": <float 0.0-1.0>,
    "categories": ["Hate Speech"|"Violence / Threats"|"Cyberbullying / Harassment"|"Sexual / Explicit Content"|"Self-Harm / Suicide"|"Misinformation"|"Other Harmful Content"],
    "selected_side": "benign"|"harmful",
    "judge_reasoning": "<why one side wins>",
    "justification": "<final moderation explanation>"
}
"""


def _build_prompt(extracted_text, image_features):
    """Build the user message sent to the LLM."""
    parts = ["Analyse the following meme content:\n"]

    if extracted_text and extracted_text.strip():
        parts.append(f"## Extracted text\n```\n{extracted_text}\n```\n")
    elif image_features.get("has_text_region"):
        parts.append("""\
## Extracted text
(OCR failed to extract text from this image. However, image analysis \
detected a text overlay region, indicating the meme very likely \
contains text that OCR could not read. Do NOT classify the meme as \
non-harmful solely because text could not be extracted – treat this \
as an uncertain case and apply appropriate caution in your analysis.)
""")
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
