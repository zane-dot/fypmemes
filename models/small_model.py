"""Small-model inference utilities for ExplainHM stage-3 decision."""

import os

try:
    import joblib
    import numpy as np
    from scipy.sparse import hstack

    _HAS_SMALL_MODEL_DEPS = True
except ImportError:
    _HAS_SMALL_MODEL_DEPS = False


_DEFAULT_SMALL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "small_decider.joblib"
)

_MODEL_CACHE = {}


def is_small_model_available(model_path=None):
    if not _HAS_SMALL_MODEL_DEPS:
        return False
    resolved_path = model_path or os.environ.get("SMALL_MODEL_PATH", _DEFAULT_SMALL_MODEL_PATH)
    return os.path.exists(resolved_path)


def predict_with_small_model(sample, model_path=None):
    if not _HAS_SMALL_MODEL_DEPS:
        return None

    resolved_path = model_path or os.environ.get("SMALL_MODEL_PATH", _DEFAULT_SMALL_MODEL_PATH)
    if not os.path.exists(resolved_path):
        return None

    bundle = _load_bundle(resolved_path)
    if bundle is None:
        return None

    model = bundle.get("model")
    vectorizer = bundle.get("vectorizer")
    if model is None or vectorizer is None:
        return None

    text_blob = _compose_text_blob(sample)
    X_text = vectorizer.transform([text_blob])
    X_num = np.array([[_numeric(sample, field) for field in _numeric_fields()]], dtype=float)
    X = hstack([X_text, X_num])

    if hasattr(model, "predict_proba"):
        harmful_prob = float(model.predict_proba(X)[0][1])
    else:
        pred = int(model.predict(X)[0])
        harmful_prob = 1.0 if pred == 1 else 0.0

    return {
        "is_harmful": harmful_prob >= 0.5,
        "harm_score": harmful_prob,
    }


def _compose_text_blob(sample):
    return "\n".join([
        str(sample.get("extracted_text", "") or ""),
        str(sample.get("pro_rationale", "") or ""),
        str(sample.get("con_rationale", "") or ""),
        str(sample.get("judge_reasoning", "") or ""),
    ]).strip()


def _numeric_fields():
    return [
        "judge_harm_score",
        "keyword_score",
        "has_text_region",
        "brightness",
        "contrast",
        "color_variance",
    ]


def _numeric(sample, key):
    value = sample.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_bundle(path):
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    try:
        bundle = joblib.load(path)
    except Exception:
        return None
    _MODEL_CACHE[path] = bundle
    return bundle
