"""Evaluate the meme classifier against the HuggingFace hateful-memes dataset.

The `neuralcatcher/hateful_memes` dataset contains meme images labelled as
hateful (1) or not-hateful (0).  This script downloads (or reads from a local
copy) each image, runs the full classification pipeline, and reports accuracy,
precision, recall, and F1 score.

Usage – automatic download (requires ``datasets`` and internet access)::

    python scripts/evaluate_hateful_memes.py

Usage – local dataset directory::

    python scripts/evaluate_hateful_memes.py --dataset-dir /path/to/hateful_memes

    The directory must contain a ``train.jsonl`` (or ``dev.jsonl`` /
    ``test.jsonl``) file and an ``img/`` subdirectory with the PNG images.

    Each JSONL line has the format::

        {"id": 12345, "img": "img/12345.png", "label": 0, "text": "..."}

Optional arguments::

    --split         Dataset split to evaluate (train / dev / test).
                    Default: dev (smallest split, ~500 images).
    --limit         Maximum number of samples to process.  Useful for a quick
                    smoke-test without running the full dataset.
    --output        Path to save the JSON report.
                    Default: evaluation_report.json in the current directory.
    --keywords-path Path to the harmful_keywords.json file.
                    Default: data/harmful_keywords.json relative to repo root.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure the repository root is on sys.path so that the project modules can
# be imported regardless of where this script is invoked from.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: E402 – must come after sys.path modification
from models.classifier import classify  # noqa: E402
from processors.image_processor import extract_image_features, extract_text  # noqa: E402
from processors.text_processor import analyse_text  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def _load_from_huggingface(split: str) -> list[dict[str, Any]]:
    """Download the dataset using the ``datasets`` library.

    Returns a list of dicts with keys: ``id``, ``label``, ``text``,
    ``image_path`` (a temporary PNG file that persists for the lifetime of
    this process).
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' library is required for automatic download.\n"
            "Install it with:  pip install datasets\n"
            "Or provide a local dataset directory with --dataset-dir."
        ) from exc

    print(f"Downloading neuralcatcher/hateful_memes ({split} split) …")
    ds = load_dataset("neuralcatcher/hateful_memes", split=split, trust_remote_code=True)

    samples: list[dict[str, Any]] = []
    tmp_dir = tempfile.mkdtemp(prefix="hateful_memes_eval_")

    for i, row in enumerate(ds):
        img_path = os.path.join(tmp_dir, f"{i:05d}.png")
        # The HuggingFace dataset stores images as PIL Image objects.
        # Older versions of the dataset use the field name "img"; newer
        # parquet-based versions may use "image".  We try both.
        pil_img = row.get("img") or row.get("image")
        if pil_img is None:
            logger.warning("Row %d has no image – skipping", i)
            continue
        pil_img.save(img_path)

        samples.append({
            "id": row.get("id", i),
            "label": int(row["label"]),
            "text": row.get("text", ""),
            "image_path": img_path,
        })

    return samples


def _load_from_directory(dataset_dir: str, split: str) -> list[dict[str, Any]]:
    """Load samples from a local dataset directory.

    Looks for ``{split}.jsonl`` then falls back to ``train.jsonl``.
    """
    base = Path(dataset_dir)
    for candidate in [f"{split}.jsonl", "train.jsonl", "dev.jsonl", "test.jsonl"]:
        jsonl_path = base / candidate
        if jsonl_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"No JSONL file found in {dataset_dir}. "
            "Expected train.jsonl, dev.jsonl, or test.jsonl."
        )

    samples: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img_rel = row.get("img", "")
            img_path = str(base / img_rel)
            if not os.path.exists(img_path):
                logger.warning("Image not found: %s – skipping", img_path)
                continue
            samples.append({
                "id": row.get("id", len(samples)),
                "label": int(row["label"]),
                "text": row.get("text", ""),
                "image_path": img_path,
            })

    return samples


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    labels: list[int],
    predictions: list[int],
) -> dict[str, float]:
    """Return accuracy, precision, recall, and F1 for binary classification."""
    tp = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(predictions, labels))

    accuracy = (tp + tn) / len(labels) if labels else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": len(labels),
    }


# ---------------------------------------------------------------------------
# Per-sample classification
# ---------------------------------------------------------------------------

def _classify_sample(
    sample: dict[str, Any],
    keywords_path: str,
) -> dict[str, Any]:
    """Run the full pipeline on one sample and return a result dict."""
    image_path = sample["image_path"]
    extracted_text = ""

    try:
        image_features = extract_image_features(image_path)
        extracted_text = extract_text(image_path)
        text_result = analyse_text(extracted_text, keywords_path)
        result = classify(
            text_result,
            image_features,
            extracted_text=extracted_text,
            image_path=image_path,
        )
    except Exception as exc:
        logger.warning("Pipeline error for sample %s: %s", sample["id"], exc)
        result = {
            "is_harmful": False,
            "harm_score": 0.0,
            "categories": "[]",
            "justification": f"Pipeline error: {exc}",
            "image_features": "{}",
            "analysis_method": "error",
        }

    return {
        "id": sample["id"],
        "true_label": sample["label"],
        "predicted_label": 1 if result["is_harmful"] else 0,
        "harm_score": result["harm_score"],
        "analysis_method": result["analysis_method"],
        "dataset_text": sample.get("text", ""),
        "extracted_text": extracted_text[:200] if extracted_text else "",
        "justification_preview": result.get("justification", "")[:200],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the meme classifier on the hateful memes dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        help=(
            "Path to a local copy of the hateful memes dataset.  "
            "When omitted the dataset is downloaded from HuggingFace."
        ),
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev", "test"],
        help="Dataset split to evaluate (default: dev).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of samples to process (0 = all).",
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="Path to write the JSON report (default: evaluation_report.json).",
    )
    parser.add_argument(
        "--keywords-path",
        default=str(_REPO_ROOT / "data" / "harmful_keywords.json"),
        help="Path to harmful_keywords.json.",
    )
    args = parser.parse_args()

    # Load dataset ---------------------------------------------------------
    if args.dataset_dir:
        samples = _load_from_directory(args.dataset_dir, args.split)
    else:
        samples = _load_from_huggingface(args.split)

    if args.limit > 0:
        samples = samples[: args.limit]

    if not samples:
        print("No samples found – aborting.")
        sys.exit(1)

    print(f"Loaded {len(samples)} samples.")

    # Run classification ---------------------------------------------------
    results: list[dict[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []

    start = time.monotonic()
    for i, sample in enumerate(samples, 1):
        if i % 50 == 0 or i == 1:
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(samples) - i) / rate if rate > 0 else 0
            print(
                f"  [{i:>4}/{len(samples)}]  "
                f"elapsed {elapsed:.0f}s  "
                f"~{remaining:.0f}s remaining"
            )

        res = _classify_sample(sample, args.keywords_path)
        results.append(res)
        labels.append(res["true_label"])
        predictions.append(res["predicted_label"])

    elapsed_total = time.monotonic() - start

    # Compute metrics ------------------------------------------------------
    metrics = _compute_metrics(labels, predictions)

    # False negatives and false positives breakdown by analysis method -----
    method_counts: dict[str, dict[str, int]] = {}
    for res in results:
        m = res["analysis_method"]
        method_counts.setdefault(m, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        tl, pl = res["true_label"], res["predicted_label"]
        if tl == 1 and pl == 1:
            method_counts[m]["tp"] += 1
        elif tl == 0 and pl == 1:
            method_counts[m]["fp"] += 1
        elif tl == 1 and pl == 0:
            method_counts[m]["fn"] += 1
        else:
            method_counts[m]["tn"] += 1

    # Produce false-negative examples for qualitative analysis ------------
    false_negatives = [
        r for r in results
        if r["true_label"] == 1 and r["predicted_label"] == 0
    ][:20]

    false_positives = [
        r for r in results
        if r["true_label"] == 0 and r["predicted_label"] == 1
    ][:20]

    report = {
        "dataset": "neuralcatcher/hateful_memes",
        "split": args.split,
        "total_samples": len(samples),
        "elapsed_seconds": round(elapsed_total, 1),
        "metrics": metrics,
        "analysis_method_breakdown": method_counts,
        "false_negative_examples": false_negatives,
        "false_positive_examples": false_positives,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Print summary --------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Dataset split : {args.split}")
    print(f"  Samples       : {len(samples)}")
    print(f"  Elapsed time  : {elapsed_total:.1f}s")
    print()
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1 score      : {metrics['f1']:.4f}")
    print()
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
          f"FN={metrics['fn']}  TN={metrics['tn']}")
    print()
    print(f"  Full report   : {output_path.resolve()}")
    print("=" * 60)

    if false_negatives:
        print(f"\nTop false negatives (missed hateful memes) – "
              f"{len(false_negatives)} shown:")
        for fn_ex in false_negatives[:5]:
            print(f"  id={fn_ex['id']}  method={fn_ex['analysis_method']}  "
                  f"score={fn_ex['harm_score']:.2f}  "
                  f"text={fn_ex['dataset_text'][:80]!r}")


if __name__ == "__main__":
    main()
