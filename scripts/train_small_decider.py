"""Train a lightweight ExplainHM stage-3 decision model."""

import argparse
import json
import os

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


NUMERIC_FIELDS = [
    "judge_harm_score",
    "keyword_score",
    "has_text_region",
    "brightness",
    "contrast",
    "color_variance",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL from build_explainhm_training_data.py")
    parser.add_argument("--output", default="data/small_decider.joblib", help="Output model bundle")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _text_blob(row):
    return "\n".join([
        str(row.get("extracted_text", "") or ""),
        str(row.get("pro_rationale", "") or ""),
        str(row.get("con_rationale", "") or ""),
        str(row.get("judge_reasoning", "") or ""),
    ]).strip()


def _num_vec(row):
    vec = []
    for field in NUMERIC_FIELDS:
        value = row.get(field, 0.0)
        try:
            vec.append(float(value))
        except (TypeError, ValueError):
            vec.append(0.0)
    return vec


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise RuntimeError("No rows found in input dataset")

    texts = [_text_blob(r) for r in rows]
    nums = np.array([_num_vec(r) for r in rows], dtype=float)
    labels = np.array([int(r.get("label", 0)) for r in rows], dtype=int)

    unique_labels, counts = np.unique(labels, return_counts=True)
    min_class_count = int(counts.min()) if len(counts) > 0 else 0
    stratify_labels = labels if (len(unique_labels) > 1 and min_class_count >= 2) else None

    X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
        texts,
        nums,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify_labels,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=12000,
        strip_accents="unicode",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    X_train = hstack([X_train_tfidf, X_train_num])
    X_test = hstack([X_test_tfidf, X_test_num])

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"accuracy={acc:.4f}")
    print(classification_report(y_test, pred, digits=4))

    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "numeric_fields": NUMERIC_FIELDS,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(bundle, args.output)
    print(f"saved model bundle: {args.output}")


if __name__ == "__main__":
    main()
