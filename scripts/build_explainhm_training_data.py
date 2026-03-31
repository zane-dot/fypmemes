"""Build ExplainHM-style supervised training data from meme dataset JSONL.

Each output row includes:
- label (ground truth)
- extracted_text
- pro_rationale / con_rationale
- judge_reasoning / judge_harm_score
- keyword_score
- image numeric features
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from processors.image_processor import extract_image_features, extract_text
from processors.llm_processor import is_explainhm_available, run_explainhm_pipeline
from processors.text_processor import analyse_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to source JSONL with label/img")
    parser.add_argument("--image-root", required=True, help="Dataset image root directory")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    return parser.parse_args()


def main():
    args = parse_args()

    if not is_explainhm_available():
        raise RuntimeError(
            "ExplainHM debate backends unavailable. Set OPENAI_* for DeepSeek and OPENAI_VISION_* for Aliyun."
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total = 0
    written = 0
    with open(args.jsonl, "r", encoding="utf-8") as src, open(args.output, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if args.max_samples and total > args.max_samples:
                break

            img_rel = row.get("img")
            label = int(row.get("label", 0))
            if not img_rel:
                continue
            image_path = os.path.join(args.image_root, img_rel)
            if not os.path.exists(image_path):
                continue

            image_features = extract_image_features(image_path)
            extracted_text = extract_text(image_path)
            text_result = analyse_text(extracted_text, config.KEYWORDS_PATH)

            debate = run_explainhm_pipeline(
                extracted_text,
                image_features,
                image_path=image_path,
            )
            if debate is None:
                debate = run_explainhm_pipeline(
                    extracted_text,
                    image_features,
                    image_path=None,
                )
            if debate is None:
                continue

            out = {
                "id": row.get("id"),
                "img": img_rel,
                "label": label,
                "extracted_text": extracted_text,
                "pro_rationale": debate.get("pro_rationale", ""),
                "con_rationale": debate.get("con_rationale", ""),
                "judge_reasoning": debate.get("judge_reasoning", ""),
                "judge_harm_score": float(debate.get("harm_score", 0.0) or 0.0),
                "keyword_score": float(text_result.get("overall_score", 0.0) or 0.0),
                "has_text_region": 1.0 if image_features.get("has_text_region") else 0.0,
                "brightness": float(image_features.get("brightness", 0.0) or 0.0),
                "contrast": float(image_features.get("contrast", 0.0) or 0.0),
                "color_variance": float(image_features.get("color_variance", 0.0) or 0.0),
            }
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
            if written % 20 == 0:
                print(f"processed={total}, written={written}")

    print(f"done. source={total}, written={written}, output={args.output}")


if __name__ == "__main__":
    main()
