# Harmful Meme Detection System

An automated system that analyses both the visual and textual content of memes to detect harmful material. Powered by **Large Language Models (LLM)** combined with image processing and text extraction.

![Screenshot](https://github.com/user-attachments/assets/1658aea8-bb4a-401b-a59e-a460eaed731d)

## Features

- **Image Processing** – Extracts visual features (colours, contrast, text-region detection) and performs OCR via a three-stage pipeline: (1) Vision API using a multimodal LLM, (2) EasyOCR with image preprocessing (5 passes), (3) pytesseract fallback. Supports both English and Chinese text.
- **LLM Analysis** – Sends extracted text and image features to an OpenAI-compatible LLM for deep contextual analysis. When `OPENAI_VISION_MODEL` is set (or when OCR fails with a text region detected), the image is sent **directly** to a vision-capable model — the approach used by most harmful-meme detection platforms — which reads embedded text and analyses harm in a single pass, bypassing OCR limitations entirely.
- **Harmfulness Classification** – Produces a harmful / not-harmful verdict with a 0–1 score.
- **Justification** – Generates a detailed, human-readable explanation of why a meme is (or is not) harmful.
- **History** – All analyses are stored in an SQLite database and viewable in the web UI.
- **ExplainHM-style Pipeline** – Uses a three-act workflow: (1) debate generation, (2) LLM judge decision, (3) optional small-model refinement.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Install Tesseract for OCR support
#    Ubuntu/Debian: sudo apt install tesseract-ocr
#    macOS:         brew install tesseract

# 3. Configure debate models
# Positive side (benign argument): DeepSeek
export OPENAI_API_KEY="sk-your-deepseek-key-here"
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-chat"

# Negative side (harmful argument): Aliyun/Qwen
export OPENAI_VISION_API_KEY="sk-your-aliyun-key-here"
export OPENAI_VISION_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_VISION_MODEL="qwen-vl-plus"

# Optional switches
# export EXPLAINHM_ENABLED="1"      # default 1, set 0 to disable debate pipeline
# export SMALL_MODEL_PATH="data/small_decider.joblib"

# 4. Run the application
python app.py
```

Open <http://127.0.0.1:5000> in your browser, upload a meme image, and view the analysis.

> **Note:** Without an `OPENAI_API_KEY` the system falls back to keyword/pattern-based classification, which still works but produces less nuanced results.

## Project Structure

```
fypmemes/
├── app.py                          # Flask web application
├── config.py                       # Configuration (paths, limits, LLM settings)
├── requirements.txt                # Python dependencies
├── models/
│   ├── database.py                 # SQLite database models
│   └── classifier.py               # Harmfulness classifier (LLM + keyword fallback)
├── processors/
│   ├── image_processor.py          # Image feature extraction & OCR
│   ├── text_processor.py           # Keyword/pattern-based text analysis
│   └── llm_processor.py            # LLM-based content analysis
├── data/
│   └── harmful_keywords.json       # Curated harmful keywords & regex patterns
├── templates/                      # Jinja2 HTML templates
├── static/css/                     # Stylesheet
└── tests/                          # pytest test suite
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Configuration

| Environment Variable   | Description                                                   | Default                          |
|-----------------------|---------------------------------------------------------------|----------------------------------|
| `OPENAI_API_KEY`      | DeepSeek (or compatible) API key                              | *(none)*                         |
| `OPENAI_BASE_URL`     | API base URL                                                  | `https://api.deepseek.com`       |
| `OPENAI_MODEL`        | Model for text-based LLM analysis; also used as a vision fallback when `OPENAI_VISION_MODEL` is unset and OCR returns nothing (see note below) | `deepseek-chat` |
| `OPENAI_VISION_MODEL` | Negative debater model (Aliyun/Qwen, e.g. `qwen-vl-plus`) | *(none)* |
| `OPENAI_VISION_API_KEY` | Negative debater API key (Aliyun) | *(none)* |
| `OPENAI_VISION_BASE_URL` | Negative debater API base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EXPLAINHM_ENABLED` | Enable ExplainHM debate pipeline | `1` |
| `SMALL_MODEL_PATH` | Trained stage-3 small-model bundle path | `data/small_decider.joblib` |
| `SECRET_KEY`          | Flask session secret                                          | dev default                      |

## Training a small model (ExplainHM Act-3)

1) Build a training dataset with debate outputs:

```bash
python scripts/build_explainhm_training_data.py \
	--jsonl data/hateful_memes_dataset/train.jsonl \
	--image-root data/hateful_memes_dataset \
	--output data/explainhm_train.jsonl
```

2) Train the lightweight decider:

```bash
python scripts/train_small_decider.py \
	--input data/explainhm_train.jsonl \
	--output data/small_decider.joblib
```

3) Restart app and it will automatically use the trained small model when present.
