# Harmful Meme Detection System

An automated system that analyses both the visual and textual content of memes to detect harmful material. Powered by **Large Language Models (LLM)** combined with image processing and text extraction.

![Screenshot](https://github.com/user-attachments/assets/1658aea8-bb4a-401b-a59e-a460eaed731d)

## Features

- **Image Processing** – Extracts visual features (colours, contrast, text-region detection) and performs OCR via a three-stage pipeline: (1) Vision API using a multimodal LLM, (2) EasyOCR with image preprocessing (5 passes), (3) pytesseract fallback. Supports both English and Chinese text.
- **LLM Analysis** – Sends extracted text and image features to an OpenAI-compatible LLM for deep contextual analysis. When `OPENAI_VISION_MODEL` is set (or when OCR fails with a text region detected), the image is sent **directly** to a vision-capable model — the approach used by most harmful-meme detection platforms — which reads embedded text and analyses harm in a single pass, bypassing OCR limitations entirely.
- **Harmfulness Classification** – Produces a harmful / not-harmful verdict with a 0–1 score.
- **Justification** – Generates a detailed, human-readable explanation of why a meme is (or is not) harmful.
- **History** – All analyses are stored in an SQLite database and viewable in the web UI.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Install Tesseract for OCR support
#    Ubuntu/Debian: sudo apt install tesseract-ocr
#    macOS:         brew install tesseract

# 3. Set your DeepSeek API key (required for LLM-powered analysis)
export OPENAI_API_KEY="sk-your-deepseek-key-here"

# (Optional) Use a different provider / model
# export OPENAI_BASE_URL="https://api.openai.com/v1"
# export OPENAI_MODEL="gpt-4o-mini"

# (Optional) Use a vision-capable model for more accurate OCR text extraction
# export OPENAI_VISION_MODEL="gpt-4o"         # OpenAI
# export OPENAI_VISION_MODEL="deepseek-vl2"   # DeepSeek

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
| `OPENAI_VISION_MODEL` | Vision model for direct image analysis (e.g. `gpt-4o`, `deepseek-vl2`). Strongly recommended – see note below. | *(none)* |
| `SECRET_KEY`          | Flask session secret                                          | dev default                      |

> **Tip – memes with hard-to-read text:** Local OCR (EasyOCR / pytesseract) can struggle with low-contrast text on neutral backgrounds (e.g. white, beige, or brown).  Set `OPENAI_VISION_MODEL` to a vision-capable model such as `gpt-4o` or `deepseek-vl2` to send the image *directly* to the LLM, which reads the embedded text and analyses it for harmful content in a single pass — the approach used by most harmful-meme detection platforms.  If `OPENAI_VISION_MODEL` is not set but `OPENAI_MODEL` is vision-capable (e.g. `gpt-4o`), the system will automatically attempt vision analysis as a fallback when OCR fails and a text region is detected.
