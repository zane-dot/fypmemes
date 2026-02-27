# Harmful Meme Detection System

An automated system that analyses both the visual and textual content of memes to detect harmful material. Powered by **Large Language Models (LLM)** combined with image processing and text extraction.

![Screenshot](https://github.com/user-attachments/assets/1658aea8-bb4a-401b-a59e-a460eaed731d)

## Features

- **Image Processing** – Extracts visual features (colours, contrast, text-region detection) and performs OCR via Tesseract to read embedded text.
- **Text Processing** – Analyses extracted text against a curated harmful-keyword database with regex pattern matching.
- **LLM Analysis** – Sends extracted text and image features to an OpenAI-compatible LLM for deep contextual analysis (supports OpenAI, DeepSeek, and other compatible providers).
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

# 3. Set your LLM API key (required for LLM-powered analysis)
export OPENAI_API_KEY="sk-your-key-here"

# (Optional) Use a different provider / model
# export OPENAI_BASE_URL="https://api.deepseek.com/v1"
# export OPENAI_MODEL="deepseek-chat"

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

| Environment Variable | Description                              | Default        |
|---------------------|------------------------------------------|----------------|
| `OPENAI_API_KEY`    | API key for LLM provider                 | *(none)*       |
| `OPENAI_BASE_URL`   | Custom API base URL                      | *(OpenAI default)* |
| `OPENAI_MODEL`      | Model name to use                        | `gpt-4o-mini`  |
| `SECRET_KEY`        | Flask session secret                     | dev default    |
