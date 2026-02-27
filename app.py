"""Harmful Meme Detection System – Flask application."""

import json
import os
import uuid

from flask import (Flask, flash, redirect, render_template, request,
                   url_for)

import config
from models.database import get_all_analyses, get_analysis, init_db, save_analysis
from models.classifier import classify
from processors.image_processor import extract_image_features, extract_text
from processors.text_processor import analyse_text

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
init_db(config.DATABASE_PATH)


def _allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    """Landing page with meme upload form."""
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    """Accept an uploaded meme image, analyse it, and redirect to results."""
    if "meme" not in request.files:
        flash("No file uploaded.", "error")
        return redirect(url_for("index"))

    file = request.files["meme"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    if not _allowed_file(file.filename):
        flash("File type not allowed. Please upload an image file.", "error")
        return redirect(url_for("index"))

    # Save uploaded file with a unique name
    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(config.UPLOAD_FOLDER, safe_name)
    file.save(filepath)

    # 1. Image processing
    image_features = extract_image_features(filepath)
    extracted_text = extract_text(filepath)

    # 2. Text processing
    text_result = analyse_text(extracted_text, config.KEYWORDS_PATH)

    # 3. Classification + justification (LLM primary, keyword fallback)
    result = classify(text_result, image_features, extracted_text=extracted_text)

    # 4. Persist to database
    record_id = save_analysis(
        config.DATABASE_PATH,
        filename=safe_name,
        extracted_text=extracted_text,
        is_harmful=result["is_harmful"],
        harm_score=result["harm_score"],
        categories=result["categories"],
        justification=result["justification"],
        image_features=result["image_features"],
    )

    return redirect(url_for("result", record_id=record_id))


@app.route("/result/<int:record_id>")
def result(record_id):
    """Display the analysis result for a single meme."""
    record = get_analysis(config.DATABASE_PATH, record_id)
    if record is None:
        flash("Analysis not found.", "error")
        return redirect(url_for("index"))

    record["categories_list"] = json.loads(record["categories"] or "[]")
    record["image_features_dict"] = json.loads(record["image_features"] or "{}")
    return render_template("result.html", record=record)


@app.route("/history")
def history():
    """Show recent meme analyses."""
    records = get_all_analyses(config.DATABASE_PATH)
    for r in records:
        r["categories_list"] = json.loads(r["categories"] or "[]")
    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1", port=5000)