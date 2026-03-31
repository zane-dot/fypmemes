"""Core analysis service shared by backend API and frontend routes."""

import json
import os
import uuid

import config
from models.classifier import classify
from models.database import get_all_analyses, get_analysis, get_analysis_dates, save_analysis
from processors.image_processor import extract_image_features, extract_text
from processors.text_processor import analyse_text


def allowed_file(filename):
	"""Return True if file extension is allowed."""
	return (
		"." in filename
		and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
	)


def hydrate_record_explanations(record):
	"""Fill missing debate explanations for legacy records."""
	if not isinstance(record, dict):
		return record

	for text_field in ("justification", "pro_rationale", "con_rationale", "judge_reasoning"):
		record[text_field] = _strip_evidence_block(record.get(text_field) or "")

	justification = (record.get("justification") or "").strip()
	extracted_text = (record.get("extracted_text") or "").strip()
	is_harmful = bool(record.get("is_harmful"))

	snippet = extracted_text
	if len(snippet) > 220:
		snippet = snippet[:220] + "..."
	quoted = f' Observed text: "{snippet}".' if snippet else ""

	if not record.get("pro_rationale"):
		if is_harmful:
			record["pro_rationale"] = (
				"Benign-side interpretation: The content may be interpreted as "
				"satire, personal opinion, or ambiguous humor; this perspective "
				"does not confirm targeted hate by itself."
				+ quoted
			)
		else:
			record["pro_rationale"] = (
				justification
				or "Benign-side interpretation: No strong hateful intent is "
				"supported by the current evidence."
			)

	if not record.get("con_rationale"):
		if is_harmful:
			record["con_rationale"] = (
				justification
				or "Harmful-side interpretation: The meme contains cues that may "
				"promote hate, discrimination, or harmful misinformation."
			)
		else:
			record["con_rationale"] = (
				"Harmful-side interpretation: Some cues could be seen as risky, "
				"but evidence is limited and remains below harmful threshold."
				+ quoted
			)

	if not record.get("judge_reasoning"):
		record["judge_reasoning"] = (
			justification or "Final decision based on combined evidence and risk score."
		)
	if not record.get("judge_side"):
		record["judge_side"] = "harmful" if is_harmful else "benign"

	return record


def _strip_evidence_block(text):
	"""Remove the auto-generated Evidence block from older saved records."""
	marker = "\n\nEvidence used in this decision:\n"
	if not isinstance(text, str):
		return text
	idx = text.find(marker)
	if idx == -1:
		return text
	return text[:idx].rstrip()


def analyse_uploaded_file(file_storage, user_id=None):
	"""Run full analysis for an uploaded image file and return record id."""
	ext = file_storage.filename.rsplit(".", 1)[1].lower()
	safe_name = f"{uuid.uuid4().hex}.{ext}"
	filepath = os.path.join(config.UPLOAD_FOLDER, safe_name)
	file_storage.save(filepath)

	image_features = extract_image_features(filepath)
	extracted_text = extract_text(filepath)
	text_result = analyse_text(extracted_text, config.KEYWORDS_PATH)
	result = classify(
		text_result,
		image_features,
		extracted_text=extracted_text,
		image_path=filepath,
	)

	return save_analysis(
		config.DATABASE_PATH,
		user_id=user_id,
		filename=safe_name,
		extracted_text=extracted_text,
		is_harmful=result["is_harmful"],
		harm_score=result["harm_score"],
		categories=result["categories"],
		justification=result["justification"],
		image_features=result["image_features"],
		analysis_method=result.get("analysis_method"),
		pro_rationale=result.get("pro_rationale"),
		con_rationale=result.get("con_rationale"),
		judge_reasoning=result.get("judge_reasoning"),
		judge_side=result.get("judge_side"),
	)


def analyse_uploaded_files(file_storages, user_id=None):
	"""Analyse many uploaded files and return created record ids + invalid filenames."""
	record_ids = []
	invalid_filenames = []

	for file_storage in file_storages:
		if not file_storage or not file_storage.filename:
			continue
		if not allowed_file(file_storage.filename):
			invalid_filenames.append(file_storage.filename)
			continue
		record_ids.append(analyse_uploaded_file(file_storage, user_id=user_id))

	return record_ids, invalid_filenames


def get_result_record(record_id, user_id=None):
	"""Load one record with parsed helper fields for rendering or API."""
	record = get_analysis(config.DATABASE_PATH, record_id, user_id=user_id)
	if record is None:
		return None

	record = hydrate_record_explanations(record)
	record["categories_list"] = json.loads(record.get("categories") or "[]")
	record["image_features_dict"] = json.loads(record.get("image_features") or "{}")
	return record


def get_history_records(
	limit=50,
	user_id=None,
	created_date=None,
	start_date=None,
	end_date=None,
	is_harmful=None,
):
	"""Load recent records with parsed category lists and optional filters."""
	records = get_all_analyses(
		config.DATABASE_PATH,
		limit=limit,
		user_id=user_id,
		created_date=created_date,
		start_date=start_date,
		end_date=end_date,
		is_harmful=is_harmful,
	)
	for record in records:
		hydrate_record_explanations(record)
		record["categories_list"] = json.loads(record.get("categories") or "[]")

	# Show a user-facing sequence id in history instead of global DB id.
	for idx, record in enumerate(records, start=1):
		record["user_history_id"] = idx
	return records


def get_history_calendar_dates(user_id=None):
	"""Load distinct analysis dates for history calendar UI."""
	return get_analysis_dates(config.DATABASE_PATH, user_id=user_id)
