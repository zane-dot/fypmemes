"""Backend JSON API routes for meme analysis."""

from flask import Blueprint, jsonify, request

from backend.services.analysis_service import (
	analyse_uploaded_files,
	get_history_records,
	get_result_record,
)

analysis_api = Blueprint("analysis_api", __name__, url_prefix="/api")


@analysis_api.route("/analyse", methods=["POST"])
def analyse():
	"""Accept one or many uploaded memes and return created record ids."""
	files = request.files.getlist("memes")
	files = [f for f in files if f and f.filename]

	if not files and "meme" in request.files:
		legacy_file = request.files["meme"]
		if legacy_file and legacy_file.filename:
			files = [legacy_file]

	if not files:
		return jsonify({"error": "No file selected."}), 400

	record_ids, invalid_filenames = analyse_uploaded_files(files)
	if not record_ids:
		return jsonify({"error": "No valid image files were uploaded."}), 400

	status = 201 if not invalid_filenames else 207
	payload = {"record_ids": record_ids}
	if len(record_ids) == 1:
		payload["record_id"] = record_ids[0]
	if invalid_filenames:
		payload["invalid_filenames"] = invalid_filenames
	return jsonify(payload), status


@analysis_api.route("/result/<int:record_id>", methods=["GET"])
def result(record_id):
	"""Return one analysis record as JSON."""
	record = get_result_record(record_id)
	if record is None:
		return jsonify({"error": "Analysis not found."}), 404
	return jsonify(record)


@analysis_api.route("/history", methods=["GET"])
def history():
	"""Return recent analysis records as JSON."""
	return jsonify({"records": get_history_records()})
