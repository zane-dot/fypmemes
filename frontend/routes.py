"""Frontend page routes for the web interface."""

from functools import wraps

from flask import (
	Blueprint,
	flash,
	g,
	redirect,
	render_template,
	request,
	session,
	url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from sqlite3 import IntegrityError

import config
from models.database import create_user, get_user_by_id, get_user_by_username

from backend.services.analysis_service import (
	analyse_uploaded_files,
	get_history_calendar_dates,
	get_history_records,
	get_result_record,
)

frontend = Blueprint("frontend", __name__)


def login_required(view_func):
	"""Require authenticated user for a route."""

	@wraps(view_func)
	def wrapped(*args, **kwargs):
		if g.user is None:
			flash("Please login first.", "error")
			return redirect(url_for("frontend.login"))
		return view_func(*args, **kwargs)

	return wrapped


@frontend.before_app_request
def load_logged_in_user():
	"""Load current user from session into flask.g."""
	user_id = session.get("user_id")
	g.user = get_user_by_id(config.DATABASE_PATH, user_id) if user_id else None


@frontend.route("/register", methods=["GET", "POST"])
def register():
	"""Create user account and auto-login."""
	if g.user:
		return redirect(url_for("frontend.index"))

	if request.method == "POST":
		username = (request.form.get("username") or "").strip()
		password = request.form.get("password") or ""

		if len(username) < 3:
			flash("Username must be at least 3 characters.", "error")
			return render_template("register.html")
		if len(password) < 6:
			flash("Password must be at least 6 characters.", "error")
			return render_template("register.html")

		try:
			user_id = create_user(
				config.DATABASE_PATH,
				username=username,
				password_hash=generate_password_hash(password),
			)
		except IntegrityError:
			flash("Username already exists. Please choose another one.", "error")
			return render_template("register.html")

		session.clear()
		session["user_id"] = user_id
		flash("Registration successful. You are now logged in.", "success")
		return redirect(url_for("frontend.index"))

	return render_template("register.html")


@frontend.route("/login", methods=["GET", "POST"])
def login():
	"""Authenticate user with username/password."""
	if g.user:
		return redirect(url_for("frontend.index"))

	if request.method == "POST":
		username = (request.form.get("username") or "").strip()
		password = request.form.get("password") or ""
		user = get_user_by_username(config.DATABASE_PATH, username)

		if user is None or not check_password_hash(user["password_hash"], password):
			flash("Invalid username or password.", "error")
			return render_template("login.html")

		session.clear()
		session["user_id"] = user["id"]
		flash("Login successful.", "success")
		return redirect(url_for("frontend.index"))

	return render_template("login.html")


@frontend.route("/logout")
def logout():
	"""Logout the current user."""
	session.clear()
	flash("You have logged out.", "success")
	return redirect(url_for("frontend.login"))


@frontend.route("/")
@login_required
def index():
	"""Landing page with meme upload form."""
	return render_template("index.html")


@frontend.route("/analyse", methods=["POST"])
@login_required
def analyse():
	"""Handle single or batch uploads and redirect accordingly."""
	files = request.files.getlist("memes")
	files = [f for f in files if f and f.filename]

	if not files and "meme" in request.files:
		legacy_file = request.files["meme"]
		if legacy_file and legacy_file.filename:
			files = [legacy_file]

	if not files:
		flash("No file selected.", "error")
		return redirect(url_for("frontend.index"))

	record_ids, invalid_filenames = analyse_uploaded_files(files, user_id=g.user["id"])

	if invalid_filenames:
		flash(
			f"Skipped {len(invalid_filenames)} unsupported file(s): "
			+ ", ".join(invalid_filenames[:5]),
			"error",
		)

	if not record_ids:
		flash("No valid image files were uploaded.", "error")
		return redirect(url_for("frontend.index"))

	if len(record_ids) == 1:
		return redirect(url_for("frontend.result", record_id=record_ids[0]))

	flash(f"Batch analysis completed: {len(record_ids)} memes analysed.", "success")
	return redirect(url_for("frontend.history"))


@frontend.route("/result/<int:record_id>")
@login_required
def result(record_id):
	"""Display one analysis record."""
	record = get_result_record(record_id, user_id=g.user["id"])
	if record is None:
		flash("Analysis not found.", "error")
		return redirect(url_for("frontend.index"))
	return render_template("result.html", record=record)


@frontend.route("/history")
@login_required
def history():
	"""Show recent meme analyses with optional filters."""
	selected_date = (request.args.get("date") or "").strip()
	start_date = (request.args.get("start_date") or "").strip()
	end_date = (request.args.get("end_date") or "").strip()
	harmful_filter = (request.args.get("harmful") or "all").strip().lower()

	is_harmful = None
	if harmful_filter == "harmful":
		is_harmful = True
	elif harmful_filter == "safe":
		is_harmful = False
	else:
		harmful_filter = "all"

	records = get_history_records(
		user_id=g.user["id"],
		created_date=selected_date or None,
		start_date=None if selected_date else (start_date or None),
		end_date=None if selected_date else (end_date or None),
		is_harmful=is_harmful,
	)
	return render_template(
		"history.html",
		records=records,
		selected_date=selected_date,
		start_date=start_date,
		end_date=end_date,
		calendar_dates=get_history_calendar_dates(user_id=g.user["id"]),
		harmful_filter=harmful_filter,
	)
