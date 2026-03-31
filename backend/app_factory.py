"""Flask application factory for the harmful meme detector."""

import os

from flask import Flask

import config
from backend.routes.analysis import analysis_api
from frontend.routes import frontend
from models.database import init_db


def create_app():
	"""Create and configure Flask app instance."""
	app = Flask(
		__name__,
		template_folder=os.path.join(config.BASE_DIR, "templates"),
		static_folder=os.path.join(config.BASE_DIR, "static"),
		static_url_path="/static",
	)
	app.config["SECRET_KEY"] = config.SECRET_KEY
	app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

	os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
	init_db(config.DATABASE_PATH)

	app.register_blueprint(frontend)
	app.register_blueprint(analysis_api)
	return app
