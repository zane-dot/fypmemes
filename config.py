import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DATABASE_PATH = os.path.join(BASE_DIR, "data", "memes.db")
KEYWORDS_PATH = os.path.join(BASE_DIR, "data", "harmful_keywords.json")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
