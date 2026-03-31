"""Application entrypoint."""

import os

from backend.app_factory import create_app

app = create_app()


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1", port=5000)