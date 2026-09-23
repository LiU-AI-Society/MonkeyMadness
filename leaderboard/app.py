import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

import db

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
WORKER = BASE_DIR / "worker.py"

# All of these should point outside the git repo in a real deployment --
# gold_labels.csv and the hidden test images must never be reachable
# through the web server's static/template roots.
TEST_IMAGE_DIR = os.environ.get("LEADERBOARD_TEST_DIR", str(REPO_ROOT / "hidden_test"))
GOLD_LABELS_CSV = os.environ.get("LEADERBOARD_GOLD_CSV", str(REPO_ROOT / "gold_labels.csv"))
LABELS_TXT = os.environ.get("LEADERBOARD_LABELS_TXT", str(REPO_ROOT / "Monkey" / "monkey_labels.txt"))
DB_PATH = os.environ.get("LEADERBOARD_DB", str(BASE_DIR / "leaderboard.db"))

MAX_UPLOAD_MB = float(os.environ.get("LEADERBOARD_MAX_UPLOAD_MB", "50"))
COOLDOWN_SECONDS = float(os.environ.get("LEADERBOARD_COOLDOWN_SECONDS", "60"))
TIMEOUT_SECONDS = float(os.environ.get("LEADERBOARD_TIMEOUT_SECONDS", "120"))
MAX_TEAM_NAME_LEN = 64

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)
app.secret_key = os.environ.get("LEADERBOARD_SECRET_KEY", "dev-only-change-me")

db.init_db(DB_PATH)


@app.route("/")
def index():
    return render_template("index.html", leaderboard=db.leaderboard(DB_PATH))


@app.route("/leaderboard.json")
def leaderboard_json():
    return jsonify(db.leaderboard(DB_PATH))


@app.route("/submit", methods=["POST"])
def submit():
    def reject(message, status):
        return render_template("index.html", leaderboard=db.leaderboard(DB_PATH), error=message), status

    team_name = request.form.get("team_name", "").strip()
    model_file = request.files.get("model_file")

    if not team_name:
        return reject("Team name is required.", 400)
    if len(team_name) > MAX_TEAM_NAME_LEN:
        return reject(f"Team name is too long (max {MAX_TEAM_NAME_LEN} characters).", 400)
    if model_file is None or not model_file.filename.lower().endswith(".onnx"):
        return reject("Please upload a .onnx model file.", 400)

    wait = db.seconds_since_last_submission(DB_PATH, team_name)
    if wait is not None and wait < COOLDOWN_SECONDS:
        remaining = int(COOLDOWN_SECONDS - wait)
        return reject(f"Please wait {remaining}s before submitting again.", 429)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save under a fixed name, not the uploaded filename -- avoids any
        # path-traversal concern from a crafted filename.
        model_path = Path(tmp_dir) / "model.onnx"
        model_file.save(model_path)
        predictions_path = Path(tmp_dir) / "predictions.csv"

        try:
            result = subprocess.run(
                [
                    sys.executable, str(WORKER),
                    "--model_path", str(model_path),
                    "--image_dir", TEST_IMAGE_DIR,
                    "--gold", GOLD_LABELS_CSV,
                    "--labels_path", LABELS_TXT,
                    "--predictions_out", str(predictions_path),
                ],
                capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return reject("Your model took too long to run and was stopped.", 408)

    scores = _parse_worker_output(result.stdout)
    if scores is None or not scores.get("ok"):
        return reject(
            "Could not score your model. Make sure it's a valid ONNX classifier "
            "with the expected input shape.",
            400,
        )

    db.record_submission(DB_PATH, team_name, scores)
    return redirect(url_for("index"))


def _parse_worker_output(stdout):
    """worker.py prints exactly one JSON line; scan from the end in case
    anything else leaked onto stdout."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
