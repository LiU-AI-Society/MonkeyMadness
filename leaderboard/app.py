import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

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
# The "model in action" image: the same public training image for every team,
# never a hidden test image (it is shown to participants).
SHOWCASE_IMAGE = os.environ.get(
    "LEADERBOARD_SHOWCASE_IMAGE", str(REPO_ROOT / "Monkey" / "training" / "training" / "n7" / "n7027.jpg")
)

MAX_UPLOAD_MB = float(os.environ.get("LEADERBOARD_MAX_UPLOAD_MB", "50"))
COOLDOWN_SECONDS = float(os.environ.get("LEADERBOARD_COOLDOWN_SECONDS", "60"))
TIMEOUT_SECONDS = float(os.environ.get("LEADERBOARD_TIMEOUT_SECONDS", "120"))
MAX_TEAM_NAME_LEN = 64

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)
app.secret_key = os.environ.get("LEADERBOARD_SECRET_KEY", "dev-only-change-me")

db.init_db(DB_PATH)


def render_index(error=None):
    return render_template(
        "index.html",
        leaderboard=db.leaderboard(DB_PATH),
        recent=db.recent_submissions(DB_PATH),
        error=error,
    )


@app.route("/")
def index():
    return render_index()


@app.route("/leaderboard.json")
def leaderboard_json():
    return jsonify(db.leaderboard(DB_PATH))


@app.route("/submission/<int:submission_id>")
def submission(submission_id):
    row = db.get_submission(DB_PATH, submission_id)
    if row is None:
        abort(404)
    board = db.leaderboard(DB_PATH)
    rank = next((i + 1 for i, r in enumerate(board) if r["team_name"] == row["team_name"]), None)
    return render_template(
        "submission.html",
        submission=row,
        rank=rank,
        showcase=db.get_showcase(DB_PATH, submission_id),
    )


@app.route("/submit", methods=["POST"])
def submit():
    def reject(message, status):
        return render_index(error=message), status

    team_name = request.form.get("team_name", "").strip()
    # Accept one or more files under the same field: an ONNX export that uses
    # "external data" (weights stored outside the .onnx protobuf) needs its
    # companion file uploaded alongside the model, not just the model itself.
    uploaded_files = [f for f in request.files.getlist("model_file") if f and f.filename]

    if not team_name:
        return reject("Team name is required.", 400)
    if len(team_name) > MAX_TEAM_NAME_LEN:
        return reject(f"Team name is too long (max {MAX_TEAM_NAME_LEN} characters).", 400)

    onnx_files = [f for f in uploaded_files if f.filename.lower().endswith(".onnx")]
    if len(onnx_files) != 1:
        return reject(
            "Please upload exactly one .onnx model file. If your export also "
            "produced a companion file (e.g. ending in .onnx.data), select "
            "both files together.",
            400,
        )

    wait = db.seconds_since_last_submission(DB_PATH, team_name)
    if wait is not None and wait < COOLDOWN_SECONDS:
        remaining = int(COOLDOWN_SECONDS - wait)
        return reject(f"Please wait {remaining}s before submitting again.", 429)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Keep each file's original (sanitized) name rather than renaming it.
        # A model exported with external data has a literal companion
        # filename (e.g. "model.onnx.data") baked into the graph -- onnx
        # resolves that name relative to the model's directory, so both
        # files must land in the same temp dir under their original names.
        model_path = None
        for f in uploaded_files:
            safe_name = secure_filename(f.filename)
            dest = Path(tmp_dir) / safe_name
            f.save(dest)
            if safe_name.lower().endswith(".onnx"):
                model_path = dest

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
                    "--showcase_image", SHOWCASE_IMAGE,
                ],
                capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return reject("Your model took too long to run and was stopped.", 408)

    scores = _parse_worker_output(result.stdout)
    if scores is None or not scores.get("ok"):
        # The participant only ever sees the generic message below -- log the
        # real detail server-side so an organizer can tell "bad model" apart
        # from "worker crashed" (e.g. missing dependency, bad env var path).
        app.logger.warning(
            "Scoring failed for team %r (exit code %s).\nstdout:\n%s\nstderr:\n%s",
            team_name, result.returncode, result.stdout, result.stderr,
        )
        error_detail = (scores or {}).get("error", "")
        if "should be stored in" in error_detail:
            return reject(
                "Your ONNX file references external weight data that wasn't "
                "uploaded (a file usually named <yourmodel>.onnx.data). Please "
                "select both that file and the .onnx file together, or "
                "re-export your model as a single self-contained .onnx file.",
                400,
            )
        return reject(
            "Could not score your model. Make sure it's a valid ONNX classifier "
            "with the expected input shape.",
            400,
        )

    showcase = scores.pop("showcase", None)
    submission_id = db.record_submission(DB_PATH, team_name, scores)
    if showcase:
        db.record_showcase(DB_PATH, submission_id, showcase)
    else:
        # The score counts either way; the organizer log says why the
        # "model in action" part is missing.
        app.logger.warning("No showcase for team %r.\nstderr:\n%s", team_name, result.stderr)
    return redirect(url_for("submission", submission_id=submission_id))


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
