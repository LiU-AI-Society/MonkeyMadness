# MonkeyMadness Leaderboard

A small Flask service participants hit directly: they upload their trained
`.onnx` model with a team name, it gets scored against the held-out test set
in an isolated subprocess, and the result is added to a public leaderboard.
Participants and their models never see the test images or gold labels.

## How a submission is scored

`POST /submit` (team name + `.onnx` file) → `app.py` spawns `worker.py` as a
**separate subprocess** with a hard timeout → `worker.py` calls
`predict.generate_predictions()` (runs the model over the hidden test
folder) then `evaluate_submission.evaluate()` (scores against the private
gold CSV) → prints one JSON line back to `app.py`, which records it in
SQLite. This reuses the same `predict.py` / `evaluate_submission.py` from
the repo root, so there's one scoring implementation, not two.

The subprocess isolation matters: a broken, huge, or malicious uploaded
model can only hang or crash that short-lived worker process, never the
web server itself, and `TIMEOUT_SECONDS` puts a hard ceiling on how long
any one submission can run.

## Setup

```bash
cd leaderboard
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Required data, kept outside the repo

Put these somewhere on the server that is **not** served by any web route
and **not** inside the git working tree (the repo's `.gitignore` already
excludes `gold_labels.csv` / `predictions.csv` as a second line of defense,
but don't rely on that alone for the actual test images):

- A flat folder of unlabeled test images (no class subfolders).
- `gold_labels.csv` with columns `filename,label`, matching those images.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `LEADERBOARD_TEST_DIR` | `../hidden_test` | Hidden test image folder |
| `LEADERBOARD_GOLD_CSV` | `../gold_labels.csv` | Private answer key |
| `LEADERBOARD_LABELS_TXT` | `../Monkey/monkey_labels.txt` | Class index → name mapping |
| `LEADERBOARD_DB` | `leaderboard.db` | SQLite file for submissions |
| `LEADERBOARD_MAX_UPLOAD_MB` | `50` | Max uploaded model size |
| `LEADERBOARD_COOLDOWN_SECONDS` | `60` | Min gap between a team's submissions |
| `LEADERBOARD_TIMEOUT_SECONDS` | `120` | Hard wall-clock limit per scoring run |
| `LEADERBOARD_SECRET_KEY` | `dev-only-change-me` | Flask secret key — **set a real one in production** |
| `PORT` | `5000` | Port for the dev server |

Point the first three at their real, non-repo locations in production, e.g.:

```bash
export LEADERBOARD_TEST_DIR=/srv/monkeymadness/hidden_test
export LEADERBOARD_GOLD_CSV=/srv/monkeymadness/gold_labels.csv
export LEADERBOARD_LABELS_TXT=/srv/monkeymadness/monkey_labels.txt
export LEADERBOARD_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
```

## Running

Dev server (fine for trying it out, not for the real event):

```bash
python app.py
```

Production: run behind a real WSGI server and a reverse proxy with TLS, e.g.:

```bash
pip install gunicorn
gunicorn -w 2 --timeout 150 -b 127.0.0.1:5000 app:app
```

(`--timeout` here is gunicorn's own worker timeout; keep it a bit above
`LEADERBOARD_TIMEOUT_SECONDS` so gunicorn doesn't kill the whole worker
process before the subprocess timeout has a chance to return cleanly.)
Put nginx/Caddy in front for HTTPS and point your existing domain's path
(e.g. `/leaderboard`) at it.

## Notes / things to revisit if this gets heavier use

- Submissions run synchronously in the request thread — fine for a
  hackathon's scale, but if submission volume grows, move scoring to a
  background queue (RQ/Celery) and poll for the result instead.
- Team identification is just a free-text name (no auth), per the
  workshop's trust model. Anyone can submit under any team name.
- `GET /leaderboard.json` is available if the existing site's frontend
  wants to fetch and render the leaderboard itself instead of using this
  service's own page.
