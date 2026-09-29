# MonkeyMadness web

Live leaderboard for the MonkeyMadness hackathon. Teams upload an `.onnx` model at
`/submit`, it is scored on the hidden test set in an isolated Python subprocess,
and every open page updates in real time. Built with SvelteKit (Svelte 5),
daisyUI and SQLite.

| Page | What |
|---|---|
| `/` | Scoreboard for the projector. ⛶ (bottom right) makes it fullscreen. Click a team for per-species stats. |
| `/submit` | Drag and drop a model, follow it through the queue, see your result. |
| `/admin` | End time (countdown), freeze/reveal, delete submissions, reset. |

## How scoring works

`POST /api/submit` saves the upload to `data/uploads/<id>/` and queues it.
The queue (`src/lib/server/queue.ts`) runs `../leaderboard/worker.py` as a
subprocess, at most `SCORING_CONCURRENCY` at a time, killed after
`LEADERBOARD_TIMEOUT_SECONDS`. The worker is shared with the Flask leaderboard:
`predict.py` runs the model over **every** image in the test folder, and
`evaluate_submission.py` scores against **every** row of the gold CSV (a
missing prediction counts as wrong). The result, including per-class stats, is
stored in `data/monkeymadness.db` and pushed to all pages over Server-Sent
Events (`/api/stream`).

## Setup (once)

```sh
# 1. Python scoring environment (the app finds leaderboard/venv automatically)
cd ..                                   # repo root
python3 -m venv leaderboard/venv
leaderboard/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
leaderboard/venv/bin/pip install onnx onnxruntime pandas scikit-learn matplotlib seaborn

# 2. Test set. Put the real one in place:
#      hidden_test/       flat folder of test images (no class subfolders)
#      gold_labels.csv    filename,label   (label = common name, e.g. mantled_howler)
#    Both paths are gitignored. Until you have them, make a FAKE set from the
#    training images (scores against it are meaningless):
python3 web/scripts/make_fake_testset.py

# 3. Web app
cd web && npm install
```

Sanity check: submit `saved_models/base_line.onnx`. It should score around
40%. If every model scores 0%, the gold labels don't match the names in
`Monkey/monkey_labels.txt`.

## Development

```sh
npm run dev -- --host          # uses .env (admin password "banana", 5 s cooldown)
SCORER=mock npm run dev        # random scores, no Python: for UI rehearsal
node scripts/simulate.mjs http://localhost:5173 4   # fake teams submitting
```

## Production (mm.jakobtor.casa)

```sh
cp .env.example .env.production   # then set ADMIN_PASSWORD (and paths if not the defaults)
npm run build
node --env-file=.env.production build     # listens on 127.0.0.1:3000
```

Keep it running with the systemd user service in `deploy/monkeymadness.service`
(install steps in the file), with your reverse proxy in front for HTTPS. The
proxy must not buffer `/api/stream` (a long-lived Server-Sent Events stream),
or the live updates stall; the app already sends `X-Accel-Buffering: no`.

Things that break if missing from `.env.production`:

- `ORIGIN=https://mm.jakobtor.casa`: without it admin login fails (SvelteKit rejects the form post).
- `BODY_SIZE_LIMIT=60M`: the default 512K rejects most models.
- `ADDRESS_HEADER=X-Forwarded-For`: otherwise every client looks like 127.0.0.1 and shares one per-IP limit.

## Event checklist

- [ ] Real `hidden_test/` + `gold_labels.csv` in place; baseline scores ~40%.
- [ ] `ADMIN_PASSWORD` set; cooldown back to 60 s (only the dev `.env` has 5 s).
- [ ] Reset in `/admin` to clear rehearsal data; set the end time.
- [ ] Projector: open `/`, press ⛶. The QR code points at `/submit` on the address the projector opened, so open it via the public URL.
- [ ] Submit once from a phone on the venue wifi.
- [ ] Afterwards: copy `data/monkeymadness.db` (all results live there).

## Limits and abuse protection

- One submission per team per `LEADERBOARD_COOLDOWN_SECONDS`, and never while its previous one is still being scored.
- `IP_LIMIT_PER_MINUTE` (default 10) per client IP. Kept generous because the venue network may put all teams behind one IP.
- New uploads are refused while `MAX_QUEUE` (default 30) are waiting.
- A model can only crash or hang its own subprocess; participants see a friendly error, the real one is logged by the server.

## Notes for participants

- Export with `train.py` as-is. Keep the validation loader at `batch_size=1`, or the exported model is fixed to that batch size and scoring fails.
- Scoring only resizes images (no `transforms.Normalize`). A model trained with normalization will score badly.
- Use the same team name every time (case doesn't matter).
