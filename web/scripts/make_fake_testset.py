"""Build a FAKE held-out test set from the training images, for rehearsing the
leaderboard before the real hidden test set and gold labels are in place.

Takes the last N images of each class (matching the real validation sizes in
monkey_labels.txt by default), copies them flat into <out>/hidden_test/ and
writes <out>/gold_labels.csv. Scores against this set are meaningless: teams
train on these exact images. Replace it with the real set before the event.

Usage: python web/scripts/make_fake_testset.py [out_dir]   (default: repo root)
"""
import csv
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "Monkey" / "training" / "training"
LABELS = REPO / "Monkey" / "monkey_labels.txt"

out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO
test_dir = out / "hidden_test"
test_dir.mkdir(parents=True, exist_ok=True)

rows = []
for line in LABELS.read_text().splitlines()[1:]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        continue
    code, name, n_val = parts[0], parts[2], int(parts[4])
    images = sorted(p for p in (TRAIN / code).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for img in images[-n_val:]:
        shutil.copy(img, test_dir / img.name)
        rows.append((img.name, name))

with open(out / "gold_labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} fake test images to {test_dir} and {out / 'gold_labels.csv'}")
