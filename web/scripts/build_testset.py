"""Build the leaderboard's hidden test set from reviewed iNaturalist candidates.

Reads <dir>/inat_candidates/<code>/*.jpg (from fetch_inaturalist.py), skips the
ids listed in <dir>/inat_candidates/excluded.txt, picks N per class with a fixed
seed, and writes <dir>/hidden_test/inat_<id>.jpg plus <dir>/gold_labels.csv.

Usage: python web/scripts/build_testset.py [per_class] [dir]   (defaults: 20, ~/.monkeymadness)
"""
import csv
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ROOT = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else Path.home() / ".monkeymadness"
CANDIDATES = ROOT / "inat_candidates"

names = {}
for line in (REPO / "Monkey" / "monkey_labels.txt").read_text().splitlines()[1:]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) > 2:
        names[parts[0]] = parts[2]

excluded = {}
for line in (CANDIDATES / "excluded.txt").read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        code, *ids = line.split()
        excluded[code] = set(ids)

test_dir = ROOT / "hidden_test"
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir(parents=True)

rng = random.Random(2026)
rows = []
for code, label in sorted(names.items()):
    usable = sorted(f for f in (CANDIDATES / code).glob("*.jpg") if f.stem not in excluded.get(code, set()))
    if len(usable) < PER_CLASS:
        sys.exit(f"{code}: only {len(usable)} usable photos, need {PER_CLASS}")
    for f in rng.sample(usable, PER_CLASS):
        name = f"inat_{f.stem}.jpg"
        shutil.copy(f, test_dir / name)
        rows.append((name, label))

with open(ROOT / "gold_labels.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["filename", "label"])
    writer.writerows(sorted(rows))

print(f"{len(rows)} test images ({PER_CLASS} per class) -> {test_dir}")
