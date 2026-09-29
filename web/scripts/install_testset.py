"""Install the leaderboard's hidden test set: the validation split of the
"10 Monkey Species" dataset (the training split is what's in Monkey/training).

Downloads the original dataset zip (or uses one you pass), copies
validation/validation/n*/ flat into ~/.monkeymadness/hidden_test/ and writes
~/.monkeymadness/gold_labels.csv. Validation images that are byte-identical to
an image in Monkey/training are skipped (the dataset has ~33 such leaks), so
nobody gets points for memorising them.

Usage: python web/scripts/install_testset.py [dataset.zip]
"""
import csv
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path.home() / ".monkeymadness"
# Public mirror of the Kaggle dataset used by the Hugging Face "Monkey-Species-Collection" loader.
URL = "https://ibm.ent.box.com/index.php?rm=box_download_shared_file&shared_name=lseef94g6rffpaglu3utaymxz2rxhhaa&file_id=f_964180830136"


def main():
    if len(sys.argv) > 1:
        zip_path = Path(sys.argv[1])
    else:
        zip_path = Path(tempfile.gettempdir()) / "10-monkey-species.zip"
        if not zip_path.exists():
            print(f"Downloading dataset (~570 MB) to {zip_path} ...")
            urllib.request.urlretrieve(URL, zip_path)

    names = {}
    for line in (REPO / "Monkey" / "monkey_labels.txt").read_text().splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) > 2:
            names[parts[0]] = parts[2]

    train = {hashlib.sha1(p.read_bytes()).hexdigest() for p in (REPO / "Monkey" / "training").rglob("*") if p.is_file()}
    test_dir = OUT / "hidden_test"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    rows, leaked = [], 0
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            parts = name.split("/")
            if parts[0] != "validation" or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            data = z.read(name)
            if hashlib.sha1(data).hexdigest() in train:
                leaked += 1
                continue
            (test_dir / parts[-1]).write_bytes(data)
            rows.append((parts[-1], names[parts[2]]))

    with open(OUT / "gold_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)
    print(f"{len(rows)} test images -> {test_dir} ({leaked} skipped: identical to a training image)")


if __name__ == "__main__":
    main()
