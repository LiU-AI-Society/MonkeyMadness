"""Download candidate test photos from iNaturalist for the leaderboard's hidden test set.

For each species in Monkey/monkey_labels.txt it fetches research-grade,
Creative-Commons-licensed observations (most-favourited first) and saves the
first photo of each into <out>/inat_candidates/<code>/<observation id>.jpg,
plus a credits.csv. Review them, delete bad ones, then run build_testset.py.

Usage: python web/scripts/fetch_inaturalist.py [per_species] [out_dir]
       (defaults: 40, ~/.monkeymadness)
"""
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PER_SPECIES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else Path.home() / ".monkeymadness"
# monkey_labels.txt misspells a couple of Latin names; iNaturalist needs the accepted ones.
TAXON_FIX = {"cebuella_pygmea": "Cebuella pygmaea"}
LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa,cc-by-nd,cc-by-nc-nd"
HEADERS = {"User-Agent": "MonkeyMadness-hackathon-testset/1.0"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
        return json.load(r)


def species():
    for line in (REPO / "Monkey" / "monkey_labels.txt").read_text().splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) > 2:
            code, latin, common = parts[0], parts[1], parts[2]
            yield code, TAXON_FIX.get(latin, latin.replace("_", " ").capitalize()), common


def main():
    credits = []
    for code, taxon, common in species():
        dest = OUT / "inat_candidates" / code
        dest.mkdir(parents=True, exist_ok=True)
        params = urllib.parse.urlencode({
            "taxon_name": taxon, "quality_grade": "research", "photos": "true",
            "photo_license": LICENSES, "per_page": min(200, PER_SPECIES * 2),
            "order_by": "votes", "order": "desc",
        })
        results = get_json(f"https://api.inaturalist.org/v1/observations?{params}")["results"]
        saved = 0
        for obs in results:
            if saved >= PER_SPECIES:
                break
            photo = obs["photos"][0]
            url = photo["url"].replace("/square.", "/large.")  # ~1024px on the long side
            path = dest / f"{obs['id']}.jpg"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
                    path.write_bytes(r.read())
            except Exception as exc:  # a missing photo shouldn't stop the run
                print(f"  skip {obs['id']}: {exc}")
                continue
            credits.append((code, common, obs["id"], photo.get("attribution", ""), photo.get("license_code", ""),
                            f"https://www.inaturalist.org/observations/{obs['id']}"))
            saved += 1
            time.sleep(0.2)
        print(f"{code} {common:<26} {saved} photos ({taxon})")
        time.sleep(1)  # be polite to the API

    with open(OUT / "inat_candidates" / "credits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "label", "observation_id", "attribution", "license", "url"])
        w.writerows(credits)


if __name__ == "__main__":
    main()
