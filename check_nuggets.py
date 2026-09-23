"""Estimate how many Neural Nuggets (NN) have been spent on the README shop, by
scanning MonkeyMadness.ipynb for signs that each upgrade has been implemented.

Usage:
    python check_nuggets.py [path/to/notebook.ipynb]

This is a heuristic, not a strict audit: it looks for the code patterns shown in the
README's "Instructions on how to implement things from the shop" section. Upgrades
implemented in an unrecognized way may be missed.
"""

import json
import math
import re
import sys
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "MonkeyMadness.ipynb"
BUDGET = 100  # Billion NN, per README "BUDGET: 100 Billion Nerual nuggets"

# Shipped-template defaults that cost nothing. Sourced from the notebook itself:
# DATASET block sets DATA_PERCENTAGE = 0.5 and marks IMAGE_SIZE "DO NOT ALTER THIS
# PARAMETER"; the starter MonkeyNET in the MODEL block ships with 2 conv layers.
BASELINE_DATA_PERCENTAGE = 0.5
BASELINE_IMAGE_SIZE = (64, 64)
BASELINE_CONV_LAYERS = 2


def load_active_source(path):
    """Return (per-cell active source list, joined active source) with full-line and
    trailing comments stripped, so commented-out code isn't mistaken for a purchase."""
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        raw = "".join(cell.get("source", []))
        active_lines = []
        for line in raw.splitlines():
            code_part = line.split("#", 1)[0]
            if code_part.strip():
                active_lines.append(code_part)
        cells.append("\n".join(active_lines))
    return cells, "\n".join(cells)


def class_body(source, class_name):
    """Return the indented body of `class <class_name>(...):` in source, stopping at
    the next top-level (column 0) line."""
    lines = source.splitlines()
    body = []
    in_class = False
    for line in lines:
        if not in_class:
            if re.match(rf"class\s+{re.escape(class_name)}\b.*:\s*$", line.strip()):
                in_class = True
            continue
        if line.strip() == "" or line.startswith((" ", "\t")):
            body.append(line)
        else:
            break
    return "\n".join(body)


def find_floats(pattern, text):
    return [float(v) for v in re.findall(pattern, text)]


def check_info_infusion(active_all):
    values = find_floats(r"DATA_PERCENTAGE\s*=\s*([0-9]*\.?[0-9]+)", active_all)
    if not values:
        return False, 0, "DATA_PERCENTAGE not found"
    data_percentage = values[-1]
    added_points = max(0.0, (data_percentage - BASELINE_DATA_PERCENTAGE) * 100)
    cost = added_points * 1
    detail = f"DATA_PERCENTAGE={data_percentage:g} ({added_points:g}% over the {BASELINE_DATA_PERCENTAGE * 100:g}% baseline)"
    return cost > 0, cost, detail


def check_resize_rocket(active_all):
    m = re.search(r"IMAGE_SIZE\s*=\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", active_all)
    if not m:
        return False, 0, "IMAGE_SIZE not found"
    w, h = int(m.group(1)), int(m.group(2))
    base_w, base_h = BASELINE_IMAGE_SIZE
    scale = max(w / base_w, h / base_h)
    if scale <= 1:
        return False, 0, f"IMAGE_SIZE=({w}, {h}) (baseline)"
    doublings = math.log2(scale)
    cost = doublings * 10
    note = "" if abs(doublings - round(doublings)) < 1e-6 else " (non power-of-2 scale, cost is approximate)"
    detail = f"IMAGE_SIZE=({w}, {h}), {scale:g}x baseline{note}"
    return True, cost, detail


def check_neural_boost(active_all):
    model_body = class_body(active_all, "MonkeyNET")
    layers = len(re.findall(r"nn\.Conv2d\(", model_body))
    extra = max(0, layers - BASELINE_CONV_LAYERS)
    if extra == 0:
        return False, 0, f"{layers} conv layer(s) (baseline)"
    groups_of_three = extra // 3
    remainder = extra % 3
    paid_layers = groups_of_three * 2 + remainder
    cost = paid_layers * 20
    detail = f"{layers} conv layers ({extra} extra, 3-for-2 deal applied)"
    return True, cost, detail


def check_dropout(active_all):
    model_body = class_body(active_all, "MonkeyNET")
    defined = re.search(r"nn\.Dropout2?d?\(", model_body)
    applied = re.search(r"=\s*self\.dropout\w*\(", model_body)
    if defined and applied:
        return True, 10, "dropout layer defined and applied in forward"
    if defined:
        return False, 0, "dropout layer defined but not applied in forward"
    return False, 0, "no dropout layer found"


def check_attention(active_all):
    defined = re.search(r"class\s+SpatialAttention\b", active_all)
    model_body = class_body(active_all, "MonkeyNET")
    applied = re.search(r"=\s*self\.attention\(", model_body)
    if defined and applied:
        return True, 15, "SpatialAttention defined and wired into forward"
    if defined:
        return False, 0, "SpatialAttention defined but not wired into forward"
    return False, 0, "no SpatialAttention found"


def check_skip_connection(active_all):
    defined = re.search(r"class\s+ResidualBlock\b", active_all)
    model_body = class_body(active_all, "MonkeyNET")
    applied = re.search(r"=\s*self\.res_block\w*\(", model_body)
    if defined and applied:
        return True, 15, "ResidualBlock defined and wired into forward"
    if defined:
        return False, 0, "ResidualBlock defined but not wired into forward"
    return False, 0, "no ResidualBlock found"


def check_ensemble(active_all):
    defined = re.search(r"class\s+Ensemble\b", active_all)
    instantiated = re.search(r"=\s*Ensemble\(", active_all)
    if defined and instantiated:
        return True, 60, "Ensemble class defined and instantiated"
    return False, 0, "no Ensemble usage found"


def check_weight_decay(active_all):
    values = find_floats(r"weight_decay\s*=\s*([0-9eE.+-]+)", active_all)
    value = max(values) if values else 0
    if value > 0:
        return True, 10, f"weight_decay={value:g}"
    return False, 0, "weight_decay not set (or 0)"


def check_momentum(active_all):
    values = find_floats(r"momentum\s*=\s*([0-9eE.+-]+)", active_all)
    value = max(values) if values else 0
    if value > 0:
        return True, 10, f"momentum={value:g}"
    return False, 0, "momentum not set (or 0)"


def check_herr_nilsson(active_all):
    m = re.search(r"class_weights\[7\]\s*=\s*([0-9.]+)", active_all)
    used = re.search(r"CrossEntropyLoss\(\s*weight\s*=\s*class_weights", active_all)
    if m and used and float(m.group(1)) != 1:
        return True, 0, f"class_weights[7]={m.group(1)}, used in CrossEntropyLoss"
    return False, 0, "no squirrel-monkey class weighting found"


def check_wisdom_extractor(active_all):
    defined = re.search(r"class\s+DistillationLoss\b", active_all)
    used = re.search(r"criterion\s*=\s*DistillationLoss\(", active_all)
    if defined and used:
        return True, 40, "DistillationLoss defined and used as criterion"
    return False, 0, "no knowledge distillation found"


def check_hypothesis_hustle(active_all):
    if re.search(r"lr_scheduler\.\w+\(", active_all):
        return True, 20, "a torch.optim.lr_scheduler is instantiated"
    return False, 0, "no learning rate scheduler found"


def check_augmentation(active_all, transform_name):
    if re.search(rf"transforms\.{transform_name}\(", active_all):
        return True, None, f"transforms.{transform_name} active"
    return False, None, f"transforms.{transform_name} not active"


def main():
    notebook_path = Path(sys.argv[1]) if len(sys.argv) > 1 else NOTEBOOK_PATH
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}")
        sys.exit(1)

    _, active_all = load_active_source(notebook_path)

    rows = []

    used, cost, detail = check_info_infusion(active_all)
    rows.append(("Info Infusion", "1B NN / % added data", used, cost, detail))

    used, cost, detail = check_resize_rocket(active_all)
    rows.append(("Resize Rocket", "10B NN / 2x size", used, cost, detail))

    used, _, detail = check_augmentation(active_all, "RandomRotation")
    rows.append(("Whirl and swirl", "10B NN", used, 10 if used else 0, detail))

    used, _, detail = check_augmentation(active_all, "GaussianBlur")
    rows.append(("Foggy Lens", "5B NN", used, 5 if used else 0, detail))

    h_flip = re.search(r"transforms\.RandomHorizontalFlip\(", active_all)
    v_flip = re.search(r"transforms\.RandomVerticalFlip\(", active_all)
    used = bool(h_flip or v_flip)
    detail = "transforms.Random{Horizontal,Vertical}Flip active" if used else "no active flip transform"
    rows.append(("Flip trick", "10B NN", used, 10 if used else 0, detail))

    used, _, detail = check_augmentation(active_all, "RandomAffine")
    rows.append(("Need for shift: Tokyo data drift", "5B NN", used, 5 if used else 0, detail))

    used, _, detail = check_augmentation(active_all, "RandomErasing")
    rows.append(("Cutout mask", "5B NN", used, 5 if used else 0, detail))

    used, cost, detail = check_ensemble(active_all)
    rows.append(("Ensemble enchanter", "60B NN", used, cost, detail))

    used, cost, detail = check_neural_boost(active_all)
    rows.append(("Neural boost", "20B NN / layer (3-for-2)", used, cost, detail))

    used, cost, detail = check_dropout(active_all)
    rows.append(("Drop shield", "10B NN", used, cost, detail))

    used, cost, detail = check_weight_decay(active_all)
    rows.append(("Weigth decay", "10B NN", used, cost, detail))

    used, cost, detail = check_herr_nilsson(active_all)
    rows.append(("Herr Nilsson's friend", "Free", used, cost, detail))

    used, cost, detail = check_wisdom_extractor(active_all)
    rows.append(("Wisdom extractor", "40B NN", used, cost, detail))

    used, cost, detail = check_momentum(active_all)
    rows.append(("Speed boost", "10B NN", used, cost, detail))

    used, cost, detail = check_attention(active_all)
    rows.append(("Focus Lens", "15B NN", used, cost, detail))

    used, cost, detail = check_skip_connection(active_all)
    rows.append(("Skip Connection", "15B NN", used, cost, detail))

    used, cost, detail = check_hypothesis_hustle(active_all)
    rows.append(("Hypothesis hustle", "20B NN", used, cost, detail))

    name_w = max(len(r[0]) for r in rows)
    price_w = max(len(r[1]) for r in rows)
    detail_w = max(len(r[4]) for r in rows)

    total = sum(r[3] for r in rows)

    header = f"{'Upgrade':<{name_w}}  {'Price':<{price_w}}  {'Status':<8}  {'Cost':>8}  Detail"
    print(f"Neural Nugget Ledger - {notebook_path.name}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, price, used, cost, detail in rows:
        status = "BOUGHT" if used else "-"
        cost_str = f"{cost:g}B" if cost else ("0B" if used else "-")
        print(f"{name:<{name_w}}  {price:<{price_w}}  {status:<8}  {cost_str:>8}  {detail:<{detail_w}}")
    print("-" * len(header))
    remaining = BUDGET - total
    print(f"{'TOTAL SPENT':<{name_w}}  {'':<{price_w}}  {'':<8}  {total:>7g}B")
    print(f"{'BUDGET':<{name_w}}  {'':<{price_w}}  {'':<8}  {BUDGET:>7g}B")
    status_word = "OVER BUDGET" if remaining < 0 else "REMAINING"
    print(f"{status_word:<{name_w}}  {'':<{price_w}}  {'':<8}  {remaining:>7g}B")


if __name__ == "__main__":
    main()
