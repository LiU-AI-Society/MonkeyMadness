"""
Runs in its own subprocess per submission (spawned by app.py with a timeout):
loads a submitted .onnx model, predicts over the hidden test image folder,
scores against the private gold-label CSV, and prints exactly one JSON line
to stdout with the result.

Isolating this in a short-lived subprocess means a broken, oversized, or
malicious submitted model can only hang/crash this process -- never the
long-running web server -- and app.py enforces a hard wall-clock timeout
on top of that.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_submission import evaluate  # noqa: E402
from predict import generate_predictions  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--predictions_out", required=True)
    args = parser.parse_args()

    try:
        generate_predictions(args.model_path, args.image_dir, args.labels_path, args.predictions_out)
        scores = evaluate(args.predictions_out, args.gold, show_confusion_matrix=False, verbose=False)
        print(json.dumps({"ok": True, **scores}))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
