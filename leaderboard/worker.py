"""
Runs in its own subprocess per submission (spawned by app.py with a timeout):
loads a submitted .onnx model, predicts over the hidden test image folder,
scores against the private gold-label CSV, builds the "model in action"
showcase (showcase.py) on one public training image, and prints exactly one
JSON line to stdout with the result.

Isolating this in a short-lived subprocess means a broken, oversized, or
malicious submitted model can only hang/crash this process -- never the
long-running web server -- and app.py enforces a hard wall-clock timeout
on top of that.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_submission import evaluate  # noqa: E402
from predict import generate_predictions, load_class_names  # noqa: E402
from showcase import occlusion_showcase  # noqa: E402

# The showcase is a bonus: if scoring alone already took this long, skip it so
# the whole run stays under app.py's timeout and the score is never lost.
SHOWCASE_SKIP_AFTER_SECONDS = 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--predictions_out", required=True)
    parser.add_argument("--showcase_image", default=None)
    args = parser.parse_args()

    start = time.monotonic()
    try:
        generate_predictions(args.model_path, args.image_dir, args.labels_path, args.predictions_out)
        scores = evaluate(args.predictions_out, args.gold, show_confusion_matrix=False, verbose=False)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(1)

    showcase = None
    if args.showcase_image and time.monotonic() - start < SHOWCASE_SKIP_AFTER_SECONDS:
        try:
            showcase = occlusion_showcase(args.model_path, args.showcase_image, load_class_names(args.labels_path))
        except Exception as exc:
            print(f"Showcase failed: {exc}", file=sys.stderr)

    print(json.dumps({"ok": True, **scores, "showcase": showcase}))


if __name__ == "__main__":
    main()
