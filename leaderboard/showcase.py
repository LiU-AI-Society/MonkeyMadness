"""
"Model in action" for the submission page: an occlusion heatmap on one fixed
image. GradCAM (GradCam.py) needs PyTorch gradients, but submissions are ONNX
run by onnxruntime, which only does forward passes. Occlusion needs nothing
else: cover one patch of the picture with grey, run the model again, and see
how much less sure it gets about its answer. Patches that make it much less
sure are the parts the model looks at.

Runs inside worker.py's subprocess, never in the web server, because it loads
the uploaded model.
"""
import base64
import io
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

GRID_STEPS = 12        # patch positions per row and per column -> 144 forward passes
PATCH_FRACTION = 0.25  # patch side as a fraction of the model's input side
GREY = 0.5             # value the covered pixels get (inputs are in [0, 1])
DISPLAY_SIZE = 256     # side of the photo sent to the browser


def _softmax(logits):
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def _data_url(image, fmt):
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _probs(values):
    # Four decimals is plenty for bars on a screen and keeps the JSON small.
    return [round(float(p), 4) for p in values]


def occlusion_showcase(model_path, image_path, class_names):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    height, width = model_input.shape[2], model_input.shape[3]

    # Same preprocessing as predict.py, so the heatmap shows the exact input
    # the model was scored on.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((height, width)),
    ])
    photo = Image.open(image_path).convert("RGB")
    tensor = transform(photo)
    x = tensor.unsqueeze(0).numpy()

    def run(batch):
        return _softmax(session.run(None, {model_input.name: batch})[0][0])

    # First guess with nothing covered.
    base_probs = run(x)

    # Slide the grey patch row by row, left to right, like reading a page.
    patch_h = max(1, round(PATCH_FRACTION * height))
    patch_w = max(1, round(PATCH_FRACTION * width))
    ys = np.linspace(0, height - patch_h, GRID_STEPS).round().astype(int)
    xs = np.linspace(0, width - patch_w, GRID_STEPS).round().astype(int)

    steps = []
    for y in ys:
        for x0 in xs:
            covered = x.copy()
            covered[:, :, y:y + patch_h, x0:x0 + patch_w] = GREY
            steps.append({"y": y / height, "x": x0 / width, "probs": _probs(run(covered))})

    # The folder name (n0..n9) is the true class of a training image.
    folder = re.fullmatch(r"n(\d)", Path(image_path).parent.name)

    return {
        "image": _data_url(photo.resize((DISPLAY_SIZE, DISPLAY_SIZE)), "JPEG"),
        "model_view": _data_url(transforms.ToPILImage()(tensor.clamp(0, 1)), "PNG"),
        "input_size": [height, width],
        "classes": [name.strip() for name in class_names],
        "true_index": int(folder.group(1)) if folder else None,
        "base_probs": _probs(base_probs),
        "patch": [patch_h / height, patch_w / width],
        "steps": steps,
    }
