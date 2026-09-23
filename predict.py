import argparse
import csv
import sys

import onnx
import onnxruntime as ort
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from Dataset import UnlabeledImageDataset


def load_class_names(labels_path):
    """Read monkey_labels.txt and return common names in n0..n9 order (matches training class indices)."""
    df = pd.read_csv(labels_path, header=None, skiprows=1)
    df.columns = ["Label", "Latin Name", "Common Name", "Train Images", "Validation Images"]
    return [name.rstrip() for name in df["Common Name"]]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def generate_predictions(model_path, image_dir, labels_path, output_csv):
    """
    Run a submitted ONNX model over a flat, unlabeled image folder and write
    predictions to a CSV. No gold labels are read or required here -- this is
    the only script that needs to touch the held-out test images.
    """
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    input_tensor = onnx_model.graph.input[0]
    input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    image_size = (input_shape[2], input_shape[3])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
    ])

    dataset = UnlabeledImageDataset(image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    class_names = load_class_names(labels_path)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    rows = []
    for inputs, filenames in loader:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        ort_inputs = {session.get_inputs()[0].name: to_numpy(inputs)}
        outputs = session.run(None, ort_inputs)[0]
        pred = torch.softmax(torch.tensor(outputs), dim=1)
        pred_idx = torch.argmax(pred, dim=1)

        for filename, idx in zip(filenames, pred_idx):
            rows.append((filename, class_names[idx.item()]))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_label"])
        writer.writerows(rows)

    # stderr, not stdout -- callers that parse worker stdout (e.g. the leaderboard
    # subprocess in leaderboard/worker.py) expect stdout to contain only their own output.
    print(f"Wrote {len(rows)} predictions -> {output_csv}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Run a submitted ONNX model over an unlabeled test image folder "
                     "and write predictions to a CSV, without exposing any gold labels."
    )
    parser.add_argument("--model_path", required=True, help="Path to submitted .onnx model")
    parser.add_argument("--image_dir", required=True, help="Folder of flat, unlabeled test images")
    parser.add_argument("--labels_path", default="Monkey/monkey_labels.txt",
                         help="Path to monkey_labels.txt for class index -> name mapping")
    parser.add_argument("--output", default="predictions.csv", help="Where to write predictions")
    args = parser.parse_args()
    generate_predictions(args.model_path, args.image_dir, args.labels_path, args.output)


if __name__ == "__main__":
    main()
