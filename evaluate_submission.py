import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score)


def plot_cm(all_labels, all_preds, classes, title="Confusion Matrix"):
    # NOTE: confusion_matrix silently drops any sample whose predicted label
    # isn't in `classes` -- pass a "no_prediction" pseudo-class in `classes`
    # so missing predictions show up as their own column instead of vanishing
    # (which would otherwise zero out that row's sum and produce a NaN).
    cm = confusion_matrix(all_labels, all_preds, labels=classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def evaluate(predictions_csv, gold_csv, show_confusion_matrix=True, verbose=True):
    """
    Score a predictions CSV (filename,predicted_label) against a private
    gold-labels CSV (filename,label). Keep gold_csv off of any shared
    machine/repo that participants can reach -- it is the only place the
    true test labels live.
    """
    preds = pd.read_csv(predictions_csv)
    gold = pd.read_csv(gold_csv)

    merged = gold.merge(preds, on="filename", how="left", validate="one_to_one")
    missing = merged["predicted_label"].isna().sum()
    if missing:
        if verbose:
            print(f"Warning: {missing} test images have no prediction (scored as incorrect).")
        merged["predicted_label"] = merged["predicted_label"].fillna("no_prediction")

    classes = sorted(gold["label"].unique())
    cm_labels = classes + (["no_prediction"] if missing else [])
    accuracy = (merged["predicted_label"] == merged["label"]).mean()
    precision = precision_score(merged["label"], merged["predicted_label"],
                                 labels=classes, average="macro", zero_division=0)
    recall = recall_score(merged["label"], merged["predicted_label"],
                           labels=classes, average="macro", zero_division=0)
    f1 = f1_score(merged["label"], merged["predicted_label"],
                  labels=classes, average="macro", zero_division=0)

    if verbose:
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"Precision: {precision:.2%} (macro)")
        print(f"Recall:    {recall:.2%} (macro)")
        print(f"F1:        {f1:.2%} (macro)")
        print()
        print(classification_report(merged["label"], merged["predicted_label"], labels=classes, zero_division=0))

    if show_confusion_matrix:
        plot_cm(merged["label"], merged["predicted_label"], cm_labels)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}


def main():
    parser = argparse.ArgumentParser(
        description="Score a predictions CSV against a private gold-labels CSV. "
                     "Never distribute the gold CSV to participants."
    )
    parser.add_argument("--predictions", required=True, help="CSV with columns: filename,predicted_label")
    parser.add_argument("--gold", required=True, help="Private CSV with columns: filename,label")
    parser.add_argument("--no-plot", action="store_true", help="Skip the confusion matrix plot")
    args = parser.parse_args()
    evaluate(args.predictions, args.gold, show_confusion_matrix=not args.no_plot)


if __name__ == "__main__":
    main()
