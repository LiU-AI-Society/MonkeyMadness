import torch
import torch.nn as nn
from torch.autograd import Variable
import csv
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from livelossplot import PlotLosses
import seaborn as sns
import numpy as np


def plot_cm(all_labels, all_preds, classes):
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def test(model, testloader, device, model_name, unique_id):
    liveloss = PlotLosses()
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    model.eval()


    for idx, (inputs, targets) in enumerate(testloader):

        # Move data to GPU if CUDA is available

        inputs, targets = inputs.to(device), targets.to(device)

        inputs = torch.tensor(inputs, requires_grad=False)

        # Feed-forward the network
        with torch.no_grad():
            outputs = model(inputs)

        pred = torch.softmax(outputs, 1)
        gt = torch.argmax(targets, 1)
        pred = torch.argmax(pred, 1)

        total += gt.size(0)
        correct += (pred == gt).sum().item()

        all_labels.extend(gt.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())

    # Compute confusion matrix

    class_names = testloader.dataset.dataset.class_names #

    plot_cm(all_labels, all_preds, class_names)

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

    return (100 * correct / total)
