import torch
import torch.nn as nn
import os
import csv
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from livelossplot import PlotLosses
import torch.nn.functional as F
import pandas as pd
from tabulate import tabulate

"""from live_plot import LivePlot

# Initialize the LivePlot class
live_plotter = LivePlot()"""


def train(model, trainloader, valloader, optimizer, objective, device, start_epoch, num_epochs, model_name, unique_id, scheduler=None, teacher_model=None):
    print("Starting training on device:", device)

    train_loss_log = []
    train_acc_log = [] 
    val_loss_log = []
    val_acc_log = []

    liveloss = PlotLosses()
    best_val_acc = 0
    onnx_path = f"saved_models/{model_name}_{unique_id}_best_model.onnx"
    for epoch in range(start_epoch, num_epochs + 1):
        print(f'\nEpoch: {epoch}')

        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if teacher_model is not None:
                teacher_logits = teacher_model(inputs).detach()  # Get soft labels
                loss = objective(outputs, teacher_logits, targets)
            else:

                loss = objective(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            outputs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)  
            targets = targets.argmax(dim=1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()

        # Training accuracy and loss
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * (correct / total)

        train_loss_log.append(train_loss)
        train_acc_log.append(train_acc)

        # Validation phase
        model.eval()
        val_correct = 0
        val_loss = 0
        val_total = 0

        with torch.no_grad():
            for val_inputs, val_targets in valloader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

                val_outputs = model(val_inputs)
                if teacher_model is not None:
                    teacher_logits = teacher_model(val_inputs).detach()  # Get soft labels
                    val_loss += objective(val_outputs, teacher_logits, val_targets).item()
                else:

                    val_loss += objective(val_outputs, val_targets).item()
                val_outputs = F.softmax(val_outputs, dim=1)  # Apply sigmoid activation

                preds = val_outputs.argmax(dim=1)
                val_targets = val_targets.argmax(dim=1)
                val_total += val_targets.size(0)
                val_correct += (preds == val_targets).sum().item()

        val_loss = val_loss / len(valloader)
        val_acc = 100. * (val_correct / val_total)

        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)
        

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Set the model to evaluation mode
            model.eval()

            # Define a dummy input tensor matching the input shape expected by the model
            # Adjust the shape based on your actual model's input


            # Export the model to ONNX format
            torch.onnx.export(
                model,
                val_inputs,
                onnx_path,
                verbose=True,
                input_names=["input"],
                output_names=["output"],
                opset_version=11,  # Use appropriate ONNX opset version
            )

            print(f"New best model saved to {onnx_path} with accuracy: {best_val_acc:.2f}%")

        # Live loss plotting
        logs = {
            'log loss': train_loss,
            'accuracy': train_acc,
            'val_log loss': val_loss,
            'val_accuracy': val_acc
        }
        liveloss.update(logs)
        liveloss.send()

    print('Training Finished!')
    return model, train_loss_log, train_acc_log, val_loss_log, val_acc_log


def training_info(model_info):
    # Add training and validation info to the model info dictionary
    # Create a DataFrame from the dictionary
    df = pd.DataFrame([model_info])  # Use list of dictionaries to create a single-row DataFrame
    filename = f'{model_info["model_name"]}_{model_info["ID"]}.csv'

    if not os.path.exists('training_metrics'):
        os.makedirs('training_metrics')

    filename = os.path.join('training_metrics', filename)

    # Append to a CSV file (or create one if it doesn't exist)
    df.to_csv(filename, mode='a', index=False, header=not pd.io.common.file_exists(filename))


def plot_experiments(data_dir):
    # Initialize lists for plotting
    data = load_csv_files(data_dir)
    plot_metrics(data)
    display_parameters(data)


def load_csv_files(directory):
    """Load all CSV files from a specified directory."""
    dataframes = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            model_name = os.path.splitext(filename)[0]  # Use filename as model name
            dataframes[model_name] = df
    return dataframes

def plot_metrics(dataframes):
    """Plot training and validation metrics for each model."""
    plt.figure(figsize=(14, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    for model_name, df in dataframes.items():
        epochs = df['epochs'].iloc[0]

        t_loss = eval(df['t_loss'].iloc[0]) if isinstance(df['t_loss'].iloc[0], str) else df['t_loss'].iloc[0]
        v_loss = eval(df['v_loss'].iloc[0]) if isinstance(df['v_loss'].iloc[0], str) else df['v_loss'].iloc[0]
        plt.plot(range(0, epochs+1), t_loss, label=f'Train loss ({model_name})', linestyle='--')
        plt.plot(range(0,epochs+1), v_loss, label=f'Val loss ({model_name})')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    for model_name, df in dataframes.items():
        epochs = df['epochs'].iloc[0]

        t_acc = eval(df['t_acc'].iloc[0]) if isinstance(df['t_acc'].iloc[0], str) else df['t_acc'].iloc[0]
        v_acc = eval(df['v_acc'].iloc[0]) if isinstance(df['v_acc'].iloc[0], str) else df['v_acc'].iloc[0]
        plt.plot(range(0, epochs+1), t_acc, label=f'Train acc ({model_name})', linestyle='--')
        plt.plot(range(0, epochs+1), v_acc, label=f'Val acc ({model_name})')

    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def display_parameters(dataframes):
    """Display model parameters in a table format, excluding metrics."""
    param_list = []
    for model_name, df in dataframes.items():
        # Extract the first row and exclude specific columns
        params = df.iloc[0].drop(['t_loss', 'v_loss', 't_acc', 'v_acc']).to_dict()  # Exclude metrics
        params['Model Name'] = model_name
        param_list.append(params)

    # Convert to DataFrame for tabulate
    param_df = pd.DataFrame(param_list)
    print(tabulate(param_df, headers='keys', tablefmt='psql', showindex=False))
