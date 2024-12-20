import os
import argparse
import torch
import onnx
import onnxruntime as ort
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import warnings
import random
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from Dataset import MonkeyImageDataset

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ModelEvaluator:
    def __init__(self, dataset_path, image_sizes = [(64, 64)], num_of_classes=10, data_percentage=1, verbose=False):
        """
        Initialize the model evaluator with dataset configuration
        
        Args:
        - dataset_path (str): Path to dataset
        - image_size (tuple): Resize dimensions for images
        - num_of_classes (int): Number of classes in the dataset
        - data_percentage (float): Percentage of dataset to use
        - verbose (bool): Enable detailed logging
        """
        # Import dataset class here to avoid potential circular import

        
        # Create transform
       
        self.dataset = MonkeyImageDataset(dataset_path)
        self.dataset_path = dataset_path
        self.num_of_classes = num_of_classes
        self.data_percentage = 1
        # Initialize plot for real-time comparison
        self.model_metrics = []
        self.model_names = []
        self.verbose = verbose
        self.class_names = self.dataset.class_names

    def to_numpy(self, tensor):
        """Convert tensor to numpy array"""
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def plot_detailed_confusion_matrix(self, all_labels, all_preds, model_name, model_metrics):
        """
        Create a more detailed confusion matrix visualization with three subplots
        
        Args:
        - all_labels (list): True labels
        - all_preds (list): Predicted labels
        - model_name (str): Name of the model for title
        - model_metrics (dict): Dictionary containing model evaluation metrics
        """
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Compute per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Generate random figure number to avoid conflicts
        random_number = random.randint(1, 100)
        plt.figure(num=random_number, figsize=(20, 6))
        
        # Confusion Matrix Subplot
        plt.subplot(1, 3, 1)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha='right')
        
        # Per-class Accuracy Subplot
        plt.subplot(1, 3, 2)
        plt.bar(self.class_names, class_accuracy)
        plt.title(f"{model_name} - Per-Class Accuracy")
        plt.xlabel("Classes")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45, ha='right')
        
        # Model Metrics Subplot
        plt.subplot(1, 3, 3)
        metrics = [
            'Accuracy', 
            'Precision', 
            'Recall', 
            'F1 Score'
        ]
        values = [
            model_metrics['accuracy']*100,
            model_metrics['precision'] * 100,
            model_metrics['recall'] * 100,
            model_metrics['f1_score'] * 100
        ]
        
        plt.bar(metrics, values)
        plt.title(f"{model_name} - Overall Model Metrics")
        plt.xlabel("Metrics")
        plt.ylabel("Percentage (%)")
        plt.ylim(0, 100)  # Set y-axis from 0 to 100%
        
        # Add percentage labels on top of each bar
        for i, v in enumerate(values):
            plt.text(i, v, f'{v:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show(block=False)

    def plot_all_models_metrics(self, results):
        """
        Plot a grouped bar chart comparing different metrics across all evaluated models,
        with an additional subplot showing performance on the 7th class.

        Args:
            results (dict): Dictionary containing evaluation results for each model
        """
        # Extract unique metric names from all models' results
        unique_metrics = set()
        for model_name, model_results in results.items():
            unique_metrics.update(model_results.keys())
        
        # Remove non-numeric keys
        unique_metrics = {metric for metric in unique_metrics if metric in ['accuracy', 'class_report']}
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # First subplot: Overall model metrics
        # Create a color map for assigning distinct colors to different models
        num_models = len(results)
        cmap = plt.get_cmap('tab20', num_models)
        colors = [cmap(i) for i in range(num_models)]

        # Prepare data for overall metrics
        metric_positions = np.arange(len(unique_metrics) - 1)  # Exclude 'class_report'
        bar_width = 0.05  # Adjust bar width for better visualization with many models

        model_counter = 0
        for model_name, model_results in results.items():
            model_color = cmap(model_counter / num_models)  # Assign unique color
            
            # Prepare metric values (exclude 'class_report')
            metric_values = [
                model_results[metric] * 100 if metric != 'class_report' else 0 
                for metric in unique_metrics if metric != 'class_report'
            ]
            
            bar_positions = [metric_positions[i] + bar_width * model_counter for i in range(len(metric_values))]
            
            ax1.bar(bar_positions, metric_values, bar_width, label=model_name, color=model_color)
            model_counter += 1

        # Customize first subplot (Overall Metrics)
        ax1.set_title("Comparison of Model Evaluation Metrics", fontsize=14)
        metric_labels = [m for m in unique_metrics if m != 'class_report']
        ax1.set_xticks(metric_positions + bar_width * (num_models - 1) / 2)
        ax1.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=10)
        ax1.set_ylabel("Percentage (%)")
        ax1.legend(title="Models", loc='upper left', bbox_to_anchor=(1.05, 1))

        # Second subplot: Performance on 8th Class
        # Determine the index of the 8th class (0-based indexing)
        seventh_class = self.class_names[7] if len(self.class_names) > 6 else None
        
        if seventh_class:
            # Prepare data for 7th class performance
            seventh_class_scores = []
            for model_name, model_results in results.items():
                # Extract precision for the 7th class from the class report
                class_report = model_results.get('class_report', {})
                class_precision = class_report.get(seventh_class, {}).get('precision', 0) * 100
                seventh_class_scores.append(class_precision)
            
            # Sort models by their performance on the 7th class
            sorted_indices = np.argsort(seventh_class_scores)[::-1]
            sorted_models = [list(results.keys())[i] for i in sorted_indices]
            sorted_scores = [seventh_class_scores[i] for i in sorted_indices]
            
            # Plot 7th class performance
            ax2.bar(range(len(sorted_models)), sorted_scores, color=[cmap(i/num_models) for i in range(len(sorted_models))])
            ax2.set_title(f"Model Performance on Herr Nillson Class", fontsize=14)
            ax2.set_xlabel("Models")
            ax2.set_ylabel("Precision (%)")
            ax2.set_xticks(range(len(sorted_models)))
            ax2.set_xticklabels(sorted_models, rotation=45, ha='right')
            
            # Add value labels on top of each bar
            for i, v in enumerate(sorted_scores):
                ax2.text(i, v, f'{v:.2f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.show(block=False)
        
    def plot_top_models_class_accuracy(self, results, top_models, model_folder):
        """
        Plot class-level accuracy for top models with an additional subplot for overall metrics
        
        Args:
        - results (dict): Dictionary of all model evaluation results
        - top_models (list): List of top models to analyze
        - model_folder (str): Path to model folder
        """
        # Create figure with subplots
        plt.figure(figsize=(20, 8))
        
        # First subplot for class-level precision
        plt.subplot(1, 2, 1)
        
        # Prepare data for plotting
        class_level_accuracy = {}
        
        # Collect class-level accuracy for top models
        for model_name, _ in top_models:
            model_path = os.path.join(model_folder, model_name)
            
            # Re-run evaluation to get detailed results
            evaluation_results = self.evaluate_onnx_model(model_path)
            class_report = evaluation_results['class_report']
            
            # Extract per-class accuracy
            class_level_accuracy[model_name] = [
                class_report[cls]['precision'] * 100 
                for cls in self.class_names
            ]
        
        # Set the width of each bar and positions
        bar_width = 0.2
        num_models = len(top_models)
        
        # Create bars for each model
        for i, (model_name, _) in enumerate(top_models):
            r = [j + i * bar_width for j in range(len(self.class_names))]
            plt.bar(r, class_level_accuracy[model_name], 
                    width=bar_width, 
                    edgecolor='white', 
                    label=model_name)
        
        # Add labels and title for class-level precision
        plt.xlabel('Classes', fontweight='bold')
        plt.ylabel('Precision (%)', fontweight='bold')
        plt.title('Top Models - Class-Level Precision')
        plt.xticks([j + bar_width * (num_models-1)/2 for j in range(len(self.class_names))], 
                self.class_names, rotation=45, ha='right')
        
        # Second subplot for overall model metrics
        plt.subplot(1, 2, 2)
        
        # Prepare data for overall metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        model_metrics = {}
        
        # Collect metrics for top models
        for model_name, _ in top_models:
            model_path = os.path.join(model_folder, model_name)
            evaluation_results = self.evaluate_onnx_model(model_path)
            
            # Convert metrics to percentage for easier comparison
            model_metrics[model_name] = [
                evaluation_results['accuracy']* 100,
                evaluation_results['precision'] * 100,
                evaluation_results['recall'] * 100,
                evaluation_results['f1_score'] * 100
            ]
        
        # Set up bar positions for overall metrics
        bar_width = 0.2
        x = np.arange(len(metrics))
        
        # Plot bars for each top model's metrics
        for i, (model_name, _) in enumerate(top_models):
            plt.bar(
                x + i * bar_width, 
                model_metrics[model_name], 
                width=bar_width, 
                label=model_name
            )
        
        # Customize overall metrics subplot
        plt.title('Top Models - Overall Performance Metrics')
        plt.xlabel('Metrics')
        plt.ylabel('Percentage (%)')
        plt.xticks(x + bar_width * (num_models-1)/2, metrics, rotation=45)
        plt.ylim(0, 100)  # Set y-axis from 0 to 100%
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()

    def evaluate_onnx_model(self, model_path, show_confusion_matrix=False):
        """
        Evaluate an ONNX model and calculate metrics more accurately
        
        Args:
        - model_path (str): Path to ONNX model file
        - show_confusion_matrix (bool): Whether to display confusion matrix
        
        Returns:
        - dict: Model evaluation metrics
        """
        
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        
        # Retrieve input details
        input_tensor = onnx_model.graph.input[0]
        input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]

        image_size = (input_shape[2], input_shape[3])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size)
        ])
        
        # Load dataset
        dataset = MonkeyImageDataset(
            self.dataset_path, 
            transform=transform, 
            amount_of_classes=self.num_of_classes, 
            data_percentage=self.data_percentage
        )
        test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        
        # Create ONNX inference session
        ort_session = ort.InferenceSession(
            model_path, 
            providers=["CPUExecutionProvider"]
        )
        
        # Tracking variables
        total = 0
        correct = 0
        all_labels = []
        all_preds = []
        
        # Inference loop
        for inputs, targets in test_loader:
            # Ensure 4D tensor shape: [batch, channels, height, width]
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)
            
            # Prepare input for ONNX Runtime
            ort_inputs = {ort_session.get_inputs()[0].name: self.to_numpy(inputs)}
            outputs = ort_session.run(None, ort_inputs)[0]
            
            # Convert outputs to PyTorch tensors for processing
            pred = torch.tensor(outputs)
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            
            # Convert targets to match original function
            gt = torch.argmax(targets, 1)
            
            # Update tracking metrics
            total += gt.size(0)
            correct += (pred == gt).sum().item()
            
            # Collect labels and predictions
            all_labels.extend(gt.numpy())
            all_preds.extend(pred.numpy())
        
        # Calculate metrics with explicit averaging strategies
        accuracy = correct / total
        
        # Use multi-class averaging strategies
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Optional confusion matrix display
        if show_confusion_matrix:
            model_name = os.path.basename(model_path)
            self.plot_detailed_confusion_matrix(all_labels, all_preds, model_name, {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        # Generate classification report
        class_report = classification_report(
            all_labels, 
            all_preds, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # Verbose logging
        if self.verbose:
            print(f"\nModel: {os.path.basename(model_path)}")
            print(f'Accuracy: {accuracy:.2%}')
            print(f'Precision: {precision:.2%}')
            print(f'Recall : {recall:.2%}')
            print(f'F1-score: {f1:.2%}')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'class_report': class_report
        }
def get_top_models(results, top_n=3):
    """
    Get top N models based on accuracy
    
    Args:
    - results (dict): Dictionary of model results
    - top_n (int): Number of top models to return
    
    Returns:
    - list: Top N models sorted by accuracy
    """
    # Sort models by accuracy in descending order
    sorted_models = sorted(
        results.items(), 
        key=lambda x: x[1]['accuracy'], 
        reverse=True
    )
    return sorted_models[:top_n]

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='ONNX Model Evaluation Script for Monkey Madness Workshop'
    )
    
    # Required arguments
    parser.add_argument(
        '--model_folder', 
        type=str, 
        default='saved_models',
        help='Path to folder containing ONNX models'
    )
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        default='Monkey/test/test',
        help='Path to validation dataset'
    )
    
    # Optional arguments
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', 
        default=True,
        help='Enable verbose output, default=TRUE'
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default='model_evaluation_results.json', 
        help='Output file for model evaluation results'
    )
    parser.add_argument(
        '-t', '--top', 
        type=int, 
        default=0, 
        help='Number of top models to analyze in detail (show confusion matrix)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        args.dataset_path, 
        verbose=args.verbose,
    )
    
    # Results dictionary
    results = {}
    
    # Iterate through ONNX models
    for model_file in os.listdir(args.model_folder):
        if model_file.endswith('.onnx'):
            model_path = os.path.join(args.model_folder, model_file)
            try:
                model_results = evaluator.evaluate_onnx_model(model_path)
                results[model_file] = model_results
            except Exception as e:
                print(f"Error evaluating {model_file}: {e}")
    
    # Always plot comparison for all models
    if results:
        evaluator.plot_all_models_metrics(results)
    
    # Optional: Find and analyze top models
    if args.top > 0:
        top_models = get_top_models(results, args.top)
        print("\n--- Top Model Detailed Analysis ---")
        
        # Modify this section to prevent duplicate processing
        for model_name, model_metrics in top_models:
            model_path = os.path.join(args.model_folder, model_name)
            print(f"\nDetailed Analysis for {model_name}:")
            # Only show confusion matrix for first top model to prevent multiple plots
            evaluator.evaluate_onnx_model(
                model_path, 
                show_confusion_matrix=(model_name == top_models[0][0])
            )
        
        # Plot class-level accuracy for top models
        evaluator.plot_top_models_class_accuracy(results, top_models, args.model_folder)

        # Ensure we see the plots
        plt.show(block=False)
    
    # Save results to JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()