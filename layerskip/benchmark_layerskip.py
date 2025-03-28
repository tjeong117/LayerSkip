import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
import argparse
from tqdm import tqdm

# Import your MoE implementations
from simpleMoE import SimpleClassifier, StandardMoEClassifier
from simpleMoE import train_model, evaluate_model

# Create directory for saving results
os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)


# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Define a more complex synthetic dataset for a better test
def generate_complex_dataset(num_samples=10000, input_dim=128, num_classes=10, difficulty_levels=3):
    """
    Generate a more complex synthetic dataset with varying difficulty levels.
    This helps demonstrate the benefits of LayerSkip.

    Args:
        num_samples: Total number of samples
        input_dim: Dimension of input features
        num_classes: Number of classes
        difficulty_levels: Number of difficulty levels (1=easy, 2=medium, 3=hard)

    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"Generating dataset with {num_samples} samples, {input_dim} features, {num_classes} classes")
    X = np.random.randn(num_samples, input_dim)
    y = np.zeros(num_samples, dtype=np.int64)

    # Assign difficulty levels
    difficulty = np.random.choice(range(difficulty_levels), size=num_samples,
                                  p=[0.6, 0.3, 0.1])  # 60% easy, 30% medium, 10% hard

    # Create different classification rules based on difficulty
    for i in range(num_samples):
        if difficulty[i] == 0:  # Easy samples - simple linear boundary
            y[i] = np.sum(X[i, :input_dim // 4]) > 0
        elif difficulty[i] == 1:  # Medium samples - more complex boundary
            y[i] = (np.sum(X[i, input_dim // 4:input_dim // 2]) > 0.5) and (np.sum(X[i, :input_dim // 4]) > -1)
        else:  # Hard samples - complex non-linear boundary
            feature1 = np.sum(X[i, :input_dim // 3])
            feature2 = np.sum(X[i, input_dim // 3:2 * input_dim // 3])
            feature3 = np.sum(X[i, 2 * input_dim // 3:])
            y[i] = (feature1 * feature2 * feature3 > 0)

    # Convert to multi-class if needed
    if num_classes > 2:
        y = y % num_classes

    # Add sample-specific noise
    noise_level = np.zeros(num_samples)
    for i in range(num_samples):
        if difficulty[i] == 0:
            noise_level[i] = 0.05  # Little noise for easy samples
        elif difficulty[i] == 1:
            noise_level[i] = 0.15  # More noise for medium samples
        else:
            noise_level[i] = 0.25  # Most noise for hard samples

    for i in range(num_samples):
        X[i] += np.random.randn(input_dim) * noise_level[i]

    # Split into train, validation, and test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create datasets with difficulty information
    train_difficulty = difficulty[:len(X_train)]
    val_difficulty = difficulty[len(X_train):len(X_train) + len(X_val)]
    test_difficulty = difficulty[len(X_train) + len(X_val):]

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, torch.LongTensor(train_difficulty))
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor, torch.LongTensor(val_difficulty))
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, torch.LongTensor(test_difficulty))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    return train_loader, val_loader, test_loader


# Enhanced training function with more metrics
def train_and_evaluate(model_with_skip, model_standard, train_loader, val_loader, test_loader,
                       epochs=10, lr=0.001, aux_loss_weight=0.3, device='cuda'):
    """
    Train and evaluate both models, collecting detailed metrics for comparison
    """
    print(f"Training models for {epochs} epochs")

    # Metrics to track
    metrics = {
        'layerskip': {
            'train_loss': [], 'val_acc': [], 'epoch_time': [], 'early_exit_rate': [],
            'layer_exits': [], 'inference_time': 0, 'test_acc': 0, 'test_acc_by_difficulty': [0, 0, 0]
        },
        'standard': {
            'train_loss': [], 'val_acc': [], 'epoch_time': [], 'inference_time': 0,
            'test_acc': 0, 'test_acc_by_difficulty': [0, 0, 0]
        }
    }

    # Train LayerSkip model
    print("Training LayerSkip MoE model...")
    train_losses_skip, val_accuracies_skip, exit_stats_skip, epoch_times_skip, avg_time_skip = train_model(
        model_with_skip, train_loader, val_loader, epochs=epochs, lr=lr,
        aux_loss_weight=aux_loss_weight, load_balance_weight=0.1
    )

    # Train standard model
    print("Training standard MoE model...")
    train_losses_std, val_accuracies_std, _, epoch_times_std, avg_time_std = train_model(
        model_standard, train_loader, val_loader, epochs=epochs, lr=lr, is_naive=True
    )

    # Store training metrics
    metrics['layerskip']['train_loss'] = train_losses_skip
    metrics['layerskip']['val_acc'] = val_accuracies_skip
    metrics['layerskip']['epoch_time'] = epoch_times_skip
    metrics['layerskip']['early_exit_rate'] = exit_stats_skip

    metrics['standard']['train_loss'] = train_losses_std
    metrics['standard']['val_acc'] = val_accuracies_std
    metrics['standard']['epoch_time'] = epoch_times_std

    # Test on the test set
    print("Evaluating models on test set...")

    # LayerSkip model
    accuracy_skip, time_skip, exits_skip, early_exit_rate = evaluate_model(model_with_skip, test_loader)
    metrics['layerskip']['inference_time'] = time_skip * 1000  # Convert to ms
    metrics['layerskip']['test_acc'] = accuracy_skip
    metrics['layerskip']['layer_exits'] = exits_skip

    # Standard model
    accuracy_std, time_std = evaluate_model(model_standard, test_loader, is_naive=True)
    metrics['standard']['inference_time'] = time_std * 1000  # Convert to ms
    metrics['standard']['test_acc'] = accuracy_std

    # Calculate accuracy by difficulty level
    metrics['layerskip']['test_acc_by_difficulty'] = evaluate_by_difficulty(model_with_skip, test_loader)
    metrics['standard']['test_acc_by_difficulty'] = evaluate_by_difficulty(model_standard, test_loader, is_naive=True)

    return metrics


# Function to evaluate models by difficulty level - fixed to handle datasets properly
def evaluate_by_difficulty(model, test_loader, is_naive=False):
    """Evaluate model accuracy separated by difficulty level"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    correct = [0, 0, 0]  # For each difficulty level
    total = [0, 0, 0]

    with torch.no_grad():
        for batch_data in test_loader:
            # Ensure we have difficulty data
            if len(batch_data) == 3:
                data, target, difficulty = batch_data
            else:
                # If no difficulty data, return default values
                return [0, 0, 0]

            data, target = data.to(device), target.to(device)

            if is_naive:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
            else:
                # May need to handle different return formats
                try:
                    outputs, _, _, _ = model(data)
                except ValueError:
                    outputs = model(data)[0]  # Just get the logits

                _, predicted = torch.max(outputs, 1)

            # Count correct predictions by difficulty
            for i in range(3):  # 3 difficulty levels
                mask = (difficulty == i)
                if mask.sum() > 0:
                    correct[i] += ((predicted == target) & mask).sum().item()
                    total[i] += mask.sum().item()

    # Calculate accuracy for each difficulty level
    accuracy = [100 * c / t if t > 0 else 0 for c, t in zip(correct, total)]
    return accuracy
# Function to plot and save all comparison graphs
def plot_metrics(metrics, epochs, save_dir="results/plots"):
    """Generate and save plots comparing LayerSkip and standard MoE"""
    print("Generating performance comparison plots...")

    # 1. Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), metrics['layerskip']['train_loss'], marker='o', label='LayerSkip MoE')
    plt.plot(range(1, epochs + 1), metrics['standard']['train_loss'], marker='s', label='Standard MoE')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/training_loss_comparison.png", dpi=300)

    # 2. Plot validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), metrics['layerskip']['val_acc'], marker='o', label='LayerSkip MoE')
    plt.plot(range(1, epochs + 1), metrics['standard']['val_acc'], marker='s', label='Standard MoE')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/val_accuracy_comparison.png", dpi=300)

    # 3. Plot epoch training time
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), metrics['layerskip']['epoch_time'], marker='o', label='LayerSkip MoE')
    plt.plot(range(1, epochs + 1), metrics['standard']['epoch_time'], marker='s', label='Standard MoE')
    plt.title('Training Time Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/training_time_comparison.png", dpi=300)

    # 4. Plot early exit rate over epochs (only for LayerSkip)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), metrics['layerskip']['early_exit_rate'], marker='o', color='green')
    plt.title('Early Exit Rate (LayerSkip MoE)')
    plt.xlabel('Epoch')
    plt.ylabel('Early Exit Rate (%)')
    plt.grid(True)
    plt.savefig(f"{save_dir}/early_exit_rate.png", dpi=300)

    # 5. Bar chart for inference time comparison
    plt.figure(figsize=(10, 6))
    models = ['LayerSkip MoE', 'Standard MoE']
    times = [metrics['layerskip']['inference_time'], metrics['standard']['inference_time']]
    plt.bar(models, times, color=['green', 'blue'])
    plt.title('Inference Time Comparison')
    plt.ylabel('Average Time per Sample (ms)')
    plt.grid(axis='y')

    # Add speedup text
    speedup = metrics['standard']['inference_time'] / metrics['layerskip']['inference_time']
    plt.text(0, times[0] + 1, f"{times[0]:.2f} ms", ha='center')
    plt.text(1, times[1] + 1, f"{times[1]:.2f} ms", ha='center')
    plt.figtext(0.5, 0.01, f"LayerSkip Speedup: {speedup:.2f}x", ha="center", fontsize=12,
                bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

    plt.savefig(f"{save_dir}/inference_time_comparison.png", dpi=300)

    # 6. Bar chart for test accuracy
    plt.figure(figsize=(10, 6))
    accuracies = [metrics['layerskip']['test_acc'], metrics['standard']['test_acc']]
    plt.bar(models, accuracies, color=['green', 'blue'])
    plt.title('Test Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(min(accuracies) - 5, 100)
    plt.grid(axis='y')

    # Add accuracy text
    plt.text(0, accuracies[0] - 3, f"{accuracies[0]:.2f}%", ha='center')
    plt.text(1, accuracies[1] - 3, f"{accuracies[1]:.2f}%", ha='center')

    plt.savefig(f"{save_dir}/test_accuracy_comparison.png", dpi=300)

    # 7. Plot accuracy by difficulty level
    plt.figure(figsize=(12, 7))
    width = 0.35
    x = np.arange(3)

    layerskip_acc = metrics['layerskip']['test_acc_by_difficulty']
    standard_acc = metrics['standard']['test_acc_by_difficulty']

    plt.bar(x - width / 2, layerskip_acc, width, label='LayerSkip MoE', color='green')
    plt.bar(x + width / 2, standard_acc, width, label='Standard MoE', color='blue')

    plt.xlabel('Task Difficulty')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance by Task Difficulty')
    plt.xticks(x, ['Easy', 'Medium', 'Hard'])
    plt.legend()
    plt.grid(axis='y')

    for i, v in enumerate(layerskip_acc):
        plt.text(i - width / 2, v + 1, f"{v:.1f}%", ha='center')
    for i, v in enumerate(standard_acc):
        plt.text(i + width / 2, v + 1, f"{v:.1f}%", ha='center')

    plt.savefig(f"{save_dir}/accuracy_by_difficulty.png", dpi=300)

    # 8. Plot layer exit distribution (only for LayerSkip)
    if 'layer_exits' in metrics['layerskip'] and metrics['layerskip']['layer_exits']:
        exits = metrics['layerskip']['layer_exits']

        # Calculate percentages
        total = sum(exits)
        exit_percentages = [100 * e / total for e in exits]

        plt.figure(figsize=(10, 6))
        layers = [f"Layer {i}" for i in range(len(exits))]
        plt.bar(layers, exit_percentages, color=['lightgreen', 'green', 'darkgreen'])
        plt.title('Layer Exit Distribution (LayerSkip MoE)')
        plt.ylabel('Percentage of Samples (%)')
        plt.grid(axis='y')

        # Add percentage text
        for i, v in enumerate(exit_percentages):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')

        plt.savefig(f"{save_dir}/layer_exit_distribution.png", dpi=300)

    # 9. Create an overall performance dashboard
    plt.figure(figsize=(15, 10))

    # Overall metrics
    plt.subplot(2, 2, 1)
    metrics_names = ['Test Accuracy (%)', 'Val Accuracy (%)', 'Training Time (s)', 'Inference Time (ms)']
    layerskip_metrics = [
        metrics['layerskip']['test_acc'],
        metrics['layerskip']['val_acc'][-1],
        sum(metrics['layerskip']['epoch_time']),
        metrics['layerskip']['inference_time']
    ]
    standard_metrics = [
        metrics['standard']['test_acc'],
        metrics['standard']['val_acc'][-1],
        sum(metrics['standard']['epoch_time']),
        metrics['standard']['inference_time']
    ]

    # Create a radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    # Normalize metrics for the radar chart
    normalized_layerskip = []
    normalized_standard = []

    # Accuracy metrics: higher is better
    for i in range(2):
        max_val = max(layerskip_metrics[i], standard_metrics[i])
        normalized_layerskip.append(layerskip_metrics[i] / max_val)
        normalized_standard.append(standard_metrics[i] / max_val)

    # Time metrics: lower is better, so invert
    for i in range(2, 4):
        min_val = min(layerskip_metrics[i], standard_metrics[i])
        normalized_layerskip.append(min_val / layerskip_metrics[i])
        normalized_standard.append(min_val / standard_metrics[i])

    # Close the loop
    normalized_layerskip = np.concatenate((normalized_layerskip, [normalized_layerskip[0]]))
    normalized_standard = np.concatenate((normalized_standard, [normalized_standard[0]]))

    # Plot radar chart
    ax = plt.subplot(2, 2, 1, polar=True)
    ax.plot(angles, normalized_layerskip, 'o-', linewidth=2, label='LayerSkip MoE')
    ax.plot(angles, normalized_standard, 'o-', linewidth=2, label='Standard MoE')
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, metrics_names)
    ax.grid(True)
    ax.set_title('Overall Performance Comparison')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Training loss
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs + 1), metrics['layerskip']['train_loss'], marker='o', label='LayerSkip MoE')
    plt.plot(range(1, epochs + 1), metrics['standard']['train_loss'], marker='s', label='Standard MoE')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Early exit rate and accuracy correlation
    plt.subplot(2, 2, 3)
    plt.scatter(metrics['layerskip']['early_exit_rate'], metrics['layerskip']['val_acc'],
                c=range(1, epochs + 1), cmap='viridis', s=100)

    for i, (x, y) in enumerate(zip(metrics['layerskip']['early_exit_rate'], metrics['layerskip']['val_acc'])):
        plt.annotate(f"Epoch {i + 1}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.colorbar(label='Epoch')
    plt.title('Early Exit Rate vs. Accuracy')
    plt.xlabel('Early Exit Rate (%)')
    plt.ylabel('Validation Accuracy (%)')
    plt.grid(True)

    # Difficulty breakdown
    plt.subplot(2, 2, 4)
    width = 0.35
    x = np.arange(3)

    layerskip_acc = metrics['layerskip']['test_acc_by_difficulty']
    standard_acc = metrics['standard']['test_acc_by_difficulty']

    plt.bar(x - width / 2, layerskip_acc, width, label='LayerSkip MoE', color='green')
    plt.bar(x + width / 2, standard_acc, width, label='Standard MoE', color='blue')

    plt.xlabel('Task Difficulty')
    plt.ylabel('Accuracy (%)')
    plt.title('Performance by Task Difficulty')
    plt.xticks(x, ['Easy', 'Medium', 'Hard'])
    plt.legend()
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_dashboard.png", dpi=300)

    print(f"All plots saved to {save_dir}")


# Main function to run experiments
def main(args):
    set_seed(args.seed)

    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate dataset
    train_loader, val_loader, test_loader = generate_complex_dataset(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        num_classes=args.num_classes
    )

    # Create models
    print("Creating models...")
    model_with_skip = SimpleClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        expert_count=args.expert_count,
        top_k=args.top_k,
        enable_layer_skip=True,
        confidence_thresholds=[0.6 + 0.1 * i for i in range(args.num_layers)]  # Progressively higher thresholds
    )

    # Standard MoE model without LayerSkip functionality
    model_standard = StandardMoEClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        expert_count=args.expert_count,
        top_k=args.top_k
    )

    # Train and evaluate both models
    metrics = train_and_evaluate(
        model_with_skip,
        model_standard,
        train_loader,
        val_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        aux_loss_weight=args.aux_loss_weight,
        device=device
    )

    # Generate all comparison plots
    plot_metrics(metrics, args.epochs)

    # Save models if requested
    if args.save_models:
        torch.save(model_with_skip.state_dict(), "results/model_with_layerskip.pt")
        torch.save(model_standard.state_dict(), "results/standard_moe_model.pt")

    # Print summary statistics
    print("\nPerformance Summary:")
    print("-" * 50)
    print(f"Test Accuracy:")
    print(f"  LayerSkip MoE: {metrics['layerskip']['test_acc']:.2f}%")
    print(f"  Standard MoE:  {metrics['standard']['test_acc']:.2f}%")

    print(f"\nInference Time:")
    print(f"  LayerSkip MoE: {metrics['layerskip']['inference_time']:.2f} ms")
    print(f"  Standard MoE:  {metrics['standard']['inference_time']:.2f} ms")

    speedup = metrics['standard']['inference_time'] / metrics['layerskip']['inference_time']
    print(f"  Speedup: {speedup:.2f}x")

    print(f"\nTraining Time (total):")
    print(f"  LayerSkip MoE: {sum(metrics['layerskip']['epoch_time']):.2f} s")
    print(f"  Standard MoE:  {sum(metrics['standard']['epoch_time']):.2f} s")

    print(f"\nFinal Early Exit Rate:")
    print(f"  {metrics['layerskip']['early_exit_rate'][-1]:.2f}%")

    print(f"\nAccuracy by Difficulty:")
    difficulties = ['Easy', 'Medium', 'Hard']
    for i, diff in enumerate(difficulties):
        ls_acc = metrics['layerskip']['test_acc_by_difficulty'][i]
        std_acc = metrics['standard']['test_acc_by_difficulty'][i]
        diff_str = f"{diff}: LayerSkip = {ls_acc:.2f}%, Standard = {std_acc:.2f}%"
        if ls_acc > std_acc:
            diff_str += f" (LayerSkip better by {ls_acc - std_acc:.2f}%)"
        else:
            diff_str += f" (Standard better by {std_acc - ls_acc:.2f}%)"
        print(f"  {diff_str}")

    print("\nAll results and plots have been saved to the 'results' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and benchmark LayerSkip MoE")

    # Dataset parameters
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples in the dataset")
    parser.add_argument("--input_dim", type=int, default=128, help="Input dimension")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of MoE layers")
    parser.add_argument("--expert_count", type=int, default=8, help="Number of experts per layer")
    parser.add_argument("--top_k", type=int, default=2, help="Number of experts to route to")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--aux_loss_weight", type=float, default=0.3, help="Weight for auxiliary loss")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_models", action="store_true", help="Save the trained models")

    args = parser.parse_args()

    main(args)