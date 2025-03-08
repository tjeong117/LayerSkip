import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class MoELayerWithSkip(nn.Module):
    def __init__(self, input_dim, hidden_dim, expert_count=4, top_k=2,
                 enable_layer_skip=True, confidence_threshold=0.5,
                 expert_dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.expert_count = expert_count
        self.top_k = top_k
        self.enable_layer_skip = enable_layer_skip
        self.confidence_threshold = confidence_threshold
        self.expert_dropout_rate = expert_dropout_rate
        self.training_mode = True

        # Router network
        self.router = nn.Linear(input_dim, expert_count)

        # Experts - simple feed-forward networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(expert_count)
        ])

        # LayerSkip components
        self.confidence_predictor = nn.Linear(input_dim, 1)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x, return_metrics=False):
        # Layer normalization
        residual = x
        x = self.layer_norm(x)

        # Get router logits
        router_logits = self.router(x)

        # Apply expert dropout during training
        if self.training and self.expert_dropout_rate > 0:
            expert_mask = torch.rand(self.expert_count, device=x.device) > self.expert_dropout_rate
            # Ensure at least one expert is active
            if not expert_mask.any():
                expert_mask[torch.randint(0, self.expert_count, (1,))] = True

            # Apply mask to router logits
            router_logits = router_logits.masked_fill(~expert_mask.unsqueeze(0), -1e10)

        # Get routing probabilities and indices
        router_probs = F.softmax(router_logits, dim=-1)

        # Get top-k experts
        vals, indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize the router probabilities
        vals = vals / vals.sum(dim=-1, keepdim=True)

        # Compute early exit confidence if enabled
        early_exit = False
        confidence_score = 0.0

        if self.enable_layer_skip and not self.training:
            confidence = torch.sigmoid(self.confidence_predictor(x))
            confidence_score = confidence.mean().item()

            # Check if we should exit early
            if confidence_score > self.confidence_threshold:
                early_exit = True
                # Just return the input if we exit early
                if return_metrics:
                    return residual + x, early_exit, vals, indices, router_probs, confidence_score
                return residual + x, early_exit, confidence_score

        # Process through experts
        batch_size = x.shape[0]

        # Initialize output tensor
        combined_output = torch.zeros_like(x)

        # Simple MoE computation
        for b in range(batch_size):
            for k in range(self.top_k):
                expert_idx = indices[b, k].item()
                weight = vals[b, k].item()
                expert_output = self.experts[expert_idx](x[b].unsqueeze(0))
                combined_output[b] += weight * expert_output.squeeze(0)

        # Residual connection
        output = residual + combined_output

        if return_metrics:
            return output, early_exit, vals, indices, router_probs, confidence_score
        return output, early_exit, confidence_score


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2,
                 num_classes=4, expert_count=4, top_k=2, enable_layer_skip=True,
                 confidence_thresholds=None):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.enable_layer_skip = enable_layer_skip

        if confidence_thresholds is None:
            confidence_thresholds = [0.5] * num_layers

        # Create a list of MoE layers with LayerSkip
        self.layers = nn.ModuleList([
            MoELayerWithSkip(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim * 2,
                expert_count=expert_count,
                top_k=top_k,
                enable_layer_skip=enable_layer_skip,
                confidence_threshold=confidence_thresholds[i]
            ) for i in range(num_layers)
        ])

        # Auxiliary classifiers for early exit loss
        self.aux_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])

        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def train(self, mode=True):
        super().train(mode)
        # Set training mode for all layers
        for layer in self.layers:
            layer.training_mode = mode
        return self

    def eval(self):
        super().eval()
        # Set eval mode for all layers
        for layer in self.layers:
            layer.training_mode = False
        return self

    def forward(self, x, return_aux_logits=False):
        # Initial layer
        x = F.relu(self.input_layer(x))

        aux_logits = []
        layer_exits = []
        confidence_scores = []

        # Process through MoE layers
        for i, (layer, aux_cls) in enumerate(zip(self.layers, self.aux_classifiers)):
            x, early_exit, conf = layer(x)

            # Store metrics
            layer_exits.append(early_exit)
            confidence_scores.append(conf)

            # Compute auxiliary logits for this layer
            aux_logits.append(aux_cls(x))

            # Exit early if enabled and confidence is high
            if early_exit and self.enable_layer_skip and not self.training:
                if return_aux_logits:
                    return aux_logits[i], layer_exits, confidence_scores, i
                return aux_logits[i], layer_exits, confidence_scores, i

        # Final prediction if no early exit
        logits = self.classifier(x)

        if return_aux_logits:
            return logits, layer_exits, confidence_scores, len(self.layers), aux_logits
        return logits, layer_exits, confidence_scores, len(self.layers)


# Generate synthetic classification dataset
def generate_synthetic_data(num_samples=10000, input_dim=64, num_classes=4):
    """Generate synthetic data for a classification task"""
    X = np.random.randn(num_samples, input_dim)

    # Generate classes based on different regions in the feature space
    y = np.zeros(num_samples, dtype=np.int64)

    # Class 0: Points in the first quadrant (all positive)
    # Class 1: Points in the second quadrant (first half negative, second half positive)
    # Class 2: Points in the third quadrant (all negative)
    # Class 3: Points in the fourth quadrant (first half positive, second half negative)

    for i in range(num_samples):
        first_half_sum = np.sum(X[i, :input_dim // 2])
        second_half_sum = np.sum(X[i, input_dim // 2:])

        if first_half_sum > 0 and second_half_sum > 0:
            y[i] = 0
        elif first_half_sum < 0 and second_half_sum > 0:
            y[i] = 1
        elif first_half_sum < 0 and second_half_sum < 0:
            y[i] = 2
        else:
            y[i] = 3

    # Add some noise to make it more challenging
    noise_indices = np.random.choice(num_samples, size=int(num_samples * 0.1), replace=False)
    y[noise_indices] = np.random.randint(0, num_classes, size=len(noise_indices))

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader, input_dim


# Training function with early exit loss
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, aux_loss_weight=0.3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []
    early_exit_stats = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass with auxiliary logits
            logits, _, _, _, aux_logits = model(data, return_aux_logits=True)

            # Main loss
            main_loss = criterion(logits, target)

            # Auxiliary losses
            aux_losses = [criterion(aux_logit, target) for aux_logit in aux_logits]

            # Combine losses with weighting
            # We weight earlier layers less since they have less information
            weighted_aux_losses = [aux_loss_weight * (idx + 1) / len(aux_losses) * loss
                                   for idx, loss in enumerate(aux_losses)]

            # Total loss
            loss = main_loss + sum(weighted_aux_losses)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        train_losses.append(total_loss / len(train_loader))

        # Validation
        model.eval()
        correct = 0
        layer_exits_count = [0] * (len(model.layers) + 1)  # +1 for final layer

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                # Forward pass
                logits, layer_exits, confidences, exit_layer = model(data)

                # Count exits per layer
                layer_exits_count[exit_layer] += 1

                # Get predictions
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(val_loader.dataset)
        val_accuracies.append(accuracy)

        # Calculate percentage of early exits per layer
        early_exit_percentages = [count / len(val_loader.dataset) * 100 for count in layer_exits_count]
        early_exit_stats.append(early_exit_percentages)

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, '
              f'Validation Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s')
        print(f'Early exits per layer: {early_exit_percentages}')

    return train_losses, val_accuracies, early_exit_stats


# Function to evaluate and visualize results
def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Regular evaluation
    correct = 0
    layer_exits_count = [0] * (len(model.layers) + 1)  # +1 for final layer
    confidence_per_layer = [[] for _ in range(len(model.layers) + 1)]

    # Timing
    inference_times = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Time inference
            start_time = time.time()
            logits, layer_exits, confidences, exit_layer = model(data)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Count exits per layer
            layer_exits_count[exit_layer] += 1

            # Store confidence scores
            for i, conf in enumerate(confidences):
                if i < exit_layer:
                    confidence_per_layer[i].append(conf)

            # Get predictions
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    avg_inference_time = sum(inference_times) / len(inference_times)

    print(f'Test accuracy: {accuracy:.2f}%')
    print(f'Average inference time: {avg_inference_time * 1000:.2f} ms per batch')

    # Calculate percentage of early exits per layer
    exit_counts = []
    for i, count in enumerate(layer_exits_count):
        layer_name = f"Layer {i}" if i < len(model.layers) else "Final"
        exit_pct = count / len(test_loader.dataset) * 100
        exit_counts.append((layer_name, exit_pct))
        print(f'{layer_name}: {exit_pct:.2f}% of samples')

    # Calculate average confidence per layer
    avg_confidence = []
    for i, confidences in enumerate(confidence_per_layer):
        if confidences:
            layer_name = f"Layer {i}" if i < len(model.layers) else "Final"
            avg_conf = sum(confidences) / len(confidences)
            avg_confidence.append((layer_name, avg_conf))
            print(f'{layer_name} average confidence: {avg_conf:.4f}')

    # Plot early exit distribution
    plt.figure(figsize=(10, 5))
    labels, values = zip(*exit_counts)
    plt.bar(labels, values)
    plt.title('Early Exit Distribution')
    plt.ylabel('Percentage of Samples')
    plt.ylim(0, 100)
    plt.savefig('early_exit_distribution.png')

    # Plot average confidence per layer
    if avg_confidence:
        plt.figure(figsize=(10, 5))
        labels, values = zip(*avg_confidence)
        plt.bar(labels, values)
        plt.title('Average Confidence per Layer')
        plt.ylabel('Confidence Score')
        plt.ylim(0, 1)
        plt.savefig('confidence_per_layer.png')

    return accuracy, avg_inference_time, exit_counts, avg_confidence


# Compare with and without LayerSkip
def compare_layerskip_performance():
    # Generate synthetic data
    print("Generating synthetic data...")
    train_loader, test_loader, input_dim = generate_synthetic_data(num_samples=10000, input_dim=64)

    # Create models
    model_with_skip = SimpleClassifier(
        input_dim=input_dim,
        enable_layer_skip=True,
        confidence_thresholds=[0.7, 0.8]  # Higher thresholds for later layers
    )

    model_without_skip = SimpleClassifier(
        input_dim=input_dim,
        enable_layer_skip=False
    )

    # Train both models
    print("Training model with LayerSkip...")
    train_losses_skip, val_accuracies_skip, exit_stats_skip = train_model(
        model_with_skip, train_loader, test_loader, epochs=5
    )

    print("\nTraining model without LayerSkip...")
    train_losses_no_skip, val_accuracies_no_skip, _ = train_model(
        model_without_skip, train_loader, test_loader, epochs=5
    )

    # Evaluate both models
    print("\nEvaluating model with LayerSkip...")
    accuracy_skip, time_skip, exits_skip, _ = evaluate_model(model_with_skip, test_loader)

    print("\nEvaluating model without LayerSkip...")
    accuracy_no_skip, time_no_skip, _, _ = evaluate_model(model_without_skip, test_loader)

    # Compare results
    print("\nComparison:")
    print(f"LayerSkip: Accuracy={accuracy_skip:.2f}%, Inference time={time_skip * 1000:.2f}ms")
    print(f"No LayerSkip: Accuracy={accuracy_no_skip:.2f}%, Inference time={time_no_skip * 1000:.2f}ms")
    print(f"Speedup: {time_no_skip / time_skip:.2f}x")

    # Plot training loss comparison
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_skip, label='With LayerSkip')
    plt.plot(train_losses_no_skip, label='Without LayerSkip')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss_comparison.png')

    # Plot validation accuracy comparison
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies_skip, label='With LayerSkip')
    plt.plot(val_accuracies_no_skip, label='Without LayerSkip')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('validation_accuracy_comparison.png')

    return model_with_skip, model_without_skip


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run comparison
    print("Starting model comparison...")
    model_with_skip, model_without_skip = compare_layerskip_performance()

    # Save trained models
    torch.save(model_with_skip.state_dict(), 'model_with_layerskip.pt')
    torch.save(model_without_skip.state_dict(), 'model_without_layerskip.pt')

    print("Completed! Models saved.")