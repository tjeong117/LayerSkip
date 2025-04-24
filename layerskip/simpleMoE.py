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


class SimpleClassifierRotation(SimpleClassifier):
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2,
                 num_classes=4, expert_count=4, top_k=2, enable_layer_skip=True,
                 confidence_thresholds=None, rotation_interval=50):
        super().__init__(input_dim, hidden_dim, num_layers, num_classes, 
                         expert_count, top_k, enable_layer_skip, confidence_thresholds)
        
        # Rotation interval (in batches) for early exit loss
        self.rotation_interval = rotation_interval
        
    def get_active_layers(self, batch_idx, num_active_layers=None):
        if num_active_layers is None:
            num_active_layers = max(1, len(self.layers) // 2)  # Default: activate half of the layers
        
        # Determine which set of layers is active for this batch
        rotation_step = (batch_idx // self.rotation_interval) % len(self.layers)
        
        # Create a circular mask of active layers starting from rotation_step
        active_layers = [(rotation_step + i) % len(self.layers) for i in range(num_active_layers)]
        
        return active_layers


# Naive model for comparison
class NaiveClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2, num_classes=4):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Create layers without MoE or LayerSkip - simple feed-forward
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.layers(x)
        return self.classifier(x)


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
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, aux_loss_weight=0.3, is_naive=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []
    early_exit_stats = []
    epoch_times = []  # Record time per epoch

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass
            if is_naive:
                logits = model(data)
                loss = criterion(logits, target)
            else:
                # Forward pass with auxiliary logits for MoE model
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

        # For non-naive models, we track early exits
        if not is_naive:
            layer_exits_count = [0] * (len(model.layers) + 1)  # +1 for final layer

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                # Forward pass
                if is_naive:
                    logits = model(data)
                    pred = logits.argmax(dim=1, keepdim=True)
                else:
                    logits, layer_exits, confidences, exit_layer = model(data)
                    # Count exits per layer
                    layer_exits_count[exit_layer] += 1
                    pred = logits.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(val_loader.dataset)
        val_accuracies.append(accuracy)

        # Calculate early exit stats if not naive model
        if not is_naive:
            early_exit_percentages = [count / len(val_loader.dataset) * 100 for count in layer_exits_count]
            early_exit_stats.append(early_exit_percentages)
            print(f'Early exits per layer: {early_exit_percentages}')

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(f'Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, '
              f'Validation Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s')

    # Calculate average epoch time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"Average epoch time: {avg_epoch_time:.2f}s")

    return train_losses, val_accuracies, early_exit_stats, epoch_times, avg_epoch_time


# Training function with rotational early exit loss 
def train_model_with_rotational_curriculum(model, train_loader, val_loader, epochs=5, lr=0.001, 
                                           aux_loss_weight=0.3, is_naive=False,
                                           rotation_interval=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training with rotational curriculum, rotation interval: {rotation_interval} batches")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []
    early_exit_stats = []
    epoch_times = []
    
    # Track which layers are active for early exit in each iteration
    active_layer_history = []
    
    step_counter = 0
    
    # Number of layers in the model
    if not is_naive:
        num_layers = len(model.layers)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass
            if is_naive:
                logits = model(data)
                loss = criterion(logits, target)
            else:
                # Forward pass with auxiliary logits for MoE model
                logits, _, _, _, aux_logits = model(data, return_aux_logits=True)

                # Main loss
                main_loss = criterion(logits, target)

                # Rotational curriculum - determine active layers
                rotation_phase = (step_counter // rotation_interval) % num_layers
                
                # Select which layers have active early exit loss
                # In rotational curriculum, we enable early exit loss for a subset of layers
                # that rotates through the network over time
                
                # Calculate how many layers to activate in this phase (at least 1)
                active_count = max(1, num_layers // 2)
                
                # Generate indices of active layers, wrapping around if needed
                active_layers = [(rotation_phase + i) % num_layers for i in range(active_count)]
                
                # Record which layers are active in this step
                if batch_idx % 50 == 0:
                    active_layer_history.append(active_layers)
                    print(f"Step {step_counter}, Rotation phase {rotation_phase}, Active layers: {active_layers}")

                # Auxiliary losses - only apply to active layers
                aux_loss_sum = 0
                for idx, aux_logit in enumerate(aux_logits):
                    if idx in active_layers:
                        # Weight the loss based on layer depth
                        weight = aux_loss_weight * (idx + 1) / len(aux_logits)
                        aux_loss = weight * criterion(aux_logit, target)
                        aux_loss_sum += aux_loss

                # Total loss
                loss = main_loss + aux_loss_sum

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_counter += 1

            if batch_idx % 50 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        train_losses.append(total_loss / len(train_loader))

        # Validation
        model.eval()
        correct = 0

        # For non-naive models, we track early exits
        if not is_naive:
            layer_exits_count = [0] * (len(model.layers) + 1)  # +1 for final layer

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                # Forward pass
                if is_naive:
                    logits = model(data)
                    pred = logits.argmax(dim=1, keepdim=True)
                else:
                    logits, layer_exits, confidences, exit_layer = model(data)
                    # Count exits per layer
                    layer_exits_count[exit_layer] += 1
                    pred = logits.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(val_loader.dataset)
        val_accuracies.append(accuracy)

        # Calculate early exit stats if not naive model
        if not is_naive:
            early_exit_percentages = [count / len(val_loader.dataset) * 100 for count in layer_exits_count]
            early_exit_stats.append(early_exit_percentages)
            print(f'Early exits per layer: {early_exit_percentages}')

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(f'Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, '
              f'Validation Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s')

    # Calculate average epoch time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"Average epoch time: {avg_epoch_time:.2f}s")

    # Plot the active layer history to visualize the rotation
    if not is_naive and active_layer_history:
        plt.figure(figsize=(12, 6))
        for step_idx, active_layers in enumerate(active_layer_history):
            for layer in active_layers:
                plt.scatter(step_idx, layer, c='blue', s=20)
        plt.title('Rotational Curriculum - Active Layers Over Time')
        plt.xlabel('Training Steps (every 50 batches)')
        plt.ylabel('Layer Index')
        plt.yticks(range(num_layers))
        plt.grid(True, alpha=0.3)
        plt.savefig('rotational_curriculum_visualization.png')
        plt.close()

    return train_losses, val_accuracies, early_exit_stats, epoch_times, avg_epoch_time, active_layer_history


# Function to evaluate and visualize results
def evaluate_model(model, test_loader, is_naive=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Regular evaluation
    correct = 0

    if not is_naive:
        layer_exits_count = [0] * (len(model.layers) + 1)  # +1 for final layer
        confidence_per_layer = [[] for _ in range(len(model.layers) + 1)]

    # Timing
    inference_times = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Time inference
            start_time = time.time()
            if is_naive:
                logits = model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                pred = logits.argmax(dim=1, keepdim=True)
            else:
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

    if not is_naive:
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

    return accuracy, avg_inference_time


# Compare all models (with LayerSkip, without LayerSkip, naive, and with rotation)
def compare_model_performance():
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
    
    model_with_rotation = SimpleClassifier(
        input_dim=input_dim,
        enable_layer_skip=True,
        confidence_thresholds=[0.7, 0.8],
    )

    naive_model = NaiveClassifier(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2
    )

    # Train all models
    print("Training model with LayerSkip...")
    train_losses_skip, val_accuracies_skip, exit_stats_skip, epoch_times_skip, avg_time_skip = train_model(
        model_with_skip, train_loader, test_loader, epochs=5
    )

    print("\nTraining model without LayerSkip...")
    train_losses_no_skip, val_accuracies_no_skip, _, epoch_times_no_skip, avg_time_no_skip = train_model(
        model_without_skip, train_loader, test_loader, epochs=5
    )
    
    print("\nTraining model with LayerSkip and Rotational Curriculum...")
    train_losses_rotation, val_accuracies_rotation, exit_stats_rotation, epoch_times_rotation, avg_time_rotation, active_layer_history = train_model_with_rotational_curriculum(
        model_with_rotation, train_loader, test_loader, epochs=5,
        rotation_interval=50  # Rotate every 50 batches
    )

    print("\nTraining naive model...")
    train_losses_naive, val_accuracies_naive, _, epoch_times_naive, avg_time_naive = train_model(
        naive_model, train_loader, test_loader, epochs=5, is_naive=True
    )

    # Evaluate all models
    print("\nEvaluating model with LayerSkip...")
    accuracy_skip, time_skip, exits_skip, _ = evaluate_model(model_with_skip, test_loader)

    print("\nEvaluating model without LayerSkip...")
    accuracy_no_skip, time_no_skip, _, _ = evaluate_model(model_without_skip, test_loader)
    
    print("\nEvaluating model with LayerSkip and Rotational Curriculum...")
    accuracy_rotation, time_rotation, exits_rotation, _ = evaluate_model(model_with_rotation, test_loader)

    print("\nEvaluating naive model...")
    accuracy_naive, time_naive = evaluate_model(naive_model, test_loader, is_naive=True)

    # Compare results
    print("\nComparison:")
    print(
        f"LayerSkip: Accuracy={accuracy_skip:.2f}%, Inference time={time_skip * 1000:.2f}ms, Training time/epoch={avg_time_skip:.2f}s")
    print(
        f"No LayerSkip: Accuracy={accuracy_no_skip:.2f}%, Inference time={time_no_skip * 1000:.2f}ms, Training time/epoch={avg_time_no_skip:.2f}s")
    print(
        f"LayerSkip with Rotation: Accuracy={accuracy_rotation:.2f}%, Inference time={time_rotation * 1000:.2f}ms, Training time/epoch={avg_time_rotation:.2f}s")
    print(
        f"Naive model: Accuracy={accuracy_naive:.2f}%, Inference time={time_naive * 1000:.2f}ms, Training time/epoch={avg_time_naive:.2f}s")
    print(f"Inference Speedup vs. Naive: {time_naive / time_skip:.2f}x")
    print(f"Training Speedup vs. Naive: {avg_time_naive / avg_time_skip:.2f}x")
    print(f"Training Speedup Rotation vs. Standard: {avg_time_skip / avg_time_rotation:.2f}x")

    # Plot training loss comparison
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_skip, label='With LayerSkip')
    plt.plot(train_losses_no_skip, label='Without LayerSkip')
    plt.plot(train_losses_rotation, label='With Rotation')
    plt.plot(train_losses_naive, label='Naive')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss_comparison.png')

    # Plot validation accuracy comparison
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies_skip, label='With LayerSkip')
    plt.plot(val_accuracies_no_skip, label='Without LayerSkip')
    plt.plot(val_accuracies_rotation, label='With Rotation')
    plt.plot(val_accuracies_naive, label='Naive')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('validation_accuracy_comparison.png')

    # Plot training time comparison
    plt.figure(figsize=(10, 5))
    epochs = range(5)
    plt.plot(epochs, epoch_times_skip, marker='o', label='With LayerSkip')
    plt.plot(epochs, epoch_times_no_skip, marker='s', label='Without LayerSkip')
    plt.plot(epochs, epoch_times_rotation, marker='d', label='With Rotation')
    plt.plot(epochs, epoch_times_naive, marker='^', label='Naive')
    plt.title('Training Time Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_time_comparison.png')

    # Bar chart for average training and inference times
    plt.figure(figsize=(14, 6))
    models = ['LayerSkip', 'No LayerSkip', 'Rotation', 'Naive']
    train_times = [avg_time_skip, avg_time_no_skip, avg_time_rotation, avg_time_naive]
    infer_times = [time_skip * 1000, time_no_skip * 1000, time_rotation * 1000, time_naive * 1000]  # Convert to ms

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    bar1 = ax1.bar(x - width / 2, train_times, width, label='Avg. Training Time (s/epoch)', color='skyblue')
    bar2 = ax2.bar(x + width / 2, infer_times, width, label='Avg. Inference Time (ms/batch)', color='salmon')

    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Training Time (seconds)')
    ax2.set_ylabel('Inference Time (milliseconds)')

    ax1.set_xticks(x)
    ax1.set_xticklabels(models)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Training vs Inference Time Comparison')
    plt.tight_layout()
    plt.savefig('time_comparison.png')

    return model_with_skip, model_without_skip, model_with_rotation, naive_model


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run comparison
    print("Starting model comparison...")
    model_with_skip, model_without_skip, model_with_rotation, naive_model = compare_model_performance()

    # Save trained models
    torch.save(model_with_skip.state_dict(), 'model_with_layerskip.pt')
    torch.save(model_without_skip.state_dict(), 'model_without_layerskip.pt')
    torch.save(model_with_rotation.state_dict(), 'model_with_rotation.pt')
    torch.save(naive_model.state_dict(), 'naive_model.pt')

    print("Completed! Models saved.")