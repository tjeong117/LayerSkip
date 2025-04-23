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

    def forward(self, x):
        # Layer normalization and save residual
        residual = x
        x_norm = self.layer_norm(x)

        # Compute confidence score from the normalized input *before* the block
        # (Though applying after the block might also be valid)
        confidence = torch.zeros(x.shape[0], 1, device=x.device) # Default confidence is 0
        if self.enable_layer_skip:
            # Sigmoid ensures confidence is between 0 and 1
            confidence = torch.sigmoid(self.confidence_predictor(x_norm))

        # --- MoE Routing and Expert Processing --- #

        # Get router logits from normalized input
        router_logits = self.router(x_norm)

        # Apply expert dropout during training
        if self.training and self.expert_dropout_rate > 0:
            expert_mask = torch.rand(self.expert_count, device=x.device) > self.expert_dropout_rate
            if not expert_mask.any():
                expert_mask[torch.randint(0, self.expert_count, (1,))] = True
            router_logits = router_logits.masked_fill(~expert_mask.unsqueeze(0), -1e10)

        router_probs = F.softmax(router_logits, dim=-1)
        vals, indices = torch.topk(router_probs, self.top_k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True)

        # Process through experts
        batch_size = x.shape[0]
        combined_output = torch.zeros_like(x) # Use original shape

        # TODO: Optimize this expert computation (e.g., using scatter/gather)
        for b in range(batch_size):
            for k in range(self.top_k):
                expert_idx = indices[b, k].item()
                weight = vals[b, k].item()
                # Pass the *normalized* input to experts
                expert_output = self.experts[expert_idx](x_norm[b].unsqueeze(0))
                combined_output[b] += weight * expert_output.squeeze(0)

        # --- Final Output Calculation --- #

        # Add residual to the expert output
        output = residual + combined_output

        # Return the main output, the per-sample confidence, and the original residual
        # We no longer return routing metrics etc. by default to simplify
        return output, confidence, residual


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2,
                 num_classes=4, expert_count=4, top_k=2, enable_layer_skip=True,
                 confidence_thresholds=None, rotation_step=0):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.enable_layer_skip = enable_layer_skip
        self.num_layers = num_layers
        self.rotation_step = rotation_step if rotation_step > 0 else 0
        self.current_rotation_offset = 0

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

    def advance_rotation(self):
        """Advances the rotation offset for the curriculum."""
        if self.rotation_step > 0:
            self.current_rotation_offset = (self.current_rotation_offset + 1) % self.rotation_step

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
        batch_size = x.shape[0]
        device = x.device
        hidden_dim = self.input_layer.out_features
        num_classes = self.classifier.out_features

        # Initial layer
        x = F.relu(self.input_layer(x))

        # Initialization for per-sample tracking
        final_logits = torch.zeros(batch_size, num_classes, device=device)
        # Default exit layer is num_layers (meaning final classifier)
        exit_layer = torch.full((batch_size,), self.num_layers, dtype=torch.long, device=device)
        has_exited = torch.zeros(batch_size, dtype=torch.bool, device=device)
        all_aux_logits = [] # Stores aux logits from *all* layers for loss calculation
        # Stores confidence scores for samples *at the point they exited* (for potential analysis)
        exit_confidences = torch.zeros(batch_size, device=device)

        # Process through MoE layers
        for i, (layer, aux_cls) in enumerate(zip(self.layers, self.aux_classifiers)):
            # Pass current state through the layer
            # Layer returns: output, confidence, residual_before_layer
            current_x, current_confidence, _ = layer(x)

            # Compute auxiliary logits using the output of the MoE layer
            # We compute this for *all* samples, even those that might exit now,
            # because aux loss might still be needed for training.
            current_aux_logits = aux_cls(current_x)
            all_aux_logits.append(current_aux_logits)

            # --- Per-Sample Early Exit Check (only during evaluation) ---
            if not self.training and self.enable_layer_skip:
                # Determine which samples *could* exit based on confidence
                # Squeeze confidence tensor for boolean indexing
                should_exit = (current_confidence.squeeze() > layer.confidence_threshold)

                # Identify samples exiting *at this specific layer*
                # (i.e., they meet threshold AND haven't exited yet)
                exiting_now = should_exit & ~has_exited

                if exiting_now.any():
                    # Store the aux logits of *this* layer as the final prediction for these samples
                    final_logits[exiting_now] = current_aux_logits[exiting_now]
                    # Record the exit layer index
                    exit_layer[exiting_now] = i
                    # Store their confidence score at exit
                    exit_confidences[exiting_now] = current_confidence[exiting_now].squeeze()
                    # Mark these samples as exited
                    has_exited[exiting_now] = True

            # Update the state for the next layer iteration
            # Samples keep propagating even if marked as exited, simplifies logic,
            # but their final_logits are already set.
            x = current_x

            # Optimization: if all samples have exited, break the loop
            if not self.training and has_exited.all():
                break

        # --- Final Classifier for Non-Exited Samples --- #

        # Identify samples that went through all layers
        non_exited_mask = ~has_exited
        if non_exited_mask.any():
            # Apply the final classifier only to these samples
            final_layer_logits = self.classifier(x[non_exited_mask])
            # Store these logits in the final_logits tensor
            final_logits[non_exited_mask] = final_layer_logits
            # Confidence for these is conventionally 1.0 (or handled differently if needed)
            exit_confidences[non_exited_mask] = 1.0 # Placeholder value

        # Return final logits, per-sample exit layer indices, and all computed aux logits
        if return_aux_logits:
            # Note: exit_confidences are also available if needed for analysis
            return final_logits, exit_layer, all_aux_logits
        else:
            # Standard return for evaluation might just be logits and exit layers
            return final_logits, exit_layer


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

        # Advance rotation if applicable
        if hasattr(model, 'advance_rotation') and not is_naive:
            model.advance_rotation()
            if model.rotation_step > 0:
                print(f"Epoch {epoch}: Rotational Curriculum - Active offset: {model.current_rotation_offset}/{model.rotation_step}")

        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass
            if is_naive:
                logits = model(data)
                loss = criterion(logits, target)
            else:
                # Forward pass now returns: final_logits, exit_layer, all_aux_logits
                # We need all_aux_logits for the loss calculation
                final_logits, exit_layer, all_aux_logits = model(data, return_aux_logits=True)

                # Main loss is calculated on the final logits returned by the model.
                # These logits correspond to the prediction made at the exit layer (or final layer)
                # for each sample.
                main_loss = criterion(final_logits, target)

                # Auxiliary losses - Apply Rotational Curriculum
                # This uses the list of auxiliary logits from *all* layers
                aux_losses = []
                active_indices = []
                if hasattr(model, 'rotation_step') and model.rotation_step > 0:
                    # Rotational curriculum enabled
                    for i in range(len(all_aux_logits)):
                        if i % model.rotation_step == model.current_rotation_offset:
                            # Calculate loss for the specific aux layer's output
                            aux_losses.append(criterion(all_aux_logits[i], target))
                            active_indices.append(i)
                else:
                    # Rotation disabled or not applicable - use all aux layers
                    aux_losses = [criterion(aux_logit, target) for aux_logit in all_aux_logits]
                    active_indices = list(range(len(all_aux_logits)))

                # Combine losses with weighting (apply only to active layers)
                weighted_aux_losses = []
                if aux_losses: # Only if there are active aux losses
                    # Weighting uses the original layer index (from active_indices)
                    # Normalization factor remains the total number of aux layers
                    num_total_aux_layers = len(all_aux_logits)
                    for idx, aux_loss in enumerate(aux_losses):
                        original_layer_index = active_indices[idx]
                        weight = aux_loss_weight * (original_layer_index + 1) / num_total_aux_layers
                        weighted_aux_losses.append(weight * aux_loss)

                # Total loss is main loss + weighted sum of *active* auxiliary losses
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
                    # For non-naive models, get final_logits and exit_layer
                    final_logits, exit_layer = model(data) # No need for aux_logits here
                    # Count exits per layer using the exit_layer tensor
                    # Ensure exit_layer is on CPU for bincount
                    exit_counts = torch.bincount(exit_layer.cpu(), minlength=len(model.layers) + 1)
                    for layer_idx, count in enumerate(exit_counts):
                        layer_exits_count[layer_idx] += count.item()

                    # Get predictions from the final logits
                    pred = final_logits.argmax(dim=1, keepdim=True)

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


# Function to evaluate and visualize results
def evaluate_model(model, test_loader, is_naive=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0
    all_inference_times = []
    all_exit_layers = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            total_samples += batch_size

            # Time inference
            start_time = time.time()
            if is_naive:
                final_logits = model(data)
                # Naive model always "exits" at the final layer
                num_layers = len(model.layers) # Need to figure out num_layers if possible
                # Or find a better way if NaiveClassifier doesn't store num_layers
                # Assuming it has a .layers Sequential block
                try:
                   # Count Linear layers in the sequential block
                   num_linear_layers = sum(1 for m in model.layers if isinstance(m, nn.Linear))
                except AttributeError:
                   num_linear_layers = 1 # Fallback
                exit_layer = torch.full((batch_size,), num_linear_layers, dtype=torch.long, device=device)

            else:
                # Model returns final_logits based on exit layer, and the exit layer index per sample
                final_logits, exit_layer = model(data)

            inference_time = time.time() - start_time
            all_inference_times.append(inference_time * batch_size) # Store total time for the batch

            # Get predictions from the final logits
            pred = final_logits.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()

            # Store exit layers for later analysis
            all_exit_layers.append(exit_layer.cpu()) # Move to CPU immediately

    # Consolidate results
    accuracy = 100. * total_correct / total_samples
    # Average inference time per sample
    avg_inference_time_ms = (sum(all_inference_times) / total_samples) * 1000

    print(f'Test accuracy: {accuracy:.2f}%')
    print(f'Average inference time: {avg_inference_time_ms:.2f} ms per sample')

    # Calculate and print exit statistics
    if not is_naive:
        num_layers = len(model.layers)
        final_exit_layer_index = num_layers # Index representing the final classifier

        all_exit_layers_tensor = torch.cat(all_exit_layers)
        layer_exits_count = torch.bincount(all_exit_layers_tensor, minlength=num_layers + 1)

        exit_distribution = []
        print("Exit Distribution:")
        for i, count in enumerate(layer_exits_count):
            layer_name = f"Layer {i}" if i < num_layers else "Final"
            exit_pct = count.item() / total_samples * 100
            exit_distribution.append((layer_name, exit_pct))
            print(f'  {layer_name}: {count.item()} samples ({exit_pct:.2f}%)')

        # Check sum is 100%
        total_pct = sum(pct for _, pct in exit_distribution)
        print(f'  Total Percentage: {total_pct:.2f}%') # Should be 100%

        # Plot early exit distribution
        plt.figure(figsize=(10, 5))
        labels, values = zip(*exit_distribution)
        plt.bar(labels, values)
        plt.title('Exit Distribution Across Layers')
        plt.ylabel('Percentage of Samples Exiting')
        plt.ylim(0, 100)
        plt.savefig('early_exit_distribution.png')
        print("Saved exit distribution plot to early_exit_distribution.png")

        return accuracy, avg_inference_time_ms, exit_distribution, None # Return None for old confidence tuple

    # Return for naive model
    # Need to return dummy exit info or adjust call site
    return accuracy, avg_inference_time_ms


# Compare all models (with LayerSkip, without LayerSkip, and naive)
def compare_model_performance():
    # Generate synthetic data
    print("Generating synthetic data...")
    train_loader, test_loader, input_dim = generate_synthetic_data(num_samples=10000, input_dim=64)

    # Create models
    model_with_skip = SimpleClassifier(
        input_dim=input_dim,
        enable_layer_skip=True,
        confidence_thresholds=[0.5, 0.55],  # Lowered thresholds
        rotation_step=3
    )

    model_without_skip = SimpleClassifier(
        input_dim=input_dim,
        enable_layer_skip=False
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

    print("\nTraining naive model...")
    train_losses_naive, val_accuracies_naive, _, epoch_times_naive, avg_time_naive = train_model(
        naive_model, train_loader, test_loader, epochs=5, is_naive=True
    )

    # Evaluate all models
    print("\nEvaluating model with LayerSkip...")
    # Now returns: accuracy, time_ms, exit_distribution, None
    accuracy_skip, time_skip, exits_skip, _ = evaluate_model(model_with_skip, test_loader)

    print("\nEvaluating model without LayerSkip...")
    # Now returns: accuracy, time_ms, exit_distribution, None
    accuracy_no_skip, time_no_skip, _, _ = evaluate_model(model_without_skip, test_loader)

    print("\nEvaluating naive model...")
    # Now returns: accuracy, time_ms
    accuracy_naive, time_naive = evaluate_model(naive_model, test_loader, is_naive=True)

    # Compare results
    print("\nComparison:")
    # Note: time_skip, time_no_skip, time_naive are now in ms per sample
    print(
        f"LayerSkip: Accuracy={accuracy_skip:.2f}%, Inference time={time_skip:.2f}ms/sample, Training time/epoch={avg_time_skip:.2f}s")
    print(
        f"No LayerSkip: Accuracy={accuracy_no_skip:.2f}%, Inference time={time_no_skip:.2f}ms/sample, Training time/epoch={avg_time_no_skip:.2f}s")
    print(
        f"Naive model: Accuracy={accuracy_naive:.2f}%, Inference time={time_naive:.2f}ms/sample, Training time/epoch={avg_time_naive:.2f}s")

    # Speedup comparison might need adjustment based on interpretation (per sample vs per batch)
    if time_skip > 0:
        print(f"Inference Speedup (LayerSkip vs. Naive, per sample): {time_naive / time_skip:.2f}x")
    else:
        print("Inference Speedup (LayerSkip vs. Naive): N/A (LayerSkip time is zero)")
    if avg_time_skip > 0:
        print(f"Training Speedup (LayerSkip vs. Naive, per epoch): {avg_time_naive / avg_time_skip:.2f}x")
    else:
        print("Training Speedup (LayerSkip vs. Naive): N/A (LayerSkip time is zero)")

    # Plot training loss comparison
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_skip, label='With LayerSkip')
    plt.plot(train_losses_no_skip, label='Without LayerSkip')
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
    plt.plot(epochs, epoch_times_naive, marker='^', label='Naive')
    plt.title('Training Time Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_time_comparison.png')

    # Bar chart for average training and inference times
    plt.figure(figsize=(12, 6))
    models = ['LayerSkip', 'No LayerSkip', 'Naive']
    train_times = [avg_time_skip, avg_time_no_skip, avg_time_naive]
    # Use the per-sample inference times in ms directly
    infer_times = [time_skip, time_no_skip, time_naive]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    bar1 = ax1.bar(x - width / 2, train_times, width, label='Avg. Training Time (s/epoch)', color='skyblue')
    bar2 = ax2.bar(x + width / 2, infer_times, width, label='Avg. Inference Time (ms/sample)', color='salmon')

    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Training Time (seconds)')
    ax2.set_ylabel('Inference Time (ms/sample)')

    ax1.set_xticks(x)
    ax1.set_xticklabels(models)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Training vs Inference Time Comparison')
    plt.tight_layout()
    plt.savefig('time_comparison.png')

    return model_with_skip, model_without_skip, naive_model


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run comparison
    print("Starting model comparison...")
    model_with_skip, model_without_skip, naive_model = compare_model_performance()

    # Save trained models
    torch.save(model_with_skip.state_dict(), 'model_with_layerskip.pt')
    torch.save(model_without_skip.state_dict(), 'model_without_layerskip.pt')
    torch.save(naive_model.state_dict(), 'naive_model.pt')

    print("Completed! Models saved.")