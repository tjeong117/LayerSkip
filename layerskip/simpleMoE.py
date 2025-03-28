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
                 expert_dropout_rate=0.2, auxiliary_loss_weight=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.expert_count = expert_count
        self.top_k = top_k
        self.enable_layer_skip = enable_layer_skip
        self.confidence_threshold = confidence_threshold
        self.expert_dropout_rate = expert_dropout_rate
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.training_mode = True

        # Router network with improved complexity awareness
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, expert_count)
        )

        # Experts - enhanced feed-forward networks with gating
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),  # Add dropout within experts for robustness
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(expert_count)
        ])

        # LayerSkip components - enhanced confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Expert utilization tracking for load balancing
        self.register_buffer('expert_utilization', torch.zeros(expert_count))
        self.layer_norm = nn.LayerNorm(input_dim)

        # Sample complexity estimator - helps determine if input is "easy" or "hard"
        self.complexity_estimator = nn.Linear(input_dim, 1)

    def reset_utilization_stats(self):
        """Reset the expert utilization tracking."""
        self.expert_utilization.zero_()

    def forward(self, x, return_metrics=False):
        # Layer normalization
        residual = x
        x = self.layer_norm(x)

        # Estimate input complexity
        complexity_score = torch.sigmoid(self.complexity_estimator(x))
        batch_size = x.shape[0]

        # Get router logits
        router_logits = self.router(x)

        # Apply expert dropout during training with dynamic rate based on complexity
        if self.training and self.expert_dropout_rate > 0:
            # Dynamic dropout rate based on input complexity
            adjusted_dropout_rate = self.expert_dropout_rate * (1 + 0.5 * complexity_score.mean().item())
            expert_mask = torch.rand(self.expert_count, device=x.device) > adjusted_dropout_rate

            # Ensure at least top_k experts are active
            if expert_mask.sum() < self.top_k:
                inactive_indices = torch.where(~expert_mask)[0]
                to_activate = min(self.top_k - expert_mask.sum().item(), len(inactive_indices))
                if to_activate > 0:
                    indices_to_activate = inactive_indices[torch.randperm(len(inactive_indices))[:to_activate]]
                    expert_mask[indices_to_activate] = True

            # Apply mask to router logits
            router_logits = router_logits.masked_fill(~expert_mask.unsqueeze(0), -1e10)

        # Get routing probabilities and indices
        router_probs = F.softmax(router_logits, dim=-1)

        # Get top-k experts
        vals, indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize the router probabilities
        vals = vals / vals.sum(dim=-1, keepdim=True)

        # Update expert utilization for load balancing
        if self.training:
            for b in range(batch_size):
                for k in range(self.top_k):
                    expert_idx = indices[b, k].item()
                    self.expert_utilization[expert_idx] += vals[b, k].item()

        # Compute early exit confidence if enabled
        early_exit = False
        confidence_score = 0.0

        # Enhanced early exit logic based on complexity and confidence
        if self.enable_layer_skip and not self.training:
            # Predict confidence for early exit
            base_confidence = torch.sigmoid(self.confidence_predictor(x))

            # Adjust confidence based on input complexity (harder inputs get lower confidence)
            adjusted_confidence = base_confidence * (1.0 - 0.5 * complexity_score)
            confidence_score = adjusted_confidence.mean().item()

            # Check if we should exit early - more conservative for complex inputs
            adjusted_threshold = self.confidence_threshold * (1 + 0.2 * complexity_score.mean().item())
            if confidence_score > adjusted_threshold:
                early_exit = True
                # Just return the input if we exit early
                if return_metrics:
                    return residual + x, early_exit, vals, indices, router_probs, confidence_score, complexity_score.mean().item()
                return residual + x, early_exit, confidence_score

        # Process through experts
        # Initialize output tensor
        combined_output = torch.zeros_like(x)

        # Enhanced MoE computation with expert-specific tracking
        expert_outputs = []
        for b in range(batch_size):
            sample_outputs = []
            for k in range(self.top_k):
                expert_idx = indices[b, k].item()
                weight = vals[b, k].item()
                expert_output = self.experts[expert_idx](x[b].unsqueeze(0))
                sample_outputs.append(weight * expert_output)

            # Combine expert outputs for this sample
            combined_sample = torch.sum(torch.stack(sample_outputs, dim=0), dim=0)
            combined_output[b] = combined_sample.squeeze(0)

        # Residual connection
        output = residual + combined_output

        if return_metrics:
            return output, early_exit, vals, indices, router_probs, confidence_score, complexity_score.mean().item()
        return output, early_exit, confidence_score


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2,
                 num_classes=4, expert_count=4, top_k=2, enable_layer_skip=True,
                 confidence_thresholds=None, auxiliary_loss_weight=0.3):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.enable_layer_skip = enable_layer_skip
        self.auxiliary_loss_weight = auxiliary_loss_weight

        if confidence_thresholds is None:
            # Gradually increase thresholds for later layers (more conservative)
            confidence_thresholds = [0.5 + 0.1 * i for i in range(num_layers)]

        # Create a list of MoE layers with LayerSkip
        self.layers = nn.ModuleList([
            MoELayerWithSkip(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim * 2,
                expert_count=expert_count,
                top_k=top_k,
                enable_layer_skip=enable_layer_skip,
                confidence_threshold=confidence_thresholds[i],
                auxiliary_loss_weight=auxiliary_loss_weight
            ) for i in range(num_layers)
        ])

        # Auxiliary classifiers for early exit loss - enhanced with layer-specific adjustments
        self.aux_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
            ) for _ in range(num_layers)
        ])

        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

        # Complexity awareness module to estimate the difficulty of the input
        self.complexity_estimator = nn.Linear(hidden_dim, 1)

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

    def reset_expert_stats(self):
        """Reset expert utilization statistics."""
        for layer in self.layers:
            layer.reset_utilization_stats()

    def get_expert_utilization(self):
        """Get expert utilization across all layers."""
        utilization = []
        for i, layer in enumerate(self.layers):
            utilization.append((f"Layer {i}", layer.expert_utilization.cpu().numpy()))
        return utilization

    def compute_load_balancing_loss(self):
        """Compute a load balancing loss to encourage uniform expert utilization."""
        loss = 0.0
        for layer in self.layers:
            # Normalize utilization to sum to 1
            normalized_util = layer.expert_utilization / (layer.expert_utilization.sum() + 1e-5)
            # Ideal uniform distribution
            uniform_util = torch.ones_like(normalized_util) / layer.expert_count
            # KL divergence from uniform
            loss += F.kl_div(normalized_util.log(), uniform_util, reduction='sum')
        return loss

    def forward(self, x, return_aux_logits=False):
        # Initial layer
        x = F.relu(self.input_layer(x))

        # Estimate input complexity
        complexity = torch.sigmoid(self.complexity_estimator(x)).mean()

        aux_logits = []
        layer_exits = []
        confidence_scores = []
        complexity_scores = []

        # Process through MoE layers with enhanced early exit
        for i, (layer, aux_cls) in enumerate(zip(self.layers, self.aux_classifiers)):
            # Dynamically adjust confidence threshold based on complexity and layer depth
            if not self.training and self.enable_layer_skip:
                layer.confidence_threshold += 0.05 * complexity.item() * (i + 1)

            # Process through layer
            x, early_exit, conf = layer(x)

            # Store metrics
            layer_exits.append(early_exit)
            confidence_scores.append(conf)
            complexity_scores.append(complexity.item())

            # Compute auxiliary logits for this layer
            aux_logits.append(aux_cls(x))

            # Exit early if enabled and confidence is high
            if early_exit and self.enable_layer_skip and not self.training:
                if return_aux_logits:
                    return aux_logits[i], layer_exits, confidence_scores, i, complexity_scores
                return aux_logits[i], layer_exits, confidence_scores, i

        # Final prediction if no early exit
        logits = self.classifier(x)

        if return_aux_logits:
            return logits, layer_exits, confidence_scores, len(self.layers), aux_logits, complexity_scores
        return logits, layer_exits, confidence_scores, len(self.layers)


# Modified train_model function to handle datasets with difficulty levels
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, aux_loss_weight=0.3,
                load_balance_weight=0.1, is_naive=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []
    early_exit_stats = []
    epoch_times = []

    # Track layer-wise exit statistics
    layer_exit_counts = [0] * len(model.layers) if hasattr(model, 'layers') else [0]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        model.reset_expert_stats()  # Reset expert utilization statistics

        start_time = time.time()
        for batch_idx, batch_data in enumerate(train_loader):
            # Handle both 2-element and 3-element batches
            if len(batch_data) == 3:
                data, target, _ = batch_data  # Ignore the difficulty level
            else:
                data, target = batch_data

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass
            if is_naive:
                # Standard MoE model
                logits = model(data)
                loss = criterion(logits, target)
            else:
                # LayerSkip MoE model with auxiliary logits
                logits, _, _, _, aux_logits, complexity_scores = model(data, return_aux_logits=True)

                # Main loss
                main_loss = criterion(logits, target)

                # Auxiliary losses with adaptive weighting based on complexity
                aux_losses = []
                for i, aux_logit in enumerate(aux_logits):
                    # Weight earlier layers less, and adjust based on complexity
                    layer_weight = aux_loss_weight * (i + 1) / len(aux_logits)
                    aux_loss = criterion(aux_logit, target)
                    aux_losses.append(layer_weight * aux_loss)

                # Load balancing loss
                load_balance_loss = model.compute_load_balancing_loss() * load_balance_weight

                # Total loss
                loss = main_loss + sum(aux_losses) + load_balance_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # End of epoch
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        early_exits_count = 0

        # Reset layer exit counts for this epoch
        for i in range(len(layer_exit_counts)):
            layer_exit_counts[i] = 0

        with torch.no_grad():
            for batch_data in val_loader:
                # Handle both 2-element and 3-element batches
                if len(batch_data) == 3:
                    data, target, _ = batch_data  # Ignore the difficulty level
                else:
                    data, target = batch_data

                data, target = data.to(device), target.to(device)

                if is_naive:
                    outputs = model(data)
                    _, predicted = torch.max(outputs, 1)
                else:
                    outputs, layer_exits, _, exit_layer = model(data)
                    _, predicted = torch.max(outputs, 1)

                    # Track early exits
                    if exit_layer < len(model.layers):
                        early_exits_count += 1
                        layer_exit_counts[exit_layer] += 1

                total += target.size(0)
                correct += (predicted == target).sum().item()

        # Calculate accuracy
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)

        # Calculate early exit rate
        early_exit_rate = 100 * early_exits_count / total if not is_naive else 0
        early_exit_stats.append(early_exit_rate)

        # Calculate epoch time
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Print statistics
        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'  Train Loss: {avg_loss:.4f}')
        print(f'  Val Accuracy: {accuracy:.2f}%')
        if not is_naive:
            print(f'  Early Exit Rate: {early_exit_rate:.2f}%')
            print(f'  Layer Exit Distribution: {[count / total * 100 for count in layer_exit_counts]}')
        print(f'  Time: {epoch_time:.2f}s')

        # Print expert utilization if available
        if hasattr(model, 'get_expert_utilization') and not is_naive:
            utilization = model.get_expert_utilization()
            print("Expert utilization:")
            for layer_name, util in utilization:
                normalized_util = util / (util.sum() + 1e-10)
                print(f"  {layer_name}: {normalized_util}")

    return train_losses, val_accuracies, early_exit_stats, epoch_times, sum(epoch_times) / len(epoch_times)
# Enhanced evaluation function
# Enhanced evaluation function with fix for dataset with difficulty levels
def evaluate_model(model, test_loader, is_naive=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    layer_exit_counts = [0] * len(model.layers) if hasattr(model, 'layers') else [0]
    early_exit_total = 0

    complexity_by_layer = {i: [] for i in range(len(layer_exit_counts))}
    confidence_by_layer = {i: [] for i in range(len(layer_exit_counts))}

    start_time = time.time()

    with torch.no_grad():
        for batch_data in test_loader:
            # Handle both 2-element and 3-element batches
            if len(batch_data) == 3:
                data, target, difficulty = batch_data
            else:
                data, target = batch_data
                difficulty = None

            data, target = data.to(device), target.to(device)

            if is_naive:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
            else:
                if difficulty is not None:
                    # If we have difficulty data available
                    outputs, layer_exits, confidence_scores, exit_layer, complexity_scores = model(data,
                                                                                                   return_aux_logits=True)
                else:
                    # For datasets without difficulty information
                    try:
                        outputs, layer_exits, confidence_scores, exit_layer, complexity_scores = model(data,
                                                                                                       return_aux_logits=True)
                    except ValueError:
                        # Fallback if the model doesn't return complexity scores
                        outputs, layer_exits, confidence_scores, exit_layer = model(data, return_aux_logits=False)
                        complexity_scores = [0] * len(layer_exits)

                _, predicted = torch.max(outputs, 1)

                # Track where we exited
                if exit_layer < len(model.layers):
                    layer_exit_counts[exit_layer] += 1
                    early_exit_total += 1

                    # Record complexity and confidence for this layer exit
                    if exit_layer < len(complexity_scores):
                        complexity_by_layer[exit_layer].append(complexity_scores[exit_layer])

                    if exit_layer < len(confidence_scores):
                        confidence_by_layer[exit_layer].append(confidence_scores[exit_layer])

            total += target.size(0)
            correct += (predicted == target).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total

    # Calculate average inference time
    inference_time = (time.time() - start_time) / total

    # Print results
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Average Inference Time: {inference_time * 1000:.2f} ms per sample')

    if not is_naive:
        early_exit_rate = 100 * early_exit_total / total
        print(f'Early Exit Rate: {early_exit_rate:.2f}%')
        print(f'Layer Exit Distribution (%): {[count / total * 100 for count in layer_exit_counts]}')

        # Print average complexity and confidence by exit layer
        print("Average complexity by exit layer:")
        for layer, values in complexity_by_layer.items():
            if values:
                print(f"  Layer {layer}: {sum(values) / len(values):.4f}")

        print("Average confidence by exit layer:")
        for layer, values in confidence_by_layer.items():
            if values:
                print(f"  Layer {layer}: {sum(values) / len(values):.4f}")

    # Return key metrics
    if is_naive:
        return accuracy, inference_time
    else:
        return accuracy, inference_time, layer_exit_counts, early_exit_rate
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardMoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, expert_count=4, top_k=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.expert_count = expert_count
        self.top_k = top_k

        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, expert_count)
        )

        # Experts - feed-forward networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(expert_count)
        ])

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Layer normalization
        residual = x
        x = self.layer_norm(x)

        # Get router logits and probabilities
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        # Get top-k experts
        vals, indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize the router probabilities
        vals = vals / vals.sum(dim=-1, keepdim=True)

        # Process through experts
        batch_size = x.shape[0]
        combined_output = torch.zeros_like(x)

        # Standard MoE computation
        for b in range(batch_size):
            sample_outputs = []
            for k in range(self.top_k):
                expert_idx = indices[b, k].item()
                weight = vals[b, k].item()
                expert_output = self.experts[expert_idx](x[b].unsqueeze(0))
                sample_outputs.append(weight * expert_output)

            # Combine expert outputs for this sample
            combined_sample = torch.sum(torch.stack(sample_outputs, dim=0), dim=0)
            combined_output[b] = combined_sample.squeeze(0)

        # Residual connection
        output = residual + combined_output
        return output


class StandardMoEClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2,
                 num_classes=4, expert_count=4, top_k=2):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Create a list of standard MoE layers
        self.layers = nn.ModuleList([
            StandardMoELayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim * 2,
                expert_count=expert_count,
                top_k=top_k
            ) for _ in range(num_layers)
        ])

        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Initial layer
        x = F.relu(self.input_layer(x))

        # Process through MoE layers
        for layer in self.layers:
            x = layer(x)

        # Final prediction
        logits = self.classifier(x)

        # Return extra values to match the interface of SimpleClassifier
        # but these are placeholder values since standard MoE doesn't have early exit
        dummy_layer_exits = [False] * len(self.layers)
        dummy_confidence_scores = [0.0] * len(self.layers)
        exit_layer = len(self.layers)  # No early exit, so we go through all layers

        return logits, dummy_layer_exits, dummy_confidence_scores, exit_layer