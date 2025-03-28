import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# Import our enhanced SparseMLPWithLayerSkip
from layer import SparseMLPWithLayerSkip, compute_layerskip_loss


class TransformerBlockWithLayerSkip(nn.Module):
    """
    Transformer block with LayerSkip-enabled MoE FFN
    """

    def __init__(
            self,
            hidden_size: int,
            num_attention_heads: int,
            intermediate_size: int,
            num_experts: int = 8,
            router_top_k: int = 2,
            dropout_prob: float = 0.1,
            enable_layerskip: bool = True,
            confidence_threshold: float = 0.8,
            expert_dropout_rate: float = 0.2,
    ):
        super().__init__()

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout_prob,
            batch_first=True
        )

        # LayerNorm for attention
        self.norm1 = nn.LayerNorm(hidden_size)

        # MoE feed-forward with LayerSkip
        self.moe_ffn = SparseMLPWithLayerSkip(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            router_top_k=router_top_k,
            enable_layerskip=enable_layerskip,
            confidence_threshold=confidence_threshold,
            expert_dropout_rate=expert_dropout_rate,
        )

        # LayerNorm for MoE
        self.norm2 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout_prob)

        # For tracking early exit info
        self.early_exit_info = None

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            return_metrics: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            return_metrics: Whether to return additional metrics

        Returns:
            hidden_states: (batch_size, seq_len, hidden_size)
            metrics: Additional metrics if requested
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        # Convert attention_mask for PyTorch attention
        attn_mask = None
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] to [batch_size, seq_len, seq_len]
            attn_mask = attention_mask.unsqueeze(1).expand(-1, attention_mask.size(1), -1)
            # Convert to boolean mask where True means to attend, False means to mask
            attn_mask = (attn_mask == 0)

        hidden_states, _ = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=attn_mask if attn_mask is not None else None
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # MoE FFN with LayerSkip
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)

        # Process through MoE with LayerSkip
        if return_metrics:
            hidden_states, metrics = self.moe_ffn(hidden_states, return_metrics=True)
            self.early_exit_info = metrics
            return hidden_states, metrics
        else:
            hidden_states = self.moe_ffn(hidden_states)
            return hidden_states, None


class TransformerModelWithLayerSkip(nn.Module):
    """
    Transformer model with LayerSkip-enabled MoE layers
    """

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            num_layers: int,
            num_attention_heads: int,
            intermediate_size: int,
            num_experts: int = 8,
            router_top_k: int = 2,
            dropout_prob: float = 0.1,
            max_seq_length: int = 512,
            enable_layerskip: bool = True,
            confidence_thresholds: Optional[List[float]] = None,
    ):
        super().__init__()

        # Default confidence thresholds if not provided
        if confidence_thresholds is None:
            # Make thresholds progressively higher for deeper layers
            confidence_thresholds = [0.7 + 0.2 * (i / (num_layers - 1)) for i in range(num_layers)]

        # Ensure we have the right number of thresholds
        assert len(confidence_thresholds) == num_layers, "Must provide a confidence threshold for each layer"

        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)

        # LayerNorm and Dropout for embeddings
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout_prob)

        # Transformer blocks with LayerSkip
        self.layers = nn.ModuleList([
            TransformerBlockWithLayerSkip(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                router_top_k=router_top_k,
                dropout_prob=dropout_prob,
                enable_layerskip=enable_layerskip,
                confidence_threshold=confidence_thresholds[i],
                expert_dropout_rate=dropout_prob,
            )
            for i in range(num_layers)
        ])

        # For classification tasks
        self.classifier = nn.Linear(hidden_size, vocab_size)

        # Auxiliary classifiers for intermediate outputs (for early exit training)
        if enable_layerskip:
            self.aux_classifiers = nn.ModuleList([
                nn.Linear(hidden_size, vocab_size)
                for _ in range(num_layers)
            ])

        # Model config
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.enable_layerskip = enable_layerskip
        self.training_mode = True

    def reset_expert_stats(self):
        """Reset expert utilization and early exit statistics."""
        for layer in self.layers:
            if hasattr(layer.moe_ffn, 'reset_utilization_stats'):
                layer.moe_ffn.reset_utilization_stats()

    def get_expert_utilization(self):
        """Get expert utilization across all layers."""
        utilization = []
        for i, layer in enumerate(self.layers):
            if hasattr(layer.moe_ffn, 'get_expert_utilization'):
                util = layer.moe_ffn.get_expert_utilization()
                if util is not None:
                    utilization.append((f"Layer {i}", util))
        return utilization

    def get_early_exit_stats(self):
        """Get early exit statistics across all layers."""
        stats = []
        for i, layer in enumerate(self.layers):
            if hasattr(layer.moe_ffn, 'get_early_exit_stats'):
                layer_stats = layer.moe_ffn.get_early_exit_stats()
                if layer_stats is not None:
                    stats.append((f"Layer {i}", layer_stats))
        return stats

    def train(self, mode=True):
        super().train(mode)
        self.training_mode = mode
        # Propagate to all layers
        for layer in self.layers:
            if hasattr(layer.moe_ffn, 'train'):
                layer.moe_ffn.train(mode)
        return self

    def eval(self):
        super().eval()
        self.training_mode = False
        # Propagate to all layers
        for layer in self.layers:
            if hasattr(layer.moe_ffn, 'eval'):
                layer.moe_ffn.eval()
        return self

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            return_aux_logits: bool = False
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            return_aux_logits: Whether to return auxiliary logits for auxiliary loss

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            aux_logits: List of auxiliary logits if requested
        """
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        # Sum embeddings
        hidden_states = inputs_embeds + position_embeds

        # Embedding preprocessing
        hidden_states = self.embedding_norm(hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        # Collect auxiliary outputs if needed
        aux_outputs = []
        early_exit_layer = self.num_layers  # Default to no early exit

        # Process through layers with potential early exit
        for i, layer in enumerate(self.layers):
            hidden_states, metrics = layer(hidden_states, attention_mask, return_metrics=True)

            # Store auxiliary outputs for training
            if return_aux_logits and self.enable_layerskip:
                aux_logits = self.aux_classifiers[i](hidden_states)
                aux_outputs.append(aux_logits)

            # Check for early exit (inference only)
            if not self.training and self.enable_layerskip:
                if metrics is not None and metrics.get('early_exit', False):
                    early_exit_layer = i
                    break

        # Apply final classifier
        logits = self.classifier(hidden_states)

        # Return results based on requested outputs
        if return_aux_logits and self.enable_layerskip:
            return logits, aux_outputs, early_exit_layer

        return logits, early_exit_layer


# Helper function to train with LayerSkip
def train_with_layerskip(
        model,
        optimizer,
        train_dataloader,
        num_epochs,
        device='cuda',
        aux_loss_weight=0.3,
        load_balance_weight=0.1
):
    """
    Train a model with LayerSkip functionality

    Args:
        model: The transformer model with LayerSkip
        optimizer: PyTorch optimizer
        train_dataloader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device to use (cuda/cpu)
        aux_loss_weight: Weight for auxiliary loss
        load_balance_weight: Weight for load balancing loss
    """
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.reset_expert_stats()
        total_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass with auxiliary outputs
            logits, aux_outputs, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_aux_logits=True
            )

            # Main loss
            main_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Auxiliary loss for LayerSkip
            if model.enable_layerskip:
                aux_loss = 0
                for i, aux_logits in enumerate(aux_outputs):
                    # Weight later layers higher
                    layer_weight = aux_loss_weight * (i + 1) / len(aux_outputs)
                    aux_loss += layer_weight * criterion(aux_logits.view(-1, aux_logits.size(-1)), labels.view(-1))

                # Load balancing loss
                load_balance_loss = 0
                for layer in model.layers:
                    if hasattr(layer.moe_ffn, 'compute_load_balancing_loss'):
                        load_balance_loss += layer.moe_ffn.compute_load_balancing_loss()

                # Total loss
                loss = main_loss + aux_loss + load_balance_weight * load_balance_loss
            else:
                loss = main_loss

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print epoch stats
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}")

        # Print expert utilization statistics
        if model.enable_layerskip:
            utilization = model.get_expert_utilization()
            print("Expert utilization:")
            for layer_name, util in utilization:
                # Normalize utilization
                if util.sum() > 0:
                    normalized_util = util / util.sum()
                    std_dev = normalized_util.std().item()
                    print(f"  {layer_name}: std_dev={std_dev:.4f}, util={normalized_util}")


# Helper function for evaluation with LayerSkip
def evaluate_with_layerskip(model, eval_dataloader, device='cuda'):
    """
    Evaluate a model with LayerSkip functionality

    Args:
        model: The transformer model with LayerSkip
        eval_dataloader: DataLoader for evaluation data
        device: Device to use (cuda/cpu)

    Returns:
        accuracy: Evaluation accuracy
        early_exit_rate: Percentage of samples that used early exit
        avg_layer: Average exit layer
    """
    model.to(device)
    model.eval()
    model.reset_expert_stats()

    correct = 0
    total = 0
    early_exits = 0
    layer_exits = [0] * (model.num_layers + 1)  # +1 for "no early exit"

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits, exit_layer = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Track which layer we exited at
            layer_exits[exit_layer] += input_ids.size(0)
            if exit_layer < model.num_layers:
                early_exits += input_ids.size(0)

            # Calculate accuracy
            _, predicted = torch.max(logits, dim=-1)
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    # Calculate metrics
    accuracy = 100 * correct / total
    early_exit_rate = 100 * early_exits / total

    # Calculate average exit layer
    weighted_sum = sum(layer * count for layer, count in enumerate(layer_exits))
    avg_layer = weighted_sum / total

    # Print layer exit distribution
    print(f"Layer exit distribution:")
    for i, count in enumerate(layer_exits):
        if i < model.num_layers:
            print(f"  Layer {i}: {100 * count / total:.2f}%")
        else:
            print(f"  No early exit: {100 * count / total:.2f}%")

    # Print early exit statistics
    if model.enable_layerskip:
        stats = model.get_early_exit_stats()
        print("Early exit statistics:")
        for layer_name, layer_stats in stats:
            print(f"  {layer_name}:")
            print(f"    Exit rate: {layer_stats.get('exit_rate', 0):.2f}")
            print(f"    Avg confidence: {layer_stats.get('avg_confidence', 0):.4f}")
            print(f"    Avg complexity: {layer_stats.get('avg_complexity', 0):.4f}")

    return accuracy, early_exit_rate, avg_layer


# Example usage
def example_usage():
    # Synthetic dataset
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000, seq_length=128, vocab_size=30000):
            self.num_samples = num_samples
            self.seq_length = seq_length
            self.vocab_size = vocab_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate random input_ids
            input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
            # Generate random attention mask (1 = attend, 0 = ignore)
            attention_mask = torch.ones_like(input_ids)
            # For simplicity, use the same sequence as labels
            labels = input_ids.clone()

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

    # Create datasets and dataloaders
    train_dataset = SyntheticDataset(num_samples=1000)
    eval_dataset = SyntheticDataset(num_samples=100)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=4, shuffle=False
    )

    # Create model
    model = TransformerModelWithLayerSkip(
        vocab_size=30000,
        hidden_size=768,
        num_layers=4,
        num_attention_heads=12,
        intermediate_size=3072,
        num_experts=8,
        router_top_k=2,
        enable_layerskip=True,
        confidence_thresholds=[0.7, 0.75, 0.8, 0.85]  # Progressively higher thresholds
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Train model
    print("Training with LayerSkip...")
    train_with_layerskip(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        num_epochs=3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Evaluate model
    print("\nEvaluating with LayerSkip...")
    accuracy, early_exit_rate, avg_layer = evaluate_with_layerskip(
        model=model,
        eval_dataloader=eval_dataloader,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Early exit rate: {early_exit_rate:.2f}%")
    print(f"  Average exit layer: {avg_layer:.2f}")

    # For comparison, disable LayerSkip and evaluate again
    for layer in model.layers:
        layer.moe_ffn.enable_layerskip = False

    print("\nEvaluating without LayerSkip...")
    accuracy_no_skip, _, _ = evaluate_with_layerskip(
        model=model,
        eval_dataloader=eval_dataloader,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nResults without LayerSkip:")
    print(f"  Accuracy: {accuracy_no_skip:.2f}%")
    print(f"  Speed improvement with LayerSkip: {model.num_layers / avg_layer:.2f}x")


if __name__ == "__main__":
    example_usage()