import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

# Add the project root to the path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
# try:
from layerskip.simpleMoE_with_LayerSkip import MoELayerWithSkip, SimpleClassifier
from analysis.colossalai_replace.layer import SparseMLP
# except ImportError:
#     print("Error importing modules. Make sure you're running from the correct directory.")
#     sys.exit(1)

# Create a mock MOE_MANAGER class for testing ColossalAI SparseMLP
class MockMOEManager:
    def get_parallel(self):
        return None

# Set deterministic behavior for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a directory for test outputs
os.makedirs("test_output", exist_ok=True)

def test_early_exit_mechanism():
    """Test if early exit works with different confidence thresholds"""
    print("\n=== Testing Early Exit Mechanism ===")

    # Create a small test input
    input_tensor = torch.randn(4, 64)  # Batch size 4, hidden dim 64

    # Create layer with different confidence thresholds
    high_threshold_layer = MoELayerWithSkip(
        input_dim=64,
        hidden_dim=128,
        confidence_threshold=0.9  # Very high threshold (rarely exits)
    )
    high_threshold_layer.eval()

    low_threshold_layer = MoELayerWithSkip(
        input_dim=64,
        hidden_dim=128,
        confidence_threshold=0.1  # Very low threshold (often exits)
    )
    low_threshold_layer.eval()

    # Force confidence predictor to give consistent results for testing
    with torch.no_grad():
        # Set confidence predictor to predict 0.5 for all inputs
        high_threshold_layer.confidence_predictor.weight.fill_(0.0)
        low_threshold_layer.confidence_predictor.weight.fill_(0.0)
        high_threshold_layer.confidence_predictor.bias.fill_(0.0)  # 0.5 after sigmoid
        low_threshold_layer.confidence_predictor.bias.fill_(0.0)  # 0.5 after sigmoid

    # Test both layers
    output_high, early_exit_high, conf_high = high_threshold_layer(input_tensor)
    output_low, early_exit_low, conf_low = low_threshold_layer(input_tensor)

    print(f"High threshold early exit: {early_exit_high}, confidence: {conf_high}")
    print(f"Low threshold early exit: {early_exit_low}, confidence: {conf_low}")

    # Verify results
    success = True
    if early_exit_high:
        print("❌ ERROR: High threshold layer should NOT exit early")
        success = False

    if not early_exit_low:
        print("❌ ERROR: Low threshold layer should exit early")
        success = False

    print(f"Test result: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

def test_expert_selection_routing():
    """Test if expert routing works correctly"""
    print("\n=== Testing Expert Selection and Routing ===")

    # Create a small test input
    input_tensor = torch.randn(4, 64)  # Batch size 4, hidden dim 64

    # Set model to training mode (disables early exit)
    moe_layer = MoELayerWithSkip(
        input_dim=64,
        hidden_dim=128,
        expert_count=4,
        top_k=2
    )
    moe_layer.train()

    # Process input
    output, _, vals, indices, router_probs, _ = moe_layer(input_tensor, return_metrics=True)

    # Verify top-k experts were selected
    print(f"Selected expert indices shape: {indices.shape}")
    print(f"Expert weights sum: {vals.sum(dim=1)}")

    # Verify results
    success = True

    # Check if exactly k experts were chosen for each input
    if indices.shape != (input_tensor.shape[0], moe_layer.top_k):
        print(f"❌ ERROR: Expected indices shape {(input_tensor.shape[0], moe_layer.top_k)}, got {indices.shape}")
        success = False

    # Check if weights sum to approximately 1
    if not torch.allclose(vals.sum(dim=1), torch.ones(input_tensor.shape[0]), atol=1e-5):
        print(f"❌ ERROR: Expert weights don't sum to 1: {vals.sum(dim=1)}")
        success = False

    print(f"Test result: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

def test_colossalai_sparse_mlp():
    """Test if the ColossalAI SparseMLP can be instantiated and run"""
    print("\n=== Testing ColossalAI SparseMLP Integration ===")

    # Monkey patch for testing
    import colossalai.moe
    import sys
    sys.modules['colossalai.moe.manager'] = type('', (), {
        'MOE_MANAGER': MockMOEManager()
    })

    try:
        # Create a test input
        input_tensor = torch.randn(4, 128)  # Batch size 4, hidden dim 128

        # Create a minimal SparseMLP
        sparse_mlp = SparseMLP(
            num_experts=4,
            hidden_size=128,
            intermediate_size=512,
            router_top_k=2,
            model_output_dir="./test_output"  # Directory for output logs
        )

        # Forward pass
        output = sparse_mlp(input_tensor)

        # Verify output shape
        success = output.shape == input_tensor.shape
        if success:
            print(f"Output shape: {output.shape} matches input shape: {input_tensor.shape}")
        else:
            print(f"❌ ERROR: Output shape {output.shape} doesn't match input shape {input_tensor.shape}")

        print(f"Test result: {'✅ PASSED' if success else '❌ FAILED'}")
        return success
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("Test result: ❌ FAILED")
        return False

def test_layer_dropout_curriculum():
    """Test if layer dropout curriculum works correctly"""
    print("\n=== Testing Layer Dropout Curriculum ===")

    # Create classifier with layer dropout
    model = SimpleClassifier(
        input_dim=64,
        hidden_dim=128,
        num_layers=4,
        expert_count=4
    )

    # Set curriculum parameters
    model.total_epochs = 5

    # Initialize an array to track which layers were used
    class LayerUsageTracker:
        def __init__(self, model):
            self.model = model
            self.layer_usage = [0] * len(model.layers)

            # Monkey patch the forward method of each layer
            for i, layer in enumerate(model.layers):
                original_forward = layer.forward

                def create_tracking_forward(idx, orig_forward):
                    def tracking_forward(x, *args, **kwargs):
                        self.layer_usage[idx] += 1
                        return orig_forward(x, *args, **kwargs)
                    return tracking_forward

                layer.forward = create_tracking_forward(i, original_forward)

        def reset(self):
            self.layer_usage = [0] * len(self.model.layers)

        def get_usage(self):
            return self.layer_usage.copy()

    # Create tracker
    tracker = LayerUsageTracker(model)

    # Test with early epoch (should have more dropout)
    model.curr_epoch = 0  # Start of training
    model.train()

    # Sample input
    input_tensor = torch.randn(8, 64)

    # Run many forward passes
    num_trials = 100
    for _ in range(num_trials):
        model(input_tensor)

    early_epoch_usage = tracker.get_usage()
    print(f"Layer usage at epoch 0: {early_epoch_usage}")
    tracker.reset()

    # Test with later epoch (should have less dropout)
    model.curr_epoch = 4  # End of training
    for _ in range(num_trials):
        model(input_tensor)

    late_epoch_usage = tracker.get_usage()
    print(f"Layer usage at epoch 4: {late_epoch_usage}")

    # Verify results - should have more layer usage in later epochs
    success = True
    if sum(late_epoch_usage) <= sum(early_epoch_usage):
        print("❌ ERROR: Expected more layer usage in later epochs")
        success = False

    print(f"Test result: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

def test_end_to_end_forward_pass():
    """Test a complete forward pass through the model"""
    print("\n=== Testing End-to-End Forward Pass ===")

    # Create input
    batch_size = 32
    input_dim = 64
    inputs = torch.randn(batch_size, input_dim)
    targets = torch.randint(0, 4, (batch_size,))

    # Create model
    model = SimpleClassifier(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=3,
        num_classes=4,
        expert_count=4,
        enable_layer_skip=True,
        confidence_thresholds=[0.3, 0.5, 0.7]  # Progressive thresholds
    )

    # Test training mode (no early exit)
    model.train()
    logits, _, _, _, aux_logits = model(inputs, return_aux_logits=True)

    # Verify shapes
    success = True
    if logits.shape != (batch_size, 4):
        print(f"❌ ERROR: Expected logits shape {(batch_size, 4)}, got {logits.shape}")
        success = False

    if len(aux_logits) != 3:
        print(f"❌ ERROR: Expected 3 auxiliary logits, got {len(aux_logits)}")
        success = False

    # Test inference mode (with early exit)
    model.eval()

    # Force confidence predictors to give specific values for testing
    with torch.no_grad():
        # Set first layer to have low confidence (won't exit)
        model.layers[0].confidence_predictor.weight.fill_(0.0)
        model.layers[0].confidence_predictor.bias.fill_(-1.0)  # ~0.27 after sigmoid

        # Set second layer to have medium confidence (some may exit)
        model.layers[1].confidence_predictor.weight.fill_(0.0)
        model.layers[1].confidence_predictor.bias.fill_(0.0)  # 0.5 after sigmoid

        # Set third layer to have high confidence (all would exit, but moot)
        model.layers[2].confidence_predictor.weight.fill_(0.0)
        model.layers[2].confidence_predictor.bias.fill_(1.0)  # ~0.73 after sigmoid

    logits, layer_exits, confidence_scores, exit_layer = model(inputs)

    # Check if exit layer makes sense
    print(f"Exit layer: {exit_layer}")
    print(f"Confidence scores: {confidence_scores}")

    # Calculate accuracy
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == targets).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")

    print(f"Test result: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

def run_all_tests():
    """Run all tests and report results"""
    print("\n====== LayerSkip MoE Tests ======")

    results = []

    try:
        results.append(test_early_exit_mechanism())
    except Exception as e:
        print(f"❌ ERROR in early exit test: {e}")
        results.append(False)

    try:
        results.append(test_expert_selection_routing())
    except Exception as e:
        print(f"❌ ERROR in expert selection test: {e}")
        results.append(False)

    try:
        results.append(test_colossalai_sparse_mlp())
    except Exception as e:
        print(f"❌ ERROR in ColossalAI test: {e}")
        results.append(False)

    try:
        results.append(test_layer_dropout_curriculum())
    except Exception as e:
        print(f"❌ ERROR in layer dropout test: {e}")
        results.append(False)

    try:
        results.append(test_end_to_end_forward_pass())
    except Exception as e:
        print(f"❌ ERROR in end-to-end test: {e}")
        results.append(False)

    # Print summary
    print("\n====== Test Summary ======")
    test_names = [
        "Early Exit Mechanism",
        "Expert Selection & Routing",
        "ColossalAI SparseMLP",
        "Layer Dropout Curriculum",
        "End-to-End Forward Pass"
    ]

    for name, result in zip(test_names, results):
        print(f"{name}: {'✅ PASSED' if result else '❌ FAILED'}")

    # Overall result
    overall = all(results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if overall else '❌ SOME TESTS FAILED'}")

    return overall

if __name__ == "__main__":
    run_all_tests()
