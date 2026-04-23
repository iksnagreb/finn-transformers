"""
ONNXSCRIPT approach to add TopK wrapper to ONNX models.
More readable and maintainable using ONNXSCRIPT patterns.

Installation:
    pip install onnxscript

This approach shows:
- How to compose TopK with existing models
- Support for multiple batch sizes
- Clean, documented code inspired by ONNXSCRIPT patterns
"""
import onnx
from onnx import helper, TensorProto

MODEL_TYPE = os.environ.get("MODEL_TYPE", "vision")
# ============================================================================
# ONNXSCRIPT-style TopK Wrapper Definition
# ============================================================================

# This is what a ONNXSCRIPT topk_indices function would look like:
#
# @script()
# def topk_indices(input_tensor: INT64) -> INT64:
#     """Extract top-5 indices from input tensor."""
#     k = op.Constant(value_int=5)
#     values, indices = op.TopK(input_tensor, k, axis=-1, largest=1, sorted=1)
#     return indices
#
# We implement this pattern below using ONNX helper functions
# for compatibility and clarity.


# ============================================================================
# Helper Functions
# ============================================================================

def find_producer_node(model, output_name):
    """Find the node that produces a given output."""
    for node in model.graph.node:
        if output_name in node.output:
            return node
    return None


def add_topk_parallel_branch_onnxscript(input_model_path, output_model_path, batch_size):
    """
    Add TopK as a parallel branch using ONNXSCRIPT pattern.
    
    This approach:
    1. Keeps the original graph intact
    2. Adds TopK as a new branch
    3. Outputs both original output and TopK indices
    
    Args:
        input_model_path: Path to original ONNX model
        output_model_path: Path to save modified model
        batch_size: Batch size for the model (used in output shape)
    """
    print(f"Loading model: {input_model_path}")
    model = onnx.load(input_model_path)
    
    original_output_name = model.graph.output[0].name
    print(f"Original output: {original_output_name}")
    print(f"Batch size: {batch_size}")
    
    # Find what produces the current output
    producer_node = find_producer_node(model, original_output_name)
    if producer_node is None:
        raise ValueError("Could not find producer node for output")
    
    print(f"Producer node: {producer_node.op_type}")
    
    # Ensure opset version supports TopK
    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = max(opset.version, 11)
    
    # Add k constant (as per ONNXSCRIPT pattern)
    k_tensor = helper.make_tensor(
        name='k_value',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[5]
    )
    model.graph.initializer.append(k_tensor)

    # Create TopK node
    # This is equivalent to what ONNXSCRIPT's op.TopK() produces internally
    topk_node = helper.make_node(
        'TopK',
        inputs=[producer_node.output[0], 'k_value'],
        outputs=['topk_values', 'top_indices'],
        axis=-1,
        largest=1,
        sorted=1
    )

    model.graph.node.append(topk_node)
    
    # Add TopK indices as an additional output (keep original output) -> otherwise tensorrt will fail to build the engine.
    model.graph.output.append(
        helper.make_tensor_value_info(
            "top_indices",
            TensorProto.INT64,
            [batch_size, 256, 5]  # Dynamic batch size
        )
    )
    
    print("Validating model...")
    onnx.checker.check_model(model)
    
    print(f"Saving model: {output_model_path}")
    onnx.save(model, output_model_path)
    print(f"✓ TopK wrapper created!")
    print(f"  Original output: {original_output_name}")
    print(f"  TopK output: top_indices (shape: [{batch_size}, 256, 5])")
    print()


# ============================================================================
# Main Functions
# ============================================================================

def create_topk_models_for_batch_sizes(base_input_path, base_output_path, batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]):
    """
    Create TopK-wrapped models for multiple batch sizes.
    
    Args:
        base_input_path: Input model path (template)
        base_output_path: Output path template (will use {batch_size} placeholder)
        batch_sizes: List of batch sizes to generate models for
    """
    print("\n" + "="*70)
    print(f"Creating TopK-wrapped models using ONNXSCRIPT")
    print("="*70 + "\n")
    
    for batch_size in batch_sizes:
        try:
            # Replace {batch_size} placeholder in paths
            actual_input = base_input_path.format(batch_size=batch_size)
            actual_output = base_output_path.format(batch_size=batch_size)
            
            print(f"Batch size {batch_size}:")
            add_topk_parallel_branch_onnxscript(actual_input, actual_output, batch_size)
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")


def create_single_batch_topk(input_path, output_path, batch_size=1):
    """
    Simple wrapper for creating a single TopK model.
    
    Args:
        input_path: Path to original ONNX model
        output_path: Path to save modified model
        batch_size: Batch size (default: 1)
    """
    print("\n" + "="*70)
    print("Creating TopK-wrapped model using ONNXSCRIPT")
    print("="*70 + "\n")
    
    add_topk_parallel_branch_onnxscript(input_path, output_path, batch_size)


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Example 1: Single batch size (most common case)
    # print("\n🎯 Example 1: Single batch size")
    # create_single_batch_topk(
    #     "outputs/language/model_brevitas_1_simple.onnx",
    #     "outputs/language/model_brevitas_1_argmax_bs1.onnx",
    #     batch_size=1
    # )
    
    # Example 2: Multiple batch sizes
    # Uncomment to generate models for different batch sizes
    if MODEL_TYPE == "language":
        print("\nMultiple batch sizes")
        create_topk_models_for_batch_sizes(
            "outputs/language/model_brevitas_{batch_size}_simple.onnx",
            "outputs/language/model_brevitas_{batch_size}_argmax.onnx",
            batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        )
