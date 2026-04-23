"""
Alternative approach: Keep original graph, add TopK as separate branch.
"""
import onnx
from onnx import helper, TensorProto


def create_topk_parallel_branch(input_model_path, output_model_path):
    """
    Instead of replacing output, create TopK as a parallel branch.
    This keeps the original graph structure intact for TensorRT.
    """
    print(f"Loading model: {input_model_path}")
    original_model = onnx.load(input_model_path)
    
    original_output_name = original_model.graph.output[0].name
    print(f"Original output: {original_output_name}")
    
    # Find what produces the current output
    producer_node = None
    for node in original_model.graph.node:
        if original_output_name in node.output:
            producer_node = node
            break
    
    if producer_node is None:
        print("ERROR: Could not find producer node")
        return
    
    print(f"Producer node: {producer_node.op_type}")
    
    # Add k constant
    k_tensor = helper.make_tensor(
        name='k_value',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[5]
    )
    original_model.graph.initializer.append(k_tensor)

    # Create TopK node that takes input from producer
    topk_node = helper.make_node(
        'TopK',
        inputs=[producer_node.output[0], 'k_value'],
        outputs=['topk_values', 'top_indices'],
        axis=-1,
        largest=1,
        sorted=1
    )

    original_model.graph.node.append(topk_node)
    
    # Add TopK indices as an additional output (keep original output)
    original_model.graph.output.append(
        helper.make_tensor_value_info(
            "top_indices",
            TensorProto.INT64,
            [1, 256, 5]
        )
    )
    
    print("Checking model...")
    onnx.checker.check_model(original_model)
    
    print(f"Saving model with TopK output: {output_model_path}")
    onnx.save(original_model, output_model_path)
    print("✓ TopK added as parallel branch!")
    print("  Original output:", original_output_name)
    print("  New output: top_indices")


def create_topk_replace_only_output_info(input_model_path, output_model_path):
    """
    Try: Just replace output info, keep graph nodes same.
    """
    print(f"Loading model: {input_model_path}")
    original_model = onnx.load(input_model_path)
    
    original_output_name = original_model.graph.output[0].name
    print(f"Original output name: {original_output_name}")
    
    # Ensure opset
    for opset in original_model.opset_import:
        if opset.domain == "":
            opset.version = max(opset.version, 11)
    
    # Add k
    k_tensor = helper.make_tensor(
        name='k_value',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[5]
    )
    original_model.graph.initializer.append(k_tensor)

    # TopK
    topk_node = helper.make_node(
        'TopK',
        inputs=[original_output_name, 'k_value'],
        outputs=['topk_values', 'topk_indices'],
        axis=-1,
        largest=1,
        sorted=1
    )

    original_model.graph.node.append(topk_node)
    
    # Change output to TopK indices
    original_model.graph.output[0].name = 'topk_indices'
    original_model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 5
    
    print("Checking model...")
    onnx.checker.check_model(original_model)
    
    print(f"Saving model: {output_model_path}")
    onnx.save(original_model, output_model_path)
    print("✓ TopK output created!")


if __name__ == "__main__":
    # Try parallel branch approach
    print("\n" + "="*70)
    print("Approach: TopK as parallel branch")
    print("="*70)
    create_topk_parallel_branch(
        "outputs/language/model_brevitas_1_simple.onnx",
        "outputs/language/model_brevitas_1_argmax.onnx"
    )
