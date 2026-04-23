"""
Inspect ONNX model structure to understand node flow and outputs.
"""
import onnx

def inspect_model(model_path):
    """Print detailed information about model structure."""
    print(f"\n{'='*70}")
    print(f"Inspecting: {model_path}")
    print('='*70)
    
    model = onnx.load(model_path)
    
    # Outputs
    print(f"\n📤 MODEL OUTPUTS ({len(model.graph.output)}):")
    for i, output in enumerate(model.graph.output):
        print(f"  [{i}] Name: {output.name}")
        if output.type.HasField('tensor_type'):
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            dtype = output.type.tensor_type.elem_type
            print(f"      Shape: {shape}, Type: {dtype}")
    
    # Last few nodes
    print(f"\n📊 LAST 5 NODES:")
    nodes = list(model.graph.node)
    for node in nodes[-5:]:
        print(f"\n  Op: {node.op_type}")
        print(f"  Name: {node.name}")
        print(f"  Inputs: {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")
        if node.attribute:
            print(f"  Attributes:")
            for attr in node.attribute:
                print(f"    - {attr.name}: {attr}")
    
    # Nodes that feed output
    print(f"\n🔗 NODES FEEDING OUTPUT:")
    output_name = model.graph.output[0].name
    print(f"  Looking for nodes that output to: '{output_name}'")
    
    for node in model.graph.node:
        if output_name in node.output:
            print(f"\n  Found: {node.op_type}")
            print(f"    Inputs: {list(node.input)}")
            print(f"    Outputs: {list(node.output)}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    inspect_model("outputs/language/model_brevitas_1_simple.onnx")
