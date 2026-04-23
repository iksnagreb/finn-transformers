"""
Analyze ONNX model structure to understand TopK branch.
"""
import onnx

model_path = "outputs/language/model_brevitas_1_argmax.onnx"
model = onnx.load(model_path)

print("=== Model Outputs ===")
for output in model.graph.output:
    print(f"  {output.name}: shape {output.type.tensor_type.shape.dim}, dtype {output.type.tensor_type.elem_type}")

print("\n=== TopK Node ===")
topk_nodes = [n for n in model.graph.node if n.op_type == 'TopK']
for node in topk_nodes:
    print(f"  Op: {node.op_type}")
    print(f"  Inputs: {node.input}")
    print(f"  Outputs: {node.output}")
    print(f"  Attributes: {[(a.name, a) for a in node.attribute]}")

print("\n=== Nodes producing outputs ===")
for output in model.graph.output:
    name = output.name
    # Find node that produces this output
    producer = None
    for node in model.graph.node:
        if name in node.output:
            producer = node
            break
    if producer:
        print(f"  {name}: produced by {producer.op_type} node (inputs: {producer.input})")
    else:
        print(f"  {name}: not produced by any node (might be input or initializer)")

print("\n=== Model IO Summary ===")
print(f"Inputs: {[inp.name for inp in model.graph.input]}")
print(f"Outputs: {[out.name for out in model.graph.output]}")
print(f"Number of nodes: {len(model.graph.node)}")

# Find path from input to TopK
print("\n=== Tracing input to TopK ===")
for topk_node in topk_nodes:
    print(f"TopK inputs: {topk_node.input}")
    # topk_node.input[0] is the data, topk_node.input[1] is k value
    data_input = topk_node.input[0]
    print(f"  Data input: {data_input}")
    
    # Find producer of data_input
    for node in model.graph.node:
        if data_input in node.output:
            print(f"  Produced by: {node.op_type} (inputs: {node.input})")
