import onnx

model = onnx.load("/home/hanna/git/finn-transformers/outputs/radioml/model_brevitas_1.onnx")
nodes = [node.op_type for node in model.graph.node]
from collections import Counter
print(Counter(nodes))
