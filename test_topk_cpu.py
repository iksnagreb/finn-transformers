"""
Test TopK node with ONNX Runtime on CPU to verify model correctness.
"""
import numpy as np
import onnxruntime as ort

# Load test data
data = np.load("/data/gitlab/language.npz")
input_ids = data["input_ids"][:1]  # First batch

print(f"Input shape: {input_ids.shape}")
print(f"Input dtype: {input_ids.dtype}")

# Test with ONNX Runtime CPU only (to isolate the issue)
print("\n=== Testing TopK with ORT CPU ===")
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 8

model_path = "outputs/language/model_brevitas_1_argmax.onnx"
session = ort.InferenceSession(model_path, sess_opts, providers=["CPUExecutionProvider"])

feed = {"input.1": input_ids.astype(np.int64)}
outputs = session.run(None, feed)

print(f"Number of outputs: {len(outputs)}")
print(f"Output 0 shape (logits): {outputs[0].shape}, dtype: {outputs[0].dtype}")
print(f"Output 1 shape (top_indices): {outputs[1].shape}, dtype: {outputs[1].dtype}")

# Check the top_indices
topk = outputs[1]
print(f"\nFirst sequence top-5 indices (first 10 tokens):")
print(topk[0, :10, :])

print(f"\nAre all sequences identical? {np.all(topk[0] == topk[0, 0])}")

# Try first 3 sequences
if topk.shape[0] >= 3:
    print(f"\nSeq 0, token 0: {topk[0, 0]}")
    print(f"Seq 0, token 1: {topk[0, 1]}")
    print(f"Seq 0, token 5: {topk[0, 5]}")
