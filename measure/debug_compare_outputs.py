"""
Compare ORT vs TensorRT outputs for language model to understand accuracy difference.
"""
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch
import yaml
from pathlib import Path

# Load config
with open("language/params.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Load test data
data = np.load("/data/gitlab/language.npz")
input_ids = data["input_ids"]
labels = data["labels"]

# Take first batch
test_input = input_ids[:1]  # shape [1, 256]
test_labels = labels[:1]   # shape [1, 256]

print(f"Test input shape: {test_input.shape}")
print(f"Test labels shape: {test_labels.shape}")

# ============================================================
# ORT Inference
# ============================================================
print("\n=== ORT CUDA Inference ===")
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 8
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

ort_model_path = "outputs/language/model_brevitas_1_argmax.onnx"
session = ort.InferenceSession(ort_model_path, sess_opts, providers=[
    ("CUDAExecutionProvider", {"device_id": 0}),
    ("CPUExecutionProvider", {})
])

feed = {"input.1": test_input.astype(np.int64)}
ort_outputs = session.run(None, feed)

ort_topk = ort_outputs[1]  # top_indices
print(f"ORT top_indices shape: {ort_topk.shape}")
print(f"ORT top_indices dtype: {ort_topk.dtype}")
print(f"ORT top_indices (first 10 tokens): {ort_topk[0, :10, :]}")
ort_pred = ort_topk[..., 0].astype(np.int64)

# Filter padding
mask = test_labels != -100
valid_labels = test_labels[mask]
valid_preds = ort_pred[mask]
ort_accuracy = (valid_preds == valid_labels).sum() / len(valid_preds) if len(valid_preds) > 0 else 0
print(f"ORT Accuracy: {ort_accuracy * 100:.2f}%")
print(f"ORT valid predictions: {len(valid_preds)}")

# ============================================================
# TensorRT Inference
# ============================================================
print("\n=== TensorRT INT8 Inference ===")

logger = trt.Logger(trt.Logger.WARNING)
parser = trt.OnnxParser(trt.Builder(logger).create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)), logger)

with open(ort_model_path, "rb") as f:
    parser.parse(f.read())

builder = trt.Builder(logger)
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB

# Enable INT8
config.set_flag(trt.BuilderFlag.INT8)

# Create optimization profile
profile = builder.create_optimization_profile()
profile.set_shape("input.1", (1, 256), (1, 256), (1, 256))
config.add_optimization_profile(profile)

engine = builder.build_engine(config)
context = engine.create_execution_context()

# Allocate GPU memory
device_input = torch.empty((1, 256), dtype=torch.int64, device='cuda')
device_topk_indices = torch.empty((1, 256, 5), dtype=torch.int64, device='cuda')
device_logits = torch.empty((1, 256, 2048), dtype=torch.float32, device='cuda')

# Copy input
device_input.copy_(torch.from_numpy(test_input).to(torch.int64))

# Set tensor addresses
context.set_tensor_address("input.1", device_input.data_ptr())
context.set_tensor_address("456", device_logits.data_ptr())  # logits
context.set_tensor_address("top_indices", device_topk_indices.data_ptr())

# Run inference
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    context.execute_async_v3(stream.cuda_stream)
stream.synchronize()

# Get results
trt_topk = device_topk_indices.cpu().numpy()
print(f"TensorRT top_indices shape: {trt_topk.shape}")
print(f"TensorRT top_indices dtype: {trt_topk.dtype}")
print(f"TensorRT top_indices (first 10 tokens): {trt_topk[0, :10, :]}")
trt_pred = trt_topk[..., 0].astype(np.int64)

# Filter padding
mask = test_labels != -100
valid_labels = test_labels[mask]
valid_preds_trt = trt_pred[mask]
trt_accuracy = (valid_preds_trt == valid_labels).sum() / len(valid_preds_trt) if len(valid_preds_trt) > 0 else 0
print(f"TensorRT Accuracy: {trt_accuracy * 100:.2f}%")
print(f"TensorRT valid predictions: {len(valid_preds_trt)}")

# ============================================================
# Compare
# ============================================================
print("\n=== Comparison ===")
print(f"ORT Accuracy:     {ort_accuracy * 100:.2f}%")
print(f"TensorRT Accuracy: {trt_accuracy * 100:.2f}%")
print(f"Difference: {(ort_accuracy - trt_accuracy) * 100:.2f}%")

# Check if indices are the same
indices_match = np.allclose(ort_pred, trt_pred)
print(f"Indices match: {indices_match}")

if not indices_match:
    diff_count = np.sum(ort_pred != trt_pred)
    print(f"Different predictions: {diff_count} out of {len(valid_preds)}")
    
    # Show some examples
    mismatches = np.where(ort_pred[0] != trt_pred[0])[0][:5]
    for idx in mismatches:
        print(f"  Token {idx}: ORT={ort_pred[0, idx]}, TRT={trt_pred[0, idx]}, Label={test_labels[0, idx]}")
