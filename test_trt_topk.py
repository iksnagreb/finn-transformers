"""
Create a simple test to isolate TensorRT TopK bug with INT8.
Test with smaller model and fixed batch size.
"""
import numpy as np
import tensorrt as trt
import torch
import onnx

# Load test data
data = np.load("/data/gitlab/language.npz")
input_ids = data["input_ids"][:2]  # First 2 samples
labels = data["labels"][:2]

print(f"Input shape: {input_ids.shape}")

# Test with TensorRT
logger = trt.Logger(trt.Logger.WARNING)
parser = trt.OnnxParser(trt.Builder(logger).create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)), logger)

model_path = "outputs/language/model_brevitas_2_argmax.onnx"
with open(model_path, "rb") as f:
    if not parser.parse(f.read()):
        print("Failed to parse model")
        exit(1)

builder = trt.Builder(logger)
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB

# Enable INT8
config.set_flag(trt.BuilderFlag.INT8)

# Create optimization profile for batch_size=2
profile = builder.create_optimization_profile()
profile.set_shape("input.1", (2, 256), (2, 256), (2, 256))
config.add_optimization_profile(profile)

engine = builder.build_engine(config)
context = engine.create_execution_context()

# Allocate GPU memory
device_input = torch.empty((2, 256), dtype=torch.int64, device='cuda')
device_logits = torch.empty((2, 256, 2048), dtype=torch.float32, device='cuda')
device_topk = torch.empty((2, 256, 5), dtype=torch.int64, device='cuda')

# Copy input
device_input.copy_(torch.from_numpy(input_ids).to(torch.int64))

# Set tensor addresses
context.set_tensor_address("input.1", device_input.data_ptr())
context.set_tensor_address("456", device_logits.data_ptr())
context.set_tensor_address("top_indices", device_topk.data_ptr())

# Run inference
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    context.execute_async_v3(stream.cuda_stream)
stream.synchronize()

# Get output
trt_topk = device_topk.cpu().numpy()

print(f"\nTensorRT TopK output shape: {trt_topk.shape}")
print(f"Sample 0, Token 0: {trt_topk[0, 0]}")
print(f"Sample 0, Token 1: {trt_topk[0, 1]}")
print(f"Sample 0, Token 10: {trt_topk[0, 10]}")
print(f"Sample 1, Token 0: {trt_topk[1, 0]}")
print(f"Sample 1, Token 1: {trt_topk[1, 1]}")

# Check if all tokens return same indices
all_same_0 = np.all(trt_topk[0, 0] == trt_topk[0])
all_same_1 = np.all(trt_topk[1, 0] == trt_topk[1])
print(f"\nSample 0: All tokens have same indices? {all_same_0}")
print(f"Sample 1: All tokens have same indices? {all_same_1}")

# Compare with logits to verify TopK is actually top-5
print(f"\nFirst sample, token 0:")
print(f"  Top-5 indices from TopK: {trt_topk[0, 0]}")
topk_logits = np.sort(device_logits[0, 0].cpu().numpy())[-5:][::-1]
topk_indices_direct = np.argsort(device_logits[0, 0].cpu().numpy())[-5:][::-1]
print(f"  Top-5 logits direct from model: {topk_logits}")
print(f"  Top-5 indices direct: {topk_indices_direct}")
