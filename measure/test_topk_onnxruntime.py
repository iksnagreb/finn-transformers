"""
Test the TopK model with ONNXRuntime (no TensorRT issues)
"""
import numpy as np
import onnxruntime as ort

def test_topk_with_onnxruntime():
    """Run the TopK model with ONNXRuntime to verify it works."""
    model_path = "outputs/language/model_brevitas_1_argmax.onnx"
    
    # Create session
    sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # Get input/output info
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    print(f"Model: {model_path}")
    print(f"Input: {input_name}, shape: {sess.get_inputs()[0].shape}")
    print(f"Output: {output_name}, shape: {sess.get_outputs()[0].shape}")
    
    # Create random input
    input_data = np.random.randint(0, 256, (1, 256), dtype=np.int64)
    
    # Run inference
    result = sess.run([output_name], {input_name: input_data})
    
    print(f"\nInput shape: {input_data.shape}")
    print(f"Output shape: {result[0].shape}")
    print(f"Output dtype: {result[0].dtype}")
    print(f"Sample output (top-5 indices):\n{result[0][0, :5, :]}")
    
    print("\n✓ Model runs successfully with ONNXRuntime!")
    return result

if __name__ == "__main__":
    test_topk_with_onnxruntime()
