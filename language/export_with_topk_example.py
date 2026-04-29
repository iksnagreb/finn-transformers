# Example script showing how to use the ModelTopKWrapper for export
# This demonstrates the complete workflow for exporting a model with TopK

import torch
import numpy as np
import yaml
from transformers import AutoTokenizer
from language.model import Model
from language.model_wrapper import ModelTopKWrapper
from measure.export_language import export


def export_with_topk(model_weights_path, tokenizer_path, export_format="qcdq", k=5):
    """
    Export the language model with TopK wrapper for efficient inference.
    
    Args:
        model_weights_path: Path to trained model weights (e.g., "outputs/language/model.pt")
        tokenizer_path: Path to tokenizer (e.g., "outputs/language/tokenizer")
        export_format: Export format ("qcdq" for INT8 or "qonnx" for FP32)
        k: Number of top predictions (default: 5 for top-5)
    """
    
    # Load configuration
    with open("language/params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create base model with same configuration
    model = Model(**params["model"], vocab_size=tokenizer.vocab_size)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_weights_path))
    
    # Set to eval mode
    model.eval()
    
    # Wrap with TopK functionality
    wrapped_model = ModelTopKWrapper(model, k=k)
    wrapped_model.eval()
    
    # Create dummy input for export
    batch_size = 1
    seq_len = 128
    dummy_input = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    
    # Export to ONNX
    export_path = f"outputs/language/model_topk_{k}.onnx"
    
    torch.onnx.export(
        wrapped_model,
        (dummy_input,),
        export_path,
        input_names=['input_ids'],
        output_names=['topk_values', 'topk_indices'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'topk_values': {0: 'batch_size'},
            'topk_indices': {0: 'batch_size'}
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"Model with TopK wrapper exported to: {export_path}")
    
    # Verify the model works
    with torch.no_grad():
        topk_values, topk_indices = wrapped_model(dummy_input)
    
    print(f"Output shapes: values={topk_values.shape}, indices={topk_indices.shape}")
    print(f"Example top-5 indices: {topk_indices[0].tolist()}")
    print(f"Example top-5 values: {topk_values[0].tolist()}")


if __name__ == "__main__":
    # Usage example
    export_with_topk(
        model_weights_path="outputs/language/model.pt",
        tokenizer_path="outputs/language/tokenizer",
        export_format="qcdq",
        k=5
    )
