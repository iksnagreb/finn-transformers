# Wrapper around the Language Model to add TopK functionality for the last token
# This reduces data transfer from GPU to CPU by only returning the top-5 predictions
# instead of the full vocabulary logits

import torch
from language.model import Model


class ModelTopKWrapper(torch.nn.Module):
    """
    Wrapper around the base Model that applies TopK to the last token's output.
    
    Instead of returning full logits (batch_size, seq_len, vocab_size), this wrapper:
    1. Runs the model forward pass
    2. Extracts the logits for the last token: (batch_size, vocab_size)
    3. Applies TopK to get top-5 predictions
    4. Returns both indices and values for the top-5 tokens
    
    This significantly reduces GPU->CPU data transfer:
    - Original: vocab_size floats per batch (e.g., 4096)
    - Wrapper: 5 floats per batch (indices + values)
    - Reduction: ~80x less data transfer for 4096 vocab size
    
    Usage:
        # Load trained model
        model = Model(**model_config, vocab_size=tokenizer.vocab_size)
        model.load_state_dict(torch.load("outputs/language/model.pt"))
        
        # Wrap with TopK functionality
        wrapped_model = ModelTopKWrapper(model, k=5)
        
        # Export to ONNX (TopK operation will be included)
        torch.onnx.export(wrapped_model, dummy_input, "model_topk.onnx")
    """
    
    def __init__(self, model: Model, k: int = 5):
        """
        Args:
            model: The base Model instance (already loaded with trained weights)
            k: Number of top predictions to return (default: 5 for top-5)
        """
        super().__init__()
        self.model = model
        self.k = k
    
    def forward(self, x):
        """
        Forward pass with TopK applied to the last token.
        
        Args:
            x: Input token indices, shape (batch_size, seq_len)
            
        Returns:
            topk_values: Top-k logit values, shape (batch_size, k)
            topk_indices: Top-k token indices, shape (batch_size, k)
        """
        # Get full model output: (batch_size, seq_len, vocab_size)
        logits = self.model(x)
        
        # Extract logits for the last token: (batch_size, vocab_size)
        # x.shape[1] is the sequence length, -1 is the last position
        last_token_logits = logits[:, -1, :]
        
        # Apply TopK to get top-k values and indices
        # torch.topk returns (values, indices) sorted in descending order
        topk_values, topk_indices = torch.topk(last_token_logits, k=self.k, dim=-1)
        
        # Return both values and indices
        # Values can be used for confidence scoring
        # Indices are the actual token IDs to generate next
        return topk_values, topk_indices
