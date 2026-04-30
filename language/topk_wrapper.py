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
    
    Usage in export_language.py:
        model = Model(**model_config, vocab_size=tokenizer.vocab_size)
        model.load_state_dict(torch.load("outputs/language/model.pt"))
        wrapped_model = ModelTopKWrapper(model, k=5)
        
    """
    
    def __init__(self, model: Model, k: int = 5):
        super().__init__()
        self.model = model
        self.k = k
    
    def forward(self, x):
        """
        Forward pass with TopK applied to all tokens in the sequence.
        
        Args:
            x: Input token indices, shape (batch_size, seq_len)
            
        Returns:
            topk_values: Top-k logit values, shape (batch_size, seq_len, k)
            topk_indices: Top-k token indices, shape (batch_size, seq_len, k)
        """
        # Get full model output: (batch_size, seq_len, vocab_size)
        logits = self.model(x)
        
        # Apply TopK to ALL positions in the sequence, not just the last token
        # torch.topk applies on the last dimension (vocab_size)
        # This returns shape (batch_size, seq_len, k) for both values and indices
        topk_values, topk_indices = torch.topk(logits, k=self.k, dim=-1)
        
        # Return both values and indices
        # Values can be used for confidence scoring
        # Indices are the actual token IDs to generate next
        return topk_values, topk_indices
