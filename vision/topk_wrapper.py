# Wrapper around the Vision Model to add TopK functionality for classification
# This reduces data transfer from GPU to CPU by only returning the top-K predictions
# instead of the full class logits

import torch
from vision.model import Model


class ModelTopKWrapper(torch.nn.Module):
    """
    Wrapper around the base Vision Model that applies TopK to the classification output.
    
    Instead of returning full logits (batch_size, num_classes), this wrapper:
    1. Runs the model forward pass
    2. Applies TopK to get top-K predictions
    3. Returns both indices and values for the top-K classes
    
    This significantly reduces GPU->CPU data transfer:
    - Original: num_classes floats per batch (e.g., 24 for RadioML, 10 for CIFAR-10)
    - Wrapper: K floats per batch (indices + values)
    - Reduction: more significant for larger class counts
    
    Usage in export_vision.py:
        model = Model(**model_config, num_classes=num_classes)
        model.load_state_dict(torch.load("outputs/vision/model.pt"))
        wrapped_model = ModelTopKWrapper(model, k=5)
        
    """
    
    def __init__(self, model: Model, k: int = 5):
        super().__init__()
        self.model = model
        self.k = k
    
    def forward(self, x):
        """
        Forward pass with TopK applied to classification logits.
        
        Args:
            x: Input features, shape (batch_size, ...)
            
        Returns:
            topk_values: Top-k logit values, shape (batch_size, k)
            topk_indices: Top-k class indices, shape (batch_size, k)
        """
        # Get full model output: (batch_size, num_classes)
        logits = self.model(x)
        
        # Apply TopK to the class dimension
        # torch.topk applies on the last dimension (num_classes)
        # This returns shape (batch_size, k) for both values and indices
        topk_values, topk_indices = torch.topk(logits, k=self.k, dim=-1)
        
        # Return both values and indices
        # Values can be used for confidence scoring
        # Indices are the actual class IDs
        return topk_values, topk_indices
