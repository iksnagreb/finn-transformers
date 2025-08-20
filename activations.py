# Standard PyTorch neural network building blocks: Activation functions
from torch.nn import ReLU, GELU, SiLU, Tanh, Sigmoid, Identity

# Dictionary of named activation functions
ACTIVATIONS = {
    "tanh": Tanh, "sigmoid": Sigmoid, "relu": ReLU, "gelu": GELU, "silu": SiLU,
    "none": Identity
}
