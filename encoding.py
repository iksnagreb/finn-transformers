# PyTorch base package: Math and Tensor Stuff, Neural Network layers
import torch
# Quantized versions of PyTorch components
import brevitas.nn

# Uses np.prod below to flatten all sequence/spatial dimensions
import numpy as np


# Quantized sinusoidal positional encoding layer
class QuantSinusoidalPositionalEncoding(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, input_quant, output_quant, return_quant_tensor):
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Adds the quantized input and positional encoding
        self.add = brevitas.nn.QuantEltwiseAdd(
            # Input quantization to be applied to the input as well as the
            # positional encodings
            input_quant=input_quant,
            # Quantize the outputs after adding input and positional encoding
            output_quant=output_quant,
            # Returns quantization information to the next layer
            return_quant_tensor=return_quant_tensor
        )

    # Forward pass adding positional encoding to the input tensor
    def forward(self, x):
        # Get the size of the inputs to dynamically generate encodings of the
        # same size: Captures multiple sequence (actually spatial) dimensions
        _, *seq, emb = x.shape
        # Flatten all sequence/spatial dimensions to only have to enumerate the
        # position for a single axis
        seq = np.prod(seq)
        # Start by enumerating all steps of the sequence
        i = torch.as_tensor([[n] for n in range(seq)])
        # Scale factor adjusting the frequency/wavelength of the sinusoid
        # depending on the embedding dimension index
        f = torch.as_tensor([1e4 ** -(i / emb) for i in range(0, emb, 2)])
        # Prepare empty positional encoding tensor of the same size as the input
        pos = torch.empty(seq, emb)
        # Fill the positional encoding with alternating sine and cosine waves
        pos[:, 0::2] = torch.sin(f * i)
        pos[:, 1::2] = torch.cos(f * i)
        # Move the encoding tensor to the same device as the input tensor
        pos = pos.to(x.device, torch.float32).reshape(x.shape[1:])
        # Add the quantized encoding to the quantized input
        return self.add(x, pos)


# Quantized learned positional encoding layer
class QuantLearnedPositionalEncoding(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(
            self,
            shape,
            input_quant,
            output_quant,
            return_quant_tensor
    ):
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Adds the quantized input and positional encoding
        self.add = brevitas.nn.QuantEltwiseAdd(
            # Input quantization to be applied to the input as well as the
            # positional encodings
            input_quant=input_quant,
            # Quantize the outputs after adding input and positional encoding
            output_quant=output_quant,
            # Returns quantization information to the next layer
            return_quant_tensor=return_quant_tensor
        )
        # Register a parameter tensor representing the not quantized positional
        # encoding
        self.pos = torch.nn.Parameter(torch.empty(shape))
        # Reset/Initialize the parameter tensor
        self.reset_parameters()

    # Resets/Initializes the positional encoding parameter tensor
    def reset_parameters(self):
        # Initialize the positional encoding from a normal distribution with
        # zero mean and unit standard deviation
        torch.nn.init.normal_(self.pos, mean=0, std=1)

    # Forward pass adding positional encoding to the input tensor
    def forward(self, x):
        # Add the quantized encoding to the quantized input
        return self.add(x, self.pos)


# Lazy version of the learned encoding not requiring input dimensions at
# initialization, inferring these at the first forward pass
class LazyQuantLearnedPositionalEncoding(
    torch.nn.modules.lazy.LazyModuleMixin,  # noqa
    QuantLearnedPositionalEncoding
):
    # Once initialized, this will become a QuantLearnedPositionalEncoding as
    # defined above
    cls_to_become = QuantLearnedPositionalEncoding
    # Parameter tensor of the QuantLearnedPositionalEncoding is uninitialized
    pos: torch.nn.UninitializedParameter

    # Initializes the model and registers the module parameters
    def __init__(self, input_quant, output_quant, return_quant_tensor):
        # Initialize the quantizer parts of QuantLearnedPositionalEncoding,
        # leaving the dimensions empty
        super().__init__((), input_quant, output_quant, return_quant_tensor)
        # Register an uninitialized parameter tensor for the positional encoding
        self.pos = torch.nn.UninitializedParameter()

    # Resets/Initializes the positional encoding parameter tensor
    def reset_parameters(self):
        # If this has already been initialized, delegate to the actual
        # implementation
        if not self.has_uninitialized_params():
            super().reset_parameters()

    # Initializes/Materializes the uninitialized parameter tensor given some
    # sample input tensor to infer the dimensions
    def initialize_parameters(self, x):
        # Only materialize the parameter tensor if it is not yet initialized
        if self.has_uninitialized_params():
            # Do not accumulate gradient information from initialization
            with torch.no_grad():
                # Get the size of the inputs to generate encodings of the same
                # size: Captures multiple sequence (actually spatial) dimensions
                _, *dims = x.shape
                # Materialize the positional encoding parameter tensor
                self.pos.materialize(dims)
                # Properly initialize the parameters by resetting the values
                self.reset_parameters()


# Quantized binary positional encoding layer
class QuantBinaryPositionalEncoding(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, input_quant, output_quant, return_quant_tensor):
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Adds the quantized input and positional encoding
        self.add = brevitas.nn.QuantEltwiseAdd(
            # Input quantization to be applied to the input as well as the
            # positional encodings
            input_quant=input_quant,
            # Quantize the outputs after adding input and positional encoding
            output_quant=output_quant,
            # Returns quantization information to the next layer
            return_quant_tensor=return_quant_tensor
        )

    # Forward pass adding positional encoding to the input tensor
    def forward(self, x):
        # Get the size of the inputs to dynamically generate encodings of the
        # same size: Captures multiple sequence (actually spatial) dimensions
        _, *seq, emb = x.shape
        # Flatten all sequence/spatial dimensions to only have to enumerate the
        # position for a single axis
        seq = np.prod(seq)
        # Binary positional encoding fills the embedding dimension with the bit
        # pattern corresponding to the position in the sequence
        pos = torch.as_tensor([
            [(n & (1 << bit)) >> bit for bit in range(emb)] for n in range(seq)
        ])
        # Move the encoding tensor to the same device as the input tensor
        pos = pos.to(x.device, dtype=torch.float32).reshape(x.shape[1:])  # noqa
        # Add the quantized encoding to the quantized input
        #   Note: Convert encoding to bipolar representation
        return self.add(x, 2 * pos - 1)


# Activation quantizer configuration generator
from quant import act_quantizer


# Constructs a positional encoding layer by key and quantization settings
def get_positional(key, bits=None, return_quant_tensor=False):
    # Cannot return the quantized tensor if there is no quantization...
    return_quant_tensor = return_quant_tensor and (bits is not None)
    # Look up named positional encoding layers and insert quantizer
    # configuration
    return {
        "none": brevitas.nn.QuantIdentity(
            act_quantizer(bits), return_quant_tensor=return_quant_tensor
        ),
        "sinusoidal": QuantSinusoidalPositionalEncoding(
            act_quantizer(bits), None, return_quant_tensor=return_quant_tensor
        ),
        "binary": QuantBinaryPositionalEncoding(
            act_quantizer(bits), None, return_quant_tensor=return_quant_tensor
        ),
        "learned": LazyQuantLearnedPositionalEncoding(
            act_quantizer(bits), None, return_quant_tensor=return_quant_tensor
        ),
    }[key]
