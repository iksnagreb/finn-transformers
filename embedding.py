# Neural Network building blocks
import torch
# Brevitas quantized equivalents of PyTorch layers
from brevitas.nn import QuantIdentity

# Selection of supported activation functions shared by different models
from activations import ACTIVATIONS
# Lazy initialization versions of Brevitas layers
from lazy import LazyQuantConv2d


# Quantized convolutional embedding layer similar to patch embeddings of ViT but
# with arbitrary kernel size, stride and padding followed by adaptive average
# pooling to aggregate into a fixed number of patches
class PatchEmbedding(torch.nn.Module):
    def __init__(
            self,
            # Number of embedding dimensions to produce at the output
            dim,
            # Number of patches, each of dim channels, to produce at the output
            patches,
            # Kernel size of the initial convolution generating the sliding
            # window
            kernel_size,
            # Activation functions to use after convolution and linear layers
            #   Options: tanh, sigmoid, relu, gelu, silu, none
            activation="relu",
            # Quantization bitwidth for weights and activations: None means no
            # quantization
            bits=None,
            # Keyword arguments going to the convolution layer configuring
            # stride, padding, etc.
            **kwargs
    ):
        super().__init__()

        # Weight quantizer configuration: Disables quantizer if bits are None
        weight_quant = (
            {"weight_bit_width": bits} if bits else {"weight_quant": None}
        )

        # Convolution plus pooling layer generating patches of the desired size
        # from the 2d input
        self.patches = torch.nn.Sequential(
            # Insert optional activation quantizer if enabled
            *([QuantIdentity(bit_width=bits)] if bits else []),
            # Quantized convolution of kernal size and stride generating
            # patches
            LazyQuantConv2d(dim, kernel_size, **kwargs, **weight_quant),
            # Normalization between convolution and activation function - this
            # is always a batch norm and not configurable
            torch.nn.LazyBatchNorm2d(),
            # Select and instantiate activation functions from the dictionary
            # defined above
            ACTIVATIONS[activation](),
            # Insert optional activation quantizer if enabled
            *([QuantIdentity(bit_width=bits)] if bits else []),
            # Pooling layer to reduce the feature map to the expected number of
            # patches
            torch.nn.AdaptiveAvgPool2d(patches),
        )

    def forward(self, x):
        return self.patches(x)
