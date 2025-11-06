# Neural Network building blocks
import torch
# Brevitas quantized equivalents of PyTorch layers
from brevitas.nn import QuantIdentity

# Lazy initialization versions of Brevitas layers
from lazy import LazyQuantLinear
# Quantized positional encoding variants
from encoding import get_positional
# Quantized patch embedding for ViT-like models
from embedding import PatchEmbedding
# Collection of reusable, named and configurable Transformer blocks
from blocks import BLOCKS, ORIGINAL, CONFORMER, TRANSFORMER_CONFIGURATIONS

# Tensor packing/unpacking operations in convenient Einstein notation
from einops import pack, unpack
# Einops layers for rearranging data with convenient Einstein notation
from einops.layers.torch import Rearrange


# Configurable Transformer implementation with patch embedding, positional
# encoding, Transformer-encoder stack (original, conformer, or custom) and
# linear classifier at the end. Can also be instantiated without any attention
# operators, e.g., when training baselines or when conducting ablation studies.
class Model(torch.nn.Module):
    def __init__(
            self,
            # Number of output classes to predict (24 RadioML classes)
            num_classes=24,
            # Configuration of the initial patch-embedding layer
            embedding=None,
            # Type of positional encoding to use at the input
            positional=None,
            # List of layers configuring the encoder stack: Either a string
            # referring to a pre-defined configuration or a list of individual
            # layer configurations
            configuration=ORIGINAL,
            # Number of layers, i.e., stacked repetitions of the encoder
            # configuration above
            num_layers=1,
            # Embedding dimension shared by all blocks, i.e., embedding,
            # attention, MLP, convolution, etc.
            emb_dim=512,
            # Number of quantization bits for weights and activations (for all
            # intermediate layers)
            bits=None,
            # Number of quantization bits for weights and activations of the
            # output classification layer
            cls_bits=None,
            # Keyword arguments are forwards as global settings to eac of the
            # blocks in configuration (alongside some of the explicit options
            # above) unless they are overwritten by local options as part of the
            # configuration dictionary
            **kwargs
    ):
        super().__init__()

        # Patch embedding layer generating embedding vectors for patches of
        # sliding windows from the input
        self.emb = torch.nn.Sequential(
            # RadioML data comes in with sequence (temporal) dimension before
            # channels but is treated as an image in channels-first layout
            Rearrange("b h w c -> b c h w"),
            # Patch embedding generating the embedding dimension from sliding
            # windows of the input
            *([PatchEmbedding(dim=emb_dim, **embedding)] if embedding else []),
            # Rearrange from channels-first back to channels-last sequence-first
            # layout
            Rearrange("b c h w -> b h w c"),
        )

        # Map named block configuration to the actual tuple configuration from
        # the configuration dictionary
        if isinstance(configuration, str):
            configuration = TRANSFORMER_CONFIGURATIONS[configuration]

        # Turn the configuration into a list supporting assignment (might be
        # tuple when coming from configurations above)
        configuration = list(configuration)

        # Convert mixed configuration (string, type, object) to a list of
        # configured layer instances
        for i, layer in enumerate(configuration):
            # Skip already fully configured layer instances, i.e., any PyTorch
            # Module instance
            if isinstance(layer, torch.nn.Module):
                continue

            # Start with a fresh global configuration dictionary for each layer,
            # potentially overwriting options with block-local configurations
            global_config = {"emb_dim": emb_dim, "bits": bits, **kwargs}

            # If the layer is a string, this must be an identifier into the
            # named layers above - lookup and replace by the corresponding class
            if isinstance(layer, str):
                layer = BLOCKS[layer]

            # If the layer is a type, this must be an unconfigured class type to
            # be configured and put into the configuration list
            if isinstance(layer, type):
                configuration[i] = layer(**global_config)

            # If layer is a dictionary, this must be a key followed by
            # configuration of the blocks above
            if isinstance(layer, dict):
                # The dictionary must be a single key-configuration pair
                key, config = tuple(*layer.items())
                # Merge block-local with global configuration with local taking
                # precedence
                global_config.update(config)
                # Load the configuration into the block type
                configuration[i] = BLOCKS[key](**global_config)

        # Default positional encoding
        if positional is None:
            positional = {"encoding": "sinusoidal"}

        # Stacked attention, MLP, convolution, pooling and normalization layers
        # as feature extractor/encoder
        self.enc = torch.nn.Sequential(
            # Insert a (quantized) positional encoding layer between embedding
            # and encoder stack
            get_positional(**positional),
            # Unpack and repeat the configured sequence of blocks
            *(num_layers * configuration),
            # RadioML data comes in with sequence (temporal) dimension before
            # channels but is treated as an image in channels-first layout
            Rearrange("b h w c -> b c h w"),
            # Global average pooling to flatten the feature map
            torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten()
        )

        # Weight quantizer configuration: Disables quantizer if bits are None
        weight_quant = (
            {"weight_bit_width": cls_bits} if bits else {"weight_quant": None}
        )

        # Linear layers as a classifier
        self.cls = torch.nn.Sequential(
            # A single Linear layer as the output layer - softmax activation
            # should be covered by the cross entropy loss
            LazyQuantLinear(num_classes, **weight_quant),
            # Insert optional activation quantizer if enabled
            *([QuantIdentity(bit_width=cls_bits)] if bits else []),
        )

    def forward(self, x):
        return self.cls(self.enc(self.emb(x)))
