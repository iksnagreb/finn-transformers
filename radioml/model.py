# Shallow copies of module containers for per-repetition normalization
import copy
# Neural Network building blocks
import torch
# Base class of all batch-norm variants, including the lazy ones
from torch.nn.modules.batchnorm import _BatchNorm
# Mixin marking lazily initialized modules, e.g., LazyBatchNorm1d
from torch.nn.modules.lazy import LazyModuleMixin
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


# Creates a copy of a module sharing all submodules (and thus all parameters,
# buffers and quantizer state) with the original, except for batch-norm layers
# which are replaced by fresh instances. Containers along the path to a
# batch-norm layer are shallow-copied to hold the new batch-norm instance.
# Batch-norm running statistics differ significantly between the stacked
# repetitions of a block, sharing them breaks the model in evaluation mode.
def share_except_batch_norm(module: torch.nn.Module):
    # Batch-norm layers get a fresh, independent instance with the same options
    # (cannot deepcopy lazy batch-norm with uninitialized buffers)
    if isinstance(module, _BatchNorm):
        options = dict(
            eps=module.eps, momentum=module.momentum, affine=module.affine,
            track_running_stats=module.track_running_stats
        )
        # Lazy variants infer the number of features at the first forward
        if isinstance(module, LazyModuleMixin):
            return type(module)(**options)
        return type(module)(module.num_features, **options)
    # Subtrees without any batch-norm are reused as is, i.e., fully shared
    if not any(isinstance(m, _BatchNorm) for m in module.modules()):
        return module
    # Shallow-copy the container, then rebuild its submodule dictionary with
    # shared or fresh children. Copy the parameter and buffer dictionaries (not
    # the tensors) so registering into the clone does not affect the original
    clone = copy.copy(module)
    clone._parameters = dict(module._parameters)
    clone._buffers = dict(module._buffers)
    clone._modules = {
        name: share_except_batch_norm(child)
        for name, child in module._modules.items()
    }
    return clone


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
            # Repeat the same block instances num_layers times (shared
            # parameters) or instantiate each repetition individually
            share_weights=True,
            # Only relevant if share_weights is true: Also share the batch-norm
            # layers (and thus their running statistics) between repetitions or
            # give each repetition its own batch-norm while all other layers
            # remain shared
            share_norm=True,
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

        # Instantiates the configuration into a fresh list of layer instances
        def instantiate(configuration):
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

            # Return the list of configured layer instances
            return configuration

        # Unpack and repeat the configured sequence of blocks: Either repeating
        # the same block instances (shared parameters) or instantiating the
        # configuration individually for each repetition (unshared parameters)
        if share_weights:
            # Configure the shared sequence of block instances once
            shared = instantiate(configuration)
            # Repeat the very same block instances, including normalization
            if share_norm:
                blocks = num_layers * shared
            # First repetition uses the original instances, all following
            # repetitions get their own batch-norm layers but share everything
            # else
            else:
                blocks = shared + [
                    share_except_batch_norm(block)
                    for _ in range(num_layers - 1) for block in shared
                ]
        else:
            blocks = [
                block for _ in range(num_layers)
                for block in instantiate(configuration)
            ]

        # Default positional encoding
        if positional is None:
            positional = {"encoding": "sinusoidal"}

        # Stacked attention, MLP, convolution, pooling and normalization layers
        # as feature extractor/encoder
        self.enc = torch.nn.Sequential(
            # Insert a (quantized) positional encoding layer between embedding
            # and encoder stack
            get_positional(**positional),
            # Stack of (shared or unshared) encoder blocks
            *blocks,
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