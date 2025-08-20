# with torch.no_grad()
import torch

# Mixin and parameter placeholder for PyTorch models lazily initializing
# parameters at the first forward pass
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn import UninitializedParameter

# Brevitas quantized layers for which lazy variants are derived in the following
from brevitas.nn import QuantLinear, QuantConv1d, QuantConv2d, QuantConv3d


class LazyQuantLinear(LazyModuleMixin, QuantLinear):  # noqa: abstract methods
    cls_to_become = QuantLinear
    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(self, out_features: int, bias: bool = True, *args, device=None,
                 dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}

        # Initializes the QuantLinear without creating actual parameter tensors
        super().__init__(
            0, 0, False, *args, device=device, dtype=dtype, **kwargs
        )

        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_features = out_features
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, x) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = x.shape[-1]
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()


# Provides most of the handling of lazy convolution layers in general
from torch.nn.modules.conv import _LazyConvXdMixin  # noqa: Protected member


class LazyQuantConv1d(_LazyConvXdMixin, QuantConv1d):  # noqa: abstract methods
    cls_to_become = QuantConv1d

    def __init__(self,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 bias=True,
                 *args,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__(
            0,
            0,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            padding_mode,
            bias,
            *args,
            device=device,
            dtype=dtype,
            **kwargs
        )

        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_channels = out_channels
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def _get_num_spatial_dims(self) -> int:
        return 1


class LazyQuantConv2d(_LazyConvXdMixin, QuantConv2d):  # noqa: abstract methods
    cls_to_become = QuantConv2d

    def __init__(self,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 bias=True,
                 *args,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__(
            0,
            0,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            padding_mode,
            bias,
            *args,
            device=device,
            dtype=dtype,
            **kwargs
        )

        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_channels = out_channels
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def _get_num_spatial_dims(self) -> int:
        return 2


class LazyQuantConv3d(_LazyConvXdMixin, QuantConv3d):  # noqa: abstract methods
    cls_to_become = QuantConv3d

    def __init__(self,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 bias=True,
                 *args,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__(
            0,
            0,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            padding_mode,
            bias,
            *args,
            device=device,
            dtype=dtype,
            **kwargs
        )

        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_channels = out_channels
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def _get_num_spatial_dims(self) -> int:
        return 3
