# Use this as the base configuration for deriving activation, weight and bias
# quantizers
from brevitas.quant import (
    Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int8Bias
)


# Derive activation quantizer form the 8-bit per-tensor float quantizer from
# brevitas as the base, overwriting bit-with and signedness
def act_quantizer(bits, _signed=True):
    if bits is not None:
        class Quantizer(Int8ActPerTensorFloat):
            bit_width = bits
            signed = _signed

        return Quantizer
    return None


# Derive weight quantizer form the 8-bit per-tensor float quantizer from
# brevitas as the base, overwriting bit-with and signedness
def weight_quantizer(bits, _signed=True):
    if bits is not None:
        class Quantizer(Int8WeightPerTensorFloat):
            bit_width = bits
            signed = _signed

        return Quantizer
    return None


# Derive bias quantizer form the 8-bit per-tensor float quantizer from brevitas
# as the base, overwriting bit-with and signedness
def bias_quantizer(bits, _signed=True):
    if bits is not None:
        class Quantizer(Int8Bias):
            bit_width = bits
            signed = _signed

        return Quantizer
    return None
