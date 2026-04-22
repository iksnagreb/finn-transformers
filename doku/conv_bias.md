# Problem:
2026-04-15 17:29:35.454313044 [W:onnxruntime:Default, conv.cc:425 UpdateState] OP Conv(/emb/emb.1/patches/patches.1/Conv) running in Fallback mode. May be extremely slow.

# Set quantize bias parameter in export
- beim export export_onnx_qcdq -> quantize_bias=True
    - macht keinen unterschied, der bias wird nicht quantisiert

# Set bias true in embeddings.py (training)

## Do it like in the brevitas doku
Brevitas Doku: https://xilinx.github.io/brevitas/v0.12.1/tutorials/quant_tensor_quant_conv2d_overview.html
Onnx QDQ Format for Convolutions doku: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- LazyQuantConv2d: 
    - bias=True
    - bias_quant=Int8Bias (from brevitas.quant.scaled_int import Int8Bias)

Error while training:
```shell
raise RuntimeError("Input scale required")
RuntimeError: Input scale required
```
-> expects a quantized input -> set return_quant_tensor to true
``` python
*([QuantIdentity(bit_width=bits, return_quant_tensor=True)] if bits else []),
LazyQuantConv2d(dim, kernel_size, **kwargs, **weight_quant, bias_quant=Int8Bias,return_quant_tensor=True), 
```

The training, export and inference is working now in general, but the error from the beginning is still there.

Netron Screenshot of Radioml Model:

![Netron Screenshot](conv_quantized_bias.png)


The warning in onnxruntime is still there, even if the format now aligns with the onnx screenshot with the correct quantized convolution:

Netron Screenshot in ORT Doku:

![Netron Onnx Doku QDQ Conv](onnx_doku_conv.png)


2026-04-22 09:51:33.129996076 [W:onnxruntime:Default, conv.cc:425 UpdateState] OP Conv(/emb/emb.1/patches/patches.1/Conv) running in Fallback mode. May be extremely slow.


## Viele ONNX Runtime backends erwarten int32 bias.
from brevitas.quant.scaled_int import Int32Bias

## Batch Norm after Conv
export_onnx_qcdq(..., fold_batch_norm=True)