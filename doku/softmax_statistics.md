-> Wert hinter softmax bei ausführung anschauen, liegen die über 0.5? uint8, und int8

denn:
softmax ist noch unsigned und es gibt warnungen für tensorrt, aber dei accuracy für das modell ist trotzdem gut.


Als die warnungen für relu (aktivierungsfunktion) kamen, wurde die accuracy sehr schlecht, damit konnte tensorrt nicht umgehen. 


[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.
[10/14/2026-13:12:49] [TRT] [W] onnxOpImporters.cpp:1458: TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.

kannst du erstmal ein skript schreiben, das das vision model ausführt und die werte nach der softmax funktion ausgibt machen? Kannst du dann die werte grafisch darstellen?


nutze dieses Modell um das zu testen: model_brevitas_1_simple