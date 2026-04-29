# Problem: 
Durch das große Output nach der inferenz beim language model, wird durch das kopieren des outputs von der gpu zur cpu sehr viel datatransfer time verbraucht. Das soll für das int8 modell in tensorrt verringert werden, indem nur die top5 zur cpu kopiert werden.

## Do the topk computation on the gpu during the inference (outside of the model)
- geändert in measure (vor dem copy zur cpu top5 bestimmen)
- geändert in tegrastats (vor dem copy zur cpu top5 bestimmen)
- RuntimeError: CUDA error: no kernel image is available for execution on the device for
    - argmax:       ```np.argmax(output_cpu, axis=-1)```
    - torch topk    ```torch.topk(device_output, k=5, dim=-1)```
    - torch sort:   ```torch.argsort(device_output, descending=True, dim=-1)[:, :, :5]```

## add a topk node in the onnx modell (as a wrapper)
    - RuntimeError: Fehler beim Bauen der TensorRT-Engine: serialized_engine ist None.
        - wrapper for argmax in onnx modell (on int 8)
        - wrapper for argmax in onnx modell (on fp32, with a cast)
    - andere library (cupy) für argmax nutzen: build failed


    Serialized engine: 
[04/15/2026-15:03:12] [TRT] [E] Error Code: 9: Skipping tactic 0x0000000000000000 due to exception [shape.cpp:~op_constraints_msg_streamer_t:136] 
Error during shape inference of

/enc/enc_1/pre_norm/pre_norm_1/BatchNormalization_scale_out = mul(ONNXTRT_unsqueezeTensor_output', /enc/enc_1/pre_norm/pre_norm_1/BatchNormalization/enc/enc_1/pre_norm/pre_norm_1/BatchNormalization_scale_wFloat), name=/enc/enc_1/pre_norm/pre_norm_1/BatchNormalization_scale
Error is:


Input 0's element type (int8) differs from input 1's element type (
Error Code 10: Internal Error (Could not find any implementation for node {ForeignNode[ONNXTRT_castHelper...ONNXTRT_castHelper_307]}.)


- dequantize node manuell hinzugefügt
- cast manuell hinzugefügt
- identity node mit fp32 manuell hinzugefügt

Chat GPT sagt, der Fehler liegt an:
- global graph optimization
- quantization re-propagation
- layer fusion before execution planning
ohne topk konnte tensorrt noch ein fallback machen, jetzt aus irgendeinem Grund nicht mehr...

--> Keine Möglichkeit gefunden, den Datentransfer GPU->CPU zu verringern ....

# use onnxscript

same error when trying to build the engine:
Serialized engine: 
[04/23/2026-09:24:20] [TRT] [E] Error Code: 9: Skipping tactic 0x0000000000000000 due to exception [shape.cpp:~op_constraints_msg_streamer_t:136] 
Error during shape inference of
/enc/enc_1/pre_norm/pre_norm_1/BatchNormalization_scale_out = mul(ONNXTRT_unsqueezeTensor_output', /enc/enc_1/pre_norm/pre_norm_1/BatchNormalization/enc/enc_1/pre_norm/pre_norm_1/BatchNormalization_scale_wFloat), name=/enc/enc_1/pre_norm/pre_norm_1/BatchNormalization_scale
Error is:
Input 0's element type (int8) differs from input 1's element type (
[04/23/2026-09:24:20] [TRT] [E] IBuilder::buildSerializedNetwork: Error Code 10: Internal Error (Could not find any implementation for node {ForeignNode[ONNXTRT_castHelper...ONNXTRT_castHelper_307]}.)
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/hanna/git/finn-transformers/measure/measure.py", line 700, in <module>
    accuracy = run_accuracy_eval(batch_size, input_info, output_info, DATA_PATH_NPZ, onnx_model_path)
  File "/home/hanna/git/finn-transformers/measure/measure.py", line 650, in run_accuracy_eval
    engine, context = build_tensorrt_engine(onnx_model_path, test_loader, 1, input_info)
  File "/home/hanna/git/finn-transformers/measure/measure.py", line 392, in build_tensorrt_engine
    raise RuntimeError("Fehler beim Bauen der TensorRT-Engine: serialized_engine ist None.")
RuntimeError: Fehler beim Bauen der TensorRT-Engine: serialized_engine ist None.


# two outputs:
but: if I leave the old output and only add the topk node, it works fine! (with onnxscript and onnxhelper)
Problem: accuracy ist bei tensorrt schlechter als bei onnxruntime, bei tesnorrt werden immer nur die gleichen labels zurückgegeben 

Das Problem: TensorRT kann die TopK-Operation bei INT8-Quantisierung nicht korrekt berechnen. Das ist eine TensorRT-Limitation.

ORT CPU produces different logits per token, but TensorRT produces identical logits. This suggests the TensorRT engine serialization/conversion is the problem, not the model itself.



# another approach: torch method??  1. pytorch methode ändern: topk (vor dem export)

- build a wrapper around the Model -> trainng not needed
- Für Top-5 nur für das letzte Token:
TopK nicht nachträglich ins ONNX-Modell wrappen, sondern direkt im ursprünglichen PyTorch-Modell als Teil des forward() einbauen und dieses geänderte Modell exportieren.



can you generate me the wrapper file and explain the workflow?