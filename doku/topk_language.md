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
