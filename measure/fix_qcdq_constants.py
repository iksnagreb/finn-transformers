import onnx
import numpy as np
from onnx import numpy_helper


def dequantize_initializers(onnx_path: str, out_path: str):
    """Safely replace quantized int8/uint8 initializers with float32 tensors.

    Rules:
    - Only replace integer initializers that are NOT used as zero_point inputs
      (input index 2) for QuantizeLinear/DequantizeLinear nodes.
    - Prefer the actual scale initializer referenced by a DequantizeLinear
      consumer (node.input[1]) when available.
    - Keep other initializers untouched to avoid creating invalid operator
      input types (which breaks ONNX Runtime).
    """
    model = onnx.load(onnx_path)
    name_to_init = {i.name: i for i in model.graph.initializer}

    # Build consumer map: initializer name -> list of (node, input_index)
    consumers = {}
    for node in model.graph.node:
        for idx, inp in enumerate(node.input):
            if inp:
                consumers.setdefault(inp, []).append((node, idx))

    replaced = 0
    new_initializers = []

    for old in model.graph.initializer:
        name = old.name
        arr = numpy_helper.to_array(old)

        # Only consider integer quantized tensors
        if arr.dtype == np.int8 or arr.dtype == np.uint8:
            # If this initializer is used as a zero_point for any Q/DQ node, skip it
            is_zero_point = False
            for (node, idx) in consumers.get(name, []):
                if node.op_type in ("QuantizeLinear", "DequantizeLinear") and idx == 2:
                    is_zero_point = True
                    break
            if is_zero_point:
                new_initializers.append(old)
                continue

            # Try to find a scale initializer. Prefer explicit DequantizeLinear consumer.
            scale_name = None
            for (node, idx) in consumers.get(name, []):
                if node.op_type == "DequantizeLinear" and idx == 0:
                    # DequantizeLinear inputs: [x, scale, zero_point]
                    if len(node.input) > 1 and node.input[1] in name_to_init:
                        scale_name = node.input[1]
                        break

            # Fallback heuristics (older heuristic kept for compatibility)
            if scale_name is None:
                if name.endswith('Constant_1_output_0'):
                    prefix = name[:-len('Constant_1_output_0')]
                    candidate = prefix + 'Constant_output_0'
                    if candidate in name_to_init:
                        scale_name = candidate
                else:
                    candidate = name.replace('_1_', '_')
                    if candidate in name_to_init:
                        scale_name = candidate

            if scale_name and scale_name in name_to_init:
                scale_init = name_to_init[scale_name]
                scale_arr = numpy_helper.to_array(scale_init).astype(np.float32)
                q = arr.astype(np.float32)
                try:
                    # multiply with broadcasting if needed
                    f = q * scale_arr
                except Exception:
                    # If shapes don't align, try using the first scalar of scale
                    scale_val = float(np.asarray(scale_arr).reshape(-1)[0])
                    f = q * scale_val

                new_init = numpy_helper.from_array(f.astype(np.float32), name)
                new_initializers.append(new_init)
                replaced += 1
                continue

        # default: keep original
        new_initializers.append(old)

    if replaced > 0:
        model.graph.ClearField('initializer')
        model.graph.initializer.extend(new_initializers)
        onnx.save(model, out_path)
    return replaced


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: fix_qcdq_constants.py input.onnx output.onnx')
        sys.exit(2)
    inp, out = sys.argv[1], sys.argv[2]
    n = dequantize_initializers(inp, out)
    print('Replaced', n, 'initializers')
