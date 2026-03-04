import onnx
import numpy as np
from onnx import numpy_helper


def dequantize_initializers(onnx_path: str, out_path: str):
    """Replace int8/uint8 quantized initializers with dequantized float tensors.

    This multiplies the integer tensor by a nearby scalar initializer (scale)
    when available. It looks for pairs like
    '<prefix>/Constant_output_0' (float scale) and
    '<prefix>/Constant_1_output_0' (int8 tensor).
    """
    model = onnx.load(onnx_path)
    name_to_init = {i.name: i for i in model.graph.initializer}

    replaced = 0
    for name, init in list(name_to_init.items()):
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.int8 or arr.dtype == np.uint8:
            # try find scale initializer with similar prefix
            # common pattern: '<prefix>/Constant_1_output_0' and '<prefix>/Constant_output_0'
            if name.endswith('Constant_1_output_0'):
                prefix = name[:-len('Constant_1_output_0')]
                scale_name = prefix + 'Constant_output_0'
            else:
                # fallback: try replace '_1_' with '_'
                scale_name = name.replace('_1_', '_')

            if scale_name in name_to_init:
                scale_init = name_to_init[scale_name]
                scale_arr = numpy_helper.to_array(scale_init)
                # scale may be scalar
                scale_val = float(np.asarray(scale_arr).reshape(-1)[0])
                q = arr.astype(np.float32)
                f = q * scale_val
                new_init = numpy_helper.from_array(f.astype(np.float32), name)
                # replace initializer in model.graph.initializer
                for i, old in enumerate(model.graph.initializer):
                    if old.name == name:
                        model.graph.initializer[i] = new_init
                        replaced += 1
                        break
    if replaced > 0:
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
