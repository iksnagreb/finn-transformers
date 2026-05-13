# source:

# outputs/radioml/model.pt
# System functionality like creating directories and reading env-vars
import os
# Use the DVC api for loading the YAML parameters
import dvc.api
# Save verification input-output pair as numpy array
import numpy as np
# PyTorch base package: Math and Tensor Stuff
import torch
# Loads shuffled batches from datasets
from torch.utils.data import DataLoader

# Export brevitas quantized models to QONNX dialect
from brevitas.export import export_qonnx

# The RadioML classification model
from radioml.model import Model
# The RadioML modulation classification dataset
from radioml.dataset import get_datasets
# Quantized custom implementation of multihead attention
from attention import QuantMultiheadAttention
# Seeding RNGs for reproducibility
from utils import seed
# TopK wrapper to reduce data transfer
from radioml.topk_wrapper import ModelTopKWrapper
import onnx
from onnxsim import simplify
import yaml


# Path to the RadioML dataset
# RADIOML_PATH = os.environ["RADIOML_PATH"]
RADIOML_PATH = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.hdf5"
RADIOML_PATH_NPZ = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"

MODEL_TYPE = "radioml"
with open(f"{MODEL_TYPE}/params.yaml", "r") as f:
    cfg = yaml.safe_load(f)

bits = cfg["model"]["embedding"].get("bits", 0)
INT8 = (bits == 8)

def remove_initializers_from_inputs_model(model: onnx.ModelProto) -> onnx.ModelProto:
    print("Removing initializers from graph inputs if they appear there...")
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    # Filter graph.input: behalte nur Inputs, die nicht Initializer sind
    new_inputs = [inp for inp in graph.input if inp.name not in initializer_names]
    if len(new_inputs) == len(graph.input):
        return model  # nichts zu tun
    del graph.input[:]
    graph.input.extend(new_inputs)
    onnx.checker.check_model(model)  # optional: Validität prüfen
    return model

# Exports the model to ONNX in conjunction with an input-output pair for
# verification
def export(model, dataset, batch_size, split_heads=False, **kwargs):  # noqa
    from brevitas.export import export_qonnx, export_onnx_qcdq
    # Do the forward pass for generating verification data and tracing the model
    # for export on CPU only
    device = "cpu"
    # Move the model to the training device
    model = model.to(device)  # noqa: Shadows model...
    # model_int8 = model_int8.to(device)  # noqa: Shadows model...
    # Set model to evaluation mode
    model = model.eval()  # noqa: Shadows model...
    # model_int8 = model_int8.eval()  # noqa: Shadows model...

    # Explicitly splits all attention heads in the model graph to be parallel
    if split_heads:
        # Iterate all modules in the model container and check for instances of
        # quantized multihead attention
        for name, module in model.named_modules():
            if isinstance(module, QuantMultiheadAttention):
                # Marks to take the split path next forward call
                module.split_heads = True
        # for name, module in model_int8.named_modules():
        #     if isinstance(module, QuantMultiheadAttention):
        #         # Marks to take the split path next forward call
        #         module.split_heads = True

    # Load the RadioML dataset splits as configured
    _, _, eval_data = get_datasets(path=RADIOML_PATH, **dataset)
    # Create a batched and shuffled data loader the ImageNet validation split
    export_data = DataLoader(eval_data, batch_size=batch_size, shuffle=True)

    # Sample the first batch from the export dataset
    inp, out, _ = next(iter(export_data))

    # Export the model to ONNX using the input example
    export_qonnx(model, (inp,), "outputs/radioml/model.onnx", **kwargs)

    # Save the input and output data for verification purposes later
    np.save("outputs/radioml/inp.npy", inp.numpy())
    
    # Handle both regular output (tensor) and wrapped output (tuple of tensors)
    with torch.no_grad():
        model_out = model(inp)

    # not needed:
    # if isinstance(model_out, tuple):
    #     # Wrapper returns (topk_values, topk_indices)
    #     topk_values, topk_indices = model_out
    #     np.save("outputs/radioml/out_topk_values.npy", topk_values.numpy())
    #     np.save("outputs/radioml/out_topk_indices.npy", topk_indices.numpy())
    #     print(f"Saved TopK outputs: values shape {topk_values.shape}, indices shape {topk_indices.shape}")
    # else:
    #     # Regular model output
    np.save("outputs/radioml/out.npy", out.numpy())

    # Standard ONNX export for reference - works with dynamic batch sizes
    onnx_path = "outputs/radioml/model_dynamic_batchsize.onnx"
    torch.onnx.export(
        model,
        (inp,),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model successfully exported as ONNX: {onnx_path}")

    # remove initializers from inputs
    m = onnx.load(onnx_path)
    m = remove_initializers_from_inputs_model(m)
    onnx.save(m, onnx_path)
    print(f"Fixed initializers and saved: {onnx_path}")


    if INT8:
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            # WICHTIG: Nicht aus dem DataLoader sampeln, da der letzte
            # Rest-Batch kleiner sein kann (z. B. 726 statt 1024) und damit
            # statische ONNX-Formen falsch exportiert werden.
            dummy_input = torch.randn(batch_size, *inp.shape[1:], dtype=inp.dtype)
            # test: wird das Ergebnis (Accuracy) besser mit echten daten?
            export_data = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
            inp, out, _ = next(iter(export_data))
            
            export_path=f"outputs/radioml/model_brevitas_{batch_size}.onnx"
            simplified_path=f"outputs/radioml/model_brevitas_{batch_size}_simple.onnx"
            export_onnx_qcdq(
                model, 
                (dummy_input,),
                export_path=export_path,
                opset_version=17,
                export_as_int8=True,
                quant_type='int'
                # quantize_bias=True,
                # fold_batch_norm=True
            )
            print(f"Quantized Model successfully exported for Batch Size: {batch_size}")

            model_load = onnx.load(export_path)
            # # Simplify mit onnxsim
            model_simplified, check = simplify(model_load)
            if not check:
                print(f"[!] Simplification failed for Batch Size {batch_size}")
                continue
            onnx.save(model_simplified, simplified_path)
            print(f"Simplified saved: {simplified_path}")
            # remove initializers from inputs
            model_simplified = remove_initializers_from_inputs_model(model_simplified)
            onnx.save(model_simplified, simplified_path)
            


# Script entrypoint
if __name__ == "__main__":

    with open("radioml/params.yaml", "r") as f:
        params = yaml.safe_load(f)
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = Model(**params["model"])
    print("Created model instance.")
    for key, value in params["model"].items():
        print(f"{key}: {value}")
    
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/radioml/model.pt")) #(int8)
    # model.load_state_dict(torch.load("outputs/radioml/model_fp32.pt")) #(fp32)
    model = ModelTopKWrapper(model, k=5)  # to minimize the output data
    export(model, dataset=params["dataset"], **params["export"])

