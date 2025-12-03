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
from brevitas.export import export_qonnx, export_onnx_qcdq

# The RadioML classification model
from radioml.model import Model
# The RadioML modulation classification dataset
from radioml.dataset import get_datasets
# Quantized custom implementation of multihead attention
from attention import QuantMultiheadAttention
<<<<<<< HEAD
# Seeding RNGs for reproducibility
from utils import seed
import onnx
from onnxsim import simplify

=======
# Seeding RNGs for reproducibility, affine parameter export patching
from utils import seed, patch_missing_affine_norms
>>>>>>> origin/main

# Path to the RadioML dataset
# RADIOML_PATH = os.environ["RADIOML_PATH"]
RADIOML_PATH = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.hdf5"
RADIOML_PATH_NPZ = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"

# Export function mapping
EXPORTERS = {"qonnx": export_qonnx, "qcdq": export_onnx_qcdq}


# Exports the model to ONNX in conjunction with an input-output pair for
# verification
<<<<<<< HEAD
def export(model, model_int8, dataset, batch_size, split_heads=False, **kwargs):  # noqa
=======
def export(model, dataset, batch_size, format="qonnx", split_heads=False,
           **kwargs):
>>>>>>> origin/main
    # Do the forward pass for generating verification data and tracing the model
    # for export on CPU only
    device = "cpu"
    # Move the model to the training device
    model = model.to(device)  # noqa: Shadows model...
    model_int8 = model_int8.to(device)  # noqa: Shadows model...
    # Set model to evaluation mode
    model = model.eval()  # noqa: Shadows model...
    model_int8 = model_int8.eval()  # noqa: Shadows model...

    # Explicitly splits all attention heads in the model graph to be parallel
    if split_heads:
        # Iterate all modules in the model container and check for instances of
        # quantized multihead attention
        for name, module in model.named_modules():
            if isinstance(module, QuantMultiheadAttention):
                # Marks to take the split path next forward call
                module.split_heads = True
        for name, module in model_int8.named_modules():
            if isinstance(module, QuantMultiheadAttention):
                # Marks to take the split path next forward call
                module.split_heads = True

    # Load the RadioML dataset splits as configured
    _, _, eval_data = get_datasets(path=RADIOML_PATH, **dataset)
    # Create a batched and shuffled data loader the RadioML validation split
    export_data = DataLoader(eval_data, batch_size=batch_size, shuffle=True)

    # Sample the first batch from the export dataset
    inp, cls, _ = next(iter(export_data))

    # Also save the model output predictions (probabilities)
    with torch.no_grad():
        out = model(inp)

    # Export the model to ONNX using the input example
    EXPORTERS[format](model, (inp,), "outputs/radioml/model.onnx", **kwargs)

    # Save the input and output data for verification purposes later
    np.save("outputs/radioml/inp.npy", inp.numpy())
    np.save("outputs/radioml/out.npy", out.numpy())
    np.save("outputs/radioml/cls.npy", cls.numpy())

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
    print(f"Modell als ONNX exportiert: {onnx_path}")


    # Brevitas 8Bit export - problem: nicht möglich mit dynamischen batch-sizes, 
    # wenn man es im nachinein patched sind die reshapes noch statisch -> funktioniert nicht mit tensorrt

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        from brevitas.export import export_onnx_qcdq
        dummy_input = torch.randn(batch_size, *inp.shape[1:], dtype=inp.dtype)
        # test: wird das Ergebnis (Accuracy) besser mit echten daten?
        export_data = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
        inp, out, _ = next(iter(export_data))



        
        export_path=f"outputs/radioml/model_brevitas_{batch_size}.onnx"
        simplified_path=f"outputs/radioml/model_brevitas_{batch_size}_simpl.onnx"
        export_onnx_qcdq(
            model_int8, 
            (inp,),
            export_path=export_path,
            opset_version=17
        )
        print(f"Quantisiertes Modell erfolgreich exportiert für Batch-Größe: {batch_size}")

        # Lade ONNX-Modell
        model = onnx.load(export_path)
        # Simplify mit onnxsim
        model_simplified, check = simplify(model)
        if not check:
            print(f"[!] Vereinfachung fehlgeschlagen für Batch-Größe {batch_size}")
            continue
        onnx.save(model_simplified, simplified_path)
        print(f"Simplified gespeichert: {simplified_path}")


# Script entrypoint
if __name__ == "__main__":
    # Load the stage parameters from the parameters file
    params = dvc.api.params_show(stages="radioml/dvc.yaml:export")
    batch_sizes = params["batch_sizes"]
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = Model(**params["model"])
    model_int8 = Model(**params["model_int8"])
    
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/radioml/model.pt"))
<<<<<<< HEAD
    model_int8.load_state_dict(torch.load("outputs/radioml/model_int8.pt"))
=======
    # Prevent export and streamlining issues for missing affine normalization
    # parameters
    model = patch_missing_affine_norms(model)
>>>>>>> origin/main
    # Pass the model and the export configuration to the evaluation loop
    export(model, model_int8, dataset=params["dataset"], **params["export"])
