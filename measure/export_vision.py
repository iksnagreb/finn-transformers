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

from torchvision import datasets, transforms

# Export brevitas quantized models to QONNX dialect
from brevitas.export import export_qonnx, export_onnx_qcdq

# The Vision classification model
from vision.model import Model
# Quantized custom implementation of multihead attention
from attention import QuantMultiheadAttention
# Seeding RNGs for reproducibility, affine parameter export patching
from utils import seed, patch_missing_affine_norms

import onnx
from onnxsim import simplify
import yaml


# Path to the RadioML dataset
# RADIOML_PATH = os.environ["RADIOML_PATH"]
# anpassen für vision
RADIOML_PATH = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.hdf5"
RADIOML_PATH_NPZ = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"










# Path to the CIFAR-10 dataset
CIFAR10_ROOT = os.environ.setdefault("CIFAR10_ROOT", "data")

# Export function mapping
EXPORTERS = {"qonnx": export_qonnx, "qcdq": export_onnx_qcdq}


# Exports the model to ONNX in conjunction with an input-output pair for
# verification
def export(model, dataset, batch_size, format="qonnx", split_heads=False,
           **kwargs):
    # Do the forward pass for generating verification data and tracing the model
    # for export on CPU only
    device = "cpu"
    # Move the model to the training device
    model = model.to(device)  # noqa: Shadows model...
    # Set model to evaluation mode
    model = model.eval()  # noqa: Shadows model...

    # Explicitly splits all attention heads in the model graph to be parallel
    if split_heads:
        # Iterate all modules in the model container and check for instances of
        # quantized multihead attention
        for name, module in model.named_modules():
            if isinstance(module, QuantMultiheadAttention):
                # Marks to take the split path next forward call
                module.split_heads = True

    # Transformation to be applied to the input images: Rather basic
    # preprocessing turning images into tensors and normalizing with only
    # minimal data augmentation
    tf = transforms.Compose([
        # Convert from PIL image to PyTorch tensors
        transforms.ToTensor(),
        # Random horizontal flip in 50% of the cases
        transforms.RandomHorizontalFlip(),
        # CIFAR-10 statistics on the whole training set
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])

    # Load the Vision test split (should already be in CIFAR10_ROOT, otherwise
    # download)
    dataset = datasets.CIFAR10(CIFAR10_ROOT, False, download=True, transform=tf)
    # Create a batched and shuffled data loader the Vision validation split
    export_data = DataLoader(dataset, batch_size=batch_size)

    # Sample the first batch from the export dataset
    inp, cls = next(iter(export_data))

    # Also save the model output predictions (probabilities)
    with torch.no_grad():
        out = model(inp)

    # Export the model to ONNX using the input example
    EXPORTERS[format](model, (inp,), "outputs/vision/model.onnx", **kwargs)

    # Save the input and output data for verification purposes later
    np.save("outputs/vision/inp.npy", inp.numpy())
    np.save("outputs/vision/out.npy", out.numpy())
    np.save("outputs/vision/cls.npy", cls.numpy())


# Script entrypoint


















# Exports the model to ONNX in conjunction with an input-output pair for
# verification
def export(model, model_int8, dataset, batch_size, split_heads=False, **kwargs):  # noqa
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
    export_qonnx(model, (inp,), "outputs/vision/model.onnx", **kwargs)

    # Save the input and output data for verification purposes later
    np.save("outputs/vision/inp.npy", inp.numpy())
    np.save("outputs/vision/out.npy", out.numpy())
    # Standard ONNX export for reference - works with dynamic batch sizes
    onnx_path = "outputs/vision/model_dynamic_batchsize.onnx"
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



# Script entrypoint
if __name__ == "__main__":
    # Load the stage parameters from the parameters file
    # params = dvc.api.params_show()
    # measure/params.yaml
    # vision/params.yaml
    with open("vision/params.yaml", "r") as f:
        params = yaml.safe_load(f)
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = Model(**params["model"])
    print("Created model instance.")
    for key, value in params["model"].items():
        print(f"{key}: {value}")
    print("int 8:")
    # for key, value in params["model_int8"].items():
    #     print(f"{key}: {value}")
    model_int8 = Model(**params["model"])
    print("Created model int8 instance.")
    
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/vision/model.pt"))
    print("loaded")

    model = patch_missing_affine_norms(model)
    # model_int8.load_state_dict(torch.load("outputs/radioml/model_int8.pt")) # model_int8.pt müsste noch hochgeladen werden
    # Pass the model and the export configuration to the evaluation loop
    export(model, model_int8, dataset=params["dataset"], **params["export"])
