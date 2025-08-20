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

# Path to the RadioML dataset
RADIOML_PATH = os.environ["RADIOML_PATH"]


# Exports the model to ONNX in conjunction with an input-output pair for
# verification
def export(model, dataset, batch_size, split_heads=False, **kwargs):  # noqa
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
    np.save("outputs/radioml/out.npy", out.numpy())


# Script entrypoint
if __name__ == "__main__":
    # Load the stage parameters from the parameters file
    params = dvc.api.params_show(stages="radioml/dvc.yaml:export")
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = Model(**params["model"])
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/radioml/model.pt"))
    # Pass the model and the export configuration to the evaluation loop
    export(model, dataset=params["dataset"], **params["export"])
