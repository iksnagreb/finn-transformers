# System functionality like creating directories and reading env-vars
import os
# Use the DVC api for loading the YAML parameters
import dvc.api
# Progressbar
from tqdm import trange
# Save verification input-output pair as numpy array
import numpy as np
# PyTorch base package: Math and Tensor Stuff
import torch

# Export brevitas quantized models to QONNX dialect
from brevitas.export import export_qonnx

# The benchmark model
from benchmark.model import Model
# Quantized custom implementation of multihead attention
from attention import QuantMultiheadAttention
# Seeding RNGs for reproducibility
from utils import seed


# Generates "training" data for quantizer/norm layer calibration
def get_data(range, shape, num):  # noqa: Shadows "range"...
    # Generate uniformly spaced, batched inputs over the range
    return torch.linspace(*range, num).reshape((-1, *[1 for _ in shape])) * torch.ones(1, *shape)


# Exports the model to ONNX in conjunction with an input-output pair for
# verification
def export(model, dataset, batch_size, split_heads=False, **kwargs):  # noqa
    # No gradient accumulation for calibration passes required
    with torch.no_grad():
        # Check whether GPU training is available and select the appropriate
        # device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move the model to the training device
        model = model.to(device)  # noqa: Shadows model...
        # Multiple passes of calibration might be necessary for larger/deep
        # models
        for _ in trange(0, 1024, desc="calibrating"):
            # Generate 32 random data samples and feed through the model
            model(get_data(**dataset, num=32).to(device))
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

    # No gradient accumulation for export passes required
    with torch.no_grad():
        # Generate input data for model verification
        inp = get_data(**dataset, num=batch_size)
        # Generate model output on this random data for verification
        out = model(inp)

    # Export the model to ONNX using the input example
    export_qonnx(model, (inp,), "outputs/benchmark/model.onnx", **kwargs)

    # Save the input and output data for verification purposes later
    np.save("outputs/benchmark/inp.npy", inp.numpy())
    np.save("outputs/benchmark/out.npy", out.numpy())


# Script entrypoint
if __name__ == "__main__":
    # Load the stage parameters from the parameters file
    params = dvc.api.params_show(stages="benchmark/dvc.yaml:export")
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = Model(**params["model"])
    # Create the output directory if it does not already exist
    os.makedirs("outputs/benchmark", exist_ok=True)
    # Pass the model and the export configuration to the evaluation loop
    export(model, dataset=params["dataset"], **params["export"])
