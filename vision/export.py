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
# PyTorch vision datasets and transformations
from torchvision import datasets, transforms

# Export brevitas quantized models to QONNX dialect
from brevitas.export import export_qonnx

# The Vision classification model
from vision.model import Model
# Quantized custom implementation of multihead attention
from attention import QuantMultiheadAttention
# Seeding RNGs for reproducibility
from utils import seed

# Path to the CIFAR-10 dataset
CIFAR10_ROOT = os.environ.setdefault("CIFAR10_ROOT", "data")


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
    export_qonnx(model, (inp,), "outputs/vision/model.onnx", **kwargs)

    # Save the input and output data for verification purposes later
    np.save("outputs/vision/inp.npy", inp.numpy())
    np.save("outputs/vision/out.npy", out.numpy())
    np.save("outputs/vision/cls.npy", cls.numpy())


# Script entrypoint
if __name__ == "__main__":
    # Load the stage parameters from the parameters file
    params = dvc.api.params_show(stages="vision/dvc.yaml:export")
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = Model(**params["model"])
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/vision/model.pt"))
    # Pass the model and the export configuration to the evaluation loop
    export(model, dataset=params["dataset"], **params["export"])
