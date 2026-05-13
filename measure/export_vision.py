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
from torchvision import datasets, transforms

# Export brevitas quantized models to QONNX dialect
from brevitas.export import export_qonnx, export_onnx_qcdq

# The Vision classification model
from vision.model import Model
# Quantized custom implementation of multihead attention
from attention import QuantMultiheadAttention
# Seeding RNGs for reproducibility, affine parameter export patching
from utils import seed, patch_missing_affine_norms
# TopK wrapper to reduce data transfer
from vision.topk_wrapper import ModelTopKWrapper

import onnx
from onnxsim import simplify
import yaml
import onnxruntime as ort

# Path to the RadioML dataset
# RADIOML_PATH = os.environ["RADIOML_PATH"]
# anpassen für vision
RADIOML_PATH = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.hdf5"
RADIOML_PATH_NPZ = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"


# Path to the CIFAR-10 dataset
CIFAR10_ROOT = R"/data/gitlab"
MODEL_TYPE = "vision"

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

    dataset = datasets.CIFAR10(CIFAR10_ROOT, False, download=True, transform=tf)
    # Create a batched and shuffled data loader the Vision validation split
    export_data = DataLoader(dataset, batch_size=batch_size)

    # accuracy on pt model
    it = iter(export_data)
    with torch.no_grad():
        for i in range(20):
            images, labels = next(it)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # Handle both tuple (TopK wrapper) and tensor (regular model) outputs
            if isinstance(outputs, tuple):
                topk_values, topk_indices = outputs
                preds = topk_indices[:, 0]  # Use top-1 prediction
            else:
                preds = outputs.argmax(dim=1)
            

    # Sample the first batch from the export dataset
    inp, cls = next(iter(export_data))

    # Also save the model output predictions (probabilities)
    with torch.no_grad():
        out = model(inp)

    # Export the model to ONNX using the input example
    # model ist wahrscheinlich schon quantisiert
    export_qonnx(model, (inp,), "outputs/vision/model.onnx", **kwargs)

    # Save the input and output data for verification purposes later
    np.save("outputs/vision/inp.npy", inp.numpy())
    
    # Handle both regular output (tensor) and wrapped output (tuple of tensors)
    if isinstance(out, tuple):
        # Wrapper returns (topk_values, topk_indices)
        topk_values, topk_indices = out
        np.save("outputs/vision/out_topk_values.npy", topk_values.numpy())
        np.save("outputs/vision/out_topk_indices.npy", topk_indices.numpy())
        print(f"Saved TopK outputs: values shape {topk_values.shape}, indices shape {topk_indices.shape}")
    else:
        # Regular model output
        np.save("outputs/vision/out.npy", out.numpy())
    
    np.save("outputs/vision/cls.npy", cls.numpy())


    # Unsupported model IR version: 10, max supported IR version: 9
    # node not valid mit opset 17
    # Standard ONNX export for reference - works with dynamic batch sizes
    if INT8 == True:
        print("quantisation -> export with qcdq")

        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            dummy_input = torch.randn(batch_size, *inp.shape[1:], dtype=inp.dtype)
            # test: wird das Ergebnis (Accuracy) besser mit echten daten?
            export_data = DataLoader(dataset, batch_size=batch_size)
            print("data now in ", CIFAR10_ROOT)

            # Sample the first batch from the export dataset
            inp, cls = next(iter(export_data))
            export_path=f"outputs/vision/model_brevitas_{batch_size}.onnx"
            simplified_path=f"outputs/vision/model_brevitas_{batch_size}_simple.onnx"

            export_onnx_qcdq(
                model, 
                (inp,),
                export_path=export_path,
                opset_version=17
            )
            print(f"Quantized Model successfully exported for Batch Size: {batch_size}")

        
            onnx_model = onnx.load(export_path)
            # Simplify mit onnxsim
            model_simplified, check = simplify(onnx_model)
            if not check:
                print(f"[!] Simplification failed for Batch Size {batch_size}")
                continue
            onnx.save(model_simplified, simplified_path)
            print(f"Simplified saved: {simplified_path}")
            # remove initializers from inputs
            model_simplified = remove_initializers_from_inputs_model(model_simplified)
            onnx.save(model_simplified, simplified_path)
    else:
        print("No quantisation -> export with qonnx")
        onnx_path = "outputs/vision/model_dynamic_batchsize.onnx"
        export_qonnx(
            model,
            (inp,),
            onnx_path,
            export_params=True,
            opset_version=18,          
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
    
    # Load the trained model parameters
    # model.load_state_dict(torch.load("outputs/vision/model_fp32.pt"))
    model.load_state_dict(torch.load("outputs/vision/model.pt"))

    model = patch_missing_affine_norms(model)
    model = ModelTopKWrapper(model, k=5)  # to minimize the output data
    # Pass the model and the export configuration to the evaluation loop
    params["export"].pop("format", None)
    export(model, dataset=params["dataset"], **params["export"])