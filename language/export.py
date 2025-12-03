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

# Generic tokenizer for loading pretrained tokenizer and data collator creating
# batches of masked sequence data
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

# The language model
from language.model import Model
# The language dataset loader
from language.dataset import get_datasets, preprocess
# Quantized custom implementation of multihead attention
from attention import QuantMultiheadAttention
# Seeding RNGs for reproducibility, affine parameter export patching
from utils import seed, patch_missing_affine_norms

# Export function mapping
EXPORTERS = {"qonnx": export_qonnx, "qcdq": export_onnx_qcdq}


# Exports the model to ONNX in conjunction with an input-output pair for
# verification
def export(model, dataset, batch_size, mlm, mlm_probability, tokenizer,
           context_length, format="qonnx", split_heads=False, **kwargs):
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

    # Load the language modeling dataset splits as configured (training and
    # validation dataset are not used here)
    _, _, export_data = get_datasets(**dataset)

    # Preprocess evaluation dataset as configured (context length is allowed to
    # deviate from training)
    export_data = preprocess(export_data, tokenizer, context_length)

    # Data collator turning sample sequences of tokens into batches of masked
    # and padded tokens as PyTorch tensors, used by each DataLoader worker
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=mlm, mlm_probability=mlm_probability
    )

    def collate(samples):
        # Use the collator for language modeling to turn the list of samples
        # into a batch of padded sequences with random masking applied
        batch = collator(samples)
        # Extract masked input tokens and target labels and rearrange into
        # batch-first layout (collator yields sequence-first)
        return batch["input_ids"], batch["labels"]

    # Create a batched and shuffled data loader for the preprocessed dataset
    export_data = DataLoader(
        export_data, batch_size, collate_fn=collate, shuffle=True
    )

    # Sample the first batch from the export dataset
    inp, cls = next(iter(export_data))

    # Also save the model output predictions (probabilities)
    with torch.no_grad():
        out = model(inp)

    # Export the model to ONNX using the input example
    EXPORTERS[format](model, (inp,), "outputs/language/model.onnx", **kwargs)

    # Save the input and output data for verification purposes later
    np.save("outputs/language/inp.npy", inp.numpy())
    np.save("outputs/language/out.npy", out.numpy())
    np.save("outputs/language/cls.npy", cls.numpy())


# Script entrypoint
if __name__ == "__main__":
    # Load the stage parameters from the parameters file
    params = dvc.api.params_show(stages="language/dvc.yaml:export")
    # Seed all RNGs
    seed(params["seed"])
    # Load the already trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("outputs/language/tokenizer")
    # Create a new model instance according to the configuration (vocabulary
    # size from the tokenizer in case this deviates from the configured)
    model = Model(**params["model"], vocab_size=tokenizer.vocab_size)
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/language/model.pt"))
    # Prevent export and streamlining issues for missing affine normalization
    # parameters
    model = patch_missing_affine_norms(model)
    # Pass the model and the export configuration to the evaluation loop
    export(model, dataset=params["dataset"], tokenizer=tokenizer,
           **params["export"])
