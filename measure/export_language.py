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
from transformers import AutoTokenizer, DataCollatorForLanguageModeling # pip install transformers==4.53.0, pip install datasets
from brevitas.export import export_qonnx, export_onnx_qcdq
# The language model
from language.model import Model
# The language dataset loader
from language.dataset import get_datasets, preprocess
# Quantized custom implementation of multihead attention
from attention import QuantMultiheadAttention
# Seeding RNGs for reproducibility, affine parameter export patching
from utils import seed, patch_missing_affine_norms
import gc
import yaml
import onnx
from onnxsim import simplify

# Export function mapping
EXPORTERS = {"qonnx": export_qonnx, "qcdq": export_onnx_qcdq}

MODEL_TYPE = "language"

with open(f"{MODEL_TYPE}/params.yaml", "r") as f:
    cfg = yaml.safe_load(f)

bits = cfg["model"]["embedding"].get("bits", 0)
INT8 = (bits == 8)
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
    export_data = export_data.shuffle(seed=42).select(range(10000))
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
    export_data_load = DataLoader(
        export_data, batch_size, collate_fn=collate, shuffle=True
    )


    def safe_dataset_to_npz(export_data_load):
        # Sample the first batch from the export dataset
        max_samples = 10000
        all_inputs = []
        all_labels = []

        num_collected = 0
        for inp, cls in export_data_load:
            all_inputs.append(inp.numpy())
            all_labels.append(cls.numpy())
            num_collected += inp.shape[0]
            if num_collected >= max_samples:
                break

        # Concatenate batches
        input_ids = np.concatenate(all_inputs, axis=0)[:max_samples]
        labels = np.concatenate(all_labels, axis=0)[:max_samples]
        # speichern
        np.savez(R"/data/gitlab/language.npz",
                input_ids=input_ids,
                labels=labels)
        print("Shapes:", input_ids.shape, labels.shape)
    
    safe_dataset_to_npz(export_data_load)
    
    inp, cls = next(iter(export_data_load))

    with torch.no_grad():
        out = model(inp)

    # Export the model to ONNX using the input example
    EXPORTERS[format](model, (inp,), "outputs/language/model.onnx", **kwargs)

    # Save the input and output data for verification purposes later
    np.save("outputs/language/inp.npy", inp.numpy())
    np.save("outputs/language/out.npy", out.numpy())
    np.save("outputs/language/cls.npy", cls.numpy())

    if INT8 == True:
        print("quantisation -> export with qcdq")
        vocab_size=tokenizer.vocab_size
        tokens = tokenizer("Hallo Welt", return_tensors="pt")["input_ids"]
        seq_len = tokens.shape[1]


        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]: 

            # test: wird das Ergebnis (Accuracy) besser mit echten daten?
            export_data_batch= DataLoader(
                export_data, batch_size=batch_size, collate_fn=collate, shuffle=True
            )

            # Sample the first batch from the export dataset
            inp, cls = next(iter(export_data_batch))
            export_path=f"outputs/language/model_brevitas_{batch_size}.onnx"
            simplified_path=f"outputs/language/model_brevitas_{batch_size}_simple.onnx"

            # uint-> int ?
            export_onnx_qcdq(
                model, 
                (inp,),
                export_path=export_path,
                opset_version=17,
                export_as_int8=True,          
                quant_type='uint',            
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
    else:
        print("No quantisation -> export with qonnx")
        onnx_path = "outputs/language/model_dynamic_batchsize.onnx"
        EXPORTERS[format](
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


# Script entrypoint
if __name__ == "__main__":
    # Load the stage parameters from the parameters file
    with open("language/params.yaml", "r") as   f:
        params = yaml.safe_load(f)
    # Seed all RNGs
    seed(params["seed"])
    # Load the already trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("outputs/language/tokenizer") #padding?

    # Create a new model instance according to the configuration (vocabulary
    # size from the tokenizer in case this deviates from the configured)
    model = Model(**params["model"], vocab_size=tokenizer.vocab_size)
    # Load the trained model parameters
    # model.load_state_dict(torch.load("outputs/language/model.pt"))      # doesn't work for not quantized params -> model.pt is not te correct unquantized one
    model.load_state_dict(torch.load("outputs/language/model_fp32.pt")) 
    # Prevent export and streamlining issues for missing affine normalization
    # parameters
    model = patch_missing_affine_norms(model)
    # Pass the model and the export configuration to the evaluation loop
    export(model, dataset=params["dataset"], tokenizer=tokenizer,
           **params["export"])

