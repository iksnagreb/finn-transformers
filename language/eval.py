# YAML for saving experiment metrics
import yaml
# Use the DVC api for loading the YAML parameters
import dvc.api
# Progressbar in for loops
import tqdm
# PyTorch base package: Math and Tensor Stuff
import torch
# Loads shuffled batches from datasets
from torch.utils.data import DataLoader

# Generic tokenizer for loading pretrained tokenizer and data collator creating
# batches of masked sequence data
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

# The language model
from language.model import Model
# The language dataset loader
from language.dataset import get_datasets, preprocess
# Seeding RNGs for reproducibility
from utils import seed


# Computes the top-k prediction accuracy for probabilities and ground truth
# labels cls
def top_k_accuracy(probabilities, cls, k=1):
    # As the model was not trained to predict anything for tokens which are not
    # masked at the input, filter all predictions which are not masked (the data
    # collator marks these with target labels -100)
    s = torch.where(torch.as_tensor(cls) >= 0)
    # Select top-k probabilities predicted along the last axis
    top_k = torch.as_tensor(probabilities[s].argsort(dim=-1)[..., -k:])
    # Classification accuracy is the fraction of correct predictions
    return torch.any(top_k == cls[..., None][s], dim=-1).sum() / cls[s].numel()


# Main evaluation loop: Takes a trained model, loads the dataset and sets and
# runs a bunch of inferences to collect metrics
def evaluate(model, dataset, batch_size, context_length, tokenizer,
             mlm, mlm_probability, loader):
    # Check whether GPU eval is available and select the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    # Move the model to the training device
    model = model.to(device)  # noqa: Shadows model...
    # Set model to evaluation mode
    model = model.eval()  # noqa: Shadows model...

    # Load the language modeling dataset splits as configured (training and
    # validation dataset are not used here)
    _, _, eval_data = get_datasets(**dataset)

    # Preprocess evaluation dataset as configured (context length is allowed to
    # deviate from training)
    eval_data = preprocess(eval_data, tokenizer, context_length)

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
    eval_data = DataLoader(eval_data, batch_size, collate_fn=collate, **loader)

    # Accumulated top-1 and top-5 prediction accuracy and count of evaluation
    # batches to average
    top_1, top_5, count = 0.0, 0.0, 0

    # Evaluation requires no gradients
    with torch.no_grad():
        # Iterate the (input, target labels) pairs
        for x, cls in tqdm.tqdm(eval_data, "eval-batch", leave=False):
            # Forward pass producing logits corresponding to token probabilities
            probabilities = model(x.to(device))
            # Evaluate and accumulate average prediction accuracy on this batch
            top_1 += top_k_accuracy(probabilities, cls.to(device), k=1).item()
            top_5 += top_k_accuracy(probabilities, cls.to(device), k=5).item()
            # Count batches to average over the entire dataset later
            count += 1

    # Average the prediction accuracies over the entire evaluation dataset
    return {"top-1": top_1 / count, "top-5": top_5 / count}


# Script entrypoint
if __name__ == "__main__":
    # Load the stage parameters from the parameters file
    params = dvc.api.params_show(stages="language/dvc.yaml:eval")
    # Seed all RNGs
    seed(params["seed"])
    # Load the already trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("outputs/language/tokenizer")
    # Create a new model instance according to the configuration (vocabulary
    # size from the tokenizer in case this deviates from the configured)
    model = Model(**params["model"], vocab_size=tokenizer.vocab_size)
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/language/model.pt"))
    # Pass the model and the evaluation configuration to the evaluation loop
    metrics = evaluate(
        model, dataset=params["dataset"], tokenizer=tokenizer, **params["eval"]
    )
    # Dump the accuracy metrics dictionary as yaml
    with open("outputs/language/accuracy.yaml", "w") as file:
        # Dictionary which can be dumped into YAML
        yaml.safe_dump(metrics, file)
