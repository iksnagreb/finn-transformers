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
# Seeding RNGs for reproducibility, configuration of optimizer and loss
from utils import seed, get_optimizer, get_criterion


# Main training loop: Takes a model, loads the dataset and sets up the
# optimizer. Runs the configured number of training epochs
def train(model, batch_size, epochs, criterion, optimizer, loader, dataset,
          context_length, mlm, mlm_probability, tokenizer, scheduler):
    # Check whether GPU training is available and select the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move the model to the training device
    model = model.to(device)  # noqa: Shadows model...
    # Set model to training mode
    model = model.train()  # noqa: Shadows model...

    # Get the optimizer and register the model parameters
    optimizer = get_optimizer(  # noqa: Shadows optimizer...
        **optimizer, parameters=model.parameters()
    )
    # Get the optimization criterion instance
    criterion = get_criterion(criterion)

    # Learning rate scheduler reducing the learning rate if the validation loss
    # stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **scheduler
    )

    # Load the language modeling dataset splits as configured (evaluation
    # dataset is not used here)
    train_data, valid_data, _ = get_datasets(**dataset)

    # Preprocess both dataset splits using the same configuration
    train_data = preprocess(train_data, tokenizer, context_length)
    valid_data = preprocess(valid_data, tokenizer, context_length)

    # Data collator turning sample sequences of tokens into batches of masked
    # and padded tokens as PyTorch tensors, used by each DataLoader worker
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=mlm, mlm_probability=mlm_probability
    )

    def collate(samples):
        # Use the collator for language modeling to turn the list of samples
        # into a batch of padded sequences with random masking applied
        batch = collator(samples)
        # Extract masked input tokens and target labels
        return batch["input_ids"], batch["labels"]

    # Create a batched and shuffled data loader for each of the preprocessed
    # dataset splits
    train_data = DataLoader(
        train_data, batch_size=batch_size, collate_fn=collate, **loader
    )
    valid_data = DataLoader(
        valid_data, batch_size=batch_size, collate_fn=collate, **loader
    )

    # Collect training and validation loss and learning rate per epoch
    _loss, _lr = [], []

    # Run the configured number of training epochs
    for _ in tqdm.trange(epochs, desc="epoch"):
        # Collect training and validation loss per epoch
        train_loss, valid_loss = (0, 0)
        # Set model to training mode
        model = model.train()  # noqa: Shadows model...
        # Iterate the batches of (input, target labels) pairs
        for x, y in tqdm.tqdm(train_data, desc="train-batch", leave=False):
            # Clear gradients of last iteration
            optimizer.zero_grad(set_to_none=True)
            # Feed input data to model to get predictions
            p = model(x.to(device))  # noqa: Duplicate, see below
            # Loss between token probabilities and true tokens per step averaged
            # over batch and sequence
            loss = criterion(p.permute(0, 2, 1), y.to(device))
            # Backpropagation of the error to compute gradients
            loss.backward()
            # Parameter update step
            optimizer.step()
            # Accumulate the loss over the whole validation dataset
            train_loss += loss.item()
        # Clear gradients of last iteration
        optimizer.zero_grad(set_to_none=True)
        # Switch the model to evaluation mode, disabling dropouts and scale
        # calibration
        model = model.eval()  # noqa: Shadows model...
        # Validation requires no gradients
        with torch.no_grad():
            # Iterate the batches of (input, target labels) pairs
            for x, y in tqdm.tqdm(valid_data, "valid-batch", leave=False):
                # Feed input data to model to get predictions
                p = model(x.to(device))  # noqa: Duplicate, see above
                # Loss between token probabilities and true tokens per step
                # averaged over batch and sequence
                loss = criterion(p.permute(0, 2, 1), y.to(device))
                # Accumulate the loss over the whole validation dataset
                valid_loss += loss.item()
        # Adjust the learning rate if necessary
        scheduler.step(valid_loss)
        # Append loss information to the log
        _loss.append({"train": train_loss, "valid": valid_loss})
        # keep track of the learning rate
        _lr.append({"last": scheduler.get_last_lr()})

    # Clear the gradients of last iteration
    optimizer.zero_grad(set_to_none=True)
    # Return the model, the optimizer state and the log after training
    return model.cpu(), optimizer, {"loss": _loss}, {"lr": _lr}


# Script entrypoint
if __name__ == "__main__":
    # Load the stage parameters from the parameters file
    params = dvc.api.params_show(stages="language/dvc.yaml:train")
    # Seed all RNGs
    seed(params["seed"])
    # Load the already trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("outputs/language/tokenizer")
    # Create a new model instance according to the configuration (vocabulary
    # size from the tokenizer in case this deviates from the configured)
    model = Model(**params["model"], vocab_size=tokenizer.vocab_size)
    # Pass the model and the training configuration to the training loop
    model, optimizer, loss, lr = train(
        model, dataset=params["dataset"], tokenizer=tokenizer, **params["train"]
    )
    # Save the model in PyTorch format
    torch.save(model.state_dict(), "outputs/language/model.pt")
    # Save the optimizer state in PyTorch format
    torch.save(optimizer.state_dict(), "outputs/language/optimizer.pt")
    # Save the training loss log as YAML
    with open("outputs/language/loss.yaml", "w") as file:
        # Dump the training log dictionary as YAML into the file
        yaml.safe_dump(loss, file)
    # Save the training learning rate log as YAML
    with open("outputs/language/lr.yaml", "w") as file:
        # Dump the training log dictionary as YAML into the file
        yaml.safe_dump(lr, file)
