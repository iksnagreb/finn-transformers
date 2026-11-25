# System functionality like creating directories and reading env-vars
import os
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
# TensorBoard writer for live training charts
from torch.utils.tensorboard import SummaryWriter
# Timestamps to give each TensorBoard run a unique name
import datetime

# The RadioML classification model
from radioml.model import Model
# The RadioML modulation classification dataset
from radioml.dataset import get_datasets
# Seeding RNGs for reproducibility, configuration of optimizer and loss
from utils import seed, get_optimizer, get_criterion

# Path to the RadioML dataset
RADIOML_PATH = os.environ["RADIOML_PATH"]
RADIOML_PATH_NPZ = os.environ["RADIOML_PATH_NPZ"]

# RADIOML_PATH = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.hdf5"
# RADIOML_PATH_NPZ = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"

INT8 = os.getenv("INT8", "0")  # Default = "0"
if INT8 == "1":
    print("INT8-Modus aktiviert")

# Main training loop: Takes a model, loads the dataset and sets up the
# optimizer. Runs the configured number of training epochs
def train(model, batch_size, epochs, criterion, optimizer, loader,  # noqa
          dataset, scheduler, tensorboard_log_dir="outputs/radioml/tensorboard"):
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

    # Load the RadioML dataset splits as configured
    train_data, valid_data, _ = get_datasets(path=RADIOML_PATH, **dataset)

    # Create a batched and shuffled data loader for each of the dataset splits
    train_data = DataLoader(train_data, batch_size=batch_size, **loader)
    valid_data = DataLoader(valid_data, batch_size=batch_size, **loader)

    # Collect training and validation loss and learning rate per epoch
    _loss, _lr = [], []

    # Log scalars for live inspection in TensorBoard:
    #   tensorboard --logdir outputs/radioml/tensorboard
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    print(f"TensorBoard logs: {tensorboard_log_dir}")

    # Run the configured number of training epochs
    for epoch in tqdm.trange(epochs, desc="epoch"):
        # Collect training and validation loss per epoch
        train_loss, valid_loss = (0, 0)
        # Count correct predictions for training and validation accuracy
        train_correct, train_total = (0, 0)
        valid_correct, valid_total = (0, 0)
        # Accumulate gradient norms to spot vanishing/exploding gradients
        grad_norm_sum = 0.0
        # Set model to training mode
        model = model.train()  # noqa: Shadows model...
        # Iterate the batches of (input, target labels, SNR) triples
        for batch_idx, (x, y, _) in enumerate(
                tqdm.tqdm(train_data, desc="train-batch", leave=False)):
            # Move the labels to the device once, used for loss and accuracy
            labels = y.to(device)
            # Clear gradients of last iteration
            optimizer.zero_grad(set_to_none=True)
            # Feed input data to model to get predictions
            p = model(x.to(device))  # noqa: Duplicate, see below
            # Loss between class probabilities and true class labels
            loss = criterion(p, labels)  # noqa: Shadows outer scope
            # Backpropagation of the error to compute gradients
            loss.backward()
            # Total gradient norm (no clipping, max_norm=inf only measures)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            ).item()
            grad_norm_sum += grad_norm
            # Parameter update step
            optimizer.step()
            # Accumulate the loss over the whole validation dataset
            train_loss += loss.item()
            # Track training accuracy for live monitoring
            correct = (p.argmax(dim=1) == labels).sum().item()
            train_correct += correct
            train_total += labels.size(0)
            # Log batch-level training scalars every few batches
            if batch_idx % 20 == 0:
                step = epoch * len(train_data) + batch_idx
                writer.add_scalar("loss/train_batch", loss.item(), step)
                writer.add_scalar(
                    "accuracy/train_batch", correct / labels.size(0), step
                )
                writer.add_scalar("grad_norm/batch", grad_norm, step)
        # Clear gradients of last iteration
        optimizer.zero_grad(set_to_none=True)
        # Switch the model to evaluation mode, disabling dropouts and scale
        # calibration
        model = model.eval()  # noqa: Shadows model...
        # Validation requires no gradients
        with torch.no_grad():
            # Iterate the batches of (input, target labels, SNR) triples
            for x, y, _ in tqdm.tqdm(valid_data, "valid-batch", leave=False):
                # Move the labels to the device once, used for loss and accuracy
                labels = y.to(device)
                # Feed input data to model to get predictions
                p = model(x.to(device))  # noqa: Duplicate, see above
                # Loss between class probabilities and true class labels
                loss = criterion(p, labels)  # noqa: Shadows outer scope
                # Accumulate the loss over the whole validation dataset
                valid_loss += loss.item()
                # Track validation accuracy for comparison against training
                valid_correct += (p.argmax(dim=1) == labels).sum().item()
                valid_total += labels.size(0)
        # Adjust the learning rate if necessary
        scheduler.step(valid_loss)
        # Append loss information to the log
        _loss.append({"train": train_loss, "valid": valid_loss})
        # keep track of the learning rate
        _lr.append({"last": scheduler.get_last_lr()})
        # Normalized per-epoch scalars to make TensorBoard runs comparable
        writer.add_scalar("loss/train_epoch", train_loss / len(train_data), epoch)
        writer.add_scalar("loss/valid_epoch", valid_loss / len(valid_data), epoch)
        writer.add_scalar(
            "accuracy/train_epoch", train_correct / max(train_total, 1), epoch
        )
        writer.add_scalar(
            "accuracy/valid_epoch", valid_correct / max(valid_total, 1), epoch
        )
        writer.add_scalar(
            "grad_norm/epoch_mean", grad_norm_sum / len(train_data), epoch
        )
        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)
        # Flush so the charts update live while training is running
        writer.flush()
    # Clear the gradients of last iteration
    optimizer.zero_grad(set_to_none=True)
    # Make sure TensorBoard writes the last events to disk
    writer.close()
    # Return the model, the optimizer state and the log after training
    return model.cpu(), optimizer, {"loss": _loss}, {"lr": _lr}


# Script entrypoint
if __name__ == "__main__":
    # Load the stage parameters from the parameters file
    if INT8 == "1":
        params = dvc.api.params_show(stages="radioml/dvc.yaml:train_INT8")
    else:
        params = dvc.api.params_show(stages="radioml/dvc.yaml:train")
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    if INT8 == "1":
        model = Model(**params["model_int8"])
    else:
        model = Model(**params["model"])
    # One TensorBoard run directory per training run, named by timestamp and
    # model size to make runs easy to tell apart in the TensorBoard UI
    cfg = params["model_int8"] if INT8 == "1" else params["model"]
    run_name = (
        f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
        f"_L{cfg['num_layers']}_E{cfg['emb_dim']}_H{cfg['num_heads']}"
        f"_share{int(cfg.get('share_weights', True))}"
        f"_norm{int(cfg.get('share_norm', True))}"
        f"_bits{cfg.get('bits')}"
    )
    # Pass the model and the training configuration to the training loop
    model, optimizer, loss, lr = train(
        model, dataset=params["dataset"], **params["train"],
        tensorboard_log_dir=os.path.join("outputs/radioml/tensorboard", run_name)
    )
    # Create the output directory if it does not already exist
    os.makedirs("outputs/radioml", exist_ok=True)
    # Save the model in PyTorch format
    if INT8 == "1":
        torch.save(model.state_dict(), "outputs/radioml/model_int8.pt")
    else:
        torch.save(model.state_dict(), "outputs/radioml/model.pt")
    # Save the optimizer state in PyTorch format
    if INT8 == "1":
        torch.save(optimizer.state_dict(), "outputs/radioml/optimizer_int8.pt")
    else:
        torch.save(optimizer.state_dict(), "outputs/radioml/optimizer.pt")
    # Save the training loss log as YAML
    with open("outputs/radioml/loss.yaml", "w") as file:
        # Dump the training log dictionary as YAML into the file
        yaml.safe_dump(loss, file)
    # Save the training learning rate log as YAML
    with open("outputs/radioml/lr.yaml", "w") as file:
        # Dump the training log dictionary as YAML into the file
        yaml.safe_dump(lr, file)
