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
from torch.utils.data import DataLoader, random_split
# PyTorch vision datasets and transformations
from torchvision import datasets, transforms

# The Vision classification model
from vision.model import Model
# Seeding RNGs for reproducibility, configuration of optimizer and loss
from utils import seed, get_optimizer, get_criterion

# Path to the CIFAR-10 dataset
CIFAR10_ROOT = os.environ.setdefault("CIFAR10_ROOT", "data")


# Main training loop: Takes a model, loads the dataset and sets up the
# optimizer. Runs the configured number of training epochs
def train(model, batch_size, epochs, criterion, optimizer, loader,  # noqa
          scheduler):
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

    # Transformation to be applied to the input images: Rather basic
    # preprocessing turning images into tensors and normalizing with only
    # minimal data augmentation
    tf = transforms.Compose([
        # Convert from PIL image to PyTorch tensors
        transforms.ToTensor(),
        # Random horizontal flip in 50% of the cases
        transforms.RandomHorizontalFlip(),
        # Radom crop of the image resized back to 32x32 pixels
        transforms.RandomResizedCrop((32, 32)),
        # CIFAR-10 statistics on the whole training set
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])

    # Load the Vision training split (should already be in CIFAR10_ROOT,
    # otherwise download)
    dataset = datasets.CIFAR10(CIFAR10_ROOT, True, download=True, transform=tf)
    # Split into 90% training and 10% validation dataset
    train_data, valid_data = random_split(dataset, [0.9, 0.1])

    # Create a batched and shuffled data loader for each of the dataset splits
    train_data = DataLoader(train_data, batch_size=batch_size, **loader)
    valid_data = DataLoader(valid_data, batch_size=batch_size, **loader)

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
            # Loss between class probabilities and true class labels
            loss = criterion(p, y.to(device))  # noqa: Shadows outer scope
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
                # Loss between class probabilities and true class labels
                loss = criterion(p, y.to(device))  # noqa: Shadows outer scope
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
    params = dvc.api.params_show(stages="vision/dvc.yaml:train")
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = Model(**params["model"])
    # Pass the model and the training configuration to the training loop
    model, optimizer, loss, lr = train(model, **params["train"])
    # Create the output directory if it does not already exist
    os.makedirs("outputs/vision", exist_ok=True)
    # Save the model in PyTorch format
    torch.save(model.state_dict(), "outputs/vision/model.pt")
    # Save the optimizer state in PyTorch format
    torch.save(optimizer.state_dict(), "outputs/vision/optimizer.pt")
    # Save the training loss log as YAML
    with open("outputs/vision/loss.yaml", "w") as file:
        # Dump the training log dictionary as YAML into the file
        yaml.safe_dump(loss, file)
    # Save the training learning rate log as YAML
    with open("outputs/vision/lr.yaml", "w") as file:
        # Dump the training log dictionary as YAML into the file
        yaml.safe_dump(lr, file)
