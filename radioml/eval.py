# System functionality like creating directories and reading env-vars
import os
# YAML for saving experiment metrics
import yaml
# Use the DVC api for loading the YAML parameters
import dvc.api
# Progressbar in for loops
import tqdm
# Pandas to handle the results as table, i.e., DataFrame
import pandas as pd
# Handling arrays
import numpy as np
# PyTorch base package: Math and Tensor Stuff
import torch
# Loads shuffled batches from datasets
from torch.utils.data import DataLoader

# The RadioML classification model
from radioml.model import Model
# The RadioML modulation classification dataset
from radioml.dataset import get_datasets
# Seeding RNGs for reproducibility
from utils import seed

# Path to the RadioML dataset
RADIOML_PATH = os.environ["RADIOML_PATH"]


# Main evaluation loop: Takes a trained model, loads the dataset and sets and
# runs a bunch of inferences to collect metrics
def evaluate(model, dataset, batch_size, loader):  # noqa: Shadows model
    # Check whether GPU eval is available and select the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move the model to the training device
    model = model.to(device)  # noqa: Shadows model...
    # Set model to evaluation mode
    model = model.eval()  # noqa: Shadows model...


    # Load the RadioML dataset splits as configured
    _, _, eval_data = get_datasets(path=RADIOML_PATH, **dataset)
    # Create a batched and shuffled data loader the RadioML validation split
    eval_data = DataLoader(eval_data, batch_size=batch_size, **loader)

    # Start collecting predictions and ground-truth in a data frame
    classes, probabilities = [], []  # noqa: Shadows classes

    # Evaluation requires no gradients
    with torch.no_grad():
        # Iterate the (input, target labels) pairs
        for x, cls, _ in tqdm.tqdm(eval_data, "eval-batch", leave=False):
            # Model forward pass producing logits corresponding to class
            # probabilities.
            classes.append(cls), probabilities.append(model(x.to(device)).cpu())

    # Concatenate lists of batched classes and predicted class probabilities
    # into a single array each. Add broadcastable dimensions.
    cls = np.concatenate(classes)[:, None]
    probabilities = np.concatenate(probabilities)

    # Get top-1 and top-5 predictions from the class probabilities
    top_1 = probabilities.argsort(axis=-1)[:, -1:]
    top_5 = probabilities.argsort(axis=-1)[:, -5:]

    # Collect true and predicted labels for creating a confusion matrix plot
    classes = pd.DataFrame.from_dict({  # noqa: Shadows from outer scope
        "cls": cls.squeeze(), "prediction": top_1.squeeze()
    })

    # Classification accuracy is the fraction of correct classifications
    top_1 = float(((top_1 == cls).any(axis=-1).sum()) / cls.size)  # noqa: sum
    top_5 = float(((top_5 == cls).any(axis=-1).sum()) / cls.size)  # noqa: sum

    # Compute the classification accuracy over the evaluation dataset
    return {"top-1": top_1, "top-5": top_5}, classes


# Script entrypoint
if __name__ == "__main__":
    # Load the stage parameters from the parameters file
    params = dvc.api.params_show(stages="radioml/dvc.yaml:eval")
    # Seed all RNGs
    seed(params["seed"])
    # Create a new model instance according to the configuration
    model = Model(**params["model"])
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/radioml/model.pt"))
    # Pass the model and the evaluation configuration to the evaluation loop
    metrics, classes = evaluate(
        model, dataset=params["dataset"], **params["eval"]
    )
    # Dump the accuracy metrics dictionary as yaml
    with open("outputs/radioml/accuracy.yaml", "w") as file:
        # Dictionary which can be dumped into YAML
        yaml.safe_dump(metrics, file)
    # Save the confusion matrix into a separate yaml to serve this as a
    # plot. Save this as CSV, as YAML stores excessive metadata
    with open("outputs/radioml/classes.csv", "w") as file:
        # Dictionary which can be dumped into YAML
        file.write(classes.to_csv(index=False))
