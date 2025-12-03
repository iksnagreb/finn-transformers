# System functionality like creating directories and reading env-vars
import os
# Use the DVC api for loading the YAML parameters
import dvc.api

# Generic tokenizer for loading pretrained tokenizer bases
from transformers import AutoTokenizer

# The language dataset loader
from language.dataset import get_datasets
# Seeding RNGs for reproducibility
from utils import seed


def train_tokenizer(data, pretrained, **kwargs):
    # Start training a custom tokenizer from a pretrained base
    base = AutoTokenizer.from_pretrained(pretrained)

    # Dataset generator for training the tokenizer
    def _generate_training_corpus():
        for index in range(0, len(data), 1024):
            yield data[index:index + 1024]["text"]

    # Train the custom tokenizer on batch data from the training corpus
    return base.train_new_from_iterator(_generate_training_corpus(), **kwargs)


# Script entrypoint when executing the module
if __name__ == "__main__":
    # Load the stage parameters from the parameters file
    params = dvc.api.params_show(stages="language/dvc.yaml:tokenizer")
    # Seed all RNGs
    seed(params["seed"])

    # Load the datasets but only keep the training split for training the
    # tokenizer
    dataset, _, _ = get_datasets(**params["dataset"])

    # Make sure the output directory exists to save the tokenizer
    os.makedirs("outputs/language", exist_ok=True)

    # Train the tokenizer on the training split and save intro the outputs
    tokenizer = train_tokenizer(dataset, **params["tokenizer"])
    tokenizer.save_pretrained("outputs/language/tokenizer")
