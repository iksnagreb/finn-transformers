# Data preprocessing on numpy arrays of tokens
import numpy as np

# Loading datasets from huggingface
from datasets import load_dataset


def get_datasets(path, name, split, splits):
    # There must be exactly three splits of the dataset
    assert len(splits) == 3, "There must be three splits of the dataset"
    # Splits must sum to 100% of the dataset
    assert sum(splits) == 1, "Splits must cover the whole dataset"

    # Load the dataset split from huggingface as configured
    data = load_dataset(path, name, split=split)

    # First split into training and test subsets
    train_test = data.train_test_split(train_size=splits[0])

    # Second split relative to the remaining fraction
    second_fraction = splits[1] / (1.0 - splits[0])
    # Second split splits test into validation and evaluation split
    test = train_test["test"].train_test_split(train_size=second_fraction)

    # Return all three splits as individual datasets
    return train_test["train"], test["train"], test["test"]


def preprocess(dataset, tokenizer, context_length):
    # Splits any sequence into non overlapping chunks of context length,
    # allowing the last chunk to be shorter
    def split(sequence):
        return np.array_split(
            sequence, range(context_length, len(sequence), context_length)
        )
        # # Alternative: Overlapping sliding windows
        # sliding_window_view(value, context_length)

    # Preprocess the dataset: Tokenize and split long sequences into chunks of
    # the maximum sequence length (context length)
    def tokenize_and_split(data):
        # First tokenize the input text into a sequence of token ids, add
        # padding to ensure at least one full context window is present
        tokenized = tokenizer(data["text"])

        # Concatenate all token sequences into a single flat sequence (should be
        # separated by [SEP] tokens) and split into chunks of context length
        return {
            key: split(sum(sequence, [])) for key, sequence in tokenized.items()
        }

    # Parallel batched preprocessing of the dataset, removes the original data
    # only leaving chunked token sequences
    return dataset.map(
        tokenize_and_split, batched=True, num_proc=10,
        remove_columns=dataset.column_names
    )
