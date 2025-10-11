# Python's builtin random number generators
import random
# Seed the numpy random number generator as well
import numpy as np

# PyTorch base package: Math and Tensor Stuff, and an RNG as well
import torch


# Seeds all relevant random number generators to the same seed for
# reproducibility
def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# Gets an optimizer instance according to configuration and register the model
# parameters
def get_optimizer(algorithm, parameters, **kwargs):
    # Supported optimizer algorithms
    algorithms = {
        "adam": torch.optim.Adam, "sgd": torch.optim.SGD
    }
    # The configured algorithm must be among the supported ones
    assert algorithm in algorithms, f"Optimizer {algorithm} is not supported"
    # Create the optimizer instance forwarding additional arguments and
    # registering the model parameters
    return algorithms[algorithm](parameters, **kwargs)


# Gets the loss functions instance according to configuration
def get_criterion(criterion, **kwargs):
    # Supported optimization criteria
    criteria = {
        "cross-entropy": torch.nn.CrossEntropyLoss, "nll": torch.nn.NLLLoss
    }
    # The configured criterion must be among the supported ones
    assert criterion in criteria, f"Criterion {criterion} is not supported"
    # Create the loss function instance forwarding additional arguments
    return criteria[criterion](**kwargs)
