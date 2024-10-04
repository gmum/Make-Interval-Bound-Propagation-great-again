import torch.nn as nn
from typing import Tuple

from IntervalTools.interval_modules import (
    IntervalConv2d,
    IntervalLinear,
    IntervalReLU,
    IntervalFlatten
)

def get_dim_in_out(dataset_name: str, neural_net_type: str = "mlp") -> Tuple[int,int]:
    """
    Returns a tuple with input and output dimensions.

    Parameters:
    -----------

        dataset_name: str
            Name of the dataset. Possible values are: 'mnist', 'cifar', 'svhn' and 'digits'.

        neural_net_type: str
            Determines a neural network type. Possible values are 'mlp' and 'cnn'.

    Returns:
    --------

        A tuple with input and output dimensions.
    """

    assert dataset_name in ["cifar", "mnist", "svhn", "digits"], "Possible values of the 'mode' are: 'cifar', 'svhn', 'digits' and 'mnist'."
    assert neural_net_type in ["mlp", "cnn"], "Possible values of the neural net type parameter are: 'mlp' and 'cnn'."

    if dataset_name == "mnist":
        dim_in      = 28
        dim_out     = 10
        in_channels = 1
    elif dataset_name == "svhn":
        dim_in      = 32
        dim_out     = 10
        in_channels = 3
    elif dataset_name == "cifar":
        dim_in      = 32
        dim_out     = 10
        in_channels = 3
    elif dataset_name == "digits":
        dim_in      = 8
        dim_out     = 10
        in_channels = 1
    
    if neural_net_type == "cnn":
        return dim_in, dim_out
    elif neural_net_type == "mlp":
        return dim_in**2 * in_channels, dim_out
    
def get_architecture(arch: str, mode: str = "mnist") -> nn.ModuleList:
    """
    Get CNN architecture based on the given arguments.

    Parameters:
    -----------
        arch: str
            Possible values are: `cnn_small`, `cnn_medium`, `cnn_large`.
        
        mode: str
            Adjust architecture to be able to process given dataset.
            Possible values are: 'mnist', 'svhn', 'cifar'.
        
    Returns:
    --------
        A list with the neural network layers.
    """

    assert mode in ["cifar", "mnist", "svhn"], "Possible values of the 'mode' are: 'cifar', 'svhn' and 'mnist'."
    assert arch in ["cnn_small", "cnn_medium", "cnn_large"], "Possible values of the 'arch' are: 'small', 'medium' and 'large'."

    if arch == "cnn_small":

        if mode == "cifar" or mode == "svhn":
            hidden_units = 4608
            in_channel = 3
        elif mode == "mnist":
            hidden_units = 3200
            in_channel = 1

        layers = nn.ModuleList([
                    IntervalConv2d(in_channel, 16, (4, 4), 2),
                    IntervalReLU(),
                    IntervalConv2d(16, 32, (4,4), 1),
                    IntervalReLU(),
                    IntervalFlatten(),
                    IntervalLinear(hidden_units, 100),
                    IntervalReLU(),
                    IntervalLinear(100, 10),
                    ])
    elif arch == "cnn_medium":

        if mode == "cifar" or mode == "svhn":
            hidden_units = 1600
            in_channel = 3
        elif mode == "mnist":
            hidden_units = 1024
            in_channel = 1

        layers = nn.ModuleList([
            IntervalConv2d(in_channel, 32, (3, 3), 1),
            IntervalReLU(),
            IntervalConv2d(32, 32, (4, 4), 2),
            IntervalReLU(),
            IntervalConv2d(32, 64, (3, 3), 1),
            IntervalReLU(),
            IntervalConv2d(64, 64, (4, 4), 2),
            IntervalReLU(),
            IntervalFlatten(),
            IntervalLinear(hidden_units, 512),
            IntervalReLU(),
            IntervalLinear(512, 512),
            IntervalReLU(),
            IntervalLinear(512, 10),
        ])
    elif arch == "cnn_large":
        if mode == "cifar" or mode == "svhn":
            hidden_units = 10368
            in_channel = 3
        elif mode == "mnist":
            hidden_units = 6272
            in_channel = 1

        layers = nn.ModuleList([
            IntervalConv2d(in_channel, 64, (3,3), 1),
            IntervalReLU(),
            IntervalConv2d(64, 64, (3,3), 1),
            IntervalReLU(),
            IntervalConv2d(64, 128, (3,3), 2),
            IntervalReLU(),
            IntervalConv2d(128, 128, (3,3), 1),
            IntervalReLU(),
            IntervalConv2d(128, 128, (3,3), 1),
            IntervalReLU(),
            IntervalFlatten(),
            IntervalLinear(hidden_units, 512),
            IntervalReLU(),
            IntervalLinear(512, 10),
        ])

    return layers