from torch import nn
from typing import List, Callable
import logging

import torch

from network.utils import load_linear_model
from network.network_abc import NetworkABC

log = logging.getLogger(__name__)


class MLP(NetworkABC):
    """
    A Multi-Layer Perceptron (MLP) implemented as a PyTorch module.
    This class constructs a feedforward neural network with a specified number of layers
    and activation functions. The network is built using `torch.nn.Sequential` for ease
    of use and modularity.
    
    Attributes:
        model (torch.nn.Sequential): The sequential model representing the MLP.
    
    Methods:
        __init__(layers: List[int], activation_fnc: Callable):
            Initializes the MLP with the specified layers and activation function.
        build(layers: List[int], activation_fnc: Callable) -> nn.Sequential:
            Constructs the sequential model based on the provided layers and activation function.
        forward(x):
            Performs a forward pass through the MLP.
    
    Args:
        layers (List[int]): A list of integers where each element represents the number
            of neurons in the corresponding layer of the MLP. The length of the list
            determines the number of layers in the network.
        activation_fnc (Callable): A callable that returns an activation function to be
            applied after each layer (except the last one).
    """

    def __init__(self, layers: List[int], activation_fnc: Callable, model_path: str = None) -> None:
        """
        Initializes the MLP (Multi-Layer Perceptron) model.
        
        Args:
            layers (List[int]): A list of integers where each integer represents 
            the number of neurons in the corresponding layer of the MLP. The 
            first element corresponds to the input size, and the last element 
            corresponds to the output size.
            activation_fnc (Callable): A callable that returns an activation 
            function to be applied between the layers of the MLP, except the 
            last layer.
            model_path (str, optional): Path to a pre-trained model file. If 
            provided, the model weights will be loaded from this file.
        """

        super().__init__()
        self.model = self.build(layers, activation_fnc)

        if model_path is not None:
            load_linear_model(self.model, model_path)
            log.info(f"Model loaded from {model_path}")

    def build(self, layers: List[int], activation_fnc: Callable) -> nn.Sequential:
        """
        Constructs a sequential neural network model based on the specified layer sizes and activation function.
        Args:
            layers (List[int]): A list of integers where each element represents the number of neurons 
                in the corresponding layer of the network. The length of the list determines the number 
                of layers in the network.
            activation_fnc (Callable): A callable that returns an activation function instance to be 
                applied after each layer, except the last one.
        Returns:
            nn.Sequential: A PyTorch Sequential model containing the specified layers and activation 
            functions in the defined order.
        """

        modules = []
        last_layer_idx = len(layers) - 1
        for i in range(last_layer_idx):
            modules.append(nn.Linear(layers[i], layers[i + 1]))

            if i < last_layer_idx-1:
                modules.append(activation_fnc)
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.
        Parameters:
            x (torch.Tensor): The input tensor to the model.
        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        
        return self.model(x)
