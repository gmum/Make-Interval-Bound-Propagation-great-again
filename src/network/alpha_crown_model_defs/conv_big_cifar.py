from torch import nn
import torch

import logging

from network.utils import load_conv_model
from network.network_abc import NetworkABC

log = logging.getLogger(__name__)


class ConvBigCIFAR(NetworkABC):
    """
    ConvBigCIFAR: A convolutional neural network model for feature extraction and classification.
    This class defines a convolutional neural network with multiple convolutional layers 
    for feature extraction, followed by fully connected layers for classification. The 
    model can optionally load pre-trained weights from a specified file path.
    Attributes:
        in_channels (int): The number of input channels for the convolutional layers.
        dim_out (int): The number of output dimensions for the final classification layer.
        model_path (str, optional): Path to a pre-trained model file. If provided, the model 
            weights will be loaded from this file.
    Methods:
        __init__(dim_out: int, model_path: str = None):
            Initializes the ConvBigCIFAR model with the specified input channels, output dimensions, 
            and an optional path to a pre-trained model.
        build() -> nn.Sequential:
            Constructs the feature extractor and classifier components of the model. Dynamically 
            determines the number of features after flattening to configure the fully connected layers.
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the model. Takes an input tensor `x` and returns the 
            output tensor after passing through the network.
    """
    def __init__(self, dim_out: int, model_path: str = None) -> None:
        """
        Initializes the class with the given parameters.
        Args:
            dim_out (int): The dimensionality of the output.
            model_path (str, optional): Path to a pre-trained model to load. Defaults to None.
        Attributes:
            model (NetworkABC): The neural network model built by `build` method.
            in_channels (int): The number of input channels for the model.
            dim_out (int): The dimensionality of the output.
        Side Effects:
            If `model_path` is provided, loads the model from the specified path and logs the action.
        """
        super().__init__()

        self.in_channels = 3
        self.input_height = 32
        self.input_width = 32
        self.dim_out = dim_out

        self.model = self.build()
        self.layer_outputs = {}

        self.model.apply(self.register_hooks)
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.input_height, self.input_width)  # Just once, at model level
            self.model(dummy_input)
        
        if model_path is not None:
            load_conv_model(self.model, model_path)
            log.info(f"Model loaded from {model_path}")

    def build(self) -> nn.Sequential:
        """
        Builds a neural network model consisting of a feature extractor and a classifier.
        The feature extractor is a sequence of convolutional layers with ReLU activations,
        followed by a flattening layer. The classifier is a sequence of fully connected
        layers with ReLU activations, culminating in an output layer with the specified
        number of output dimensions.
        Returns:
            nn.Sequential: A sequential container combining the feature extractor and classifier.
        Notes:
            - The number of features after the flattening layer is dynamically determined
              using a dummy input tensor with the specified input dimensions.
            - The input dimensions (self.in_channels, self.input_height, self.input_width)
              and output dimensions (self.dim_out) must be defined as attributes of the class.
        """

        feature_layers = [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        ]

        # Dynamically determine the number of features after flattening
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.input_height, self.input_width)
            hidden_units = nn.Sequential(*feature_layers)(dummy_input).shape[1]

        classifier = [
            nn.Linear(hidden_units, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.dim_out)
        ]

        all_layers = feature_layers + classifier

        return nn.Sequential(*all_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.
        Parameters:
            x (torch.Tensor): The input tensor to the network.
        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """

        return self.model(x)
