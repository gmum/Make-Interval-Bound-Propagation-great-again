from torch import nn
import torch

import logging

from network.utils import load_conv_model
from network.network_abc import NetworkABC

log = logging.getLogger(__name__)


class ConvSmallCIFAR(NetworkABC):
    """
    A convolutional neural network model designed for small-scale image processing tasks.
    The model consists of a feature extractor using convolutional layers followed by a classifier.
    Attributes:
        in_channels (int): Number of input channels for the convolutional layers.
        dim_out (int): Dimension of the output layer (number of classes or regression targets).
        model_path (str, optional): Path to a pre-trained model to load. Defaults to None.
    Methods:
        __init__(dim_out: int, model_path: str = None):
            Initializes the ConvSmall model.
        build() -> nn.Sequential:
            Builds the architecture of the ConvSmall model.
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the ConvSmall model.
    """
        
    def __init__(self, dim_out: int, model_path: str = None) -> None:
        """
        Initializes the ConvSmall model.
        Args:
            dim_out (int): Dimension of the output layer (number of classes or regression targets).
            model_path (str, optional): Path to a pre-trained model to load. If provided, the model
                                         weights will be loaded from this path. Defaults to None.
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
        Builds the architecture of the ConvSmall model as a single flat nn.Sequential.
        Returns:
            nn.Sequential: A flat sequential model including feature extractor and classifier.
        """
        feature_layers = [
            nn.Conv2d(self.in_channels, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        ]
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.input_height, self.input_width)
            hidden_units = nn.Sequential(*feature_layers)(dummy_input).shape[1]

        classifier_layers = [
            nn.Linear(hidden_units, 100),
            nn.ReLU(),
            nn.Linear(100, self.dim_out),
        ]

        all_layers = feature_layers + classifier_layers

        return nn.Sequential(*all_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the ConvSmall model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim_out).
        """
        return self.model(x)
