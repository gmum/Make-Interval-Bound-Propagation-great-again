from torch import nn
import torch

import logging

from ..network_abc import NetworkABC

log = logging.getLogger(__name__)

class DMSmall(NetworkABC):
        
    def __init__(
            self, 
            in_channels: int, 
            dim_out: int, 
            input_width: int, 
            input_height: int,
        ) -> None:
        """
        Initializes the DM-small convolutional neural network model (Table A from https://arxiv.org/pdf/1906.06316)

        Args:
            in_channels (int): Number of input channels for the first convolutional layer.
            dim_out (int): Number of output features for the final layer.
            input_width (int): Width of the input images (note: likely a typo, should be 'input_width').
            input_height (int): Height of the input images.

        Attributes:
            in_channels (int): Stores the number of input channels.
            input_height (int): Stores the input image height.
            input_width (int): Stores the input image width.
            dim_out (int): Stores the output dimension.
            model (NetworkABC): The constructed sequential model.
            layer_outputs (dict): Dictionary to store outputs of layers for debugging or analysis.

        Side Effects:
            Registers forward hooks on model layers to capture their outputs.
            Performs a dummy forward pass with zero input to initialize layer outputs.
        """
        super().__init__()

        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.dim_out = dim_out

        self.model = self.build()

        self.layer_outputs = {}

        self.model.apply(self.register_hooks)
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.input_height, self.input_width)  # Just once, at model level
            self.model(dummy_input)

    def build(self) -> nn.Sequential:
        """
        Builds the architecture of the DMSmall model as a single flat nn.Sequential.
        Returns:
            nn.Sequential: A flat sequential model including feature extractor and classifier.
        """
        feature_layers = [
            nn.Conv2d(self.in_channels, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        ]
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.input_height, self.input_width)
            hidden_units = nn.Sequential(*feature_layers)(dummy_input).shape[1]

        classifier_layers = [
            nn.Linear(hidden_units, 100),
            nn.ReLU(),
            nn.Linear(100, self.dim_out)
        ]

        all_layers = feature_layers + classifier_layers

        return nn.Sequential(*all_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the DMSmall model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim_out).
        """
        return self.model(x)
