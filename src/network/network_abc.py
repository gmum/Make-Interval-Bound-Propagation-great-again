from abc import ABCMeta, abstractmethod

from torch import nn
import torch

class NetworkABC(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for used network architectures.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the NetworkABC.

        Args:
            *args: Additional positional arguments for the nn.Module constructor.
            **kwargs: Additional keyword arguments for the nn.Module constructor.
        """
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        """
        pass

    @abstractmethod
    def build(self) -> nn.Sequential:
        """
        Builds the architecture of the network as a single flat nn.Sequential.
        
        Returns:
            nn.Sequential: A sequential model including all layers.
        """
        pass

    def register_hooks(self, m):
        def hook_fn(module, input, output):
            self.layer_outputs[module] = output.shape[1:]  # Exclude batch dim
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.register_forward_hook(hook_fn)