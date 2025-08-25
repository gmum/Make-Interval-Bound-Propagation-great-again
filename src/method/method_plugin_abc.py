from abc import ABCMeta, abstractmethod

import torch

from network.network_abc import NetworkABC
from method.interval_arithmetic import Interval

class MethodPluginABC(metaclass=ABCMeta):    
    """
    An abstract base class for method plugins that operate on neural network modules.
    This class defines the interface for plugins that can be attached to a neural 
    network module and perform specific operations during the forward pass.
    Methods:
        set_module(module: NetworkABC):
            Sets the neural network module for the plugin to operate on.
        forward(x,y,eps):
            Abstract method that defines the internal forward pass logic. 
            Must be implemented by subclasses.
    Attributes:
        module (NetworkABC):
            The neural network module associated with the plugin.
    """
    def set_module(self, module: NetworkABC) -> None:
        """
        Set the module for the plugin.
        
        Args:
            module (NetworkABC): The model to be set.
        """

        self.module = module

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor, eps: torch.Tensor) -> Interval:
        """
        Internal forward pass.
        """
        pass