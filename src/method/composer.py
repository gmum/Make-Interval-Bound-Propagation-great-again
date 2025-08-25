import logging
from typing import Optional

from torch import nn
import torch

from method.method_plugin_abc import MethodPluginABC
from method.interval_arithmetic import Interval
from network.network_abc import NetworkABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Composer:
    """
    Composer is a class that facilitates the composition of a neural network module with a list of plugins. 
    Each plugin is expected to implement the `MethodPluginABC` interface and can modify the behavior of the 
    module during the forward pass.
    Attributes:
        module (NetworkABC): The neural network module to which plugins are applied.
        plugins (Optional[list[MethodPluginABC]]): A list of plugins that modify the behavior of the module. 
            Defaults to an empty list.
    Methods:
        __init__(NetworkABC, plugins: Optional[list[MethodPluginABC]] = []):
            Initializes the Composer with a module and an optional list of plugins. 
            Each plugin is associated with the module.
        forward(x: torch.Tensor, y: torch.Tensor) -> Interval:
            Executes the forward pass through the module and applies each plugin in sequence.
    """
    def __init__(self, 
        module: NetworkABC,
        plugins: Optional[list[MethodPluginABC]]=[]
    ) -> None:

        self.module = module
        self.plugins = plugins
        for plugin in self.plugins:
            
            plugin.set_module(self.module)
            log.info(f'Plugin {plugin.__class__.__name__} added to composer')


    def forward(self, x: torch.Tensor, y: torch.Tensor, eps: torch.Tensor) -> Interval:
        """
        Performs the forward pass through the module and its associated plugins.
        This method iteratively applies each plugin's forward method to the input tensor `x`,
        using the target tensor `y` and epsilon tensor `eps` as additional inputs. The final
        transformed tensor is returned as the output.
            x (torch.Tensor): The input tensor to be processed.
            y (torch.Tensor): The target tensor used for processing.
            eps (torch.Tensor): A tensor representing the perturbation range. The final radii
                for the perturbation are computed as `self.epsilon * eps` within a plugin.
            Interval: The resulting interval bounds after processing through all plugins.
        """
        for plugin in self.plugins:
            x = plugin.forward(x, y, eps)
        
        return x
        


   