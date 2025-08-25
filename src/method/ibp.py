import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import Tuple, List
import logging

from method.method_plugin_abc import MethodPluginABC
from method.interval_arithmetic import Interval
from network.network_abc import NetworkABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class IBP(MethodPluginABC):
    """
    IBP (Interval Bound Propagation) class for handling interval-based forward propagation 
    through neural network layers.
    This class extends the `MethodPluginABC` and is designed to propagate intervals 
    through a given neural network module. It supports a variety of common layer types 
    and applies specific interval propagation logic for each supported layer.
    Attributes:
        module (NetworkABC): The neural network module through which intervals will be propagated.
    Methods:
        __init__() -> None:
            Initializes the IBP class with the given neural network module.
        forward(x: torch.Tensor, y: torch.Tensor, eps: torch.Tensor) -> Interval:
            Performs interval bound propagation through the layers of the neural network.
            Supports specific layer types and applies the corresponding interval propagation 
            logic. Logs a warning for unsupported layers.
            Args:
                x (torch.Tensor): The input tensor.
                y (torch.Tensor): An additional tensor (purpose depends on the specific use case).
                eps (torch.Tensor): A perturbation tensor.
            Returns:
                Interval: The propagated interval bounds.
    """

    def __init__(self) -> None:
        super().__init__()

        self._layer_handlers = {}
        self._register_handlers()

        log.info(f"IBP plugin initialized.")
    
    def _register_handlers(self):
        """
        Register handlers for different layer types.
        """
        self._layer_handlers: List[Tuple[type, callable]] = [
            (nn.Linear, lambda layer, x, epsilon: IntervalLinear(
                layer.in_features, layer.out_features
            ).forward(x, epsilon, weight=layer.weight, bias=layer.bias)),

            (nn.ReLU, lambda layer, x, epsilon: IntervalReLU().forward(x, epsilon)),

            (nn.Conv2d, lambda layer, x, epsilon: IntervalConv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                padding_mode=layer.padding_mode
            ).forward(x, epsilon, weight=layer.weight, bias=layer.bias)),

            (nn.Flatten, lambda layer, x, epsilon: IntervalFlatten(
                start_dim=layer.start_dim,
                end_dim=layer.end_dim
            ).forward(x, epsilon)),

            ((nn.BatchNorm1d, nn.BatchNorm2d), lambda layer, x, epsilon: IntervalBatchNorm(
                eps=layer.eps,
                momentum=layer.momentum
            ).forward(x, epsilon, layer.weight, layer.bias, layer.running_mean, layer.running_var, layer.training)),

            (nn.MaxPool2d, lambda layer, x, epsilon: IntervalMaxPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation
            ).forward(x, epsilon)),

            (nn.AvgPool2d, lambda layer, x, epsilon: IntervalAvgPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding
            ).forward(x, epsilon))
        ]

    def forward(self, x: torch.Tensor, y: torch.Tensor, eps: torch.Tensor) -> Interval:
        with torch.no_grad():
            for m in self.module.module.children():
                for layer in m:
                    for layer_type, handler in self._layer_handlers:
                        if isinstance(layer, layer_type):
                            x, eps = handler(layer, x, eps)
                            break
        return Interval(x - eps, x + eps)
class IntervalLinear:
    """
    A class that performs interval-based linear transformations for interval arithmetic.
    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
    Methods:
        __init__(in_features: int, out_features: int) -> None:
            Initializes the IntervalLinear object with the specified input and output features.
        forward(mu: torch.Tensor, eps: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
            Performs the forward pass of the interval-based linear transformation.
            Args:
                mu (torch.Tensor): The mean tensor representing the center of the interval.
                eps (torch.Tensor): The epsilon tensor representing the deviation from the center.
                weight (torch.Tensor): The weight tensor for the linear transformation.
                bias (torch.Tensor, optional): The bias tensor for the linear transformation. Defaults to None.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed mean tensor (new_mu) and the transformed epsilon tensor (new_eps).
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, mu: torch.Tensor, 
                    eps: torch.Tensor,
                    weight: torch.Tensor,
                    bias: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        # Perform linear transformation
        new_mu = F.linear(
            input=mu,
            weight=weight,
            bias=bias
        )

        new_eps = F.linear(
            input=eps,
            weight=weight.abs(),
            bias=None
        )

        return new_mu, new_eps
    
class IntervalReLU:
    """
    A class implementing the Interval ReLU operation for interval arithmetic.
    The IntervalReLU class applies the ReLU (Rectified Linear Unit) activation 
    function to interval bounds represented by their center (`mu`) and radius (`eps`).
    Attributes:
        inplace (bool): Whether to perform the operation in-place. Defaults to False.
    Methods:
        forward(mu: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Applies the ReLU activation function to the interval bounds and 
            computes the new center (`mu`) and radius (`eps`) of the resulting interval.
    """
    
    def __init__(self, inplace: bool = False):
       
       self.inplace = inplace

    def forward(self, mu: torch.Tensor, 
                eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        z_l, z_u = mu - eps, mu + eps
        z_l, z_u = F.relu(z_l), F.relu(z_u)

        new_mu, new_eps  = (z_u + z_l) / 2, (z_u - z_l) / 2

        return new_mu, new_eps
    
class IntervalConv2d:
    """
    A class that performs interval-based convolution operations for interval arithmetic.
    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple, optional): Stride of the convolution. Default is 1.
        padding (int, tuple, or str, optional): Padding added to all four sides of the input. Default is 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default is 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
        padding_mode (str, optional): Padding mode. Can be 'zeros', 'reflect', 'replicate', or 'circular'. Default is 'zeros'.
    Methods:
        forward(mu: torch.Tensor, eps: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Performs the forward pass of the interval convolution operation.
            Args:
                mu (torch.Tensor): The mean tensor of the input interval.
                eps (torch.Tensor): The epsilon tensor of the input interval.
                weight (torch.Tensor): The weight tensor for the convolution.
                bias (torch.Tensor): The bias tensor for the convolution.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - new_mu (torch.Tensor): The mean tensor of the output interval.
                    - new_eps (torch.Tensor): The epsilon tensor of the output interval.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | Tuple[int, int], 
                 stride: int | Tuple[int, int] = 1, padding: int | Tuple[int, int] = 0,
                 dilation: int | Tuple[int, int] = 1, groups: int = 1, 
                 padding_mode: str = 'zeros') -> None:

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        
    def forward(self, mu: torch.Tensor, 
                eps: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Perform convolutional transformation
        new_mu = F.conv2d(
              input=mu,
              weight=weight,
              bias=bias,
              stride=self.stride,
              padding=self.padding,
              dilation=self.dilation,
              groups=self.groups
          )

        new_eps = F.conv2d(
              input=eps,
              weight=weight.abs(),
              bias=None,
              stride=self.stride,
              padding=self.padding,
              dilation=self.dilation,
              groups=self.groups
          )
        
        return new_mu, new_eps
    
class IntervalFlatten:
    """
    A class that flattens tensors along specified dimensions for interval arithmetic.
    Attributes:
        start_dim (int): The starting dimension for flattening. Defaults to 1.
        end_dim (int): The ending dimension for flattening. Defaults to -1.
    Methods:
        forward(mu: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Flattens the input tensors `mu` and `eps` along the specified dimensions
            and returns the flattened tensors.
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, mu: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        mu  = mu.flatten(self.start_dim, self.end_dim)
        eps = eps.flatten(self.start_dim, self.end_dim)

        return mu, eps


class IntervalBatchNorm:
    """
    A class implementing interval-based batch normalization for interval arithmetic.
    Attributes:
        eps (float): A small value added to the denominator for numerical stability. Default is 1e-05.
        momentum (float): The momentum for the running mean and variance. Default is 0.1.
    Methods:
        forward(mu, eps, weight, bias, running_mean, running_var, training) -> Tuple[torch.Tensor, torch.Tensor]:
            Applies interval-based batch normalization to the input interval represented by its mean (mu) 
            and deviation (eps). Returns the updated mean and deviation after normalization.
        Applies interval-based batch normalization to the input interval.
            Args:
                mu (torch.Tensor): The mean of the input interval.
                eps (torch.Tensor): The deviation of the input interval.
                weight (torch.Tensor): The learnable scale parameter of the batch normalization.
                bias (torch.Tensor): The learnable shift parameter of the batch normalization.
                running_mean (torch.Tensor): The running mean used during evaluation.
                running_var (torch.Tensor): The running variance used during evaluation.
                training (bool): A flag indicating whether the model is in training mode.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - new_mu (torch.Tensor): The updated mean after batch normalization.
                    - new_eps (torch.Tensor): The updated deviation after batch normalization.
    """

    def __init__(self, eps: float=1e-05, momentum: float=0.1) -> None:
        self.eps = eps
        self.momentum = momentum

    def forward(self, 
                mu: torch.Tensor, 
                eps: torch.Tensor, 
                weight: torch.Tensor, 
                bias: torch.Tensor, 
                running_mean: torch.Tensor, 
                running_var: torch.Tensor, 
                training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        z_lower, z_upper = mu - eps, mu + eps

        z_lower = F.batch_norm(
            input=z_lower,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps
        )

        z_upper = F.batch_norm(
            input=z_upper,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps
        )

        # It's just possible that the batchnorm's scale is negative.
        z_lower, z_upper = torch.minimum(z_lower, z_upper), torch.maximum(z_lower, z_upper)
        new_mu = (z_lower + z_upper) / 2.0
        new_eps = (z_upper - z_lower) / 2.0

        return new_mu, new_eps
    

class IntervalMaxPool2d:
    """
    A class that performs interval-based max pooling for 2D inputs. This is useful in scenarios
    where inputs are represented as intervals (mu ± eps) and the operation needs to propagate
    these intervals through a max pooling layer.
    Attributes:
        kernel_size (int or tuple): The size of the window to take a max over.
        stride (int or tuple, optional): The stride of the window. Default is None, which means it is set to `kernel_size`.
        padding (int or tuple, optional): Implicit zero padding to be added on both sides. Default is 0.
        dilation (int or tuple, optional): A parameter that controls the stride of elements in the window. Default is 1.
    Methods:
        forward(mu: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Applies interval-based max pooling to the input tensors `mu` and `eps`.
            Args:
                mu (torch.Tensor): The mean tensor of the input interval.
                eps (torch.Tensor): The epsilon tensor representing the deviation from the mean.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: A tuple containing the new mean tensor (`new_mu`) 
                and the new epsilon tensor (`new_eps`) after max pooling.
    """
    
    def __init__(self, kernel_size: int | Tuple[int, int], stride: int | Tuple[int, int] = None, 
                 padding: int | Tuple[int, int] = 0, dilation: int | Tuple[int, int] = 1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, mu: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_lower = mu - eps
        z_upper = mu + eps
        z_lower = F.max_pool2d(
            z_lower,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )

        z_upper = F.max_pool2d(
            z_upper,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )

        new_mu, new_eps = (z_upper+z_lower)/2.0, (z_upper-z_lower)/2.0

        return new_mu, new_eps
    
class IntervalAvgPool2d:
    """
    A class that performs interval-based average pooling for 2D inputs. This operation
    applies average pooling separately to the mean (`mu`) and deviation (`eps`) tensors.
    Attributes:
        kernel_size (int or tuple): The size of the pooling window.
        stride (int or tuple, optional): The stride of the pooling operation. If None, it defaults to `kernel_size`.
        padding (int or tuple, optional): The amount of zero-padding added to both sides of the input. Default is 0.
    Methods:
        forward(mu: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Applies average pooling to the `mu` and `eps` tensors and returns the resulting tensors.
    """
    
    def __init__(self, kernel_size: int | Tuple[int, int], stride: int | Tuple[int, int] = None, 
                 padding: int | Tuple[int, int] = 0) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, mu: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        new_mu = F.avg_pool2d(mu, self.kernel_size, self.stride, self.padding)
        new_eps = F.avg_pool2d(eps, self.kernel_size, self.stride, self.padding)
        return new_mu, new_eps

    