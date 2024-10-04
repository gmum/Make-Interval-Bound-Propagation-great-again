import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class IntervalLinear(nn.Linear):
    """
    Implementation of an interval version of a linear layer.

    Please see docs of 'https://pytorch.org/docs/stable/generated/torch.nn.Linear.html'
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=None) -> None:
        
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, mu: torch.Tensor, 
                        eps: torch.Tensor,
                        device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies an interval version of a linear transformation.

        Parameters:
        ----------
            mu: torch.Tensor
                The centre of an interval. 

            eps: torch.Tensor
                The radii of an interval.
            
            device: str
                A string representing the device on which
                calculations will be performed. Possible
                values are "cpu" or "cuda".

        Returns:
        --------
            new_mu: torch.Tensor
                'mu' after the linear transformation.
            
            new_eps: torch.Tensor
                'eps' after the linear transformation.
        """

        # Send tensors into devices
        mu     = mu.to(device)
        eps    = eps.to(device)
        weight = self.weight.to(device)
        bias   = self.bias.to(device)

        
        # Perform linear transformations
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
    
class IntervalReLU(nn.ReLU):
    """
    Implementation of an interval version of ReLU activation function.

    For the description of rest arguments please see docs of
    https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    """

    def __init__(self, inplace: bool = False):
       
        super().__init__(inplace)

    def forward(self, mu: torch.Tensor, 
                eps: torch.Tensor,
                device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies an interval version of ReLU transformation.

        Parameters:
        ----------
            mu: torch.Tensor
                The centre of an interval. 

            eps: torch.Tensor
                The radii of an interval.

            device: str
                A string representing the device on which
                calculations will be performed. Possible
                values are "cpu" or "cuda".

        Returns:
        --------
            new_mu: torch.Tensor
                'mu' after ReLU transformation.
            
            new_eps: torch.Tensor
                'eps' after ReLU transformation.
        """

        # Send tensors into devices
        mu  = mu.to(device)
        eps = eps.to(device)

        z_l, z_u = mu - eps, mu + eps
        z_l, z_u = F.relu(z_l), F.relu(z_u)

        new_mu, new_eps  = (z_u + z_l) / 2, (z_u - z_l) / 2

        return new_mu, new_eps
    
class IntervalConv2d(nn.Conv2d):

    """
    For the description of the rest arguments please see docs of
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | Tuple[int], 
                 stride: int | Tuple[int] = 1, padding: str | int | Tuple[int] = 0,
                 dilation: int | Tuple[int] = 1, groups: int = 1, bias: bool = True, 
                 padding_mode: str = 'zeros', device=None, dtype=None) -> None:

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                          groups, bias, padding_mode, device, dtype)
        
        self._input_shape  = None
        self._output_shape = None
        
    def forward(self, mu: torch.Tensor, 
                eps: torch.Tensor,
                device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies an interval version of a convolutional transformation.

        Parameters:
        ----------
            mu: torch.Tensor
                The centre of an interval. 

            eps: torch.Tensor
                The radii of an interval.
            
            device: str
                A string representing the device on which
                calculations will be performed. Possible
                values are "cpu" or "cuda".

        Returns:
        --------
            new_mu: torch.Tensor
                'mu' after the convolutional transformation.
            
            new_eps: torch.Tensor
                'eps' after the convolutional transformation.
        """

        # Send tensors into devices
        mu     = mu.to(device)
        eps    = eps.to(device)
        weight = self.weight.to(device)
        bias   = self.bias.to(device)

        # Save the input's shape
        self._input_shape = mu.shape
        
        # Perform convolutional transformations
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
        
        # Save the otuput's shape           
        self._output_shape = new_mu.shape

        return new_mu, new_eps
    

class IntervalSoftmax(nn.Softmax):
    """
    Implementation of an interval version of softmax function.

    Attributes:
    -----------

        which_class: int
            Indicates the class number against which the gradient will be backpropagated.

        dim_out: int
            A number of classes.
    """

    def __init__(self, dim_out: int, which_class: int) -> None:
        super().__init__(dim_out)

        self.dim_out     = dim_out
        self.which_class = which_class

    def forward(self, mu: torch.Tensor,
                 eps: torch.Tensor, 
                 device: str = "cpu") -> torch.Tensor:
        """
        Applies an interval version of softmax function.

        Parameters:
        ----------
            mu: torch.Tensor
                The centre of an interval. 

            eps: torch.Tensor
                The radii of an interval.

            device: str
                A string representing the device on which
                calculations will be performed. Possible
                values are "cpu" or "cuda".
        
        Returns:
        --------
            new_mu: torch.Tensor
                'mu' after softmax transformation.
            
            new_eps: torch.Tensor
                'eps' after softmax transformation.

        """
        
        # Send tensors to the desired device
        mu  = mu.to(device)
        eps = eps.to(device)

        mu  = mu.T
        eps = eps.T

        z_l, z_u = mu - eps, mu + eps   # We have dim x num_points

        # Create tensors Z_L and Z_U
        Z_L = torch.zeros((self.dim_out, *z_l.shape))
        Z_U = torch.zeros((self.dim_out, *z_u.shape))
    
        # The worst and best case scenarios
        for dim in range(self.dim_out):

            # The worst case
            Z_L[dim, :, :]   = torch.clone(z_u)
            Z_L[dim, dim, :] = torch.clone(z_l)[dim]

            # The best case
            Z_U[dim, :, :]   = torch.clone(z_l)
            Z_U[dim, dim, :] = torch.clone(z_u)[dim]

        # Choose k-th class
        Z_L = Z_L[self.which_class]
        Z_U = Z_U[self.which_class]

        # Calculate softmax for k-th class
        z_l = F.softmax(Z_L, dim=0)[self.which_class]
        z_u = F.softmax(Z_U, dim=0)[self.which_class]

        z_l = z_l.unsqueeze(0)
        z_u = z_u.unsqueeze(0)

        new_mu, new_eps  = (z_u + z_l) / 2, (z_u - z_l) / 2

        return new_mu, new_eps

    
class IntervalFlatten(nn.Flatten):

    """
    Simple class with tool neccessary to properly flat interval arrays.

    Attributes:
    -----------

        For the description of arguments please see docs: https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__(start_dim, end_dim)
        self._in_shape = None

    def forward(self, mu: torch.Tensor, eps: torch.Tensor,
                device: str = "cpu") -> Tuple[torch.Tensor,torch.Tensor]:
        
        """
        Applies an interval version of softmax function.

        Parameters:
        ----------
            mu: torch.Tensor
                The centre of an interval. 

            eps: torch.Tensor
                The radii of an interval.

            device: str
                A string representing the device on which
                calculations will be performed. Possible
                values are "cpu" or "cuda". It is used just for
                convenience to simplify a forward method of NNs.
        
        Returns:
        --------
            new_mu: torch.Tensor
                Flatten 'mu'.
            
            new_eps: torch.Tensor
                Flatten 'eps'.

        """

        self._in_shape = mu.shape

        mu  = mu.flatten(self.start_dim, self.end_dim)
        eps = eps.flatten(self.start_dim, self.end_dim)

        return mu, eps
    
    
if __name__ == "__main__":

    lin = IntervalLinear(5, 3)
    print(lin)
    mu = torch.randn((2,5))
    eps = torch.randn((2,5))

    print(lin(mu, eps))





