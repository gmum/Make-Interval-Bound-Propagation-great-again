import torch
import torch.nn as nn
import sys
import path

# Directory reach
directory = path.Path(__file__).abspath()
 
# Setting path
sys.path.append(directory.parent.parent)

from typing import Tuple
from IntervalTools.interval_modules import (
    IntervalLinear,
    IntervalSoftmax,
    IntervalReLU
)

class IntervalMLP(nn.Module):
    """
    This class implements interval-based fully-connected networks
    with ReLU activation functions.

    Attributes:
        -----------
            num_layers: int
                Number of fully-connected layers.
            
            dim_in: int
                Dimensionality of input.
            
            dim_out: int
                Number of classes.
            
            which_class: int
                Class number against which the gradient will be backpropagated.
    """

    def __init__(self, num_layers: int = 4, 
                dim_in: int = 2, 
                dim_hidden: int = 100, 
                dim_out: int = 2, 
                which_class: int = 0):
        
        super().__init__()

        # Architecture Parameters
        self.dim_in     = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out    = dim_out
        self.num_layers = num_layers

        # Backward parameters
        assert which_class < dim_out, "Choose a correct class label!"
        self.which_class = which_class

        # Fully conntected layers + ReLU activation function
        self.linears = nn.ModuleList([IntervalLinear(dim_in, dim_hidden),
                                    IntervalReLU()])
        for _ in range(num_layers):
            self.linears.append(IntervalLinear(dim_hidden, dim_hidden))
            self.linears.append(IntervalReLU())

        self.linears.append(IntervalLinear(dim_hidden, dim_out))

        self.softmax_layer = IntervalSoftmax(dim_out, which_class)

    def forward(self, mu: torch.Tensor, 
                eps: torch.Tensor,
                use_softmax: bool = True,
                device: str = "cpu") -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        """
        Parameters:
        -----------
            mu: torch.Tensor
                Interval centres.

            eps: torch.Tensor
                Interval radii.

            use_softmax: bool
                Decides to use nn.Softmax() as the last layer or not.

            device: str
                Device on which calculations will be performed.

        Returns:
        --------
            A tuple with, respectively, lower logit, upper logit, middle logit and
            radii transformed by the neural network.
        """
        # Send mu and eps to the GPU
        mu  = mu.to(device)
        eps = eps.to(device)

        for layer in self.linears:
            mu, eps = layer(mu, eps, device=device)
            
        # If softmax is used as a conversion of final logits into
        # probabilities
        if use_softmax:
            mu, eps = self.softmax_layer(mu=mu, eps=eps, device=device)

        return mu-eps, mu+eps, mu, eps
  

if __name__ == '__main__':
   
    # Test
    interval_nn = IntervalMLP(dim_in=4, which_class=0, dim_out=2)

    X = torch.randn(3, 4)
    eps = 0.005 * torch.ones_like(X)

    zl, zu, mu_pred, eps_pred = interval_nn(X, eps, use_softmax=True)
    print(f"Mu pred: {mu_pred}")