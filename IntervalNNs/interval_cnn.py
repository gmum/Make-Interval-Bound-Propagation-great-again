import torch
import torch.nn as nn
import sys
import path

# Directory reach
directory = path.Path(__file__).abspath()
 
# Setting path
sys.path.append(directory.parent.parent)

from typing import Tuple

from IntervalTools.interval_modules import IntervalSoftmax
from IntervalNNs.architecture_utils import get_architecture, get_dim_in_out

class IntervalCNN(nn.Module):
    """
    This class implements interval-based fully-connected networks
    with ReLU activation functions.

    Attributes:
    -----------
        arch: str
            Possible values are: `cnn_small`, `cnn_medium`, `cnn_large`.
            
        mode: str
            Adjust architecture to be able to process given dataset.
            Possible values are: 'mnist', 'svhn', 'cifar'.
        
        which_class: int
            Class number to be considered.
    """

    def __init__(self, arch: str = "cnn_small",
               mode: str = "mnist",
               which_class: int = 0):
        
        super().__init__()

        # Get architecture
        self.layers = get_architecture(
        arch=arch,
        mode=mode
        )

        # Get architecture parameters
        self.dim_out = get_dim_in_out(mode)[1]

        # Backward parameters
        assert which_class < self.dim_out, "Choose a correct class label!"
        self.which_class = which_class

        # Interval softmax layer
        self.softmax_layer = IntervalSoftmax(self.dim_out, which_class)

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

        for layer in self.layers:
            mu, eps = layer(mu, eps, device=device)
            
        # If softmax is used as a conversion of final logits into
        # probabilities
        if use_softmax:
            mu, eps = self.softmax_layer(mu=mu, eps=eps, device=device)

        return mu-eps, mu+eps, mu, eps
  

if __name__ == '__main__':

    torch.manual_seed(1)
   
    # Test
    interval_nn = IntervalCNN(mode="mnist", arch="cnn_small")

    X = torch.randn((1, 1, 28, 28))
    eps = 0.05 * torch.ones_like(X)

    zl, zu, mu_pred, eps_pred = interval_nn(X, eps, use_softmax=True)

    print(f"Lower logit: {zl}\n")
    print(f"Upper logit: {zu}\n")
    print(f"Mu pred: {mu_pred}\n")