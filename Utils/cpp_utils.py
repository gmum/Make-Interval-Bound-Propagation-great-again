import torch
import numpy as np

from typing import Iterable, Tuple
import csv
import json
import path
import sys

# Directory reach
directory = path.Path(__file__).abspath()
 
# Setting path
sys.path.append(directory.parent.parent)

from IntervalNNs.interval_cnn import IntervalCNN
from IntervalTools.interval_modules import IntervalConv2d


def save_weights2txt(path_to_weights: str, 
                     path_to_save: str,
                     arch: str = None,
                     mode: str = None,
                     save_conv_weights_as_Toeplitz_matrix: bool = False) -> None:
    """
    Load weights from a pickle file and saves them with .txt extension
    to be able to be processed by C++ scripts.

    Parameters:
    -----------

        path_to_weights: str
            Path to weights saved with .pth extension. The path
            should ends with .pth extension.

        path_to_save: str
            Path where weight with .txt extension should
            be saved. The path should ends with name of the file
            with .txt extension.

        arch: str
            Allowed values are: 'cnn_small', 'cnn_medium', 'cnn_large'.
            This parameter is used only when 'save_conv_weights_as_Toeplitz_matrix'
            is set to be 'True'.

        mode: str
            Allowed values are: 'mnist', 'cifar', 'svhn'.

        save_conv_weights_as_Toeplitz_matrix: bool
            Saves convolutional kernels as linear weights.
    """

    assert arch in [
        "cnn_small",
        "cnn_medium",
        "cnn_large",
        "mlp"
    ]

    assert mode in [
        "mnist",
        "cifar",
        "svhn",
        "digits"
    ]

    if save_conv_weights_as_Toeplitz_matrix:
        if arch == "cnn_small":
            conv_idx_list = [_ for _ in range(4)]
        elif arch == "cnn_medium":
            conv_idx_list = [_ for _ in range(8)]
        elif arch == "cnn_large":
            conv_idx_list = [_ for _ in range(10)]

        net = IntervalCNN(arch=arch, mode=mode)

        if mode == "mnist":
            input_shape = (1,1,28,28)
        elif mode in ["cifar", "svhn"]:
            input_shape = (1,3,32,32)

        input_shapes = []
        strides = []
        num_patches = []

        layers = next(iter(net.children()))
        for layer in layers:
            if isinstance(layer, IntervalConv2d):
                input_shapes.append(input_shape[1:])
                strides.append(layer.stride)
                
                # Calculate input shape to the next convolutional layer
                h = (input_shapes[-1][-2] - layer.kernel_size[0] + 2*layer.padding[0]) // layer.stride[0] + 1
                w = (input_shapes[-1][-1] - layer.kernel_size[1] + 2*layer.padding[1]) // layer.stride[1] + 1

                input_shape = (1,layer.weight.size(0),h,w)
                num_patches.append(h*w)
   
    # Load the weights from the .pth file
    model_weights = torch.load(path_to_weights, map_location="cpu")

    weights_dict = dict()
    for layer_idx, (name, weights) in enumerate(model_weights.items()):
        if save_conv_weights_as_Toeplitz_matrix and \
            layer_idx in conv_idx_list and \
            layer_idx % 2 == 0:

            weights_dict[name] = create_toeplitz_mult_ch(
                kernel=weights,
                input_size=input_shapes[layer_idx//2],
                stride=strides[layer_idx//2]
            ).numpy().tolist()

        elif save_conv_weights_as_Toeplitz_matrix and \
            layer_idx in conv_idx_list and \
            layer_idx % 2 == 1:
            
            weights_dict[name] = torch.tensor(np.repeat(weights.numpy(), num_patches[layer_idx // 2])).tolist()
        else:
            weights_dict[name] = weights.data.numpy().tolist()

    # Convert and write JSON object to file
    with open(path_to_save, "w") as outfile:
        json.dump(weights_dict, outfile)

def save_data2txt(data: Iterable[torch.Tensor], path_to_save: str, 
              mode: str = "digits",
              use_toeplitz_transform: bool = True,
              n_points_to_save: int = 10) -> None:
    """
    Saves dataset to format acceptable by C++ scripts.

    Parameters:
    -----------

        data: Iterable[torch.Tensor,torch.Tensor]
            Data in form (X,y), where 'y' is a tensor with
            real labels.

        path_to_save: str
            Path where teh dataset should be saved.

        mode: bool
            If an MLP is used by C++ scripts, then data are saved
            as flattened vectors. Possible values are:
            - digits,
            - mnist,
            - cifar,
            - svhn.

        use_toeplitz_transform: bool
            Flag to indicate whether Toeplitz matrices are used instead
            of classical convolutional kernels.

        n_points_to_save: int
            Number of points to be saved. Default value is set to
            10 first tensors with corresponding labels.

    """ 

    assert mode in [
        "digits",
        "mnist",
        "cifar",
        "svhn"
    ]

    input_output_dict = {}
    for idx, (X_data, _) in enumerate(data):
        if use_toeplitz_transform:
            input_output_dict[f"input_{idx}"] = X_data.flatten().unsqueeze(0).detach().numpy().tolist()
        else:
            input_output_dict[f"input_{idx}"] = X_data.detach().numpy().tolist()

        if idx == n_points_to_save-1:
            break

    # Convert and write JSON object to file
    with open(path_to_save, "w") as out:
        json.dump(input_output_dict, out)

def create_toeplitz_1_ch(kernel: torch.Tensor, input_size: Tuple[int], stride: Tuple[int]) -> torch.Tensor:
    """
    Calculates Toeplitz matrix from the 'kernel' tensor for only one dimension.

    Parameters:
    ------------

        kernel: torch.Tensor
            Kernel tensor used in a convolutional layer.

        input_size: Tuple[int]
            Shape of the input to a convolutional layer.

        stride: int
            Stride parameter.

    Returns:
    --------

        torch.Tensor: A Toeplitz matrix for one channel.
    """
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    o_h = (i_h - k_h) // stride[0] + 1
    o_w = (i_w - k_w) // stride[1] + 1

    T = torch.zeros((o_h * o_w, i_h * i_w))

    for i in range(o_h):
        for j in range(o_w):
            start_i = i * stride[0]
            start_j = j * stride[1]
            end_i = start_i + k_h
            end_j = start_j + k_w
            patch = torch.zeros((i_h, i_w))
            patch[start_i:end_i, start_j:end_j] = kernel
            T[i * o_w + j, :] = patch.flatten()

    return T
def create_toeplitz_mult_ch(kernel: torch.Tensor, input_size: Tuple[int], stride: Tuple[int]) -> torch.Tensor:
    """
    Compute Toeplitz matrix for 2D convolution with multiple input and output channels and stride.
    
    Parameters:
    -----------

        kernel: torch.Tensor
            Kernel tensor used in a convolutional layer.

        input_size: Tuple[int]
            Shape of the input to a convolutional layer.

        stride: Tuple[int]
            Stride parameter.

    Returns:
    --------

        torch.Tensor: A Toeplitz matrix for multiple channels.
    """
    kernel_size = kernel.shape
    i_h, i_w = input_size[1:]
    o_h = (i_h - kernel_size[2]) // stride[0] + 1
    o_w = (i_w - kernel_size[3]) // stride[1] + 1
    num_patches = o_h * o_w
    num_elements = torch.prod(torch.tensor(input_size))
    T = torch.zeros((kernel_size[0] * num_patches, num_elements))

    for i, ks in enumerate(kernel):  # loop over output channels
        for j, k in enumerate(ks):  # loop over input channels
            T_k = create_toeplitz_1_ch(k, input_size[1:], stride)
            T[i * num_patches:(i + 1) * num_patches, j * i_h * i_w:(j + 1) * i_h * i_w] = T_k

    return T

if __name__ == '__main__':

    pass