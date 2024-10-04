from typing import Iterable, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, TensorDataset

from IntervalNNs.interval_cnn import IntervalCNN
from IntervalNNs.interval_mlp import IntervalMLP

def mask_points(data: torch.Tensor, p: float = 0.3,
                use_toeplitz_transform: bool = False) -> DataLoader:
    """
    Function zeroes 'p' percent of values contained in the 'data'.

    Parameters:
    -----------
        
        data: torch.Tensor
            Batch of tensors.

        use_toeplitz_transform: bool
            Flag to indicate whether Toeplitz transform is applied
            to data or not.

    Returns:
    --------

        DataLoader with perturbations.
    """

    assert p <= 1 and p >= 0

    X_list = []
    y_list = []

    for X,y in data:
        mask = (torch.rand_like(X) >= p).int()
        out = mask * X

        if use_toeplitz_transform:
            out = out.flatten()

        X_list.append(out)
        y_list.append(y)

    X_tensor = torch.as_tensor(np.array(X_list)).squeeze(1)
    y_tensor = torch.as_tensor(np.array(y_list))

    tensors = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(tensors, batch_size=1, shuffle=False)

    return dl



def select_points_from_each_class(dataloader: DataLoader, no_classes: int = 10,
                                  return_dl: bool = False) -> Union[dict,DataLoader]:
    """
    Functions samples points from the dataset wrapped into
    DataLoader such that each point belongs to distinct class.

    Parameters:
    -----------

        dataloader: DataLoader
            The dataset wrapped into 'DataLoader' object.

        no_classes: int
            A number of classes in the dataset.

        return_dl: bool
            Returns DataLoader object if needed.

    Returns:
    --------

        dict | DataLoader: A dictionary with keys being class labels and values being
        corresponding points. If 'return_dl' is set to be 'True', then a DataLoader
        is returned.
    """
    class_points = {}

    for X,y in dataloader:

        # Assuming target is a tensor with class labels
        for i in range(len(y)):
            label = y[i].item()
            if label not in class_points:
                class_points[label] = X[i]
                
            # Stop if we've collected a point from each class
            if len(class_points) == no_classes:
                break
                # return class_points
            
    if return_dl:
        # Convert dictionary values to a list of tensors (features) and dictionary keys to a list of labels
        tensors = np.array(list(class_points.values()))
        tensors = torch.tensor(tensors).squeeze(1)
        labels  = torch.tensor(list(class_points.keys()))

        # Create a TensorDataset from features and labels
        data = TensorDataset(tensors, labels)

        # Create a DataLoader
        dl = DataLoader(data, shuffle=False)

        return dl

    return class_points

def get_affine_doubleton_arithmetic_bounds(
        file_path: str,
        mode: str = "yAffn",
) -> np.ndarray:
    """
    Functions returns bounds for affine and doubleton arithmetic
    from the file, which is output of the 'main.cpp' file saved in
    AffineAndDoubletonArithmetic folder.

    Parameters:
    ------------

        file_path: str
            Path to the file with results of affine and doubleton
            arithmetic.

        perturbation_eps: float
            Perturbation applied to the data.

        mode: str
            String to indicate whether affine or doubleton arithmetic
            results should be returned. There is a possibility to return
            an ideal estimate of network's output cube. Possible values are:
            - yAffn,
            - yDltn,
            - yRand,
            - yIntv

    Returns:
    --------
        np.ndarray: An array with interval bounds
    """

    # Initialize a dictionary to store the results
    result_dict = defaultdict(lambda: [])

    assert mode in ["yDltn", "yAffn", "yRand", "yIntv"]

    # Open the file and read the content
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into sections based on the test separator
    sections = content.split('#######################')

    # Iterate over each section to extract epsilon and bounds
    for section in sections:

        if 'Test for :' in section:
            
            # Find the epsilon value
            epsilon_match = re.search(r'Test for : ([\deE\-\+\.]+)', section)
            epsilon_idx = epsilon_match.group(1)

            if epsilon_match:

                # Find all the bound types (yIntv, yDltn, yAffn, yRand) and their values
                bounds = re.findall(r'(y\w+)=\{+(\[[^\}]+\])\}+', section)
                bounds_dict = {item[0]:item[1:][0] for item in bounds if item[0] == mode}

                if mode in bounds_dict.keys():
                    # Convert the bound values to a numpy array
                    bound_array = np.array(eval(bounds_dict[mode]))

                    # Store the bound array in the dictionary
                    result_dict[epsilon_idx].append(np.max([arr[1] - arr[0] for arr in bound_array]))

    return result_dict

def plot_intervals_length(
        save_path: str,
        file_path_with_affine_and_doubleton_results: str,
        perturbation_eps: float,
        start_eps_value: float = 1e-5,
        figsize: Tuple[int] = (8,4),
        fontsize = 13,
        add_doubleton_bounds: bool = True,
        no_epsilons: int = 20,
        legend_place: str = "upper left",
        bottom_ylim: float = 1e-10,
        top_ylim: float = 1.
) -> None:
    """
    Function plots and saves plots of dependency of neural net's 
    output interval lenghts on perturbation size applied to input.

    Parameters:
    -----------

        no_epsilons: int
            Number of perturbation sizes taken sequentially between 'start_eps_value'
            and 'perturbation_eps'.

        save_path: str
            String respresenting save path to folder where plot will be
            saved.

        file_path_with_affine_and_doubleton_results: str
            Path to a txt file with results of affine and doubleton arithmetic.

        perturbation_eps: float
            Maximum perturbation size applied to data.

        start_eps_value: float
            Minimum pertubration size applied to data.

        figsize: tuple
            Represents the size of the plot.

        fontsize: int
            Size of font in title, OX, and OY axes.

        add_doubleton_bounds: bool
            Flag to indicate whether doubleton bounds length should be plotted.

        no_epsilons: int
            Number of epsilons.

        legend_place: str
            Possible values are desribed here: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

        bottom_ylim: float
            Bottom ylim of the plots.

        top_ylim: float
            Top ylim of the plots.
    """

    # Make a grid of perturbation sizes
    epsilons = np.linspace(
        start=start_eps_value,
        stop=perturbation_eps,
        num=no_epsilons
    )

    fig, ax = plt.subplots(figsize=figsize)

    method_types = ["Affine Arithmetic", "IBP", "Lower Bound"]

    if add_doubleton_bounds:
        method_types.append("Doubleton Arithmetic")
    
    with torch.no_grad():
        for method_type in method_types:
            outputs = []

            for eps_idx, eps in enumerate(epsilons):        
                if method_type == "IBP":
                    tmp = get_affine_doubleton_arithmetic_bounds(
                        file_path=file_path_with_affine_and_doubleton_results,
                        mode="yIntv",
                    )
                elif method_type == "Affine Arithmetic":
                    tmp = get_affine_doubleton_arithmetic_bounds(
                        file_path=file_path_with_affine_and_doubleton_results,
                        mode="yAffn",
                    )
                elif method_type == "Doubleton Arithmetic":
                    tmp = get_affine_doubleton_arithmetic_bounds(
                        file_path=file_path_with_affine_and_doubleton_results,
                        mode="yDltn",
                    )
                elif method_type == "Lower Bound":
                    tmp = get_affine_doubleton_arithmetic_bounds(
                        file_path=file_path_with_affine_and_doubleton_results,
                        mode="yRand",
                    )
                    
                diffs = np.array(tmp[f"{eps_idx}"])
                diffs_mean = np.mean(diffs)

                outputs.append(
                    diffs_mean
                )
                print(f"Epsilon: {eps}/{perturbation_eps} is processed.")
            
            if add_doubleton_bounds and method_type == "Affine Arithmetic":
                ax.plot(epsilons, outputs, marker="o", label=f"{method_type}")
            elif add_doubleton_bounds and method_type == "Doubleton Arithmetic":
                ax.plot(epsilons, outputs, marker='v', label=f"{method_type}")
            else:
                ax.plot(epsilons, outputs, label=f"{method_type}")

        ax.set_title("Dependence of output maximal interval length on input perturbation", fontsize=fontsize)
        ax.set_xlabel("Perturbation size", fontsize=fontsize)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=bottom_ylim, top=top_ylim)
        ax.set_ylabel("Maximal output interval length", fontsize=fontsize)
        ax.grid()

        ax.legend(loc=legend_place, fontsize=fontsize)
        plt.tight_layout()
        plt.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlim(right=perturbation_eps)
        fig.savefig(f"{save_path}")
        plt.close()

def find_the_most_uncertain_points(
        model: Union[IntervalCNN,IntervalMLP],
        dataset: Iterable[torch.Tensor],
        perturbation_eps: float,
) -> DataLoader:
    """
    The function connects points from 'dataset' each by each by line segment.
    Then, functions samples points which lie near the decision boundary and
    returns them wrapped into the DataLoader object.

    Parameters:
    -----------
        model: Union[IntervalCNN,IntervalMLP]
            Architecture model with loaded trained weights.

        dataset: dict
            Dictionary with keys being class labels and values being
            corresponding tensors.

        perturbation_eps: float
            Perturbation size applied to the input data.

    Returns:
    --------

        DataLoader: Tensor with corresponding class labels.
    """

    alphas = torch.linspace(0, 1.0, 100)

    results = {}

    with torch.no_grad():

        # Get the 0-th class label and the corresponding tensor
        first_label  = 0
        first_tensor = dataset[first_label]

        for class_label in dataset.keys():
            if class_label != first_label:
                current_tensor = dataset[class_label]
                prev_connected = current_tensor

                if len(prev_connected.shape) == 3:
                    prev_connected = prev_connected.unsqueeze(0)

                for alpha in alphas:
                    current_connected = (1-alpha) * first_tensor + alpha * current_tensor

                    if len(current_connected.shape) == 3:
                        current_connected = current_connected.unsqueeze(0)
                    eps = perturbation_eps * torch.ones_like(current_connected)

                    _, _, y_pred, _ = model(
                            current_connected,
                            eps,
                            use_softmax=False,
                        )
                    y_pred = torch.argmax(y_pred, dim=-1)
                   
                    if y_pred == class_label:
                        results[class_label] = prev_connected
                        break

                    prev_connected = current_connected

    # Convert dictionary values to a list of tensors (features) and dictionary keys to a list of labels
    tensors = np.array(list(results.values()))
    tensors = torch.tensor(tensors).squeeze(1)
    labels  = torch.tensor(list(results.keys()))

    # Create a TensorDataset from features and labels
    data = TensorDataset(tensors, labels)

    # Create a DataLoader
    dl = DataLoader(data, shuffle=False)

    return dl

if __name__ == "__main__":

    pass