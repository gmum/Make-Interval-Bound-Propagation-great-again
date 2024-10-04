import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from IntervalNNs.interval_cnn import IntervalCNN
from IntervalNNs.interval_mlp import IntervalMLP
from IntervalNNs.architecture_utils import get_dim_in_out

import numpy as np
import random
from typing import Tuple

def get_data_loaders(dataset_name: str, 
                     val_size: float = 0.1,
                     train_batch_size: int = 32,
                     test_batch_size: int = 1,
                     arch_type: str = "cnn_small",
                     shuffle_test_data: bool = False,
                     data_path: str = "./Data") -> Tuple[DataLoader,DataLoader,DataLoader]:
    """
    Get train, val and test dataloaders for the given dataset.

    Parameters:
    -----------
        dataset_name: str
            Possible values are: 'mnist', 'cifar', 'svhn', 'digits'.

        val_size: float
            Ratio of sampples to be taken for validation purpose.

        train_batch_size: int
            Number of samples in one single batch of train data.

        test_batch_size: int
            Number of samples in one single batch of test data.

        arch_type: str
            Defines an architecture type. Possible values are: 'cnn_small', 'cnn_medium', 'cnn_large' and 'mlp'.

        shuffle_test_data: bool
            A flag to indicate whether test data should be shuffled or not.

        data_path: str
            Path to folder, where the data will be downloaded.

    Returns:
    --------
        A tuple with, respectively, train, val and test dataloaders.
    """

    assert dataset_name in ["mnist", "cifar", "svhn", "digits"], "Possible datasets name are: 'mnist', 'cifar', 'svhn' and 'digits'"
    assert val_size > 0 and val_size < 1
    assert arch_type in ["mlp", "cnn_small", "cnn_medium", "cnn_large"], "Possible values are: 'cnn_small', 'cnn_medium', 'cnn_large'" \
                                                                          "and 'mlp'."

    if dataset_name == "mnist":
        transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        train = datasets.MNIST(
            data_path,
            train=True,
            download=True,
            transform=transform
        )
        test = datasets.MNIST(
            data_path,
            train=False,
            download=True,
            transform=transform
        )
    elif dataset_name == "cifar":
        normalize = transforms.Normalize((0.4915, 0.4823, 0.4468),
                         (0.2470, 0.2435, 0.2616))
        
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=[32,32], padding=4),
                transforms.ToTensor(),
                normalize
            ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train = datasets.CIFAR10(
            data_path,
            train=True,
            download=True,
            transform=train_transform
        )
        test = datasets.CIFAR10(
            data_path,
            train=False,
            download=True,
            transform=test_transform
        )
    elif dataset_name == "svhn":
        normalize = transforms.Normalize((0.4376821, 0.4437697, 0.47280442), 
                                         (0.19803012, 0.20101562, 0.19703614))

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=[32,32], padding=4),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train = datasets.SVHN(
            data_path,
            split="train",
            download=True,
            transform=train_transform
        )
        test = datasets.SVHN(
            data_path,
            split="test",
            download=True,
            transform=test_transform
        )

    elif dataset_name == "digits":

        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, 
                                                            test_size=val_size)
        X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train,
                                                          test_size=val_size)
        
        X_train = torch.Tensor(X_train)
        y_train = torch.Tensor(y_train).long()

        X_val = torch.Tensor(X_val)
        y_val = torch.Tensor(y_val).long()

        X_test = torch.Tensor(X_test)
        y_test = torch.Tensor(y_test).long()

        train_split = TensorDataset(X_train, y_train)
        val_split   = TensorDataset(X_val, y_val)
        test        = TensorDataset(X_test, y_test)

    # Generate indices: instead of the actual data we pass in integers instead
    if dataset_name == "svhn":
        train_indices, val_indices, _, _ = train_test_split(
            range(len(train)),
            train.labels,
            stratify=train.labels,
            test_size=val_size,
        )
    elif dataset_name in [
        "mnist",
        "cifar"
    ]:
        train_indices, val_indices, _, _ = train_test_split(
            range(len(train)),
            train.targets,
            stratify=train.targets,
            test_size=val_size,
        )

    if dataset_name in [
        "mnist",
        "cifar",
        "svhn"
    ]:
        # Generate a subset based on indices
        train_split = Subset(train, train_indices)
        val_split   = Subset(train, val_indices)

    # Create batches
    train_dl = DataLoader(train_split, batch_size=train_batch_size, shuffle=True)
    val_dl   = DataLoader(val_split, batch_size=train_batch_size, shuffle=True)
    test_dl  = DataLoader(test, batch_size=test_batch_size, shuffle=shuffle_test_data)

    return train_dl, val_dl, test_dl

def create_model(hyperparams: dict) -> torch.nn.Module:
    """
    Prepare the desired architecture to training.

    Parameters:
    -----------
        hyperparams: dict
            A dictionary with hyperparameters loaded from a json file.
    
    Returns:
    --------
        Neural network architecture which will be trained / evaluated.
    """
    
    arch         = hyperparams["arch"]
    dataset_name = hyperparams["dataset_name"]

    assert arch in [
       "cnn_small",
       "cnn_medium",
       "cnn_large",
       "mlp"
    ], "Possible values are: 'cnn_small', 'cnn_medium', 'cnn_large', 'mlp'."

    assert dataset_name in [
       "mnist",
       "digits",
       "cifar",
       "svhn"
    ], "Possible values are: 'mnist', 'digits', 'cifar' and 'svhn'."

    if "cnn" in arch:
       
       assert dataset_name != "digits", "CNNs are not implemented for the Digits dataset!"

       return IntervalCNN(
            arch=arch,
            mode=dataset_name,
       )
    elif arch == "mlp":
       dim_in, dim_out = get_dim_in_out(dataset_name)
       
       return IntervalMLP(
           num_layers=hyperparams["num_layers"],
           dim_in=dim_in,
           dim_hidden=hyperparams["dim_hidden"],
           dim_out=dim_out,
       )
    
def set_seed(seed: int) -> None:
    """
    Sets a global seed within the following modules: 'PyTorch', 'NumPy' and 'random'
    """
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_model(model: torch.nn.Module, save_path: str) -> None:
    torch.save(model.state_dict(), save_path)


def get_optimizer(optimizer_name: str, params: nn.Module.parameters,
                  lr: float, weight_decay: float = None) -> torch.optim:
    """
    Returns an optimizer to optimize `params` during training.

    Parameters:
    -----------
        optimizer_name: str
            Name of an optimizer. Currently supported values are: `adam`, 'sgd'.
        
        params: torch.nn.Module.parameters
            Parameter to be optimized during training.
        
        lr: float
            Learning rate.
        
        weight_decay: float
            A number indicating strenght of regularization applied to the optimized
            parameters. Default value is set to be None.

    Returns:
    --------
        A 'torch.optim' object.
    """

    assert optimizer_name in ["adam", "sgd"], "Currently supported optimizers are: 'adam' and 'sgd'."

    if optimizer_name == "adam":
        return torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(params=params, lr=lr, weight_decay=weight_decay)
    
def load_model(model: nn.Module, weight_path: str, device: str = "cpu") -> None:
   """
   Loads weights from the 'weight_path' to the given model.

   Parameters:
   -----------
        model: torch.nn.Module

        weight_path: str
            Path to the file with stored weights

        device: str
            Device on which calculations will be performed.
    Returns:
    --------
       None
    """
   
   model.load_state_dict(torch.load(weight_path, map_location=device))

def sample_points(dataset_name: str, arch_type: str, no_points: int) -> DataLoader:
    """
    Samples 'no_points' from the test data and wrap them into
    DataLoader object.

    Parameters:
    -----------
        dataset_name: str
            Possible values are: 'mnist', 'cifar', 'svhn', 'digits'.

        arch_type: str
            Defines an architecture type. Possible values are: 'cnn_small', 'cnn_medium', 'cnn_large' and 'mlp'.
        
        no_points: int
            Number of points to be sampled.
    """

    _, _, test_dl = get_data_loaders(
        dataset_name=dataset_name,
        arch_type=arch_type,
        test_batch_size=no_points,
        shuffle_test_data=True
    )
    X, y = next(iter(test_dl))
    data = TensorDataset(X,y)    
    test_dl = DataLoader(data, batch_size=1, shuffle=False)

    return test_dl
