from Utils.handy_functions import (
    set_seed,
    create_model,
    get_data_loaders,
)
from Utils.logger import create_logger
from Utils.logger_config import Color
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np

from IntervalNNs.interval_cnn import IntervalCNN
from IntervalNNs.interval_mlp import IntervalMLP

from typing import Union
import time
import argparse
import json
import pandas as pd
import os
from copy import deepcopy

def main() -> None:
    """
    This method runs the training of interval neural networks. Hyperparameters
    are defined in the 'Experiments' folder.
    """
    args = load_hyperparams()

    # Set seed for reproducibility
    seed = args["seed"][0]
    set_seed(seed)

    # Create a list of keys and a list of values, where each value is a list of possible values
    keys, values = zip(*args.items())

    # Generate all combinations of hyperparameter values
    for combination in product(*values):
        params = dict(zip(keys, combination))

        # Filter out the non-list parameters
        params     = {k: v for k, v in params.items() if not isinstance(v, list)}
        neural_net = create_model(params)

        # Add the logger to the args dictionary
        logger, out_data_folder_name = create_logger(params["dataset_name"])
        params["logger"]             = logger
        params["save_path"]          = out_data_folder_name

        # Create dataloaders
        train_dl, val_dl, test_dl = get_data_loaders(
            dataset_name=params["dataset_name"],
            val_size=params["val_set_size"],
            train_batch_size=params["batch_size"],
            test_batch_size=1,
            arch_type=params["arch"],
            data_path=params["data_path"]
        )

        # Save hyperparams
        save_hyperparams(params, out_data_folder_name)

        # Run training        
        start_training_time = time.time()

        logger.info(f"Training of the {Color.YELLOW}{params['arch']}{Color.RESET} model has started "
                    f"for the {Color.YELLOW}{params['dataset_name']}{Color.RESET} dataset.")
        

        neural_net = train_model(
                    net=neural_net,
                    train_dl=train_dl,
                    val_dl=val_dl,
                    params=params
                )

        elapsed_training_time = time.time() - start_training_time

        logger.info(f"After {Color.YELLOW}{elapsed_training_time:.4f}{Color.RESET} seconds "
                f"the training is done and the model is saved \n")

        

def load_json(settings_path: str) -> dict:
    """
    Parameters:
    -----------
        settings_path: str
            A name of JSON file with hyperparameters
    
    Returns:
    --------
        params: dict
            A dictionary with loaded hyperparameters.
    """
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser object.
    """
    parser = argparse.ArgumentParser(description="Graph autoencoder training")
    parser.add_argument('--config', type=str, default='../experiments/mnist_hyperparams.json',
                        help='Json file of settings.')

    return parser


def load_hyperparams() -> dict:
    """
    Function loads hyperparameters used in the framework.
    """

    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)
    args.update(param)
    del args["config"]

    return args


def save_hyperparams(params: dict, folder_path: str) -> None:
    """
    Saves currently chosen hyperparameters into .csv file
    in the 'folder_path' folder

    Parameters:
    -----------
        params: dict
            A dictionary with hyperparameters.
        
        folder_path: str
            A name of folder where results will be stored.
    
    Returns:
    --------
        None
    """
    df = pd.DataFrame(params, index=[0])

    if df["logger"] is not None:
        del df["logger"]

    df.to_csv(f"{folder_path}/hyperparams.csv", index=False)


def train_model(
        net: Union[IntervalMLP,IntervalCNN],
        train_dl: DataLoader,
        val_dl: DataLoader,
        params: dict
        ) -> nn.Module:

    """
    Function performs interval training.

    Parameters:
    -----------
        net: Union[IntervalMLP,IntervalCNN]
            An interval version of neural network architecture
            that will be trained by 'n_epochs' epochs.

        train_dl: DataLoader
            Dataloader with training batches of data.

        val_dl: Dataloader
            Dataloader with validation batches of data.

        perturbation_epsilon: float
            A magnitude of perturbation applied to input data.

        params: dict
            Dictionary with the following keys:
                - kappa_max: float
                    A float controlling trade-off between worst-case loss
                    and vanilla cross-entropy loss.

                - n_epochs: int
                    Number of epochs used in training.

                - lr: float
                    Learning rate.

                - path_to_folder: str
                    Path, where the best model will be saved.

                - device: str
                    Specifies the device on which calculations will be performed.

                - batch_size: int
                    Number of samples per batch.

    Returns:
    --------
        The trained model.
    """

    perturbation_epsilon = params["epsilon"]
    kappa_max            = params["kappa"]
    n_epochs             = params["epochs"]
    lr                   = params["lr"]
    path_to_folder       = params["save_path"]
    device               = params["device"]
    batch_size           = params["batch_size"]

    os.makedirs(path_to_folder, exist_ok=True)
    n_epochs_to_adjust_eps = n_epochs // 2

    # Send the model to the GPU
    net = net.to(device)

    # Create optimizer
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    no_iterations = int(len(train_dl.dataset) * batch_size * n_epochs)

    # Create scheduler
    steps = [
        no_iterations // 4,
        25 * no_iterations // 60
    ]
    scheduler = MultiStepLR(opt, milestones=steps, gamma=0.1)

    # Create cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # Initialize empty losses over epochs
    train_fit_loss_list        = []
    train_worst_case_loss_list = []
    train_total_loss_list      = []
    train_acc_list             = []

    val_fit_loss_list        = []
    val_worst_case_loss_list = []
    val_total_loss_list      = []
    val_acc_list             = []

    best_val_acc = 0.0

    for epoch_i in range(n_epochs):

        # Initialize empty losses over batches
        batch_train_fit_loss_list        = []
        batch_train_worst_case_loss_list = []
        batch_train_total_loss_list      = []
        batch_train_acc_list             = []

        batch_val_fit_loss_list        = []
        batch_val_worst_case_loss_list = []
        batch_val_total_loss_list      = []
        batch_val_acc_list             = []

        net.train()

        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            if epoch_i < n_epochs_to_adjust_eps:
                eps_temp = (epoch_i/(n_epochs_to_adjust_eps-1))*perturbation_epsilon
                eps = eps_temp * torch.ones_like(x_batch)
                kappa = max(1 - 0.0005*epoch_i, kappa_max)
            else:
                kappa = kappa_max
                eps   = perturbation_epsilon * torch.ones_like(x_batch)

            # Send eps to the device
            eps = eps.to(device)

            ### Forward pass ###
            z_l, z_u, mu_pred, _ = net(x_batch, eps, use_softmax=False)

            ### Loss calculations ###
            mu_pred = mu_pred.to(device)
            z_l = z_l.to(device)
            z_u = z_u.to(device)
            
            loss_fit = criterion(mu_pred, y_batch)

            tmp = nn.functional.one_hot(y_batch, mu_pred.size(-1))
            z = torch.where(tmp.bool(), z_l, z_u)

            loss_spec = criterion(z, y_batch)
            loss = kappa * loss_fit + (1-kappa) * loss_spec

            is_correct = (torch.argmax(mu_pred, dim=1) == y_batch).float().cpu().numpy()

            # Get errors
            batch_train_fit_loss_list.append(loss_fit.cpu().item())
            batch_train_worst_case_loss_list.append(loss_spec.cpu().item())
            batch_train_total_loss_list.append(loss.cpu().item())
            batch_train_acc_list.append(100 * np.mean(is_correct))

            # Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Average train losses
        train_fit_loss_list.append(np.mean(batch_train_fit_loss_list))
        train_worst_case_loss_list.append(np.mean(batch_train_worst_case_loss_list))
        train_total_loss_list.append(np.mean(batch_train_total_loss_list))
        train_acc_list.append(np.mean(batch_train_acc_list))

        # Validation
        net.eval()

        with torch.no_grad():
            for x_batch_val, y_batch_val in val_dl:
                x_batch_val = x_batch_val.to(device)
                y_batch_val = y_batch_val.to(device)

                if epoch_i < n_epochs_to_adjust_eps:
                    eps_temp = (epoch_i/(n_epochs_to_adjust_eps-1))*perturbation_epsilon
                    eps      = eps_temp * torch.ones_like(x_batch_val)
                    kappa    = max(1 - 0.0005*epoch_i, kappa_max)
                else:
                    kappa = kappa_max
                    eps   = perturbation_epsilon * torch.ones_like(x_batch_val)

                # Send eps to the GPU
                eps = eps.to(device)

                ### Forward pass ###
                z_l, z_u, mu_pred, _ = net(x_batch_val, eps, use_softmax=False)

                ### Loss calculations ###
                mu_pred = mu_pred.to(device)
                z_l = z_l.to(device)
                z_u = z_u.to(device)
                
                loss_fit = criterion(mu_pred, y_batch_val)

                tmp = nn.functional.one_hot(y_batch_val, mu_pred.size(-1))
                z = torch.where(tmp.bool(), z_l, z_u)

                loss_spec = criterion(z, y_batch_val)
                loss = kappa * loss_fit + (1-kappa) * loss_spec

                is_correct = (torch.argmax(mu_pred, dim=1) == y_batch_val).float().cpu().numpy()

                # Get errors
                batch_val_fit_loss_list.append(loss_fit.cpu().item())
                batch_val_worst_case_loss_list.append(loss_spec.cpu().item())
                batch_val_total_loss_list.append(loss.cpu().item())
                batch_val_acc_list.append(100 * np.mean(is_correct))

        # Average val losses
        val_fit_loss_list.append(np.mean(batch_val_fit_loss_list))
        val_worst_case_loss_list.append(np.mean(batch_val_worst_case_loss_list))
        val_total_loss_list.append(np.mean(batch_val_total_loss_list))
        val_acc_list.append(np.mean(batch_val_acc_list))

        print(f"Epoch: {epoch_i}, "
                f"Total val loss: {val_total_loss_list[-1]:.4f}, "
                f"worst case loss: {val_worst_case_loss_list[-1]:.4f}, "
                f"accuracy: {val_acc_list[-1]:.4f}")

        # Save the best model based on the val set
        if val_acc_list[-1] > best_val_acc and kappa == kappa_max:
            best_model    = deepcopy(net)
            best_val_acc = val_acc_list[-1]

        # Write train and val losses to csv file
        train_dataframe = {
            "fit_loss": train_fit_loss_list,
            "worst_case_loss": train_worst_case_loss_list,
            "total_loss": train_total_loss_list
        }
        train_dataframe = pd.DataFrame.from_dict(train_dataframe)
        train_dataframe.to_csv(f'{path_to_folder}/train_loss.csv', index=False)

        val_dataframe = {
            "fit_loss": val_fit_loss_list,
            "worst_case_loss": val_worst_case_loss_list,
            "total_loss": val_total_loss_list
        }
        val_dataframe = pd.DataFrame(val_dataframe)
        val_dataframe.to_csv(f'{path_to_folder}/val_loss.csv', index=False)

        # Scheduler step
        scheduler.step()

    # Save weights of the best model
    torch.save(best_model.state_dict(), f'{path_to_folder}/best_weights.pth')

    return best_model


if __name__ == '__main__':
    main()