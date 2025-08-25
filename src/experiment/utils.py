import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import uuid
from typing import Tuple, Optional

from method.interval_arithmetic import Interval
from network.network_abc import NetworkABC

from omegaconf import DictConfig
from hydra.utils import instantiate


def squeeze_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Squeeze batch dimension if batch size is 1, else return unchanged."""
    return tensor.squeeze(0) if tensor.size(0) == 1 else tensor

def get_dataloader(config: DictConfig, fabric) -> DataLoader:
    """
    Initializes and returns a dataloader using the provided configuration and fabric.
    Args:
        config (dict): A configuration object containing the dataset settings.
        fabric (object): An object responsible for setting up dataloaders.
    Returns:
        DataLoader: A dataloader instance prepared using the specified configuration and fabric.
    """
    return fabric.setup_dataloaders(instantiate(config.dataset))

def verify_point(output_bounds: Interval, y_pred: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
    """
    Verifies robust classification only for correctly predicted samples.

    Args:
        output_bounds (Interval): Contains `.lower` and `.upper` bounds (B x C).
        y_pred (torch.Tensor): Predicted labels (B,).
        y_gt (torch.Tensor): Ground truth labels (B,).

    Returns:
        torch.Tensor: Tensor of shape (N,), where N is the number of correct predictions.
                      Values are 1.0 if verified, 0.0 if not.
    """
    with torch.no_grad():
        lb = output_bounds.lower 
        ub = output_bounds.upper

        correct_mask = (y_pred == y_gt)
        if correct_mask.sum() == 0:
            return []

        y_pred = y_pred[correct_mask]
        y_gt = y_gt[correct_mask]
        lb = lb[correct_mask]
        ub = ub[correct_mask]

        batch_size = lb.size(0)
        idx = torch.arange(batch_size, device=lb.device)

        gt_lb = lb[idx, y_gt]

        # Mask out the correct class
        non_gt_mask = torch.ones_like(ub, dtype=torch.bool)
        non_gt_mask[idx, y_gt] = False
        ub_non_gt = ub[non_gt_mask].view(batch_size, -1)

        # Verified if ground truth lower bound is greater than all upper bounds of other classes
        is_verified = (gt_lb.unsqueeze(1) > ub_non_gt).all(dim=1)
       
        return is_verified.float()



    
def check_correct_prediction(module: NetworkABC, x: torch.Tensor, y_gt: torch.Tensor) -> Tuple[bool,torch.Tensor]:
    """
    Checks whether a given neural network module correctly classifies an input tensor.
    Args:
        module (NetworkABC): The neural network module to evaluate.
        x (torch.Tensor): The input tensor to be classified by the module.
        y_gt (torch.Tensor): The ground truth tensor representing the correct classification.
    Returns: Tuple
        bool: True if the module's prediction matches the ground truth, False otherwise.
        torch.tensor: 
    """
    y_pred = module(x)
    y_pred = torch.argmax(y_pred, dim=-1)
    correctly_classified = y_pred == y_gt
    
    return correctly_classified, y_pred

def save_deteriotated_image(x: torch.Tensor, flatten: bool, folder: str, 
                            dataset: str, path_suffix: str = None) -> None:
    """
    Save a deteriorated image from a tensor to disk.

    Args:
        x (torch.Tensor): Image tensor (either flattened or in CHW format).
        flatten (bool): Whether the tensor is flattened (e.g., from a dataset like MNIST).
        folder (str): Path to the directory where the deteriorated image will be saved.
        dataset (str): Name of the dataset (needed if flatten=True).
        path_suffix (Optional, str): Name suffix.
    """
    x = x.clone().detach()
    x = squeeze_batch_dim(x)

    if flatten:
        if "MNIST" in dataset:
            x = x.view(1, 28, 28)
        elif "CIFAR10" in dataset or "SVHN" in dataset:
            x = x.view(3, 32, 32)
        else:
            raise ValueError(f"Flattened image saving not implemented for dataset: {dataset}")

    if x.max() <= 1.0:
        x = x * 255.0
    x = x.to(torch.uint8)

    to_pil = transforms.ToPILImage()
    img = to_pil(x)

    os.makedirs(f"{folder}/deteriorated_images", exist_ok=True)
    if path_suffix is not None:
        img_path = f"{folder}/deteriorated_images/image_{uuid.uuid4().hex[:8]}{path_suffix}.png"
    else:
        img_path = f"{folder}/deteriorated_images/image_{uuid.uuid4().hex[:8]}.png"
    img.save(img_path)

def get_eps(config: DictConfig, shape: Tuple[int,...], device: torch.device) -> torch.Tensor:
    """
    Adjusts the epsilon tensor by dividing it by the dataset's standard deviation
    if the standard deviation is defined in the dataset configuration.

    Args:
        config (DictConfig): The configuration object containing dataset information.
            It is expected to have a nested structure where `config.dataset.dataset.std`
            holds the standard deviation values as a list or tensor.
        shape (Tuple[int,...]): The shape of the epsilon tensor to be created.
        device (torch.device): The device on which the tensor should be created.

    Returns:
        torch.Tensor: The adjusted epsilon tensor. If the dataset's standard deviation
            is defined, the epsilon tensor is divided by it. Otherwise, the original
            epsilon tensor is returned unchanged.
    """
    eps = torch.ones(shape, device=device)
    if hasattr(config.dataset.dataset, "std"):
        std = config.dataset.dataset.std
        std = torch.tensor(std, device=eps.device).view(1, 3, 1, 1)
        eps = eps / std
    return eps

@torch.no_grad()
def margin_lowers_from_bounds(bounds: Interval, y: torch.Tensor) -> torch.Tensor:
    """
    Compute robust lower bounds on margins z_y - z_j for all j != y
    from per-logit lower/upper bounds.

    Args:
        bounds (Interval): Interval with .lower and .upper tensors, shape [B, C]
        y (torch.Tensor): labels, shape [B]

    Returns:
        margin_lower (torch.Tensor): [B, C-1], lower bounds on (z_y - z_j) for all j != y
    """
    lower, upper = bounds.lower, bounds.upper  # [B, C]
    B, C = lower.shape
    arange = torch.arange(B, device=lower.device)

    lower_y = lower[arange, y].unsqueeze(1)       # [B, 1]
    mask = torch.ones_like(upper, dtype=torch.bool)
    mask[arange, y] = False
    upper_others = upper[mask].view(B, C - 1)     # [B, C-1]

    # (z_y - z_j)_lower = lower_y - upper_j
    margin_lower = lower_y - upper_others         # [B, C-1]
    return margin_lower

@torch.no_grad()
def compute_verified_error(bounds: Interval, y: torch.Tensor) -> float:
    """
    Verified error: percentage of samples where any margin lower bound < 0.
    """
    m_lower = margin_lowers_from_bounds(bounds, y)
    violations = (m_lower < 0).any(dim=1).float()
    return 100.0 * violations.mean().item()

def pgd_linf_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int = 200,
    alpha: Optional[float] = None,
    clip: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    Untargeted PGD under L_infty. Returns adversarial examples.

    Args:
        model (nn.Module): classifier producing logits.
        x (torch.Tensor): inputs, shape [B, ...], assumed in [clip_min, clip_max]
        y (torch.Tensor): labels, shape [B]
        eps (float): L_infty radius
        steps (int): number of PGD steps (default 200)
        alpha (float): step size; default 2*eps/steps
        clip (Tuple[float, float]): (min, max) clamp range

    Returns:
        adv_x (torch.Tensor): adversarial examples
    """
    model.eval()
    clip_min, clip_max = clip
    if alpha is None:
        alpha = 2.0 * eps / steps

    # Random start in the Linf ball
    delta = torch.empty_like(x).uniform_(-eps, eps)
    adv_x = (x + delta).clamp(clip_min, clip_max).detach().requires_grad_(True)

    for _ in range(steps):
        logits = model(adv_x)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, adv_x, retain_graph=False, create_graph=False)[0]
        adv_x = adv_x.detach() + alpha * torch.sign(grad.detach())
        # Project to Linf ball around x
        adv_x = torch.max(torch.min(adv_x, x + eps), x - eps)
        adv_x = adv_x.clamp(clip_min, clip_max).detach().requires_grad_(True)

    return adv_x.detach()