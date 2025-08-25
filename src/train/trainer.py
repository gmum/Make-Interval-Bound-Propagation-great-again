import torch
import torch.nn.functional as F
import logging
from omegaconf import DictConfig
from typing import Tuple

from method.interval_arithmetic import Interval
from method.method_plugin_abc import MethodPluginABC
from experiment.utils import get_eps

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Trainer:
    """
    Trainer class responsible for scheduling epsilon and kappa during training,
    and computing loss for robust training using interval bounds.

    Epsilon is linearly increased and kappa linearly decreased per batch
    after the warmup period, over a fixed number of scheduling epochs.
    """

    def __init__(
        self,
        method: MethodPluginABC,
        eps_train: float,
        warmup_epochs: int = 0,
        schedule_epochs: int = 10,
        num_batches_per_epoch: int = 100,
        kappa_end: float = 0.5
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            method (MethodPluginABC): The training method (plugin) with an interval bound forward pass.
            eps_train (float): The maximum epsilon value for training (non-scheduled).
            warmup_epochs (int): Number of warmup epochs.
            schedule_epochs (int): Number of epochs to linearly schedule epsilon and kappa after warmup.
            num_batches_per_epoch (int): Number of batches per training epoch.
            kappa_end (float): The final value of kappa after the scheduling period.
        """
        super().__init__()
        self.method = method.plugins[0]
        self.epsilon_train = eps_train
        self.kappa_end = kappa_end

        self.current_epsilon = 0.0
        self.current_kappa = 1.0

        self.warmup_epochs = warmup_epochs
        self.schedule_epochs = schedule_epochs
        self.num_batches_per_epoch = num_batches_per_epoch

        self.schedule_total_steps = schedule_epochs * num_batches_per_epoch
        self.schedule_step = 0

        log.info(f"Trainer initialized with epsilon = {self.epsilon_train}, "
                 f"schedule steps = {self.schedule_total_steps}")

    def update_schedule(self, epoch: int) -> None:
        """
        Updates epsilon and kappa based on the current epoch and batch progress.

        Args:
            epoch (int): Sets the current epoch.
        """
        if epoch < self.warmup_epochs:
            self.current_epsilon = 0.0
            self.current_kappa = 1.0
        elif self.schedule_total_steps == 0:
            self.current_epsilon = self.epsilon_train
            self.current_kappa = self.kappa_end
        else:
            progress = min(1.0, self.schedule_step / self.schedule_total_steps)
            self.current_epsilon = progress * self.epsilon_train
            self.current_kappa = max(1.0 - progress, self.kappa_end)
            self.schedule_step += 1

        # log.info(f"[Epoch {epoch}]: epsilon = {self.current_epsilon:.5f}, "
        #         f"kappa = {self.current_kappa:.5f}")


    def forward(self, x: torch.Tensor, y: torch.Tensor, config: DictConfig) -> Tuple[torch.Tensor, Interval]:
        """
        Performs a forward pass using the current epsilon, computes robust interval bounds,
        and returns the loss and bounds.

        Args:
            x (torch.Tensor): Input batch.
            y (torch.Tensor): Ground truth labels.
            config (DictConfig): The configuration object containing dataset information.
                It is expected to have a nested structure where `config.dataset.dataset.std`
                holds the standard deviation values as a list or tensor.

        Returns:
            Tuple[torch.Tensor, Interval]: Loss and interval bounds for the batch.
        """
        if self.method.module.training:
            eps_factor = self.current_epsilon
        else:
            eps_factor = config.training.eps_test
        eps = eps_factor * get_eps(config, shape=x.shape, device=x.device)
        out_bounds = self.method.forward(x, y, eps)
        logits = self.method.module(x)
        loss = self.calculate_loss(logits, out_bounds, y)
        return loss, out_bounds

    def calculate_loss(self, logits: torch.Tensor, bounds: Interval, y: torch.Tensor) -> torch.Tensor:
        """
        Mixed natural + robust cross-entropy loss using a CROWN-IBP-style
        margin bound, computed without constructing the specification matrix.

        Args:
            logits (torch.Tensor): Model outputs (natural forward pass), shape [B, C].
            bounds (Interval): Interval bounds of the outputs with .lower and .upper, each [B, C].
            y (torch.Tensor): Ground truth labels, shape [B].

        Returns:
            torch.Tensor: Combined loss.
        """
        # ===== Natural loss =====
        natural_loss = F.cross_entropy(logits, y)

        lower, upper = bounds.lower, bounds.upper  # [B, C]
        B, C = lower.shape
        device = logits.device

        # ===== Robust margin lower bounds for all j != y =====
        # For each sample: margin_lower_j = lower_y - upper_j
        arange = torch.arange(B, device=device)
        lower_y = lower[arange, y]                       # [B]
        margin_left = lower_y.unsqueeze(1)               # [B, 1]

        # Collect upper bounds of all *other* classes
        mask = torch.ones_like(upper, dtype=torch.bool)
        mask[arange, y] = False
        upper_others = upper[mask].view(B, C - 1)        # [B, C-1]

        margin_lower = margin_left - upper_others        # [B, C-1]

        # ===== Robust loss (CE over logits [0, -margin_lower]) =====
        # CE(z, y) = logsumexp(0, -m_j) with target=0
        robust_logits = torch.cat([torch.zeros(B, 1, device=device), -margin_lower], dim=1)  # [B, C]
        robust_targets = torch.zeros(B, dtype=torch.long, device=device)
        robust_loss = F.cross_entropy(robust_logits, robust_targets)

        # ===== Mix losses =====
        loss = self.current_kappa * natural_loss + (1.0 - self.current_kappa) * robust_loss
        return loss