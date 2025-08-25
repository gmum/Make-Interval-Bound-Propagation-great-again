import logging
import numpy as np
import time
import torch
import wandb
import json
import os

from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
from typing import Dict

from utils.fabric import setup_fabric
from utils.hydra import extract_output_dir
from experiment.utils import compute_verified_error
from dataset.dataset_factory import DatasetFactory
from method.method_plugin_abc import MethodPluginABC
from train import Trainer

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def evaluate_split(
    split_name: str,
    dataloader: DataLoader,
    fabric: Fabric,
    trainer: Trainer,
    method: MethodPluginABC,
    config: DictConfig
) -> Dict[str, float]:
    """
    Evaluate model performance on a given dataset split, logging both
    standard metrics and robust verification metrics to W&B.
    """
    log.info(f"Evaluating on {split_name} set")

    method.module.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_verified_error: list[float] = []

    all_bounds = []

    for _, (X, y) in enumerate(tqdm(dataloader)):
        X = fabric.to_device(X)
        y = fabric.to_device(y)

        # --- No gradients needed for these parts ---
        with torch.no_grad():
            # Forward pass to get natural loss and bounds
            loss, bounds = trainer.forward(X, y, config)
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

            # Clean predictions
            y_pred_clean = torch.argmax(method.module(X), dim=1)
            correct = (y_pred_clean == y).sum().item()
            total_correct += correct

            # Robustness verification
            verified_err_batch = compute_verified_error(bounds, y)
            verified_err_val = float(verified_err_batch.item() if hasattr(verified_err_batch, "item") else verified_err_batch)
            total_verified_error.append(verified_err_val)

            # Track bound widths
            bound_width = (bounds.upper - bounds.lower)
            all_bounds.append(bound_width.detach().cpu())

    # Aggregate results
    avg_loss = total_loss / max(total_samples, 1)
    clean_error = 100.0 * (1.0 - (total_correct / max(total_samples, 1)))
    verified_error = float(np.mean(total_verified_error)) if total_verified_error else 0.0

    all_bounds_tensor = torch.cat(all_bounds) if len(all_bounds) > 0 else torch.tensor([])
    if all_bounds_tensor.numel() > 0:
        flat_bounds = all_bounds_tensor.flatten().numpy()
        avg_bound = all_bounds_tensor.mean().item()
        max_bound = all_bounds_tensor.max().item()
        min_bound = all_bounds_tensor.min().item()
    else:
        flat_bounds = np.array([])
        avg_bound = max_bound = min_bound = 0.0

    # Log to W&B (ensure JSON/serializable-friendly values)
    wandb.log({
        f"{split_name}/avg_loss": float(avg_loss),
        f"{split_name}/clean_error": float(clean_error),
        f"{split_name}/verified_error": float(verified_error),
        f"{split_name}/avg_bound_width": float(avg_bound),
        f"{split_name}/max_bound_width": float(max_bound),
        f"{split_name}/min_bound_width": float(min_bound),
        f"{split_name}/bound_width_hist": wandb.Histogram(flat_bounds),
    })

    # Console logging
    print(
        f"[{split_name.upper()}] Avg Loss: {avg_loss:.6f}, "
        f"Clean Err: {clean_error:.2f}%, "
        f"Verified Err: {verified_error:.2f}%, "
        f"Avg Bound Width: {avg_bound:.6f}, "
        f"Max Bound Width: {max_bound:.6f}, "
        f"Min Bound Width: {min_bound:.6f}"
    )

    return {
        "avg_loss": float(avg_loss),
        "clean_error": float(clean_error),
        "verified_error": float(verified_error),
        "avg_bound_width": float(avg_bound),
        "max_bound_width": float(max_bound),
        "min_bound_width": float(min_bound),
    }

def run(config: DictConfig) -> None:
    """
    Main training and evaluation loop with metric logging.
    """
    wandb.init(project=os.environ['WANDB_PROJECT'], config=dict(config))

    log.info(f'Launching Fabric')
    fabric = setup_fabric(config)

    log.info(f'Building model')
    model = fabric.setup(instantiate(config.network))

    log.info(f'Setting up training method')
    method = instantiate(config.method)(model)

    log.info(f'Setting up dataloaders')

    train_dataset, val_dataset, test_dataset = DatasetFactory.get_dataset(
        dataset_type=config.dataset.dataset.type,
        data_path=config.dataset.dataset.data_path,
        flatten=config.dataset.dataset.flatten,
        split_ratio=config.dataset.dataset.split_ratio,
        seed=config.dataset.dataset.seed,
        download=config.dataset.dataset.download
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.train_batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.val_batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataset.test_batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
    )

    train_loader, val_loader, test_loader = fabric.setup_dataloaders(
        train_loader, val_loader, test_loader
    )

    extracted_output_dir = extract_output_dir(config)
    output_file = f"{extracted_output_dir}/training_log.json"

    log.info(f'Initializing the trainer')
    trainer = Trainer(
        method=method,
        eps_train=config.training.eps_train,
        warmup_epochs=config.training.warmup_epochs,
        schedule_epochs=config.training.schedule_epochs_after_warmup,
        num_batches_per_epoch=len(train_loader)
    )

    optimizer = torch.optim.Adam(trainer.method.module.parameters(), lr=config.training.lr)

    epochs = config.training.epochs
    use_scheduler = config.training.get("use_scheduler", False)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    epoch_logs = []
    best_val_loss = float("inf")
    processing_times = []
    checkpoint_path = os.path.join(extracted_output_dir, "best_model.pth")

    n_steps = config.training.get("n_steps", 1)

    for epoch in range(epochs):
        epoch_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_verified_error = []
        batch_processing_times = []

        log.info(f"Epoch {epoch+1}/{epochs} — epsilon: {trainer.current_epsilon:.5f}, kappa: {trainer.current_kappa:.5f}")

        trainer.method.module.train()
        for batch_idx, (X, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} — Training")):
            trainer.update_schedule(epoch)

            X = fabric.to_device(X)
            y = fabric.to_device(y)

            start_time = time.time()

            loss, bounds = trainer.forward(X, y, config)
            loss = loss / n_steps
            loss.backward()

            if (batch_idx + 1) % n_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            elapsed_time = time.time() - start_time
            batch_processing_times.append(elapsed_time)

            # Clean accuracy
            y_pred = torch.argmax(trainer.method.module(X), dim=1)
            correct = (y_pred == y).sum().item()
            total_correct += correct
            batch_accuracy = correct / X.size(0)

            # Verified error (batch)
            verified_err_batch = compute_verified_error(bounds, y)
            total_verified_error.append(verified_err_batch)

            epoch_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

            # Bound widths
            bound_width = (bounds.upper - bounds.lower)
            avg_bound_width = bound_width.mean().item()
            max_bound_width = bound_width.max().item()
            min_bound_width = bound_width.min().item()
            flat_bound_width = bound_width.detach().cpu().flatten().numpy()

            # Batch logs
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/epsilon": trainer.current_epsilon,
                "train/kappa": trainer.current_kappa,
                "train/epoch": epoch,
                "train/batch_clean_error": 100.0 * (1.0 - batch_accuracy),
                "train/batch_verified_error": verified_err_batch,
                "train/batch_bound_avg": avg_bound_width,
                "train/batch_bound_max": max_bound_width,
                "train/batch_bound_min": min_bound_width,
                "train/bound_width_hist": wandb.Histogram(flat_bound_width)
            })

            torch.cuda.empty_cache()

        # Epoch aggregates
        avg_train_loss = epoch_loss / total_samples
        train_clean_error = 100.0 * (1.0 - (total_correct / total_samples))
        train_verified_error = np.mean(total_verified_error)
        train_processing_time = np.sum(batch_processing_times)
        processing_times.append(train_processing_time)

        if use_scheduler:
            scheduler.step(avg_train_loss)

        # Validation
        val_stats = evaluate_split("val", val_loader, fabric, trainer, method, config)

        # Save best model
        if val_stats["avg_loss"] < best_val_loss:
            best_val_loss = val_stats["avg_loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": trainer.method.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "epsilon": trainer.current_epsilon,
                "kappa": trainer.current_kappa
            }, checkpoint_path)
            log.info(f"Saved new best model at epoch {epoch+1} with val loss {best_val_loss:.6f}")
            print(f"Saved new best model at epoch {epoch+1} with val loss {best_val_loss:.6f}")

        # Epoch logs
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "train/epoch_clean_error": train_clean_error,
            "train/epoch_verified_error": train_verified_error,
            "train/epoch": epoch,
            "train/epoch_elapsed_time": train_processing_time,
        })

        epoch_logs.append({
            "epoch": epoch,
            "train_processing_time": train_processing_time,
            "train_loss": avg_train_loss,
            "train_clean_error": train_clean_error,
            "train_verified_error": train_verified_error,
            "train_epsilon": trainer.current_epsilon,
            "train_kappa": trainer.current_kappa,
            "val_loss": val_stats["avg_loss"],
            "val_clean_error": val_stats["clean_error"],
            "val_verified_error": val_stats["verified_error"],
            "val_avg_bound_width": val_stats["avg_bound_width"],
            "val_max_bound_width": val_stats["max_bound_width"],
            "val_min_bound_width": val_stats["min_bound_width"],
        })

        torch.cuda.empty_cache()

    # Test evaluation
    test_stats = evaluate_split("test", test_loader, fabric, trainer, method, config)

    total_time = np.sum(processing_times)
    wandb.log({
        "total_time": total_time,
    })

    output = {
        "training_log": epoch_logs,
        "final_test_stats": test_stats,
        "total_time": total_time
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)

    wandb.finish()
    log.info(f"Training complete. Logs saved to {output_file}")
    print(f"Training complete. Logs saved to {output_file}")
