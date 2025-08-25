import torch

from dataset.dataset_factory import DatasetType

class DatasetFactory:
    @staticmethod
    def get_scheduler(
        optimizer: torch.optim.Optimizer,
        dataset_type: str
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns the appropriate LR scheduler instance for the dataset.

        Args:
            optimizer (Optimizer): The optimizer
            dataset_type (str): Dataset name ("mnist", "cifar10", etc.)

        Returns:
            torch.optim.lr_scheduler._LRScheduler
        """

        dataset_type = DatasetType(dataset_type.lower())

        if dataset_type == DatasetType.MNIST:
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(130, 190), gamma=0.1)
        else:
            raise ValueError(f"Unsupported dataset type for scheduler: {dataset_type}")

