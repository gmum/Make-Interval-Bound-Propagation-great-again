import math
from enum import Enum
from typing import Sequence, Optional
import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms


class DatasetType(Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"


class DatasetFactory:
    @staticmethod
    def get_dataset(
        dataset_type: str,
        data_path: str = "./data",
        flatten: bool = False,
        split_ratio: Optional[Sequence[float]] = (0.8, 0.2),
        seed: int = 0,
        download: bool = True,
    ) -> tuple[Dataset, Dataset, Dataset]:
        """
        Returns (train, val, test) datasets for the given dataset type.
        """

        dataset_type = DatasetType(dataset_type.lower())

        if dataset_type == DatasetType.MNIST:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)) if flatten else (lambda x: x),
            ])
            train_full = datasets.MNIST(root=data_path, train=True, transform=transform, download=download)
            test_set = datasets.MNIST(root=data_path, train=False, transform=transform, download=download)

        elif dataset_type == DatasetType.CIFAR10:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train_full = datasets.CIFAR10(root=data_path, train=True, transform=transform, download=download)
            test_set = datasets.CIFAR10(root=data_path, train=False, transform=transform, download=download)

        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        if split_ratio is not None:
            if len(split_ratio) != 2:
                raise ValueError("split_ratio must have exactly two values: [train_ratio, val_ratio]")
            total = len(train_full)
            train_size = int(math.floor(split_ratio[0] * total))
            val_size = total - train_size
            g = torch.Generator().manual_seed(seed)
            train_set, val_set = random_split(train_full, [train_size, val_size], generator=g)
        else:
            train_set, val_set = train_full, None

        return train_set, val_set, test_set
