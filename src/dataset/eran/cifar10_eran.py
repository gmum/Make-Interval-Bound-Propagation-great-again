from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

class CIFAR_ERAN(Dataset):
    """
    CIFAR10 Dataset Wrapper
    This class provides a wrapper around the CIFAR-10 dataset, allowing for easy access
    to either training or testing data. It supports indexing and provides the total length
    of the dataset.
    Attributes:
        data (torch.utils.data.Dataset): The CIFAR-10 dataset (either training or testing).
        no_points (int or None): Optional limit on the number of data points used.
    Methods:
        __init__(data_path: str = './data', train: bool = False, flatten: bool = True, no_points=None):
            Initializes the CIFAR10 dataset wrapper by downloading and preparing
            either the training or testing dataset based on the `train` flag.
        __len__():
            Returns the total number of samples in the dataset.
        __getitem__(idx: int):
            Retrieves a single sample from the dataset based on the provided index.
    Args:
        data_path (str): The path where the CIFAR-10 dataset will be downloaded and stored.
        train (bool): A flag indicating whether to load the training dataset (True) or testing (False).
        flatten (bool): A flag to flatten images to 1D tensors.
        no_points (int or list): Optional number of data points to use.
    """

    def __init__(self, data_path: str = './data', train: bool = False, flatten: bool = True, no_points: int = None,
                 **kwargs):
        if isinstance(no_points, (list, tuple)):
            no_points = no_points[0]

        mean = kwargs["mean"]
        std = kwargs["std"]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)) if flatten else lambda x: x,
            transforms.Normalize(mean=mean, std=std)
        ])

        full_data = datasets.CIFAR10(root=data_path, train=train, download=True, transform=transform)

        self.data = full_data if no_points is None else Subset(full_data, range(no_points))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
