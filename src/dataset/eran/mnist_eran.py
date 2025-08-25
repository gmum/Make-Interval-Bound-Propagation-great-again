from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

class MNIST_ERAN(Dataset):
    """
    MNIST Dataset Wrapper
    This class provides a wrapper around the MNIST dataset, allowing for easy access
    to either training or testing data. It supports indexing and provides the total length
    of the dataset.
    Attributes:
        data (torch.utils.data.Dataset): The MNIST dataset (either training or testing).
        no_points (int or None): Optional limit on the number of data points used.
    Methods:
        __init__(data_path: str = './data', train: bool = False, flatten: bool = True, no_points=None):
            Initializes the MNIST dataset wrapper by downloading and preparing
            either the training or testing dataset based on the `train` flag.
        __len__():
            Returns the total number of samples in the dataset.
        __getitem__(idx: int):
            Retrieves a single sample from the dataset based on the provided index.
    """
    def __init__(self, data_path: str = './data', train: bool = False, flatten: bool = True, no_points=None):
        if isinstance(no_points, (list, tuple)):
            no_points = no_points[0]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)) if flatten else lambda x: x
        ])

        full_data = datasets.MNIST(root=data_path, train=train, download=True, transform=transform)
        self.data = full_data if no_points is None else Subset(full_data, range(no_points))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
