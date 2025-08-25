import numpy as np
import torch
from torch.utils.data import Dataset

class MNIST_SDP(Dataset):
    """
    Custom Dataset for loading MNIST data from .npy files.
    
    Args:
        data_path (str): Path to the .npy file containing image data.
    """
    def __init__(self, data_path: str):
        self.images = np.load(f"{data_path}/X_sdp.npy")
        self.labels = np.load(f"{data_path}/y_sdp.npy")

        assert len(self.images) == len(self.labels), "Image and label counts do not match."

        self.images = self.images.transpose(0, 3, 1, 2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = torch.tensor(image, dtype=torch.float32)

        return image, int(label)
