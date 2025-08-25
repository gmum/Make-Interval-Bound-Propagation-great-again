from enum import Enum
import torch
import torch.nn as nn
import logging

from network.network_abc import NetworkABC

log = logging.getLogger(__name__)

from .sdp_models_archs import (
    cnn_4layer,
    cnn_4layer_adv,
    cnn_4layer_adv4,
    cnn_4layer_mix4,
    mnist_cnn_4layer
)


class ModelArch(Enum):
    CIFAR_CNN_A_ADV = "cifar_cnn_a_adv"
    CIFAR_CNN_A_ADV4 = "cifar_cnn_a_adv4"
    CIFAR_CNN_A_MIX = "cifar_cnn_a_mix"
    CIFAR_CNN_A_MIX4 = "cifar_cnn_a_mix4"
    MNIST_CNN_A_ADV = "mnist_cnn_a_adv"


# Mapping from architecture enum to model constructor
ARCH_CONSTRUCTORS = {
    ModelArch.CIFAR_CNN_A_ADV: cnn_4layer_adv,
    ModelArch.CIFAR_CNN_A_ADV4: cnn_4layer_adv4,
    ModelArch.CIFAR_CNN_A_MIX: cnn_4layer,
    ModelArch.CIFAR_CNN_A_MIX4: cnn_4layer_mix4,
    ModelArch.MNIST_CNN_A_ADV: mnist_cnn_4layer,
}


class SDPModelLoader(NetworkABC):
    """
    Loads and introspects an SDP model from file.

    Attributes:
        model (NetworkABC): The loaded model.
        layer_outputs (dict): Dictionary mapping layers to their output shapes (excluding batch dim).
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): Path to the .model file.
        """
        super().__init__()

        _arch_name = model_path.split("/")[-1].split(".")[0]

        try:
            arch_enum = ModelArch(_arch_name)
            arch_constructor = ARCH_CONSTRUCTORS[arch_enum]
        except (ValueError, KeyError):
            raise ValueError(f"Unknown model architecture: {_arch_name}")

        self.model = arch_constructor()

        try:
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            log.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            log.error(f"Failed to load model from {model_path}: {e}")
            raise RuntimeError(f"Could not load model: {e}")

        if not isinstance(self.model, nn.Module):
            raise TypeError("Loaded object is not a PyTorch nn.Module.")

        # Auto-detect dummy input shape
        model_path_lower = model_path.lower()
        if "mnist" in model_path_lower:
            self.input_shape = (1, 1, 28, 28)
        elif "cifar" in model_path_lower:
            self.input_shape = (1, 3, 32, 32)
        else:
            self.input_shape = (1, 3, 32, 32)
            log.warning(f"Could not infer input shape from model path: {model_path}. Using default CIFAR input.")

        self.layer_outputs = {}

        self.model.apply(self.register_hooks)

        with torch.no_grad():
            dummy_input = torch.zeros(self.input_shape)
            self.model(dummy_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
