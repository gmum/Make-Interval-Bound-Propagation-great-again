import torch.nn as nn
import torch

def load_linear_model(model: nn.Module, weight_path: str, device: str = "cpu") -> None:
    """
    Loads the weights of an MLP model from a specified file path.
    Args:
        model (NetworkABC): The PyTorch model instance to load the weights into.
        weight_path (str): The file path to the saved model weights.
        device (str, optional): The device to map the model weights to. Defaults to "cpu".
    Notes:
        - The function attempts to extract the "state_dict" key from the loaded weights.
        - If the state dictionary is a list, it assumes a specific naming convention for keys
          and adjusts them accordingly before loading into the model.
        - The `strict` parameter in `load_state_dict` is set to `False`, allowing for partial
          loading of weights if the model architecture does not match exactly.
    Returns:
        None
    """
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = state_dict.get("state_dict", state_dict)

    if isinstance(state_dict, list):
        state_dict = state_dict[0]
    state_dict = {f"{int(key.split('.')[0])-1}.{key.split('.')[1]}": val for key, val in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def load_conv_model(model: nn.Module, weight_path: str, device: str = "cpu") -> None:
    """
    Loads the weights of a CNN model from a specified file path.
    Args:
        model (NetworkABC): The PyTorch model instance to load the weights into.
        weight_path (str): The file path to the saved model weights.
        device (str, optional): The device to map the model weights to. Defaults to "cpu".
    Notes:
        - Attempts to extract the "state_dict" key from the loaded weights.
        - Maps state dictionary keys to match the model's expected naming convention.
        - Logs mismatched keys and raises an error if strict loading fails.
    Returns:
        None
    """
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = state_dict.get("state_dict", state_dict)

    if isinstance(state_dict, list):
        state_dict = state_dict[0]

    model.load_state_dict(state_dict)