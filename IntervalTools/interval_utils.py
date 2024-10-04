import torch
import numpy as np

from IntervalTools.interval_arithmetic import Interval

def interval_arr_from_tensor(M_l : torch.Tensor, M_u : torch.Tensor) -> np.ndarray:
    """
    Combines two tensors N-D into numpy array of intervals.

    Parameters:
    -----------
        M_l: torch.Tensor
            A lower bound tensor.
        
        M_u: torch.Tensor
            An upper bound tensor.
    
    Returns:
    --------
        An array representing the interval N-D tensor.
    """
    assert M_l.shape == M_u.shape, "the shapes of lower and upper matrix should be equal"

    if isinstance(M_l, torch.Tensor):
        M_l = M_l.detach().cpu().numpy()

    if isinstance(M_u, torch.Tensor):
        M_u = M_u.detach().cpu().numpy()

    vectorized_intervals = np.vectorize(Interval, otypes=[Interval])
    intervals = vectorized_intervals(M_l, M_u)

    return intervals


if __name__ == "__main__":

   pass




