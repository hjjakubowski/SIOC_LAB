import numpy as np
from numpy.typing import NDArray

def sample_hold_kernel(points: NDArray, offset: float | NDArray, width: float) -> NDArray:
    
    distances = points - offset
    within_range = (0 <= distances[:, 0]) & (distances[:, 0] < width) & (0 <= distances[:, 1]) & (distances[:, 1] < width)
    return np.where(within_range, 1.0, 0.0)

def nearest_neighbour_kernel(points: NDArray, offset: float | NDArray, width: float) -> NDArray:
    
    distances = points - offset
    within_range = (-width / 2 <= distances[:, 0]) & (distances[:, 0] < width / 2) & (-width / 2 <= distances[:, 1]) & (distances[:, 1] < width / 2)
    
    return np.where(within_range, 1.0, 0.0)