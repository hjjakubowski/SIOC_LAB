import numpy as np
from numpy.typing import NDArray

def sample_hold_kernel(x: NDArray, x0: float, w: float) -> NDArray:
    return np.where((x >= x0) & (x < x0 + w), 1, 0)

def nearest_neighbour_kernel(x: NDArray, x0: float, w: float) -> NDArray:
    half_width = w / 2
    return np.where((x >= x0 - half_width) & (x < x0 + half_width), 1, 0)

def linear_kernel(x: NDArray, x0: float, w: float) -> NDArray:
    normalized_x = (x - x0) / w
    return np.where(np.abs(normalized_x) < 1, 1 - np.abs(normalized_x), 0)
