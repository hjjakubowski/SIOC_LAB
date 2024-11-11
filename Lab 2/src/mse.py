from sklearn.metrics import mean_squared_error as mse
from numpy.typing import NDArray

def calculate_mse(original: NDArray, resampled: NDArray) -> float:
    
    if original.shape != resampled.shape:
        raise ValueError("Images must have the same dimensions for MSE calculation.")
    
    return mse(original.flatten(), resampled.flatten())


