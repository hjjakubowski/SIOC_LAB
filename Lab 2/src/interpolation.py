import numpy as np
from numpy.typing import NDArray

def conv_interpolate(x: NDArray, y: NDArray, xp: NDArray, kernel_func: callable) -> NDArray:

    interpolated_values = np.zeros_like(xp, dtype=float)
    width = np.mean(np.diff(x)) 

    for i, x_point in enumerate(xp):
        kernel_values = kernel_func(x, x_point, width)
        interpolated_values[i] = np.sum(y * kernel_values)

    return interpolated_values

def grey_image_interpolate(image: NDArray, kernel: callable, ratio: int) -> NDArray:
    
    def row_column_interpolate(row: NDArray) -> NDArray:
        x_measure = np.arange(len(row))
        x_interp= np.linspace(0, len(row), ratio*len(row), endpoint= 0 )
        return conv_interpolate(x_measure, row, x_interp, kernel)

    interpolated = np.apply_along_axis(row_column_interpolate, 1 , image)
    return np.apply_along_axis(row_column_interpolate, 0, interpolated)

def color_image_intepolate(image: NDArray, kernel: callable, ratio: int) -> NDArray:
    
    return np.stack([grey_image_interpolate(image[:, :, channel], kernel , ratio) for channel in range(3) ], axis=2)
