import numpy as np
from numpy.typing import NDArray
from kernels import kernel


def image_interpolate2d(image: NDArray, ratio: int) -> NDArray:
    """
    Interpolate image using 2D kernel interpolation
    :param image: grayscale image to interpolate as 2D NDArray
    :param ratio: up-scaling factor
    :return: interpolated image as 2D NDArray
    """
    w = 1 # C
    target_shape = None # C
    image_grid = None # C
    interpolate_grid = None # C
    kernels = []
    for point, value in zip(image_grid, image.ravel()):
        kernel = value * kernel(interpolate_grid, offset=point, width=w)
        kernels.append(kernel.reshape())
    return np.sum(np.asarray(kernels), axis=0)

