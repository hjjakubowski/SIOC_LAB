import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve

def downsample_grey(image: NDArray, kernel_size: int = 2) -> np.array:
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    convolved_image = convolve(image, kernel, mode='constant', cval=0.0)
    return convolved_image[::kernel_size, ::kernel_size]

def downsample_universal(image: np.ndarray, kernel_size: int = 2) -> np.array:
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2

    if image.ndim == 3:
        convolved_image = np.stack([convolve(image[:, :, channel], kernel, mode='constant', cval=0.0)for channel in range(3)], axis=2)
    else:
        convolved_image = convolve(image, kernel, mode='constant', cval=0.0)
    return convolved_image[::kernel_size, ::kernel_size, ...]