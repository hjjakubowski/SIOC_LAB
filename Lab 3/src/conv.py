import numpy as np
from scipy.signal import convolve2d
from numpy.typing import NDArray

def apply_convolution(image: NDArray, kernel: NDArray) -> NDArray:
    if image.ndim == 2:  
        return convolve2d(image, kernel, mode='same')

    elif image.ndim == 3 and image.shape[2] == 3: 
        edited_image = np.zeros_like(image, dtype=np.float32)
        for channel in range(3):
            edited_image[:, :, channel] = convolve2d(image[:, :, channel], kernel, mode='same')
        return edited_image

    else:
        raise ValueError("Input image must be a 2D grayscale image or a 3D RGB image with shape (height, width, 3).")
    
def convolution_for_edge_finders(image: NDArray, kernels: tuple) -> NDArray:
    plane_x, plane_y = kernels

    if image.ndim == 2:  
        edges_x = convolve2d(image, plane_x, mode='same')
        edges_y = convolve2d(image, plane_y, mode='same')
        edges = np.sqrt(edges_x ** 2 + edges_y ** 2)
        return edges

    elif image.ndim == 3 and image.shape[2] == 3:  
        edges = np.zeros_like(image, dtype=np.float32)
        for channel in range(3):
            edges_x = convolve2d(image[:, :, channel], plane_x, mode='same')
            edges_y = convolve2d(image[:, :, channel], plane_y, mode='same')
            edges[:, :, channel] = np.sqrt(edges_x ** 2 + edges_y ** 2)
        return edges

    else:
        raise ValueError("Input image must be a 2D grayscale image or a 3D RGB image with shape (height, width, 3).")
