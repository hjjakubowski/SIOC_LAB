import numpy as np
from scipy.signal import convolve2d
from numpy.typing import NDArray

def demosaic_bayer(image: NDArray) -> NDArray:
    """
    Performs demosaicing on a Bayer-filtered RGB image using NumPy.
    Args:
        image: Input Bayer-filtered image as a 3D numpy array (height x width x 3).
    Returns:
        Demosaicked RGB image as a 3D numpy array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3D array with 3 color channels (RGB).")

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]


    kernel_R = np.array([[0.25, 0.5, 0.25], 
                         [0.5, 1.0, 0.5],
                         [0.25, 0.5, 0.25]], dtype=np.float32)
    
    kernel_B = np.array([[0.25, 0.5, 0.25],
                         [0.5, 1.0, 0.5], 
                         [0.25, 0.5, 0.25]], dtype=np.float32)
    
    kernel_G = np.array([[0, 0.25, 0],
                         [0.25, 1.0, 0.25],
                         [0, 0.25, 0]], dtype=np.float32)

    R_interp = convolve2d(R, kernel_R, mode='same', boundary='symm')
    G_interp = convolve2d(G, kernel_G, mode='same', boundary='symm')
    B_interp = convolve2d(B, kernel_B, mode='same', boundary='symm')

    demosaiced_image = np.stack([R_interp, G_interp, B_interp], axis=-1)
    return demosaiced_image
