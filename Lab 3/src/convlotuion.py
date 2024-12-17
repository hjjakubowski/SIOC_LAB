import numpy as np
from scipy.signal import convolve2d
from numpy.typing import NDArray

def apply_convolution(image: NDArray, kernel: NDArray, stride: int = 1, padding: int = 0) -> NDArray:
    
    if image.ndim == 2: #Greyscale image
        image_padded = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
        edited_image = convolve2d(image_padded, kernel, mode='same', boundary='fill', fillvalue=0)
        return edited_image

    elif image.ndim == 3 and image.shape[2] == 3:  #RGB image
        edited_image = np.zeros_like(image, dtype=np.float32)
        for channel in range(3): 
            image_padded = np.pad(image[:, :, channel], pad_width=padding, mode='constant', constant_values=0)
            edited_image[:, :, channel] = convolve2d(image_padded, kernel, mode='same', boundary='fill', fillvalue=0)
        return edited_image

    else:
        raise ValueError("Input image must be a 2D grayscale image or a 3D RGB image with shape (height, width, 3).")


def convolution_for_edge_finders(image: NDArray, kernels: tuple, stride: int = 1, padding: int = 0) -> NDArray:

    
    plane_x, plane_y = kernels  

    if image.ndim == 2: #Greyscale image
        image_padded = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
        
        edges_x = convolve2d(image_padded, plane_x, mode='same', boundary='fill', fillvalue=0)
        edges_y = convolve2d(image_padded, plane_y, mode='same', boundary='fill', fillvalue=0)
        
        edges = np.sqrt(edges_x ** 2 + edges_y ** 2)
        return edges

    elif image.ndim == 3 and image.shape[2] == 3:  #RGB image
        edges = np.zeros_like(image, dtype=np.float32)
        for channel in range(3):  
            image_padded = np.pad(image[:, :, channel], pad_width=padding, mode='constant', constant_values=0)
            
            edges_x = convolve2d(image_padded, plane_x, mode='same', boundary='fill', fillvalue=0)
            edges_y = convolve2d(image_padded, plane_y, mode='same', boundary='fill', fillvalue=0)
 
            edges[:, :, channel] = np.sqrt(edges_x ** 2 + edges_y ** 2)
        
        return edges

    else:
        raise ValueError("Input image must be a 2D grayscale image or a 3D RGB image with shape (height, width, 3).")

