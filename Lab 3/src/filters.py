import numpy as np
from numpy.typing import NDArray

def blur_kernel(size: int ) -> NDArray:

    return np.ones((size, size), dtype=np.float32) / (size ** 2)

def gauss_blur_kernel() -> NDArray:
 
    return np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16.0
    
def sharpen_kernel() -> NDArray:

    return np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

def laplace_kernel() -> NDArray:
  
    return np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)
    
def sobel_kernel() -> NDArray:
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)
    
    return sobel_x, sobel_y

def scharr_kernel() -> NDArray:
    scharr_x = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3]
    ], dtype=np.float32)

    scharr_y = np.array([
        [ 3, 10, 3],
        [ 0,  0,  0],
        [ -3, -10, -3]
    ], dtype=np.float32)

    return scharr_x, scharr_y

def prewitt_kernel() -> NDArray:
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    prewitt_y = np.array([
        [ -1, -1, -1],
        [ 0,  0,  0],
        [ 1, 1, 1]
    ], dtype=np.float32)

    return prewitt_x, prewitt_y

bayer_filters = {
    "red": np.array([[0.25, 0.5, 0.25],
                     [0.5, 1.0, 0.5],
                     [0.25, 0.5, 0.25]], dtype=np.float32),
    "green": np.array([[0, 0.25, 0],
                       [0.25, 1.0, 0.25],
                       [0, 0.25, 0]], dtype=np.float32),
    "blue": np.array([[0.25, 0.5, 0.25],
                      [0.5, 1.0, 0.5],
                      [0.25, 0.5, 0.25]], dtype=np.float32)
}

