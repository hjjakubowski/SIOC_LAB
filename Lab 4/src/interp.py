import numpy as np
from numpy.typing import NDArray

def image_interpolate2d(image: NDArray, ratio: int, kernel_function: callable) -> NDArray:
    """
    Interpolate image using 2D kernel interpolation
    :param image: grayscale image to interpolate as 2D NDArray
    :param ratio: up-scaling factor
    :param kernel_funct: kernel function to use for interpolation
    :return: interpolated image as 2D NDArray
    """
    w = 1 * ratio 

    img_rows, img_cols = image.shape
    target_shape = (img_rows * ratio, img_cols * ratio)  

    image_grid = np.array([[row, col] for row in range(img_rows) for col in range(img_cols)])  

    interpolate_grid = np.array([[row, col] for row in range(target_shape[0]) for col in range(target_shape[1])])  

    kernels = []
    for point, value in zip(image_grid, image.ravel()):
        kernel = value * kernel_function(interpolate_grid, offset=point * ratio, width=w)  
        kernels.append(kernel.reshape(target_shape))  

    return np.sum(np.asarray(kernels), axis=0)

def image_interpolate2d_rgb(image: NDArray, ratio: int, kernel_function: callable) -> NDArray:
    """
    Interpolate an RGB image using 2D kernel interpolation with adjustable kernel width.
    :param image: RGB image to interpolate as a 3D NDArray (height, width, channels)
    :param ratio: up-scaling factor
    :param kernel_funct: kernel function to use for interpolation
    :param w: effective width of the kernel in the interpolated coordinate space
    :return: interpolated RGB image as a 3D NDArray
    """
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3D array representing an RGB image.")

    img_rows, img_cols, num_channels = image.shape

    target_shape = (img_rows * ratio, img_cols * ratio, num_channels)
    interpolated_image = np.zeros(target_shape)

    for channel in range(num_channels):
        single_channel = image[:, :, channel]

        interpolated_channel = image_interpolate2d(single_channel, ratio, kernel_function)

        interpolated_image[:, :, channel] = interpolated_channel

    return interpolated_image

