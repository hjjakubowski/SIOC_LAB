import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as Funct

def downsample_grey(image: NDArray, kernel_size: int = 2, step: int = 2) -> NDArray:
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
    kernel = torch.ones((1, 1, kernel_size, kernel_size)) / (kernel_size ** 2)
    
    convolved_image = Funct.conv2d(image_tensor, kernel, stride = step, padding=0)
    return convolved_image.squeeze().numpy()  

def downsample_RGB(image: NDArray, kernel_size: int = 2, step: int = 2) -> NDArray:
    
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3D RGB image with shape (height, width, 3).")

    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) 
    kernel = torch.ones((3, 1, kernel_size, kernel_size)) / (kernel_size ** 2)

    convolved_image = Funct.conv2d(image_tensor, kernel, stride=step, padding=0, groups=3)
    return convolved_image.squeeze(0).permute(1, 2, 0).numpy()
