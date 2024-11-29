import torch
import torch.nn.functional as Funct
from numpy.typing import NDArray

def sharpen_RGB(image: NDArray) -> NDArray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3D RGB image with shape (height, width, 3).")

    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    sharpen_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    kernel = sharpen_kernel.expand(3, 1, 3, 3)  
    sharpened_image = Funct.conv2d(image_tensor, kernel, stride=1, padding=1, groups=3)
    
    return sharpened_image.squeeze(0).permute(1, 2, 0).numpy()
