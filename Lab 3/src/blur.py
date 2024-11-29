import torch
import torch.nn.functional as Funct
from numpy.typing import NDArray

def blur_RGB(image: NDArray, kernel_size: int = 3) -> NDArray:
    
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3D RGB image with shape (height, width, 3).")

    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    kernel = torch.ones((3, 1, kernel_size, kernel_size)) / (kernel_size ** 2)
    blurred_image = Funct.conv2d(image_tensor, kernel, stride=1, padding=kernel_size // 2, groups=3)
    
    return blurred_image.squeeze(0).permute(1, 2, 0).numpy()

'''
def Gauss_blur_RGB(image: NDArray, blur_strength: int) -> NDArray:
    
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3D RGB image with shape (height, width, 3).")

    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    Gauss_blur_kernel = torch.tensor([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=torch.float32) / 16.0
    
    kernel = Gauss_blur_kernel.expand(3, 1, 3, 3) 
     
    blurred_image = image_tensor
    for _ in range(blur_strength):
        blurred_image = Funct.conv2d(image_tensor, kernel, stride=1, padding=1, groups=3)
        
    return blurred_image.squeeze(0).permute(1, 2, 0).numpy()
    '''
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

def Gauss_blur_RGB(image: NDArray, iterations: int ) -> NDArray:
    """
    Apply a Gaussian blur to an RGB image with adjustable blur strength by repeated application.
    
    Args:
        image (NDArray): Input RGB image with shape (height, width, 3), pixel values normalized to [0, 1].
        iterations (int): Number of times the Gaussian blur is applied. More iterations = stronger blur.
    
    Returns:
        NDArray: Blurred RGB image, normalized to [0, 1].
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3D RGB image with shape (height, width, 3).")
    
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    Gauss_blur_kernel = torch.tensor([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=torch.float32) / 16.0
    
    kernel = Gauss_blur_kernel.expand(3, 1, 3, 3)
    
    for _ in range(iterations):
        image_tensor = F.conv2d(image_tensor, kernel, stride=1, padding=1, groups=3)
    
    blurred_image = image_tensor
    return blurred_image.squeeze(0).permute(1, 2, 0).numpy()
