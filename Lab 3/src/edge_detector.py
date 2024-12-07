import torch
import torch.nn.functional as Funct
from numpy.typing import NDArray

def detect_edges_RGB(image: NDArray) -> NDArray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3D RGB image with shape (height, width, 3).")

    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    
    sobel_filter_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) 

    sobel_filter_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  

    kernel_x = sobel_filter_x.expand(3, 1, 3, 3) 
    kernel_y = sobel_filter_y.expand(3, 1, 3, 3)  

    edges_x = Funct.conv2d(image_tensor, kernel_x, stride=1, padding=1, groups=3)
    edges_y = Funct.conv2d(image_tensor, kernel_y, stride=1, padding=1, groups=3)

    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

    return edges.squeeze(0).permute(1, 2, 0).numpy()
