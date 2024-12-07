import numpy as np
import torch
import torch.nn.functional as Funct
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
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) 

    R = image_tensor[:, 0:1, :, :]  
    G = image_tensor[:, 1:2, :, :]  
    B = image_tensor[:, 2:2+1, :, :]


    red_filter = torch.tensor([[0.25, 0.5, 0.25], 
                               [0.5, 1.0, 0.5],
                               [0.25, 0.5, 0.25]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
    
    green_filter = torch.tensor([[0, 0.25, 0],
                                 [0.25, 1.0, 0.25],
                                 [0, 0.25, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
    
    blue_filter = torch.tensor([[0.25, 0.5, 0.25],
                                [0.5, 1.0, 0.5], 
                                [0.25, 0.5, 0.25]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) 

    R_interp = Funct.conv2d(R, red_filter, padding=1)
    G_interp = Funct.conv2d(G, green_filter, padding=1)
    B_interp = Funct.conv2d(B, blue_filter, padding=1)

    demosaiced_tensor = torch.cat([R_interp, G_interp, B_interp], dim=1)
    demosaiced_image = demosaiced_tensor.squeeze(0).permute(1, 2, 0).numpy()
    return demosaiced_image
