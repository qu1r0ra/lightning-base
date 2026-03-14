import numpy as np
import torch


def denormalize(
    img_tensor: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Denormalize a tensor image for visualization.
    """
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)

    image = img_tensor.permute(1, 2, 0).cpu().numpy()
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image
