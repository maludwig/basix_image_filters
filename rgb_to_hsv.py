import torch
from torch import Tensor

def rgb_tensor_to_hsv_tensor(image_tensor: Tensor, normalize: bool = False) -> Tensor:
    """
    Convert an RGB image tensor to an HSV image tensor.

    Args:
        image_tensor (Tensor): A tensor of shape (batch_size, image_width, image_height, 3) with RGB channels.
                               The RGB values should be in the range [0, 1]. If `normalize` is True and the tensor
                               is of an integer type, it will be normalized to [0, 1].
        normalize (bool): Whether to normalize integer inputs to [0, 1].

    Returns:
        Tensor: A tensor of shape (batch_size, image_width, image_height, 3) in HSV color space.
                - Hue (H) is in the range [0, 1), representing the angle in degrees divided by 360.
                - Saturation (S) and Value (V) are in the range [0, 1].
    """
    if image_tensor.dim() != 4 or image_tensor.size(-1) != 3:
        raise ValueError("Input tensor must have shape (batch_size, width, height, 3)")

    # Optionally normalize integer inputs to [0, 1]
    if normalize and not torch.is_floating_point(image_tensor):
        image = image_tensor.float() / 255.0
    else:
        image = image_tensor.float()

    # Separate the RGB channels
    r, g, b = image[..., 0], image[..., 1], image[..., 2]

    # Compute the maximum and minimum values among R, G, B
    max_val, _ = torch.max(image, dim=-1)
    min_val, _ = torch.min(image, dim=-1)
    delta = max_val - min_val

    # Initialize Hue, Saturation, and Value tensors
    h = torch.zeros_like(max_val)
    s = torch.zeros_like(max_val)
    v = max_val

    # Compute Saturation
    # If max_val is 0, saturation remains 0 to avoid division by zero
    s[max_val != 0] = delta[max_val != 0] / max_val[max_val != 0]

    # Compute Hue
    # Avoid division by zero by ensuring delta != 0
    mask = delta != 0

    # Where max is Red
    mask_r = (max_val == r) & mask
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360

    # Where max is Green
    mask_g = (max_val == g) & mask
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360

    # Where max is Blue
    mask_b = (max_val == b) & mask
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

    # Normalize Hue to [0, 1)
    h = (h % 360.0) / 360.0

    # Clamp hue to [0, 1) to avoid floating-point precision issues
    h = torch.clamp(h, 0, 1 - 1e-6)

    # Stack the H, S, V channels back into a single tensor
    hsv = torch.stack((h, s, v), dim=-1)

    return hsv
