import torch
from torch import Tensor

def rgb_tensor_to_hsl_tensor(image_tensor: Tensor, normalize: bool = False) -> Tensor:
    """
    Convert an RGB image tensor to an HSL image tensor.

    Args:
        image_tensor (Tensor): A tensor of shape (batch_size, width, height, 3) 
                               or (width, height, 3) with RGB channels.
                               The RGB values should be in the range [0, 1]. If `normalize` is True and the tensor
                               is of an integer type, it will be normalized to [0, 1].
        normalize (bool): Whether to normalize integer inputs to [0, 1].

    Returns:
        Tensor: A tensor of shape (batch_size, width, height, 3) in HSL color space,
                or (width, height, 3) if the input tensor is not batched.
                - Hue (H) is in the range [0, 1), representing the angle in degrees divided by 360.
                - Saturation (S) and Lightness (L) are in the range [0, 1].
    """
    if image_tensor.dim() == 3:
        is_single_image = True
        image_tensor = image_tensor.unsqueeze(0)
    else:
        is_single_image = False

    if image_tensor.dim() != 4 or image_tensor.size(-1) != 3:
        error_msg = "Input tensor must have shape (batch_size, width, height, 3) or (width, height, 3)\n"
        error_msg += f"Got: {image_tensor.size()} of dim {image_tensor.dim()}"
        raise ValueError(error_msg)

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

    # Initialize Hue, Saturation, and Lightness tensors
    h = torch.zeros_like(max_val)
    s = torch.zeros_like(max_val)
    l = (max_val + min_val) / 2.0

    # Compute Saturation
    # If delta is zero, saturation remains zero
    mask = delta != 0
    s[mask] = torch.where(
        l[mask] < 0.5,
        delta[mask] / (max_val[mask] + min_val[mask]),
        delta[mask] / (2.0 - max_val[mask] - min_val[mask])
    )

    # Compute Hue
    # Avoid division by zero by ensuring delta != 0
    mask_r = (max_val == r) & mask
    mask_g = (max_val == g) & mask
    mask_b = (max_val == b) & mask

    h[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360.0
    h[mask_g] = (60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120.0) % 360.0
    h[mask_b] = (60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240.0) % 360.0

    # Normalize Hue to [0, 1)
    h = (h % 360.0) / 360.0

    # Clamp hue to [0, 1) to avoid floating-point precision issues
    h = torch.clamp(h, 0.0, 1.0 - 1e-6)

    # Stack the H, S, L channels back into a single tensor
    hsl = torch.stack((h, s, l), dim=-1)

    if is_single_image:
        return hsl.squeeze(0)
    else:
        return hsl
