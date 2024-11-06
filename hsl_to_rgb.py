import torch
from torch import Tensor

def hsl_tensor_to_rgb_tensor(hsl_tensor: Tensor, normalize: bool = False) -> Tensor:
    """
    Convert an HSL image tensor to an RGB image tensor.

    Args:
        hsl_tensor (Tensor): A tensor of shape (batch_size, width, height, 3) with HSL channels.
                             - Hue (H) should be in the range [0, 1).
                             - Saturation (S) and Lightness (L) should be in the range [0, 1].
                             If `normalize` is True and the tensor is of an integer type, it will be normalized to [0, 1].
        normalize (bool): Whether to normalize integer inputs to [0, 1].

    Returns:
        Tensor: A tensor of shape (batch_size, width, height, 3) in RGB color space.
                The RGB values are in the range [0, 1].
    """
    if hsl_tensor.dim() != 4 or hsl_tensor.size(-1) != 3:
        raise ValueError("Input tensor must have shape (batch_size, width, height, 3)")

    # Optionally normalize integer inputs to [0, 1]
    if normalize and not torch.is_floating_point(hsl_tensor):
        hsl = hsl_tensor.float() / 255.0
    else:
        hsl = hsl_tensor.float()

    # Separate the H, S, L channels
    h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    # Compute chroma
    c = (1.0 - torch.abs(2.0 * l - 1.0)) * s

    # Compute hue prime
    h_prime = h * 6.0  # Scale hue to [0,6)

    # Compute intermediate value X
    x = c * (1.0 - torch.abs((h_prime % 2) - 1.0))

    # Initialize R', G', B' tensors
    zeros = torch.zeros_like(h)
    r_prime = torch.zeros_like(h)
    g_prime = torch.zeros_like(h)
    b_prime = torch.zeros_like(h)

    # Masks for different sectors
    mask_0 = (h_prime >= 0) & (h_prime < 1)
    mask_1 = (h_prime >= 1) & (h_prime < 2)
    mask_2 = (h_prime >= 2) & (h_prime < 3)
    mask_3 = (h_prime >= 3) & (h_prime < 4)
    mask_4 = (h_prime >= 4) & (h_prime < 5)
    mask_5 = (h_prime >= 5) & (h_prime < 6)

    # Assign R', G', B' based on the sector
    r_prime = torch.where(mask_0 | mask_5, c, r_prime)
    g_prime = torch.where(mask_0, x, g_prime)
    b_prime = torch.where(mask_0, zeros, b_prime)

    r_prime = torch.where(mask_1, x, r_prime)
    g_prime = torch.where(mask_1 | mask_2, c, g_prime)
    b_prime = torch.where(mask_1 | mask_2, zeros, b_prime)

    # Fix: Assign b_prime = x for mask_2
    b_prime = torch.where(mask_2, x, b_prime)

    r_prime = torch.where(mask_2, zeros, r_prime)
    g_prime = torch.where(mask_3, x, g_prime)
    b_prime = torch.where(mask_3 | mask_4, c, b_prime)

    r_prime = torch.where(mask_4, x, r_prime)
    g_prime = torch.where(mask_5, zeros, g_prime)
    b_prime = torch.where(mask_5, c, b_prime)

    # Compute m
    m = l - c / 2.0

    # Compute final RGB values
    r = r_prime + m
    g = g_prime + m
    b = b_prime + m

    # Stack the R, G, B channels back into a single tensor
    rgb = torch.stack((r, g, b), dim=-1)

    # Clamp the results to [0, 1] to avoid numerical issues
    rgb = torch.clamp(rgb, 0.0, 1.0)

    return rgb
