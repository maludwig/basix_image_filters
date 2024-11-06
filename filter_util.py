from torch import Tensor


def composite_by_mask(image1: Tensor, image2: Tensor, mask: Tensor):
    """
    Composite two images using a mask.
    The mask is used to select pixels from the first image where it is 1,
    and from the second image where it is 0.
    @param image1: The first image tensor.
    @param image2: The second image tensor.
    @param mask: The mask tensor.
    @return: The composited image tensor.
    """
    if mask.dim() == image1.dim() - 1:
        # If the mask does not have a channel dimension, add it
        # mask.shape = (batch_size, width, height)
        mask = mask.unsqueeze(-1)
        # mask.shape = (batch_size, width, height, 1)

    masked_image1 = image1 * mask
    masked_image2 = image2 * (1 - mask)
    # Combine the masked images
    composited_image = masked_image1 + masked_image2
    return composited_image
