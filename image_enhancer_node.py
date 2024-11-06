from typing import Optional

import torch
from torch import Tensor

from comfyui_image_filters.filter_util import composite_by_mask


def as_list(image: Tensor | list[Tensor] | None) -> list[Tensor] | None:
    if isinstance(image, list):
        return image
    elif isinstance(image, Tensor):
        return [image[i] for i in range(image.size(0))]
    else:
        return None


def as_tensor(image_list: Tensor | list[Tensor] | None) -> Tensor | None:
    if isinstance(image_list, list):
        return torch.stack(image_list)
    elif isinstance(image_list, Tensor):
        return image_list
    else:
        return None


class ImageEnhancerNode:
    """
    This is a base class for image enhancer nodes.
    It provides a common interface, with an image and optional mask input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "masked_area": ("BOOLEAN", {"default": True, "label_on": "modify_masked", "label_off": "modify_unmasked"}),
            },
            "optional": {
                "mask_opt": ("MASK",),
            },
        }

    def change_image(self, image: Tensor, **kwargs) -> Tensor:
        """
        This method must be implemented by the subclass. It changes a single image
        Args:
            image: A single image tensor
            **kwargs: The custom arguments for the node

        Returns:
            The modified image tensor
        """
        raise NotImplementedError()

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        """
        This method applies the change_image method to a list of images.
        If your node only works with a single image, you can use the default implementation.
        If your node is more efficient when working with a list/batch of images, you can override this method.
        Args:
            images: A Tensor or list of Tensors representing the image(s)
            **kwargs: The custom arguments for the node

        Returns:
            A Tensor or list of Tensors representing the modified images
        """
        new_images = []
        for i in range(len(images)):
            new_images.append(self.change_image(images[i], **kwargs))
        return new_images

    def sanitize_inputs(self, image: Tensor | list[Tensor], masked_area=True, mask_opt: Tensor | list[Tensor] = None) -> (Tensor, Optional[Tensor]):
        """
        This method ensures that the inputs are in a common format, and applies the transformation to the mask area if needed.
        Args:
            image: A Tensor or list of Tensors representing the image(s)
            masked_area: If true, the image will be modified where the mask is 1. If false, the image will be modified where the mask is 0.
            mask_opt: A Tensor or list of Tensors representing the mask(s), or None if no mask is provided

        Returns:
            A tuple of the image(s) and the mask(s) in the correct format
        """
        if mask_opt is None:
            mask = None
        else:
            if masked_area:
                mask = mask_opt
            else:
                if isinstance(mask_opt, list):
                    mask = [1 - mask_opt[i] for i in range(len(mask_opt))]
                elif isinstance(mask_opt, Tensor):
                    mask = 1 - mask_opt
                else:
                    raise ValueError("The mask must be a list or a Tensor.")
        return image, mask

    def finalize_output(
        self, original_images: list[Tensor] | Tensor, new_images: list[Tensor] | Tensor, mask_opt: Tensor | list[Tensor] = None
    ) -> list[Tensor] | Tensor:
        """
        This method applies the mask to the modified images if needed, and returns the final output as a list or Tensor.
        Args:
            original_images: A Tensor or list of Tensors representing the original image(s)
            new_images: A Tensor or list of Tensors representing the modified image(s)
            mask_opt: A Tensor or list of Tensors representing the mask(s), or None if no mask is provided

        Returns:
            A Tensor or list of Tensors representing the final output image(s)
        """
        if mask_opt is None:
            masked_images = new_images
        else:
            # The images must be composited with the mask
            if isinstance(mask_opt, Tensor) and isinstance(new_images, Tensor):
                masked_images = composite_by_mask(new_images, original_images, mask_opt)
            elif isinstance(mask_opt, list) and isinstance(new_images, list):
                new_images_composited = []
                for i in range(len(new_images)):
                    new_images_composited.append(composite_by_mask(new_images[i], original_images[i], mask_opt[i]))
                masked_images = new_images_composited
            else:
                raise ValueError("The mask and images must be of the same type (list or Tensor).")
        if self.OUTPUT_IS_LIST[0]:
            return as_list(masked_images)
        else:
            return as_tensor(masked_images)

    def doit(self, image: Tensor, masked_area=True, mask_opt: Tensor = None, **kwargs):
        image, mask = self.sanitize_inputs(image, masked_area, mask_opt)
        new_images = self.change_images(image, **kwargs)
        return (self.finalize_output(image, new_images, mask),)

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)

    FUNCTION = "doit"

    CATEGORY = "Image Filters"
