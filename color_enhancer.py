import torch
from torch import Tensor
from enum import Enum
from basix_image_filters.rgb_to_hsl import rgb_tensor_to_hsl_tensor
from basix_image_filters.hsl_to_rgb import hsl_tensor_to_rgb_tensor
from basix_image_filters.rgb_to_hsv import rgb_tensor_to_hsv_tensor
from basix_image_filters.hsv_to_rgb import hsv_tensor_to_rgb_tensor


class ColorSpace(Enum):
    HSL = "HSL"
    HSV = "HSV"
    RGB = "RGB"


class RGBChannels(Enum):
    red = 0
    green = 1
    blue = 2


class HSLChannels(Enum):
    hue = 0
    saturation = 1
    lightness = 2


class HSVChannels(Enum):
    hue = 0
    saturation = 1
    value = 2


CHANNEL_MAP = {ColorSpace.HSL: HSLChannels, ColorSpace.HSV: HSVChannels, ColorSpace.RGB: RGBChannels}


class ColorEnhancer:
    """
    A unified class for enhancing images in either RGB, HSL, or HSV color space.

    Attributes:
        color_space (ColorSpace): The color space of the input images (HSL or HSV).
    """

    def __init__(self, color_space: ColorSpace):
        """
        Initializes the ColorEnhancer with the specified color space.

        Args:
            color_space (ColorSpace): The color space of the input images (HSL or HSV).

        Raises:
            ValueError: If an unsupported color space is provided.
        """
        if not isinstance(color_space, ColorSpace):
            raise ValueError(f"Unsupported color space: {color_space}")
        self.color_space = color_space
        self.channels = CHANNEL_MAP[color_space]
        # Both HSL and HSV have brightness at index 2
        self.brightness_channel = 2  # H, S, L/V

    def to_rgb(self, image: Tensor) -> Tensor:
        """
        Converts an image tensor from the specified color space to RGB.

        Args:
            image (Tensor): A tensor of shape (batch_size, width, height, 3) in the specified color space.

        Returns:
            Tensor: The RGB image tensor.
        """
        if self.color_space == ColorSpace.HSL:
            return hsl_tensor_to_rgb_tensor(image)
        elif self.color_space == ColorSpace.HSV:
            return hsv_tensor_to_rgb_tensor(image)
        elif self.color_space == ColorSpace.RGB:
            return image

    def from_rgb(self, image: Tensor) -> Tensor:
        """
        Converts an image tensor from RGB to the specified color space.

        Args:
            image (Tensor): A tensor of shape (batch_size, width, height, 3) in RGB color space.

        Returns:
            Tensor: The image tensor in the specified color space.
        """
        if self.color_space == ColorSpace.HSL:
            return rgb_tensor_to_hsl_tensor(image)
        elif self.color_space == ColorSpace.HSV:
            return rgb_tensor_to_hsv_tensor(image)
        elif self.color_space == ColorSpace.RGB:
            return image

    def lighten(self, image: Tensor, amount: float) -> Tensor:
        """
        Lightens the image by increasing the brightness component.

        Args:
            image (Tensor): A tensor of shape (batch_size, width, height, 3) in the specified color space.
            amount (float): The amount to increase the brightness component. Must be between 0 and 1.

        Returns:
            Tensor: The lightened image tensor.
        """

        lightened_image = image.clone()
        lightened_image[..., self.brightness_channel] += amount
        lightened_image[..., self.brightness_channel] = torch.clamp(lightened_image[..., self.brightness_channel], 0.0, 1.0)
        return lightened_image

    def darken(self, image: Tensor, amount: float) -> Tensor:
        """
        Darkens the image by decreasing the brightness component.

        Args:
            image (Tensor): A tensor of shape (batch_size, width, height, 3) in the specified color space.
            amount (float): The amount to decrease the brightness component. Must be between 0 and 1.

        Returns:
            Tensor: The darkened image tensor.
        """

        darkened_image = image.clone()
        darkened_image[..., self.brightness_channel] -= amount
        darkened_image[..., self.brightness_channel] = torch.clamp(darkened_image[..., self.brightness_channel], 0.0, 1.0)
        return darkened_image

    def saturate(self, image: Tensor, amount: float) -> Tensor:
        """
        Increases the Saturation (S) component.

        Args:
            image (Tensor): A tensor of shape (batch_size, width, height, 3) in the specified color space.
            amount (float): The amount to increase the Saturation component. Must be between 0 and 1.

        Returns:
            Tensor: The saturated image tensor.
        """

        saturated_image = image.clone()
        saturated_image[..., 1] += amount
        saturated_image[..., 1] = torch.clamp(saturated_image[..., 1], 0.0, 1.0)
        return saturated_image

    def desaturate(self, image: Tensor, amount: float) -> Tensor:
        """
        Decreases the Saturation (S) component.

        Args:
            image (Tensor): A tensor of shape (batch_size, width, height, 3) in the specified color space.
            amount (float): The amount to decrease the Saturation component. Must be between 0 and 1.

        Returns:
            Tensor: The desaturated image tensor.
        """

        desaturated_image = image.clone()
        desaturated_image[..., 1] -= amount
        desaturated_image[..., 1] = torch.clamp(desaturated_image[..., 1], 0.0, 1.0)
        return desaturated_image

    def shift_hue(self, image: Tensor, degrees: float) -> Tensor:
        """
        Shifts the hue by a specified number of degrees.

        Args:
            image (Tensor): A tensor of shape (batch_size, width, height, 3) in the specified color space.
            degrees (float): The number of degrees to shift the hue. Can be positive or negative.

        Returns:
            Tensor: The hue-shifted image tensor.
        """

        shifted_image = image.clone()
        shift_normalized = degrees / 360.0  # Normalize the degree shift
        shifted_image[..., 0] = torch.remainder(shifted_image[..., 0] + shift_normalized, 1.0)
        return shifted_image

    def adjust_levels(self, image: Tensor, shadows: float, gamma: float, highlights: float, channel=2) -> Tensor:
        """
        Adjusts the levels of the image by modifying the shadows, midtones, and highlights.

        Args:
            image (Tensor): A tensor of shape (batch_size, width, height, 3) in the specified color space.
            shadows (float): The amount to darken the shadows, any brightness below "shadows" becomes black. Must be between 0 and 1.
            gamma (float): The amount to adjust the midtones, any brightness between "shadows" and "highlights" is adjusted. Represents a gamma curve.
                           Values below 1.0 darken the midtones, while values above 1.0 brighten the midtones.
            highlights (float): The amount to brighten the highlights, any brightness above "highlights" becomes white. Must be between "shadows" and 1.
            channel (int): The channel to adjust the levels of (0 = Hue, 1 = Saturation, 2 = Lightness/Value).

        Returns:
            Tensor: The image tensor with adjusted levels.
        """

        # The magnitude of the difference between the shadows and highlights
        magnitude = highlights - shadows
        scale_factor = 1 / magnitude

        adjusted_image = image.clone()
        brightness = adjusted_image[..., channel]
        brightness -= shadows
        brightness *= scale_factor
        brightness = torch.clamp(brightness, 0.0, 1.0)

        # Adjust gamma
        brightness = torch.pow(brightness, gamma)
        adjusted_image[..., channel] = torch.clamp(brightness, 0.0, 1.0)
        return adjusted_image
