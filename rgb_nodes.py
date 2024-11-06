from torch import Tensor

from basix_image_filters.color_enhancer import ColorEnhancer, ColorSpace, RGBChannels
from basix_image_filters.image_enhancer_node import ImageEnhancerNode

rgb_enhancer = ColorEnhancer(ColorSpace.RGB)


class RGBLevelsNode(ImageEnhancerNode):
    @classmethod
    def INPUT_TYPES(cls):
        enhancer_inputs = ImageEnhancerNode.INPUT_TYPES()
        enhancer_inputs["required"]["channel"] = (["red", "green", "blue", "all"], {"default": "all"})
        enhancer_inputs["required"]["shadows"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1})
        enhancer_inputs["required"]["midtones"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1})
        enhancer_inputs["required"]["highlights"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1})
        return enhancer_inputs

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        channel_name = kwargs["channel"]
        if channel_name == "all":
            channels = [0, 1, 2]
        else:
            channels = [RGBChannels[channel_name].value]
        shadows = kwargs["shadows"]
        midtones = kwargs["midtones"]
        highlights = kwargs["highlights"]
        for channel in channels:
            images = rgb_enhancer.adjust_levels(images, shadows, midtones, highlights, channel)
        return images

    DESCRIPTION = "Adjust the levels of an image in the RGB color space."
    UI_NAME = "Levels (RGB)"


# A dictionary that contains all nodes you want to export with their names
RGB_NODES = [RGBLevelsNode]
