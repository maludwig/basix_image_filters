from torch import Tensor

from custom_nodes.basix_image_filters.color_enhancer import ColorEnhancer, ColorSpace
from custom_nodes.basix_image_filters.image_enhancer_node import ImageEnhancerNode

hsl_enhancer = ColorEnhancer(ColorSpace.HSL)


class HSLLevelsNode(ImageEnhancerNode):
    @classmethod
    def INPUT_TYPES(cls):
        enhancer_inputs = ImageEnhancerNode.INPUT_TYPES()
        enhancer_inputs["required"]["shadows"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1})
        enhancer_inputs["required"]["midtones"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1})
        enhancer_inputs["required"]["highlights"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1})
        return enhancer_inputs

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        shadows = kwargs["shadows"]
        midtones = kwargs["midtones"]
        highlights = kwargs["highlights"]
        images_hsl = hsl_enhancer.from_rgb(images)
        images_hsl = hsl_enhancer.adjust_levels(images_hsl, shadows, midtones, highlights)
        images_rgb = hsl_enhancer.to_rgb(images_hsl)
        return images_rgb

    DESCRIPTION = "Adjust the levels of an image in the HSL color space. A levels value of 0.0 results in a completely black image, while a value of 1.0 gives the original image. Values above 1.0 will lighten the image."
    UI_NAME = "Levels (HSL)"


class HSLRotateHueNode(ImageEnhancerNode):
    @classmethod
    def INPUT_TYPES(cls):
        enhancer_inputs = ImageEnhancerNode.INPUT_TYPES()
        enhancer_inputs["required"]["degrees"] = ("FLOAT", {"default": 15.0, "min": -360, "max": 360, "step": 5})
        return enhancer_inputs

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        images_hsl = hsl_enhancer.from_rgb(images)
        images_hsl = hsl_enhancer.shift_hue(images_hsl, kwargs["degrees"])
        images_rgb = hsl_enhancer.to_rgb(images_hsl)
        return images_rgb

    DESCRIPTION = "Rotate the hue of an image in the HSL color space."
    UI_NAME = "Rotate Hue (HSL)"


class HSLDarkenImageNodeNode(ImageEnhancerNode):
    @classmethod
    def INPUT_TYPES(cls):
        enhancer_inputs = ImageEnhancerNode.INPUT_TYPES()
        enhancer_inputs["required"]["factor"] = ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.1})
        return enhancer_inputs

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        images_hsl = hsl_enhancer.from_rgb(images)
        images_hsl = hsl_enhancer.darken(images_hsl, kwargs["factor"])
        images_rgb = hsl_enhancer.to_rgb(images_hsl)
        return images_rgb

    DESCRIPTION = "Darken an image by adjusting its brightness in HSL color space. A factor of 0.0 results in a completely black image, while a factor of 1.0 gives the original image. Values above 1.0 will lighten the image."
    UI_NAME = "Darken (HSL)"


class HSLLightenImageNodeNode(ImageEnhancerNode):
    @classmethod
    def INPUT_TYPES(cls):
        enhancer_inputs = ImageEnhancerNode.INPUT_TYPES()
        enhancer_inputs["required"]["factor"] = ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.1})
        return enhancer_inputs

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        images_hsl = hsl_enhancer.from_rgb(images)
        images_hsl = hsl_enhancer.lighten(images_hsl, kwargs["factor"])
        images_rgb = hsl_enhancer.to_rgb(images_hsl)
        return images_rgb

    DESCRIPTION = "Darken an image by adjusting its brightness in HSL color space. A factor of 0.0 results in a completely black image, while a factor of 1.0 gives the original image. Values above 1.0 will lighten the image."
    UI_NAME = "Lighten (HSL)"


class HSLSaturateImageNodeNode(ImageEnhancerNode):
    @classmethod
    def INPUT_TYPES(cls):
        enhancer_inputs = ImageEnhancerNode.INPUT_TYPES()
        enhancer_inputs["required"]["factor"] = ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.1})
        return enhancer_inputs

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        images_hsl = hsl_enhancer.from_rgb(images)
        images_hsl = hsl_enhancer.saturate(images_hsl, kwargs["factor"])
        images_rgb = hsl_enhancer.to_rgb(images_hsl)
        return images_rgb

    DESCRIPTION = "Saturate an image by adjusting its saturation in HSL color space. A factor of 0.0 results in a completely desaturated image, while a factor of 1.0 gives the original image. Values above 1.0 will saturate the image."
    UI_NAME = "Saturate (HSL)"


class HSLDesaturateImageNodeNode(ImageEnhancerNode):
    @classmethod
    def INPUT_TYPES(cls):
        enhancer_inputs = ImageEnhancerNode.INPUT_TYPES()
        enhancer_inputs["required"]["factor"] = ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.1})
        return enhancer_inputs

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        images_hsl = hsl_enhancer.from_rgb(images)
        images_hsl = hsl_enhancer.desaturate(images_hsl, kwargs["factor"])
        images_rgb = hsl_enhancer.to_rgb(images_hsl)
        return images_rgb

    DESCRIPTION = "Desaturate an image by adjusting its saturation in HSL color space. A factor of 0.0 results in a completely desaturated image, while a factor of 1.0 gives the original image. Values above 1.0 will saturate the image."
    UI_NAME = "Desaturate (HSL)"


# A dictionary that contains all nodes you want to export with their names
HSL_NODES = [HSLLevelsNode, HSLRotateHueNode, HSLDarkenImageNodeNode, HSLLightenImageNodeNode, HSLSaturateImageNodeNode, HSLDesaturateImageNodeNode]
