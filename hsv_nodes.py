from torch import Tensor

from basix_image_filters.color_enhancer import ColorEnhancer, ColorSpace
from basix_image_filters.image_enhancer_node import ImageEnhancerNode

hsv_enhancer = ColorEnhancer(ColorSpace.HSV)


class HSVLevelsNode(ImageEnhancerNode):
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
        images_hsl = hsv_enhancer.from_rgb(images)
        images_hsl = hsv_enhancer.adjust_levels(images_hsl, shadows, midtones, highlights)
        images_rgb = hsv_enhancer.to_rgb(images_hsl)
        return images_rgb

    DESCRIPTION = "Adjust the levels of an image in the HSV color space."
    UI_NAME = "Levels (HSV)"


class HSVRotateHueNode(ImageEnhancerNode):
    @classmethod
    def INPUT_TYPES(cls):
        enhancer_inputs = ImageEnhancerNode.INPUT_TYPES()
        enhancer_inputs["required"]["degrees"] = ("FLOAT", {"default": 15.0, "min": -360, "max": 360, "step": 5})
        return enhancer_inputs

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        images_hsv = hsv_enhancer.from_rgb(images)
        images_hsv = hsv_enhancer.shift_hue(images_hsv, kwargs["degrees"])
        images_rgb = hsv_enhancer.to_rgb(images_hsv)
        return images_rgb

    DESCRIPTION = "Rotate the hue of an image in the HSV color space."
    UI_NAME = "Rotate Hue (HSV)"


class HSVDarkenImageNodeNode(ImageEnhancerNode):
    @classmethod
    def INPUT_TYPES(cls):
        enhancer_inputs = ImageEnhancerNode.INPUT_TYPES()
        enhancer_inputs["required"]["factor"] = ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.1})
        return enhancer_inputs

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        images_hsv = hsv_enhancer.from_rgb(images)
        images_hsv = hsv_enhancer.darken(images_hsv, kwargs["factor"])
        images_rgb = hsv_enhancer.to_rgb(images_hsv)
        return images_rgb

    DESCRIPTION = "Darken an image by adjusting its brightness value in HSV color space. A factor of 1.0 results in a completely black image, while a factor of 0.0 gives the original image. Negative values will lighten the image."
    UI_NAME = "Darken (HSV)"


class HSVBrightenImageNodeNode(ImageEnhancerNode):
    @classmethod
    def INPUT_TYPES(cls):
        enhancer_inputs = ImageEnhancerNode.INPUT_TYPES()
        enhancer_inputs["required"]["factor"] = ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.1})
        return enhancer_inputs

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        images_hsv = hsv_enhancer.from_rgb(images)
        images_hsv = hsv_enhancer.lighten(images_hsv, kwargs["factor"])
        images_rgb = hsv_enhancer.to_rgb(images_hsv)
        return images_rgb

    DESCRIPTION = "Brighten an image by adjusting its brightness value in HSV color space. A factor of 1.0 results in a completely bright and saturated image, while a factor of 0.0 gives the original image. Negative values will darken the image."
    UI_NAME = "Brighten (HSV)"


class HSVSaturateImageNodeNode(ImageEnhancerNode):
    @classmethod
    def INPUT_TYPES(cls):
        enhancer_inputs = ImageEnhancerNode.INPUT_TYPES()
        enhancer_inputs["required"]["factor"] = ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.1})
        return enhancer_inputs

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        images_hsv = hsv_enhancer.from_rgb(images)
        images_hsv = hsv_enhancer.saturate(images_hsv, kwargs["factor"])
        images_rgb = hsv_enhancer.to_rgb(images_hsv)
        return images_rgb

    DESCRIPTION = "Saturate an image by adjusting its saturation in HSV color space. A factor of 1.0 results in a completely saturated image, while a factor of 0.0 gives the original image. Negative values will desaturate the image."
    UI_NAME = "Saturate (HSV)"


class HSVDesaturateImageNodeNode(ImageEnhancerNode):
    @classmethod
    def INPUT_TYPES(cls):
        enhancer_inputs = ImageEnhancerNode.INPUT_TYPES()
        enhancer_inputs["required"]["factor"] = ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.1})
        return enhancer_inputs

    def change_images(self, images: Tensor | list[Tensor], **kwargs):
        images_hsv = hsv_enhancer.from_rgb(images)
        images_hsv = hsv_enhancer.desaturate(images_hsv, kwargs["factor"])
        images_rgb = hsv_enhancer.to_rgb(images_hsv)
        return images_rgb

    DESCRIPTION = "Desaturate an image by adjusting its saturation in HSV color space. A factor of 1.0 results in a completely desaturated image, while a factor of 0.0 gives the original image. Negative values will saturate the image."
    UI_NAME = "Desaturate (HSV)"


# A dictionary that contains all nodes you want to export with their names
HSV_NODES = [HSVLevelsNode, HSVRotateHueNode, HSVDarkenImageNodeNode, HSVBrightenImageNodeNode, HSVSaturateImageNodeNode, HSVDesaturateImageNodeNode]
