import unittest
import torch
from torch import Tensor

# Assuming the color_enhancer.py is in the same directory or properly installed as a module
from comfyui_image_filters.color_enhancer import (
    ColorEnhancer,
    ColorSpace
)


class TestColorEnhancer(unittest.TestCase):
    def setUp(self):
        """
        Set up common tensors and instances for tests.
        """
        # Define some common colors in RGB
        self.rgb_black = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # Black
        self.rgb_white = torch.tensor([[[[1.0, 1.0, 1.0]]]])  # White
        self.rgb_red = torch.tensor([[[[1.0, 0.0, 0.0]]]])    # Red
        self.rgb_cyan = torch.tensor([[[[0.0, 1.0, 1.0]]]])   # Cyan

        # Define expected HSL and HSV for these colors
        self.hsl_black = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # HSL: (0,0,0)
        self.hsl_white = torch.tensor([[[[0.0, 0.0, 1.0]]]])  # HSL: (0,0,1)
        self.hsl_red = torch.tensor([[[[0.0, 1.0, 0.5]]]])    # HSL: (0,1,0.5)
        self.hsl_cyan = torch.tensor([[[[0.5, 1.0, 0.5]]]])   # HSL: (0.5,1,0.5)

        self.hsv_black = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # HSV: (0,0,0)
        self.hsv_white = torch.tensor([[[[0.0, 0.0, 1.0]]]])  # HSV: (0,0,1)
        self.hsv_red = torch.tensor([[[[0.0, 1.0, 1.0]]]])    # HSV: (0,1,1)
        self.hsv_cyan = torch.tensor([[[[0.5, 1.0, 1.0]]]])   # HSV: (0.5,1,1)

    def assertTensorAlmostEqual(self, tensor1: Tensor, tensor2: Tensor, tol=1e-4):
        """
        Helper method to assert that two tensors are almost equal.
        """
        self.assertTrue(torch.allclose(tensor1, tensor2, atol=tol),
                        msg=f"Tensors are not almost equal.\nTensor1: {tensor1}\nTensor2: {tensor2}")

    def test_initialization_valid_color_spaces(self):
        """
        Test that ColorEnhancer initializes correctly with valid color spaces.
        """
        for color_space in [ColorSpace.HSL, ColorSpace.HSV]:
            enhancer = ColorEnhancer(color_space)
            self.assertEqual(enhancer.color_space, color_space)

    def test_initialization_invalid_color_space(self):
        """
        Test that ColorEnhancer raises ValueError when initialized with an invalid color space.
        """
        with self.assertRaises(ValueError):
            ColorEnhancer("INVALID_COLOR_SPACE")  # Passing a string instead of ColorSpace Enum

    def test_to_rgb_hsl(self):
        """
        Test to_rgb method for HSL color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Black
        rgb = enhancer.to_rgb(self.hsl_black)
        self.assertTensorAlmostEqual(rgb, self.rgb_black)

        # White
        rgb = enhancer.to_rgb(self.hsl_white)
        self.assertTensorAlmostEqual(rgb, self.rgb_white)

        # Red
        rgb = enhancer.to_rgb(self.hsl_red)
        self.assertTensorAlmostEqual(rgb, self.rgb_red)

        # Cyan
        rgb = enhancer.to_rgb(self.hsl_cyan)
        self.assertTensorAlmostEqual(rgb, self.rgb_cyan)

    def test_to_rgb_hsv(self):
        """
        Test to_rgb method for HSV color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Black
        rgb = enhancer.to_rgb(self.hsv_black)
        self.assertTensorAlmostEqual(rgb, self.rgb_black)

        # White
        rgb = enhancer.to_rgb(self.hsv_white)
        self.assertTensorAlmostEqual(rgb, self.rgb_white)

        # Red
        rgb = enhancer.to_rgb(self.hsv_red)
        self.assertTensorAlmostEqual(rgb, self.rgb_red)

        # Cyan
        rgb = enhancer.to_rgb(self.hsv_cyan)
        self.assertTensorAlmostEqual(rgb, self.rgb_cyan)

    def test_from_rgb_hsl(self):
        """
        Test from_rgb method for HSL color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Black
        hsl = enhancer.from_rgb(self.rgb_black)
        self.assertTensorAlmostEqual(hsl, self.hsl_black)

        # White
        hsl = enhancer.from_rgb(self.rgb_white)
        self.assertTensorAlmostEqual(hsl, self.hsl_white)

        # Red
        hsl = enhancer.from_rgb(self.rgb_red)
        self.assertTensorAlmostEqual(hsl, self.hsl_red)

        # Cyan
        hsl = enhancer.from_rgb(self.rgb_cyan)
        self.assertTensorAlmostEqual(hsl, self.hsl_cyan)

    def test_from_rgb_hsv(self):
        """
        Test from_rgb method for HSV color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Black
        hsv = enhancer.from_rgb(self.rgb_black)
        self.assertTensorAlmostEqual(hsv, self.hsv_black)

        # White
        hsv = enhancer.from_rgb(self.rgb_white)
        self.assertTensorAlmostEqual(hsv, self.hsv_white)

        # Red
        hsv = enhancer.from_rgb(self.rgb_red)
        self.assertTensorAlmostEqual(hsv, self.hsv_red)

        # Cyan
        hsv = enhancer.from_rgb(self.rgb_cyan)
        self.assertTensorAlmostEqual(hsv, self.hsv_cyan)

    def test_lighten_hsl(self):
        """
        Test the lighten method for HSL color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Lighten red by 0.1
        lightened_hsl = enhancer.lighten(self.hsl_red, 0.1)
        expected_hsl = self.hsl_red.clone()
        expected_hsl[..., 2] += 0.1
        expected_hsl[..., 2] = torch.clamp(expected_hsl[..., 2], 0.0, 1.0)
        self.assertTensorAlmostEqual(lightened_hsl, expected_hsl)

    def test_lighten_hsv(self):
        """
        Test the lighten method for HSV color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Lighten red by 0.1
        lightened_hsv = enhancer.lighten(self.hsv_red, 0.1)
        expected_hsv = self.hsv_red.clone()
        expected_hsv[..., 2] += 0.1
        expected_hsv[..., 2] = torch.clamp(expected_hsv[..., 2], 0.0, 1.0)
        self.assertTensorAlmostEqual(lightened_hsv, expected_hsv)

    def test_darken_hsl(self):
        """
        Test the darken method for HSL color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Darken white by 0.3
        darkened_hsl = enhancer.darken(self.hsl_white, 0.3)
        expected_hsl = self.hsl_white.clone()
        expected_hsl[..., 2] -= 0.3
        expected_hsl[..., 2] = torch.clamp(expected_hsl[..., 2], 0.0, 1.0)
        self.assertTensorAlmostEqual(darkened_hsl, expected_hsl)

    def test_darken_hsv(self):
        """
        Test the darken method for HSV color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Darken white by 0.3
        darkened_hsv = enhancer.darken(self.hsv_white, 0.3)
        expected_hsv = self.hsv_white.clone()
        expected_hsv[..., 2] -= 0.3
        expected_hsv[..., 2] = torch.clamp(expected_hsv[..., 2], 0.0, 1.0)
        self.assertTensorAlmostEqual(darkened_hsv, expected_hsv)

    def test_saturate_hsl(self):
        """
        Test the saturate method for HSL color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Saturate cyan by 0.2
        saturated_hsl = enhancer.saturate(self.hsl_cyan, 0.2)
        expected_hsl = self.hsl_cyan.clone()
        expected_hsl[..., 1] += 0.2
        expected_hsl[..., 1] = torch.clamp(expected_hsl[..., 1], 0.0, 1.0)
        self.assertTensorAlmostEqual(saturated_hsl, expected_hsl)

    def test_saturate_hsv(self):
        """
        Test the saturate method for HSV color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Saturate cyan by 0.2
        saturated_hsv = enhancer.saturate(self.hsv_cyan, 0.2)
        expected_hsv = self.hsv_cyan.clone()
        expected_hsv[..., 1] += 0.2
        expected_hsv[..., 1] = torch.clamp(expected_hsv[..., 1], 0.0, 1.0)
        self.assertTensorAlmostEqual(saturated_hsv, expected_hsv)

    def test_desaturate_hsl(self):
        """
        Test the desaturate method for HSL color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Desaturate red by 0.5
        desaturated_hsl = enhancer.desaturate(self.hsl_red, 0.5)
        expected_hsl = self.hsl_red.clone()
        expected_hsl[..., 1] -= 0.5
        expected_hsl[..., 1] = torch.clamp(expected_hsl[..., 1], 0.0, 1.0)
        self.assertTensorAlmostEqual(desaturated_hsl, expected_hsl)

    def test_desaturate_hsv(self):
        """
        Test the desaturate method for HSV color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Desaturate red by 0.5
        desaturated_hsv = enhancer.desaturate(self.hsv_red, 0.5)
        expected_hsv = self.hsv_red.clone()
        expected_hsv[..., 1] -= 0.5
        expected_hsv[..., 1] = torch.clamp(expected_hsv[..., 1], 0.0, 1.0)
        self.assertTensorAlmostEqual(desaturated_hsv, expected_hsv)

    def test_shift_hue_positive(self):
        """
        Test shifting hue positively in HSL color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Shift hue of red by 120 degrees to get green
        hsl_red = torch.tensor([[[[0.0, 1.0, 0.5]]]])  # Red in HSL
        shifted_hsl = enhancer.shift_hue(hsl_red, 120.0)
        expected_hsl = torch.tensor([[[[(0.0 + 120.0 / 360.0) % 1.0, 1.0, 0.5]]]])  # Should be green
        self.assertTensorAlmostEqual(shifted_hsl, expected_hsl)

    def test_shift_hue_negative(self):
        """
        Test shifting hue negatively in HSV color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Shift hue of cyan by -120 degrees to get green
        hsv_cyan = torch.tensor([[[[0.5, 1.0, 1.0]]]])  # Cyan in HSV
        shifted_hsv = enhancer.shift_hue(hsv_cyan, -120.0)
        expected_hsv = torch.tensor([[[[(0.5 - 120.0 / 360.0) % 1.0, 1.0, 1.0]]]])  # Should be green
        self.assertTensorAlmostEqual(shifted_hsv, expected_hsv)

    def test_adjust_levels_hsl(self):
        """
        Test adjust_levels method for HSL color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Create a gradient in lightness
        hsl = torch.tensor([[[[0.0, 0.5, 0.2], [0.0, 0.5, 0.8]]]])  # Shape: (1, 1, 2, 3)

        # Adjust levels: shadows=0.1, gamma=1.0, highlights=0.9 on lightness channel (2)
        adjusted_hsl = enhancer.adjust_levels(hsl, shadows=0.1, gamma=1.0, highlights=0.9, channel=2)
        # Expected calculation:
        # For first pixel: (0.2 - 0.1) / (0.9 - 0.1) = 0.125, gamma=1.0 => 0.125
        # For second pixel: (0.8 - 0.1) / (0.9 - 0.1) = 0.875, gamma=1.0 => 0.875
        expected_hsl = torch.tensor([[[[0.0, 0.5, 0.125], [0.0, 0.5, 0.875]]]])

        self.assertTensorAlmostEqual(adjusted_hsl, expected_hsl)

    def test_adjust_levels_hsv_gamma(self):
        """
        Test adjust_levels method with gamma correction for HSV color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Create a gradient in value
        hsv = torch.tensor([[[[0.0, 0.5, 0.2], [0.0, 0.5, 0.8]]]])  # Shape: (1, 1, 2, 3)

        # Adjust levels: shadows=0.1, gamma=2.0, highlights=0.9 on value channel (2)
        adjusted_hsv = enhancer.adjust_levels(hsv, shadows=0.1, gamma=2.0, highlights=0.9, channel=2)
        # Expected calculation:
        # For first pixel: (0.2 - 0.1) / (0.9 - 0.1) = 0.125, gamma=2.0 => 0.125^2 = 0.015625
        # For second pixel: (0.8 - 0.1) / (0.9 - 0.1) = 0.875, gamma=2.0 => 0.875^2 = 0.765625
        expected_hsv = torch.tensor([[[[0.0, 0.5, 0.015625], [0.0, 0.5, 0.765625]]]])

        self.assertTensorAlmostEqual(adjusted_hsv, expected_hsv)

    def test_adjust_levels_clamping(self):
        """
        Test that adjust_levels correctly clamps values to [0, 1].
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Values that will exceed the [0, 1] range after adjustment
        hsl = torch.tensor([[[[0.0, 0.5, 0.95], [0.0, 0.5, 0.05]]]])  # Shape: (1, 1, 2, 3)

        # Adjust levels: shadows=0.0, gamma=1.0, highlights=1.0 on lightness channel (2)
        adjusted_hsl = enhancer.adjust_levels(hsl, shadows=0.0, gamma=1.0, highlights=1.0, channel=2)
        # Expected calculation:
        # For first pixel: (0.95 - 0.0) / (1.0 - 0.0) = 0.95, gamma=1.0 => 0.95
        # For second pixel: (0.05 - 0.0) / (1.0 - 0.0) = 0.05, gamma=1.0 => 0.05
        # No clamping needed here, but let's set gamma=10 to push 0.95^10 and 0.05^10
        adjusted_hsl = enhancer.adjust_levels(hsl, shadows=0.0, gamma=10.0, highlights=1.0, channel=2)
        expected_hsl = torch.tensor([[[[0.0, 0.5, 0.95 ** 10], [0.0, 0.5, 0.05 ** 10]]]])
        expected_hsl = torch.clamp(expected_hsl, 0.0, 1.0)
        self.assertTensorAlmostEqual(adjusted_hsl, expected_hsl)

    def test_shift_hue_wrap_around(self):
        """
        Test that shifting hue wraps around correctly.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Hue near the upper boundary
        hsl = torch.tensor([[[[0.9, 0.5, 0.5]]]])  # Hue = 0.9 (324 degrees)
        shifted_hsl = enhancer.shift_hue(hsl, 90.0)   # Shift by +90 degrees -> 324 + 90 = 414 % 360 = 54 degrees -> 54/360=0.15
        expected_hsl = torch.tensor([[[[(0.9 + 90.0 / 360.0) % 1.0, 0.5, 0.5]]]])
        self.assertTensorAlmostEqual(shifted_hsl, expected_hsl)

    def test_shift_hue_no_change(self):
        """
        Test that shifting hue by 0 degrees results in no change.
        """
        for color_space, hsl, hsv in [
            (ColorSpace.HSL, self.hsl_red, self.hsv_red),
            (ColorSpace.HSV, self.hsv_cyan, self.hsv_cyan)
        ]:
            enhancer = ColorEnhancer(color_space)
            shifted = enhancer.shift_hue(hsl if color_space == ColorSpace.HSL else hsv, 0.0)
            expected = hsl.clone() if color_space == ColorSpace.HSL else hsv.clone()
            self.assertTensorAlmostEqual(shifted, expected)

    def test_lighten_clamping(self):
        """
        Test that lighten method clamps the brightness correctly.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Lighten white by 0.5, should clamp to 1.0
        lightened_hsl = enhancer.lighten(self.hsl_white, 0.5)
        expected_hsl = self.hsl_white.clone()
        expected_hsl[..., 2] = 1.0  # Clamped
        self.assertTensorAlmostEqual(lightened_hsl, expected_hsl)

    def test_darken_clamping(self):
        """
        Test that darken method clamps the brightness correctly.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Darken black by 0.5, should remain 0.0
        darkened_hsv = enhancer.darken(self.hsv_black, 0.5)
        expected_hsv = self.hsv_black.clone()
        expected_hsv[..., 2] = 0.0  # Clamped
        self.assertTensorAlmostEqual(darkened_hsv, expected_hsv)

    def test_saturate_clamping(self):
        """
        Test that saturate method clamps the saturation correctly.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Saturate already fully saturated color
        saturated_hsv = enhancer.saturate(self.hsv_red, 0.5)
        expected_hsv = self.hsv_red.clone()
        expected_hsv[..., 1] = 1.0  # Clamped
        self.assertTensorAlmostEqual(saturated_hsv, expected_hsv)

    def test_desaturate_clamping(self):
        """
        Test that desaturate method clamps the saturation correctly.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Desaturate already desaturated color
        desaturated_hsl = enhancer.desaturate(self.hsl_black, 0.5)
        expected_hsl = self.hsl_black.clone()
        expected_hsl[..., 1] = 0.0  # Clamped
        self.assertTensorAlmostEqual(desaturated_hsl, expected_hsl)

    def test_full_2x2_image_adjustments_hsl(self):
        """
        Test adjustments on a full 2x2 image in HSL color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Define a 2x2 HSL image
        hsl_image = torch.tensor([
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # Top row: Black, White
                [[0.5, 1.0, 0.5], [0.0, 1.0, 0.5]]   # Bottom row: Cyan, Red
            ]
        ])  # Shape: (1, 2, 2, 3)

        # Lighten by 0.1
        lightened = enhancer.lighten(hsl_image, 0.1)
        expected_lightened = hsl_image.clone()
        expected_lightened[..., 2] += 0.1
        expected_lightened[..., 2] = torch.clamp(expected_lightened[..., 2], 0.0, 1.0)
        self.assertTensorAlmostEqual(lightened, expected_lightened)

        # Saturate by 0.2
        saturated = enhancer.saturate(hsl_image, 0.2)
        expected_saturated = hsl_image.clone()
        expected_saturated[..., 1] += 0.2
        expected_saturated[..., 1] = torch.clamp(expected_saturated[..., 1], 0.0, 1.0)
        self.assertTensorAlmostEqual(saturated, expected_saturated)

        # Shift hue by 90 degrees
        shifted = enhancer.shift_hue(hsl_image, 90.0)
        expected_shifted = hsl_image.clone()
        expected_shifted[..., 0] = torch.remainder(expected_shifted[..., 0] + (90.0 / 360.0), 1.0)
        self.assertTensorAlmostEqual(shifted, expected_shifted)

    def test_full_2x2_image_adjustments_hsv(self):
        """
        Test adjustments on a full 2x2 image in HSV color space.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Define a 2x2 HSV image
        hsv_image = torch.tensor([
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # Top row: Black, White
                [[0.5, 1.0, 1.0], [0.0, 1.0, 1.0]]   # Bottom row: Cyan, Red
            ]
        ])  # Shape: (1, 2, 2, 3)

        # Darken by 0.2
        darkened = enhancer.darken(hsv_image, 0.2)
        expected_darkened = hsv_image.clone()
        expected_darkened[..., 2] -= 0.2
        expected_darkened[..., 2] = torch.clamp(expected_darkened[..., 2], 0.0, 1.0)
        self.assertTensorAlmostEqual(darkened, expected_darkened)

        # Desaturate by 0.3
        desaturated = enhancer.desaturate(hsv_image, 0.3)
        expected_desaturated = hsv_image.clone()
        expected_desaturated[..., 1] -= 0.3
        expected_desaturated[..., 1] = torch.clamp(expected_desaturated[..., 1], 0.0, 1.0)
        self.assertTensorAlmostEqual(desaturated, expected_desaturated)

        # Adjust levels: shadows=0.1, gamma=1.0, highlights=0.9 on value channel (2)
        adjusted_levels = enhancer.adjust_levels(hsv_image, shadows=0.1, gamma=1.0, highlights=0.9, channel=2)
        # For each pixel:
        # New V = clamp((V - 0.1) / 0.8, 0.0, 1.0)
        # Since gamma=1.0, no change after scaling
        expected_levels = hsv_image.clone()
        expected_levels[..., 2] = torch.clamp((expected_levels[..., 2] - 0.1) / 0.8, 0.0, 1.0)
        self.assertTensorAlmostEqual(adjusted_levels, expected_levels)

    def test_shift_hue_multiple_wraps(self):
        """
        Test shifting hue by more than 360 degrees to ensure proper wrapping.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Shift hue of red by 720 degrees (2 full rotations)
        hsv_red = torch.tensor([[[[0.0, 1.0, 1.0]]]])  # Hue = 0.0
        shifted_hsv = enhancer.shift_hue(hsv_red, 720.0)
        expected_hsv = torch.tensor([[[[(0.0 + 720.0 / 360.0) % 1.0, 1.0, 1.0]]]])  # Should remain red
        self.assertTensorAlmostEqual(shifted_hsv, expected_hsv)

    def test_shift_hue_large_negative_shift(self):
        """
        Test shifting hue by a large negative degree to ensure proper wrapping.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Shift hue of cyan by -450 degrees (-1.25 rotations)
        hsv_cyan = torch.tensor([[[[0.5, 1.0, 1.0]]]])  # Hue = 0.5
        shifted_hsv = enhancer.shift_hue(hsv_cyan, -450.0)
        # -450 degrees = -1.25 rotations -> equivalent to shifting by +270 degrees
        # New hue = (0.5 + 270/360) % 1.0 = (0.5 + 0.75) % 1.0 = 0.25
        expected_hsv = torch.tensor([[[[(0.5 + (-450.0 / 360.0)) % 1.0, 1.0, 1.0]]]])  # 0.25
        self.assertTensorAlmostEqual(shifted_hsv, torch.tensor([[[[0.25, 1.0, 1.0]]]]))

    def test_adjust_levels_gamma_less_than_one(self):
        """
        Test adjust_levels with gamma < 1 (brighten midtones).
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Midtone lightness
        hsl = torch.tensor([[[[0.0, 0.5, 0.5]]]])  # Lightness = 0.5
        adjusted = enhancer.adjust_levels(hsl, shadows=0.0, gamma=0.5, highlights=1.0, channel=2)
        # New lightness = (0.5 - 0) / 1 * sqrt(0.5) = sqrt(0.5) ~ 0.7071
        expected = torch.tensor([[[[0.0, 0.5, torch.sqrt(torch.tensor(0.5))]]]])
        self.assertTensorAlmostEqual(adjusted, expected)

    def test_adjust_levels_gamma_greater_than_one(self):
        """
        Test adjust_levels with gamma > 1 (darken midtones).
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Midtone value
        hsv = torch.tensor([[[[0.0, 0.5, 0.5]]]])  # Value = 0.5
        adjusted = enhancer.adjust_levels(hsv, shadows=0.0, gamma=2.0, highlights=1.0, channel=2)
        # New value = (0.5 - 0) / 1 * (0.5)^2 = 0.25
        expected = torch.tensor([[[[0.0, 0.5, 0.25]]]])
        self.assertTensorAlmostEqual(adjusted, expected)

    def test_adjust_levels_full_range(self):
        """
        Test adjust_levels across the full range of brightness.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Create an image with varying lightness
        hsl = torch.tensor([
            [
                [[0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, 1.0]]
            ]
        ])  # Shape: (1, 1, 3, 3)

        # Adjust levels with shadows=0.0, gamma=1.0, highlights=1.0 (no change)
        adjusted = enhancer.adjust_levels(hsl, shadows=0.0, gamma=1.0, highlights=1.0, channel=2)
        expected = hsl.clone()
        self.assertTensorAlmostEqual(adjusted, expected)

    def test_adjust_levels_zero_magnitude(self):
        """
        Test adjust_levels when highlights and shadows are the same, expecting clamping to zero.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # shadows = highlights = 0.5
        hsl = torch.tensor([[[[0.0, 0.5, 0.5]]]])
        with self.assertRaises(ZeroDivisionError):
            enhancer.adjust_levels(hsl, shadows=0.5, gamma=1.0, highlights=0.5, channel=2)

    def test_adjust_levels_invalid_channel(self):
        """
        Test adjust_levels with an invalid channel index.
        """
        enhancer = ColorEnhancer(ColorSpace.HSL)

        # Attempt to adjust an invalid channel (e.g., channel=3)
        hsl = torch.tensor([[[[0.0, 0.5, 0.5]]]])
        with self.assertRaises(IndexError):
            enhancer.adjust_levels(hsl, shadows=0.0, gamma=1.0, highlights=1.0, channel=3)

    def test_large_image_performance(self):
        """
        Optional: Test the ColorEnhancer on a larger image tensor to ensure it handles larger data.
        Note: This is a basic performance test and does not assert correctness.
        """
        enhancer = ColorEnhancer(ColorSpace.HSV)

        # Create a large image tensor (e.g., 100x100 pixels)
        large_image = torch.rand((1, 100, 100, 3))  # Random HSV image

        try:
            # Perform multiple operations
            rgb = enhancer.to_rgb(large_image)
            hsv = enhancer.from_rgb(rgb)
            lightened = enhancer.lighten(large_image, 0.1)
            darkened = enhancer.darken(large_image, 0.1)
            saturated = enhancer.saturate(large_image, 0.1)
            desaturated = enhancer.desaturate(large_image, 0.1)
            shifted = enhancer.shift_hue(large_image, 45.0)
            adjusted = enhancer.adjust_levels(large_image, shadows=0.1, gamma=1.0, highlights=0.9, channel=2)
        except Exception as e:
            self.fail(f"ColorEnhancer failed on large image with exception: {e}")


if __name__ == '__main__':
    unittest.main()
