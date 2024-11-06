import unittest

import torch

from basix_image_filters.hsl_to_rgb import hsl_tensor_to_rgb_tensor
from basix_image_filters.rgb_to_hsl import rgb_tensor_to_hsl_tensor


class TestHSLToRGBColorConversion(unittest.TestCase):
    def setUp(self):
        """
        Set up any required variables for the tests.
        """
        pass

    def assertTensorAlmostEqual(self, tensor1, tensor2, tol=1e-4):
        """
        Helper method to assert that two tensors are almost equal.
        """
        self.assertTrue(torch.allclose(tensor1, tensor2, atol=tol), msg=f"Tensors are not almost equal.\nTensor1: {tensor1}\nTensor2: {tensor2}")

    def test_single_pixel_black(self):
        """
        Test HSL to RGB conversion for a single black pixel.
        HSL: (0, 0, 0) => RGB: (0, 0, 0)
        """
        hsl = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # Shape: (1, 1, 1, 3)
        expected_rgb = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # Black

        rgb = hsl_tensor_to_rgb_tensor(hsl)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_single_pixel_white(self):
        """
        Test HSL to RGB conversion for a single white pixel.
        HSL: (0, 0, 1) => RGB: (1, 1, 1)
        """
        hsl = torch.tensor([[[[0.0, 0.0, 1.0]]]])  # Shape: (1, 1, 1, 3)
        expected_rgb = torch.tensor([[[[1.0, 1.0, 1.0]]]])  # White

        rgb = hsl_tensor_to_rgb_tensor(hsl)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_single_pixel_red(self):
        """
        Test HSL to RGB conversion for a single red pixel.
        HSL: (0, 1, 0.5) => RGB: (1, 0, 0)
        """
        hsl = torch.tensor([[[[0.0, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        expected_rgb = torch.tensor([[[[1.0, 0.0, 0.0]]]])  # Red

        rgb = hsl_tensor_to_rgb_tensor(hsl)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_single_pixel_cyan(self):
        """
        Test HSL to RGB conversion for a single cyan pixel.
        HSL: (0.5, 1, 0.5) => RGB: (0, 1, 1)
        """
        hsl = torch.tensor([[[[0.5, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        expected_rgb = torch.tensor([[[[0.0, 1.0, 1.0]]]])  # Cyan

        rgb = hsl_tensor_to_rgb_tensor(hsl)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_batch_single_pixels(self):
        """
        Test HSL to RGB conversion for a batch of single pixels: black, white, red, cyan.
        """
        hsl = torch.tensor(
            [[[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 1.0]]], [[[0.0, 1.0, 0.5]]], [[[0.5, 1.0, 0.5]]]]  # Black  # White  # Red  # Cyan
        )  # Shape: (4, 1, 1, 3)

        expected_rgb = torch.tensor(
            [[[[0.0, 0.0, 0.0]]], [[[1.0, 1.0, 1.0]]], [[[1.0, 0.0, 0.0]]], [[[0.0, 1.0, 1.0]]]]  # Black  # White  # Red  # Cyan
        )  # Shape: (4, 1, 1, 3)

        rgb = hsl_tensor_to_rgb_tensor(hsl)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_2x2_full_black(self):
        """
        Test HSL to RGB conversion for a 2x2 full black image.
        """
        hsl = torch.zeros((1, 2, 2, 3))  # All pixels HSL: (0, 0, 0)
        expected_rgb = torch.zeros((1, 2, 2, 3))  # All pixels RGB: (0, 0, 0)

        rgb = hsl_tensor_to_rgb_tensor(hsl)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_2x2_full_white(self):
        """
        Test HSL to RGB conversion for a 2x2 full white image.
        """
        hsl = torch.zeros((1, 2, 2, 3))
        hsl[..., 0] = 0.0  # Hue irrelevant
        hsl[..., 1] = 0.0  # Saturation
        hsl[..., 2] = 1.0  # Lightness
        expected_rgb = torch.ones((1, 2, 2, 3))  # All pixels RGB: (1, 1, 1)

        rgb = hsl_tensor_to_rgb_tensor(hsl)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_2x2_full_red(self):
        """
        Test HSL to RGB conversion for a 2x2 full red image.
        """
        hsl = torch.zeros((1, 2, 2, 3))
        hsl[..., 0] = 0.0  # Hue = 0 (Red)
        hsl[..., 1] = 1.0  # Saturation = 1
        hsl[..., 2] = 0.5  # Lightness = 0.5
        expected_rgb = torch.zeros((1, 2, 2, 3))
        expected_rgb[..., 0] = 1.0  # Red channel
        expected_rgb[..., 1] = 0.0  # Green channel
        expected_rgb[..., 2] = 0.0  # Blue channel

        rgb = hsl_tensor_to_rgb_tensor(hsl)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_2x2_mixed_pixels(self):
        """
        Test HSL to RGB conversion for a 2x2 image with:
        - Top Left: Black
        - Top Right: White
        - Bottom Left: Cyan
        - Bottom Right: Red
        """
        hsl = torch.zeros((1, 2, 2, 3))

        # Top Left: Black (0, 0, 0)
        hsl[0, 0, 0, :] = torch.tensor([0.0, 0.0, 0.0])

        # Top Right: White (0, 0, 1)
        hsl[0, 0, 1, :] = torch.tensor([0.0, 0.0, 1.0])

        # Bottom Left: Cyan (0.5, 1, 0.5)
        hsl[0, 1, 0, :] = torch.tensor([0.5, 1.0, 0.5])

        # Bottom Right: Red (0, 1, 0.5)
        hsl[0, 1, 1, :] = torch.tensor([0.0, 1.0, 0.5])

        expected_rgb = torch.zeros((1, 2, 2, 3))
        # Top Left: Black (0,0,0)
        expected_rgb[0, 0, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        # Top Right: White (1,1,1)
        expected_rgb[0, 0, 1, :] = torch.tensor([1.0, 1.0, 1.0])
        # Bottom Left: Cyan (0,1,1)
        expected_rgb[0, 1, 0, :] = torch.tensor([0.0, 1.0, 1.0])
        # Bottom Right: Red (1,0,0)
        expected_rgb[0, 1, 1, :] = torch.tensor([1.0, 0.0, 0.0])

        rgb = hsl_tensor_to_rgb_tensor(hsl)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_rgb_to_hsl_black(self):
        """
        Test RGB to HSL conversion for a single black pixel.
        RGB: (0, 0, 0) => HSL: (0, 0, 0)
        """
        rgb = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # Shape: (1, 1, 1, 3)
        expected_hsl = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # Black

        hsl = rgb_tensor_to_hsl_tensor(rgb)
        self.assertTensorAlmostEqual(hsl, expected_hsl)

    def test_rgb_to_hsl_white(self):
        """
        Test RGB to HSL conversion for a single white pixel.
        RGB: (1, 1, 1) => HSL: (0, 0, 1)
        """
        rgb = torch.tensor([[[[1.0, 1.0, 1.0]]]])  # Shape: (1, 1, 1, 3)
        expected_hsl = torch.tensor([[[[0.0, 0.0, 1.0]]]])  # White

        hsl = rgb_tensor_to_hsl_tensor(rgb)
        self.assertTensorAlmostEqual(hsl, expected_hsl)

    def test_rgb_to_hsl_red(self):
        """
        Test RGB to HSL conversion for a single red pixel.
        RGB: (1, 0, 0) => HSL: (0, 1, 0.5)
        """
        rgb = torch.tensor([[[[1.0, 0.0, 0.0]]]])  # Shape: (1, 1, 1, 3)
        expected_hsl = torch.tensor([[[[0.0, 1.0, 0.5]]]])  # Red

        hsl = rgb_tensor_to_hsl_tensor(rgb)
        self.assertTensorAlmostEqual(hsl, expected_hsl)

    def test_rgb_to_hsl_cyan(self):
        """
        Test RGB to HSL conversion for a single cyan pixel.
        RGB: (0, 1, 1) => HSL: (0.5, 1, 0.5)
        """
        rgb = torch.tensor([[[[0.0, 1.0, 1.0]]]])  # Shape: (1, 1, 1, 3)
        expected_hsl = torch.tensor([[[[0.5, 1.0, 0.5]]]])  # Cyan

        hsl = rgb_tensor_to_hsl_tensor(rgb)
        self.assertTensorAlmostEqual(hsl, expected_hsl)

    def test_rgb_to_hsl_batch_single_pixels(self):
        """
        Test RGB to HSL conversion for a batch of single pixels: black, white, red, cyan.
        """
        rgb = torch.tensor(
            [[[[0.0, 0.0, 0.0]]], [[[1.0, 1.0, 1.0]]], [[[1.0, 0.0, 0.0]]], [[[0.0, 1.0, 1.0]]]]  # Black  # White  # Red  # Cyan
        )  # Shape: (4, 1, 1, 3)

        expected_hsl = torch.tensor(
            [[[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 1.0]]], [[[0.0, 1.0, 0.5]]], [[[0.5, 1.0, 0.5]]]]  # Black  # White  # Red  # Cyan
        )  # Shape: (4, 1, 1, 3)

        hsl = rgb_tensor_to_hsl_tensor(rgb)
        self.assertTensorAlmostEqual(hsl, expected_hsl)

    def test_rgb_to_hsl_2x2_mixed_pixels(self):
        """
        Test RGB to HSL conversion for a 2x2 image with:
        - Top Left: Black
        - Top Right: White
        - Bottom Left: Cyan
        - Bottom Right: Red
        """
        rgb = torch.zeros((1, 2, 2, 3))
        # Top Left: Black (0,0,0)
        rgb[0, 0, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        # Top Right: White (1,1,1)
        rgb[0, 0, 1, :] = torch.tensor([1.0, 1.0, 1.0])
        # Bottom Left: Cyan (0,1,1)
        rgb[0, 1, 0, :] = torch.tensor([0.0, 1.0, 1.0])
        # Bottom Right: Red (1,0,0)
        rgb[0, 1, 1, :] = torch.tensor([1.0, 0.0, 0.0])

        expected_hsl = torch.zeros((1, 2, 2, 3))
        # Top Left: Black (0,0,0)
        expected_hsl[0, 0, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        # Top Right: White (0,0,1)
        expected_hsl[0, 0, 1, :] = torch.tensor([0.0, 0.0, 1.0])
        # Bottom Left: Cyan (0.5,1,0.5)
        expected_hsl[0, 1, 0, :] = torch.tensor([0.5, 1.0, 0.5])
        # Bottom Right: Red (0,1,0.5)
        expected_hsl[0, 1, 1, :] = torch.tensor([0.0, 1.0, 0.5])

        hsl = rgb_tensor_to_hsl_tensor(rgb)
        self.assertTensorAlmostEqual(hsl, expected_hsl)


if __name__ == "__main__":
    unittest.main()
