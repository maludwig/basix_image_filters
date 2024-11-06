import unittest

import torch

from basix_image_filters.hsv_to_rgb import hsv_tensor_to_rgb_tensor
from basix_image_filters.rgb_to_hsl import rgb_tensor_to_hsl_tensor

import unittest

import torch

from basix_image_filters.hsv_to_rgb import hsv_tensor_to_rgb_tensor
from basix_image_filters.rgb_to_hsv import rgb_tensor_to_hsv_tensor


class TestHSVToRGBColorConversion(unittest.TestCase):
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
        Test HSV to RGB conversion for a single black pixel.
        HSV: (0, 0, 0) => RGB: (0, 0, 0)
        """
        hsv = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # Shape: (1, 1, 1, 3)
        expected_rgb = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # Black

        rgb = hsv_tensor_to_rgb_tensor(hsv)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_single_pixel_white(self):
        """
        Test HSV to RGB conversion for a single white pixel.
        HSV: (0, 0, 1) => RGB: (1, 1, 1)
        """
        hsv = torch.tensor([[[[0.0, 0.0, 1.0]]]])  # Shape: (1, 1, 1, 3)
        expected_rgb = torch.tensor([[[[1.0, 1.0, 1.0]]]])  # White

        rgb = hsv_tensor_to_rgb_tensor(hsv)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_single_pixel_red(self):
        """
        Test HSV to RGB conversion for a single red pixel.
        HSV: (0, 1, 1) => RGB: (1, 0, 0)
        """
        hsv = torch.tensor([[[[0.0, 1.0, 1.0]]]])  # Shape: (1, 1, 1, 3)
        expected_rgb = torch.tensor([[[[1.0, 0.0, 0.0]]]])  # Red

        rgb = hsv_tensor_to_rgb_tensor(hsv)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_single_pixel_cyan(self):
        """
        Test HSV to RGB conversion for a single cyan pixel.
        HSV: (0.5, 1, 1) => RGB: (0, 1, 1)
        """
        hsv = torch.tensor([[[[0.5, 1.0, 1.0]]]])  # Shape: (1, 1, 1, 3)
        expected_rgb = torch.tensor([[[[0.0, 1.0, 1.0]]]])  # Cyan

        rgb = hsv_tensor_to_rgb_tensor(hsv)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_batch_single_pixels(self):
        """
        Test HSV to RGB conversion for a batch of single pixels: black, white, red, cyan.
        """
        hsv = torch.tensor(
            [[[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 1.0]]], [[[0.0, 1.0, 1.0]]], [[[0.5, 1.0, 1.0]]]]  # Black  # White  # Red  # Cyan
        )  # Shape: (4, 1, 1, 3)

        expected_rgb = torch.tensor(
            [[[[0.0, 0.0, 0.0]]], [[[1.0, 1.0, 1.0]]], [[[1.0, 0.0, 0.0]]], [[[0.0, 1.0, 1.0]]]]  # Black  # White  # Red  # Cyan
        )  # Shape: (4, 1, 1, 3)

        rgb = hsv_tensor_to_rgb_tensor(hsv)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_2x2_full_black(self):
        """
        Test HSV to RGB conversion for a 2x2 full black image.
        """
        hsv = torch.zeros((1, 2, 2, 3))  # All pixels HSV: (0, 0, 0)
        expected_rgb = torch.zeros((1, 2, 2, 3))  # All pixels RGB: (0, 0, 0)

        rgb = hsv_tensor_to_rgb_tensor(hsv)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_2x2_full_white(self):
        """
        Test HSV to RGB conversion for a 2x2 full white image.
        """
        hsv = torch.zeros((1, 2, 2, 3))
        hsv[..., 0] = 0.0  # Hue irrelevant
        hsv[..., 1] = 0.0  # Saturation
        hsv[..., 2] = 1.0  # Value
        expected_rgb = torch.ones((1, 2, 2, 3))  # All pixels RGB: (1, 1, 1)

        rgb = hsv_tensor_to_rgb_tensor(hsv)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_2x2_full_red(self):
        """
        Test HSV to RGB conversion for a 2x2 full red image.
        """
        hsv = torch.zeros((1, 2, 2, 3))
        hsv[..., 0] = 0.0  # Hue = 0 (Red)
        hsv[..., 1] = 1.0  # Saturation = 1
        hsv[..., 2] = 1.0  # Value = 1
        expected_rgb = torch.zeros((1, 2, 2, 3))
        expected_rgb[..., 0] = 1.0  # Red channel
        expected_rgb[..., 1] = 0.0  # Green channel
        expected_rgb[..., 2] = 0.0  # Blue channel

        rgb = hsv_tensor_to_rgb_tensor(hsv)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_2x2_mixed_pixels(self):
        """
        Test HSV to RGB conversion for a 2x2 image with:
        - Top Left: Black
        - Top Right: White
        - Bottom Left: Cyan
        - Bottom Right: Red
        """
        hsv = torch.zeros((1, 2, 2, 3))

        # Top Left: Black (0, 0, 0)
        hsv[0, 0, 0, :] = torch.tensor([0.0, 0.0, 0.0])

        # Top Right: White (0, 0, 1)
        hsv[0, 0, 1, :] = torch.tensor([0.0, 0.0, 1.0])

        # Bottom Left: Cyan (0.5, 1, 1)
        hsv[0, 1, 0, :] = torch.tensor([0.5, 1.0, 1.0])

        # Bottom Right: Red (0, 1, 1)
        hsv[0, 1, 1, :] = torch.tensor([0.0, 1.0, 1.0])

        expected_rgb = torch.zeros((1, 2, 2, 3))
        # Top Left: Black (0,0,0)
        expected_rgb[0, 0, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        # Top Right: White (1,1,1)
        expected_rgb[0, 0, 1, :] = torch.tensor([1.0, 1.0, 1.0])
        # Bottom Left: Cyan (0,1,1)
        expected_rgb[0, 1, 0, :] = torch.tensor([0.0, 1.0, 1.0])
        # Bottom Right: Red (1,0,0)
        expected_rgb[0, 1, 1, :] = torch.tensor([1.0, 0.0, 0.0])

        rgb = hsv_tensor_to_rgb_tensor(hsv)
        self.assertTensorAlmostEqual(rgb, expected_rgb)

    def test_rgb_to_hsv_black(self):
        """
        Test RGB to HSV conversion for a single black pixel.
        RGB: (0, 0, 0) => HSV: (0, 0, 0)
        """
        rgb = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # Shape: (1, 1, 1, 3)
        expected_hsv = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # Black

        hsv = rgb_tensor_to_hsv_tensor(rgb)
        self.assertTensorAlmostEqual(hsv, expected_hsv)

    def test_rgb_to_hsv_white(self):
        """
        Test RGB to HSV conversion for a single white pixel.
        RGB: (1, 1, 1) => HSV: (0, 0, 1)
        """
        rgb = torch.tensor([[[[1.0, 1.0, 1.0]]]])  # Shape: (1, 1, 1, 3)
        expected_hsv = torch.tensor([[[[0.0, 0.0, 1.0]]]])  # White

        hsv = rgb_tensor_to_hsv_tensor(rgb)
        self.assertTensorAlmostEqual(hsv, expected_hsv)

    def test_rgb_to_hsv_red(self):
        """
        Test RGB to HSV conversion for a single red pixel.
        RGB: (1, 0, 0) => HSV: (0, 1, 1)
        """
        rgb = torch.tensor([[[[1.0, 0.0, 0.0]]]])  # Shape: (1, 1, 1, 3)
        expected_hsv = torch.tensor([[[[0.0, 1.0, 1.0]]]])  # Red

        hsv = rgb_tensor_to_hsv_tensor(rgb)
        self.assertTensorAlmostEqual(hsv, expected_hsv)

    def test_rgb_to_hsv_cyan(self):
        """
        Test RGB to HSV conversion for a single cyan pixel.
        RGB: (0, 1, 1) => HSV: (0.5, 1, 1)
        """
        rgb = torch.tensor([[[[0.0, 1.0, 1.0]]]])  # Shape: (1, 1, 1, 3)
        expected_hsv = torch.tensor([[[[0.5, 1.0, 1.0]]]])  # Cyan

        hsv = rgb_tensor_to_hsv_tensor(rgb)
        self.assertTensorAlmostEqual(hsv, expected_hsv)

    def test_rgb_to_hsv_batch_single_pixels(self):
        """
        Test RGB to HSV conversion for a batch of single pixels: black, white, red, cyan.
        """
        rgb = torch.tensor(
            [[[[0.0, 0.0, 0.0]]], [[[1.0, 1.0, 1.0]]], [[[1.0, 0.0, 0.0]]], [[[0.0, 1.0, 1.0]]]]  # Black  # White  # Red  # Cyan
        )  # Shape: (4, 1, 1, 3)

        expected_hsv = torch.tensor(
            [[[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 1.0]]], [[[0.0, 1.0, 1.0]]], [[[0.5, 1.0, 1.0]]]]  # Black  # White  # Red  # Cyan
        )  # Shape: (4, 1, 1, 3)

        hsv = rgb_tensor_to_hsv_tensor(rgb)
        self.assertTensorAlmostEqual(hsv, expected_hsv)

    def test_rgb_to_hsv_2x2_mixed_pixels(self):
        """
        Test RGB to HSV conversion for a 2x2 image with:
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

        expected_hsv = torch.zeros((1, 2, 2, 3))
        # Top Left: Black (0,0,0)
        expected_hsv[0, 0, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        # Top Right: White (0,0,1)
        expected_hsv[0, 0, 1, :] = torch.tensor([0.0, 0.0, 1.0])
        # Bottom Left: Cyan (0.5,1,1)
        expected_hsv[0, 1, 0, :] = torch.tensor([0.5, 1.0, 1.0])
        # Bottom Right: Red (0,1,1)
        expected_hsv[0, 1, 1, :] = torch.tensor([0.0, 1.0, 1.0])

        hsv = rgb_tensor_to_hsv_tensor(rgb)
        self.assertTensorAlmostEqual(hsv, expected_hsv)


if __name__ == "__main__":
    unittest.main()
