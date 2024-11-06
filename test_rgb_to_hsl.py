import unittest

import torch

from comfyui_image_filters.rgb_to_hsl import rgb_tensor_to_hsl_tensor


class TestRgbToHslConversion(unittest.TestCase):
    def test_basic_colors(self):
        """Test conversion of basic colors with known HSL values."""
        # Define RGB values and their corresponding HSL
        test_cases = [
            # (R, G, B) in [0,1], (H, S, L) expected
            (torch.tensor([[[[1.0, 0.0, 0.0]]]]), torch.tensor([[[[0.0, 1.0, 0.5]]]])),      # Red
            (torch.tensor([[[[0.0, 1.0, 0.0]]]]), torch.tensor([[[[1/3, 1.0, 0.5]]]])),    # Green
            (torch.tensor([[[[0.0, 0.0, 1.0]]]]), torch.tensor([[[[2/3, 1.0, 0.5]]]])),    # Blue
            (torch.tensor([[[[1.0, 1.0, 1.0]]]]), torch.tensor([[[[0.0, 0.0, 1.0]]]])),    # White
            (torch.tensor([[[[0.0, 0.0, 0.0]]]]), torch.tensor([[[[0.0, 0.0, 0.0]]]])),    # Black
            (torch.tensor([[[[1.0, 1.0, 0.0]]]]), torch.tensor([[[[1/6, 1.0, 0.5]]]])),    # Yellow
            (torch.tensor([[[[0.0, 1.0, 1.0]]]]), torch.tensor([[[[0.5, 1.0, 0.5]]]])),    # Cyan
            (torch.tensor([[[[1.0, 0.0, 1.0]]]]), torch.tensor([[[[5/6, 1.0, 0.5]]]])),    # Magenta
            (torch.tensor([[[[0.5, 0.5, 0.5]]]]), torch.tensor([[[[0.0, 0.0, 0.5]]]])),    # Gray
        ]

        for idx, (rgb, expected_hsl) in enumerate(test_cases):
            with self.subTest(i=idx):
                hsl = rgb_tensor_to_hsl_tensor(rgb)
                # Use a tolerance for floating point comparisons
                self.assertTrue(torch.allclose(hsl, expected_hsl, atol=1e-4),
                                msg=f"Failed for RGB input: {rgb}")

    def test_hue_wraparound(self):
        """Test hue normalization and wraparound."""
        # Pure red can have hue 0 or 360, but should be normalized to 0
        rgb_red = torch.tensor([[[[1.0, 0.0, 0.0]]]])
        hsl_red = rgb_tensor_to_hsl_tensor(rgb_red)
        expected_hsl_red = torch.tensor([[[[0.0, 1.0, 0.5]]]])
        self.assertTrue(torch.allclose(hsl_red, expected_hsl_red, atol=1e-4),
                        msg="Hue wraparound failed for pure red.")

    def test_zero_saturation(self):
        """Test colors with zero saturation (grayscale)."""
        grayscales = [
            (torch.tensor([[[[0.5, 0.5, 0.5]]]]), torch.tensor([[[[0.0, 0.0, 0.5]]]])),
            (torch.tensor([[[[0.2, 0.2, 0.2]]]]), torch.tensor([[[[0.0, 0.0, 0.2]]]])),
            (torch.tensor([[[[0.8, 0.8, 0.8]]]]), torch.tensor([[[[0.0, 0.0, 0.8]]]])),
        ]

        for idx, (rgb, expected_hsl) in enumerate(grayscales):
            with self.subTest(i=idx):
                hsl = rgb_tensor_to_hsl_tensor(rgb)
                self.assertTrue(torch.allclose(hsl, expected_hsl, atol=1e-4),
                                msg=f"Failed for grayscale RGB input: {rgb}")

    def test_non_float_input(self):
        """Test that integer inputs are correctly handled by converting to float."""
        # Test without normalization: should not normalize, leading to incorrect HSL
        rgb_int = torch.tensor([[[[255, 0, 0]]]], dtype=torch.int)
        hsl = rgb_tensor_to_hsl_tensor(rgb_int, normalize=False)
        # Since normalization is False, HSL should be incorrect
        # Therefore, we should pass normalize=True or adjust the test
        # Here, we'll test with normalization
        hsl_normalized = rgb_tensor_to_hsl_tensor(rgb_int, normalize=True)
        expected_hsl = torch.tensor([[[[0.0, 1.0, 0.5]]]], dtype=torch.float)
        self.assertTrue(torch.allclose(hsl_normalized, expected_hsl, atol=1e-4),
                        msg="Integer input was not correctly normalized when normalize=True.")

    def test_random_values(self):
        """Test random RGB values and verify HSL ranges."""
        # Generate random RGB tensor
        rgb_random = torch.rand((1, 100, 100, 3))
        hsl_random = rgb_tensor_to_hsl_tensor(rgb_random)

        # Check that H is in [0,1), S and L are in [0,1]
        self.assertTrue(torch.all((hsl_random[..., 0] >= 0) & (hsl_random[..., 0] < 1)),
                        msg="Hue values out of range [0,1).")
        self.assertTrue(torch.all((hsl_random[..., 1] >= 0) & (hsl_random[..., 1] <= 1)),
                        msg="Saturation values out of range [0,1].")
        self.assertTrue(torch.all((hsl_random[..., 2] >= 0) & (hsl_random[..., 2] <= 1)),
                        msg="Lightness values out of range [0,1].")

    def test_batch_processing(self):
        """Test that the function works correctly with batch size >1."""
        # Generate a batch of 5 random images
        batch_size = 5
        rgb_batch = torch.rand((batch_size, 224, 224, 3))
        hsl_batch = rgb_tensor_to_hsl_tensor(rgb_batch)

        # Check the shape
        self.assertEqual(hsl_batch.shape, rgb_batch.shape,
                         msg="Output HSL tensor shape mismatch for batch input.")

        # Check value ranges
        self.assertTrue(torch.all((hsl_batch[..., 0] >= 0) & (hsl_batch[..., 0] < 1)),
                        msg="Hue values out of range [0,1) in batch input.")
        self.assertTrue(torch.all((hsl_batch[..., 1] >= 0) & (hsl_batch[..., 1] <= 1)),
                        msg="Saturation values out of range [0,1] in batch input.")
        self.assertTrue(torch.all((hsl_batch[..., 2] >= 0) & (hsl_batch[..., 2] <= 1)),
                        msg="Lightness values out of range [0,1] in batch input.")

    def test_high_resolution(self):
        """Test the function with a high-resolution image tensor."""
        high_res = torch.rand((1, 4096, 4096, 3))
        try:
            hsl_high_res = rgb_tensor_to_hsl_tensor(high_res)
            self.assertEqual(hsl_high_res.shape, high_res.shape,
                             msg="Output HSL tensor shape mismatch for high-resolution input.")
            # Check value ranges
            self.assertTrue(torch.all((hsl_high_res[..., 0] >= 0) & (hsl_high_res[..., 0] < 1)),
                            msg="Hue values out of range [0,1) in high-resolution input.")
            self.assertTrue(torch.all((hsl_high_res[..., 1] >= 0) & (hsl_high_res[..., 1] <= 1)),
                            msg="Saturation values out of range [0,1] in high-resolution input.")
            self.assertTrue(torch.all((hsl_high_res[..., 2] >= 0) & (hsl_high_res[..., 2] <= 1)),
                            msg="Lightness values out of range [0,1] in high-resolution input.")
        except Exception as e:
            self.fail(f"Function failed on high-resolution input: {e}")

if __name__ == '__main__':
    unittest.main()
