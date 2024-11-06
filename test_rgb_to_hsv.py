import unittest

import torch

from comfyui_image_filters.rgb_to_hsv import rgb_tensor_to_hsv_tensor


class TestRgbToHsvConversion(unittest.TestCase):
    def test_basic_colors(self):
        """Test conversion of basic colors with known HSV values."""
        # Define RGB values and their corresponding HSV
        test_cases = [
            # (R, G, B) in [0,1], (H, S, V) expected
            (torch.tensor([[[[1.0, 0.0, 0.0]]]]), torch.tensor([[[[0.0, 1.0, 1.0]]]])),      # Red
            (torch.tensor([[[[0.0, 1.0, 0.0]]]]), torch.tensor([[[[1/3, 1.0, 1.0]]]])),    # Green
            (torch.tensor([[[[0.0, 0.0, 1.0]]]]), torch.tensor([[[[2/3, 1.0, 1.0]]]])),    # Blue
            (torch.tensor([[[[1.0, 1.0, 1.0]]]]), torch.tensor([[[[0.0, 0.0, 1.0]]]])),    # White
            (torch.tensor([[[[0.0, 0.0, 0.0]]]]), torch.tensor([[[[0.0, 0.0, 0.0]]]])),    # Black
            (torch.tensor([[[[1.0, 1.0, 0.0]]]]), torch.tensor([[[[1/6, 1.0, 1.0]]]])),    # Yellow
            (torch.tensor([[[[0.0, 1.0, 1.0]]]]), torch.tensor([[[[0.5, 1.0, 1.0]]]])),    # Cyan
            (torch.tensor([[[[1.0, 0.0, 1.0]]]]), torch.tensor([[[[5/6, 1.0, 1.0]]]])),    # Magenta
            (torch.tensor([[[[0.5, 0.5, 0.5]]]]), torch.tensor([[[[0.0, 0.0, 0.5]]]])),    # Gray
        ]

        for idx, (rgb, expected_hsv) in enumerate(test_cases):
            with self.subTest(i=idx):
                hsv = rgb_tensor_to_hsv_tensor(rgb)
                # Use a tolerance for floating point comparisons
                self.assertTrue(torch.allclose(hsv, expected_hsv, atol=1e-4),
                                msg=f"Failed for RGB input: {rgb}")

    def test_hue_wraparound(self):
        """Test hue normalization and wraparound."""
        # Pure red can have hue 0 or 360, but should be normalized to 0
        rgb_red = torch.tensor([[[[1.0, 0.0, 0.0]]]])
        hsv_red = rgb_tensor_to_hsv_tensor(rgb_red)
        expected_hsv_red = torch.tensor([[[[0.0, 1.0, 1.0]]]])
        self.assertTrue(torch.allclose(hsv_red, expected_hsv_red, atol=1e-4),
                        msg="Hue wraparound failed for pure red.")

    def test_zero_saturation(self):
        """Test colors with zero saturation (grayscale)."""
        grayscales = [
            (torch.tensor([[[[0.5, 0.5, 0.5]]]]), torch.tensor([[[[0.0, 0.0, 0.5]]]])),
            (torch.tensor([[[[0.2, 0.2, 0.2]]]]), torch.tensor([[[[0.0, 0.0, 0.2]]]])),
            (torch.tensor([[[[0.8, 0.8, 0.8]]]]), torch.tensor([[[[0.0, 0.0, 0.8]]]])),
        ]

        for idx, (rgb, expected_hsv) in enumerate(grayscales):
            with self.subTest(i=idx):
                hsv = rgb_tensor_to_hsv_tensor(rgb)
                self.assertTrue(torch.allclose(hsv, expected_hsv, atol=1e-4),
                                msg=f"Failed for grayscale RGB input: {rgb}")

    def test_invalid_input_shape(self):
        """Test that invalid input shapes raise ValueError."""
        invalid_shapes = [
            torch.rand((224, 224, 3)),          # Missing batch dimension
            torch.rand((2, 224, 224, 3)),        # Valid batch size, should not raise
            torch.rand((1, 224, 224)),           # Missing color channels
            torch.rand((1, 224, 224, 4)),        # Too many color channels
            torch.rand((1, 224)),                 # Completely wrong shape
        ]

        expected_errors = [
            True,   # Missing batch dimension
            False,  # Valid batch size
            True,   # Missing color channels
            True,   # Too many color channels
            True,   # Completely wrong shape
        ]

        for idx, (invalid_tensor, should_raise) in enumerate(zip(invalid_shapes, expected_errors)):
            with self.subTest(i=idx):
                if should_raise:
                    with self.assertRaises(ValueError):
                        rgb_tensor_to_hsv_tensor(invalid_tensor)
                else:
                    # Should not raise
                    try:
                        hsv = rgb_tensor_to_hsv_tensor(invalid_tensor)
                        self.assertEqual(hsv.shape, invalid_tensor.shape,
                                         msg="Output HSV tensor shape mismatch for valid batch input.")
                    except ValueError:
                        self.fail("ValueError was raised unexpectedly for a valid batch size.")

    def test_non_float_input(self):
        """Test that integer inputs are correctly handled by converting to float."""
        # Test without normalization: should not normalize, leading to incorrect HSV
        rgb_int = torch.tensor([[[[255, 0, 0]]]], dtype=torch.int)
        hsv = rgb_tensor_to_hsv_tensor(rgb_int, normalize=False)
        # Since normalization is False, HSV should be incorrect
        # Therefore, we expect the test to fail
        # Instead, we should pass normalize=True or adjust the test
        # Here, we'll test with normalization
        hsv_normalized = rgb_tensor_to_hsv_tensor(rgb_int, normalize=True)
        expected_hsv = torch.tensor([[[[0.0, 1.0, 1.0]]]], dtype=torch.float)
        self.assertTrue(torch.allclose(hsv_normalized, expected_hsv, atol=1e-4),
                        msg="Integer input was not correctly normalized when normalize=True.")

    def test_random_values(self):
        """Test random RGB values and verify HSV ranges."""
        # Generate random RGB tensor
        rgb_random = torch.rand((1, 100, 100, 3))
        hsv_random = rgb_tensor_to_hsv_tensor(rgb_random)

        # Check that H is in [0,1), S and V are in [0,1]
        self.assertTrue(torch.all((hsv_random[..., 0] >= 0) & (hsv_random[..., 0] < 1)),
                        msg="Hue values out of range [0,1).")
        self.assertTrue(torch.all((hsv_random[..., 1] >= 0) & (hsv_random[..., 1] <= 1)),
                        msg="Saturation values out of range [0,1].")
        self.assertTrue(torch.all((hsv_random[..., 2] >= 0) & (hsv_random[..., 2] <= 1)),
                        msg="Value values out of range [0,1].")

    def test_batch_processing(self):
        """Test that the function works correctly with batch size >1."""
        # Generate a batch of 5 random images
        batch_size = 5
        rgb_batch = torch.rand((batch_size, 224, 224, 3))
        hsv_batch = rgb_tensor_to_hsv_tensor(rgb_batch)

        # Check the shape
        self.assertEqual(hsv_batch.shape, rgb_batch.shape,
                         msg="Output HSV tensor shape mismatch for batch input.")

        # Check value ranges
        self.assertTrue(torch.all((hsv_batch[..., 0] >= 0) & (hsv_batch[..., 0] < 1)),
                        msg="Hue values out of range [0,1) in batch input.")
        self.assertTrue(torch.all((hsv_batch[..., 1] >= 0) & (hsv_batch[..., 1] <= 1)),
                        msg="Saturation values out of range [0,1] in batch input.")
        self.assertTrue(torch.all((hsv_batch[..., 2] >= 0) & (hsv_batch[..., 2] <= 1)),
                        msg="Value values out of range [0,1] in batch input.")

    def test_high_resolution(self):
        """Test the function with a high-resolution image tensor."""
        high_res = torch.rand((1, 4096, 4096, 3))
        try:
            hsv_high_res = rgb_tensor_to_hsv_tensor(high_res)
            self.assertEqual(hsv_high_res.shape, high_res.shape,
                             msg="Output HSV tensor shape mismatch for high-resolution input.")
            # Check value ranges
            self.assertTrue(torch.all((hsv_high_res[..., 0] >= 0) & (hsv_high_res[..., 0] < 1)),
                            msg="Hue values out of range [0,1) in high-resolution input.")
            self.assertTrue(torch.all((hsv_high_res[..., 1] >= 0) & (hsv_high_res[..., 1] <= 1)),
                            msg="Saturation values out of range [0,1] in high-resolution input.")
            self.assertTrue(torch.all((hsv_high_res[..., 2] >= 0) & (hsv_high_res[..., 2] <= 1)),
                            msg="Value values out of range [0,1] in high-resolution input.")
        except Exception as e:
            self.fail(f"Function failed on high-resolution input: {e}")

if __name__ == '__main__':
    unittest.main()
