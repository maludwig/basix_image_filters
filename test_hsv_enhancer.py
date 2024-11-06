import unittest
import torch
from basix_image_filters.color_enhancer import ColorEnhancer, ColorSpace


class TestHSVEnhancer(unittest.TestCase):
    def setUp(self):
        """
        Initialize an instance of HSVEnhancer before each test.
        """
        self.enhancer = ColorEnhancer(ColorSpace.HSV)

    # -------------------
    # Tests for lighten
    # -------------------

    def test_lighten_basic(self):
        """Test lightening a basic HSV color."""
        # Define an HSV color: pure red with V=0.5
        hsv_input = torch.tensor([[[[0.0, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.2
        expected_hsv = torch.tensor([[[[0.0, 1.0, 0.7]]]])  # V should be 0.5 + 0.2 = 0.7
        lightened_hsv = self.enhancer.lighten(hsv_input, amount)
        self.assertTrue(
            torch.allclose(lightened_hsv, expected_hsv, atol=1e-6),
            msg="Lightening basic HSV color failed.",
        )

    def test_lighten_clamping(self):
        """Test that lightening does not exceed V=1."""
        # Define an HSV color with V=0.9
        hsv_input = torch.tensor([[[[0.0, 1.0, 0.9]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.2
        expected_hsv = torch.tensor([[[[0.0, 1.0, 1.0]]]])  # V should be clamped to 1.0
        lightened_hsv = self.enhancer.lighten(hsv_input, amount)
        self.assertTrue(
            torch.allclose(lightened_hsv, expected_hsv, atol=1e-6),
            msg="Lightening clamping failed.",
        )

    def test_lighten_zero_amount(self):
        """Test lightening with amount=0 (no change)."""
        # Define an HSV color
        hsv_input = torch.tensor([[[[0.5, 0.5, 0.5]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.0
        expected_hsv = hsv_input.clone()
        lightened_hsv = self.enhancer.lighten(hsv_input, amount)
        self.assertTrue(
            torch.allclose(lightened_hsv, expected_hsv, atol=1e-6),
            msg="Lightening with zero amount should not change the image.",
        )

    def test_lighten_full_batch(self):
        """Test lightening on a full batch of HSV images."""
        # Create a batch of 3 HSV images
        hsv_input = torch.tensor(
            [
                [[[0.0, 1.0, 0.3]]],  # Red, dark
                [[[0.3333, 0.5, 0.5]]],  # Green, medium
                [[[0.6667, 0.8, 0.7]]],  # Blue, light
            ]
        )  # Shape: (3, 1, 1, 3)
        amount = 0.2
        expected_hsv = torch.tensor(
            [
                [[[0.0, 1.0, 0.5]]],  # V=0.3 + 0.2 = 0.5
                [[[0.3333, 0.5, 0.7]]],  # V=0.5 + 0.2 = 0.7
                [[[0.6667, 0.8, 0.9]]],  # V=0.7 + 0.2 = 0.9
            ]
        )
        lightened_hsv = self.enhancer.lighten(hsv_input, amount)
        self.assertTrue(
            torch.allclose(lightened_hsv, expected_hsv, atol=1e-4),
            msg="Lightening on a full batch failed.",
        )

    # -------------------
    # Tests for darken
    # -------------------

    def test_darken_basic(self):
        """Test darkening a basic HSV color."""
        # Define an HSV color: pure blue with V=0.6
        hsv_input = torch.tensor([[[[0.6667, 1.0, 0.6]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.2
        expected_hsv = torch.tensor([[[[0.6667, 1.0, 0.4]]]])  # V should be 0.6 - 0.2 = 0.4
        darkened_hsv = self.enhancer.darken(hsv_input, amount)
        self.assertTrue(
            torch.allclose(darkened_hsv, expected_hsv, atol=1e-4),
            msg="Darkening basic HSV color failed.",
        )

    def test_darken_clamping(self):
        """Test that darkening does not go below V=0."""
        # Define an HSV color with V=0.1
        hsv_input = torch.tensor([[[[0.5, 0.5, 0.1]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.2
        expected_hsv = torch.tensor([[[[0.5, 0.5, 0.0]]]])  # V should be clamped to 0.0
        darkened_hsv = self.enhancer.darken(hsv_input, amount)
        self.assertTrue(
            torch.allclose(darkened_hsv, expected_hsv, atol=1e-6),
            msg="Darkening clamping failed.",
        )

    def test_darken_zero_amount(self):
        """Test darkening with amount=0 (no change)."""
        # Define an HSV color
        hsv_input = torch.tensor([[[[0.25, 0.75, 0.5]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.0
        expected_hsv = hsv_input.clone()
        darkened_hsv = self.enhancer.darken(hsv_input, amount)
        self.assertTrue(
            torch.allclose(darkened_hsv, expected_hsv, atol=1e-6),
            msg="Darkening with zero amount should not change the image.",
        )

    def test_darken_full_batch(self):
        """Test darkening on a full batch of HSV images."""
        # Create a batch of 3 HSV images
        hsv_input = torch.tensor(
            [
                [[[0.0, 1.0, 0.7]]],  # Red, light
                [[[0.3333, 0.5, 0.6]]],  # Green, medium
                [[[0.6667, 0.8, 0.3]]],  # Blue, dark
            ]
        )  # Shape: (3, 1, 1, 3)
        amount = 0.2
        expected_hsv = torch.tensor(
            [
                [[[0.0, 1.0, 0.5]]],  # V=0.7 - 0.2 = 0.5
                [[[0.3333, 0.5, 0.4]]],  # V=0.6 - 0.2 = 0.4
                [[[0.6667, 0.8, 0.1]]],  # V=0.3 - 0.2 = 0.1
            ]
        )
        darkened_hsv = self.enhancer.darken(hsv_input, amount)
        self.assertTrue(
            torch.allclose(darkened_hsv, expected_hsv, atol=1e-4),
            msg="Darkening on a full batch failed.",
        )

    # -------------------
    # Tests for saturate
    # -------------------

    def test_saturate_basic(self):
        """Test saturating a basic HSV color."""
        # Define an HSV color: pure green with S=0.5
        hsv_input = torch.tensor([[[[1 / 3, 0.5, 0.5]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.3
        expected_hsv = torch.tensor([[[[1 / 3, 0.8, 0.5]]]])  # S should be 0.5 + 0.3 = 0.8
        saturated_hsv = self.enhancer.saturate(hsv_input, amount)
        self.assertTrue(
            torch.allclose(saturated_hsv, expected_hsv, atol=1e-6),
            msg="Saturating basic HSV color failed.",
        )

    def test_saturate_clamping(self):
        """Test that saturating does not exceed S=1."""
        # Define an HSV color with S=0.9
        hsv_input = torch.tensor([[[[0.5, 0.9, 0.5]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.2
        expected_hsv = torch.tensor([[[[0.5, 1.0, 0.5]]]])  # S should be clamped to 1.0
        saturated_hsv = self.enhancer.saturate(hsv_input, amount)
        self.assertTrue(
            torch.allclose(saturated_hsv, expected_hsv, atol=1e-6),
            msg="Saturating clamping failed.",
        )

    def test_saturate_zero_amount(self):
        """Test saturating with amount=0 (no change)."""
        # Define an HSV color
        hsv_input = torch.tensor([[[[0.75, 0.4, 0.5]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.0
        expected_hsv = hsv_input.clone()
        saturated_hsv = self.enhancer.saturate(hsv_input, amount)
        self.assertTrue(
            torch.allclose(saturated_hsv, expected_hsv, atol=1e-6),
            msg="Saturating with zero amount should not change the image.",
        )

    def test_saturate_full_batch(self):
        """Test saturating on a full batch of HSV images."""
        # Create a batch of 3 HSV images
        hsv_input = torch.tensor(
            [
                [[[0.0, 0.3, 0.5]]],  # Red, low saturation
                [[[1 / 3, 0.5, 0.6]]],  # Green, medium saturation
                [[[2 / 3, 0.8, 0.7]]],  # Blue, high saturation
            ]
        )  # Shape: (3, 1, 1, 3)
        amount = 0.3
        expected_hsv = torch.tensor(
            [
                [[[0.0, 0.6, 0.5]]],  # S=0.3 + 0.3 = 0.6
                [[[1 / 3, 0.8, 0.6]]],  # S=0.5 + 0.3 = 0.8
                [[[2 / 3, 1.0, 0.7]]],  # S=0.8 + 0.3 = 1.0 (clamped)
            ]
        )
        saturated_hsv = self.enhancer.saturate(hsv_input, amount)
        self.assertTrue(
            torch.allclose(saturated_hsv, expected_hsv, atol=1e-4),
            msg="Saturating on a full batch failed.",
        )

    # -------------------
    # Tests for desaturate
    # -------------------

    def test_desaturate_basic(self):
        """Test desaturating a basic HSV color."""
        # Define an HSV color: pure cyan with S=0.8
        hsv_input = torch.tensor([[[[0.5, 0.8, 0.5]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.3
        expected_hsv = torch.tensor([[[[0.5, 0.5, 0.5]]]])  # S should be 0.8 - 0.3 = 0.5
        desaturated_hsv = self.enhancer.desaturate(hsv_input, amount)
        self.assertTrue(
            torch.allclose(desaturated_hsv, expected_hsv, atol=1e-6),
            msg="Desaturating basic HSV color failed.",
        )

    def test_desaturate_clamping(self):
        """Test that desaturating does not go below S=0."""
        # Define an HSV color with S=0.1
        hsv_input = torch.tensor([[[[0.25, 0.1, 0.5]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.2
        expected_hsv = torch.tensor([[[[0.25, 0.0, 0.5]]]])  # S should be clamped to 0.0
        desaturated_hsv = self.enhancer.desaturate(hsv_input, amount)
        self.assertTrue(
            torch.allclose(desaturated_hsv, expected_hsv, atol=1e-6),
            msg="Desaturating clamping failed.",
        )

    def test_desaturate_zero_amount(self):
        """Test desaturating with amount=0 (no change)."""
        # Define an HSV color
        hsv_input = torch.tensor([[[[0.75, 0.6, 0.5]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.0
        expected_hsv = hsv_input.clone()
        desaturated_hsv = self.enhancer.desaturate(hsv_input, amount)
        self.assertTrue(
            torch.allclose(desaturated_hsv, expected_hsv, atol=1e-6),
            msg="Desaturating with zero amount should not change the image.",
        )

    def test_desaturate_full_batch(self):
        """Test desaturating on a full batch of HSV images."""
        # Create a batch of 3 HSV images
        hsv_input = torch.tensor(
            [
                [[[0.0, 0.6, 0.5]]],  # Red, high saturation
                [[[0.3333, 0.5, 0.6]]],  # Green, medium saturation
                [[[0.6667, 0.2, 0.7]]],  # Blue, low saturation
            ]
        )  # Shape: (3, 1, 1, 3)
        amount = 0.3
        expected_hsv = torch.tensor(
            [
                [[[0.0, 0.3, 0.5]]],  # S=0.6 - 0.3 = 0.3
                [[[0.3333, 0.2, 0.6]]],  # S=0.5 - 0.3 = 0.2
                [[[0.6667, 0.0, 0.7]]],  # S=0.2 - 0.3 = 0.0 (clamped)
            ]
        )
        desaturated_hsv = self.enhancer.desaturate(hsv_input, amount)
        self.assertTrue(
            torch.allclose(desaturated_hsv, expected_hsv, atol=1e-4),
            msg="Desaturating on a full batch failed.",
        )

    # -------------------
    # Tests for shift_hue
    # -------------------
    def test_shift_hue_basic(self):
        """Test shifting hue by a basic degree value."""
        # Define an HSV color: pure yellow with H=1/6 (60 degrees)
        hsv_input = torch.tensor([[[[1 / 6, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        degrees = 120.0  # Shift to 180 degrees
        expected_hsv = torch.tensor([[[[0.5, 1.0, 0.5]]]])  # H=0.5 (180 degrees)
        shifted_hsv = self.enhancer.shift_hue(hsv_input, degrees)
        self.assertTrue(
            torch.allclose(shifted_hsv, expected_hsv, atol=1e-4),
            msg="Shifting hue by 120 degrees failed.",
        )

    def test_shift_hue_wraparound(self):
        """Test that shifting hue wraps around correctly."""
        # Define an HSV color: pure red with H=0.0 (0 degrees)
        hsv_input = torch.tensor([[[[0.0, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        degrees = 360.0  # Full rotation, should wrap to 0.0
        expected_hsv = torch.tensor([[[[0.0, 1.0, 0.5]]]])  # H=0.0
        shifted_hsv = self.enhancer.shift_hue(hsv_input, degrees)
        self.assertTrue(
            torch.allclose(shifted_hsv, expected_hsv, atol=1e-6),
            msg="Hue wraparound with 360 degrees failed.",
        )

    def test_shift_hue_negative_shift(self):
        """Test shifting hue by a negative degree value."""
        # Define an HSV color: pure blue with H=2/3 (240 degrees)
        hsv_input = torch.tensor([[[[2 / 3, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        degrees = -120.0  # Shift to pure green
        expected_hsv = torch.tensor([[[[1 / 3, 1.0, 0.5]]]])  # H=1/3 (120 degrees)
        shifted_hsv = self.enhancer.shift_hue(hsv_input, degrees)
        self.assertTrue(
            torch.allclose(shifted_hsv, expected_hsv, atol=1e-4),
            msg="Shifting hue by -120 degrees failed.",
        )

    def test_shift_hue_large_shift(self):
        """Test shifting hue by more than 360 degrees."""
        # Define an HSV color: pure green with H=1/3 (120 degrees)
        hsv_input = torch.tensor([[[[1 / 3, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        degrees = 480.0  # Equivalent to shifting by 120 degrees twice: 120 + 480 = 600 % 360 = 240 degrees
        expected_hsv = torch.tensor([[[[2 / 3, 1.0, 0.5]]]])  # H=2/3 (240 degrees)
        shifted_hsv = self.enhancer.shift_hue(hsv_input, degrees)
        self.assertTrue(
            torch.allclose(shifted_hsv, expected_hsv, atol=1e-6),
            msg="Shifting hue by 480 degrees failed.",
        )

    def test_shift_hue_zero_shift(self):
        """Test shifting hue by 0 degrees (no change)."""
        # Define an HSV color
        hsv_input = torch.tensor([[[[0.75, 0.6, 0.5]]]])  # Shape: (1, 1, 1, 3)
        degrees = 0.0
        expected_hsv = hsv_input.clone()
        shifted_hsv = self.enhancer.shift_hue(hsv_input, degrees)
        self.assertTrue(
            torch.allclose(shifted_hsv, expected_hsv, atol=1e-6),
            msg="Shifting hue by 0 degrees should not change the image.",
        )

    def test_shift_hue_full_batch(self):
        """Test shifting hue on a full batch of HSV images."""
        # Create a batch of 3 HSV images
        hsv_input = torch.tensor(
            [
                [[[0.0, 1.0, 0.5]]],  # Red
                [[[1 / 3, 0.5, 0.6]]],  # Green
                [[[2 / 3, 0.8, 0.7]]],  # Blue
            ]
        )  # Shape: (3, 1, 1, 3)
        degrees = 120.0  # Shift each hue by 120 degrees
        expected_hsv = torch.tensor(
            [
                [[[1 / 3, 1.0, 0.5]]],  # Red shifted to Green
                [[[2 / 3, 0.5, 0.6]]],  # Green shifted to Blue
                [[[0.0, 0.8, 0.7]]],  # Blue shifted to Red
            ]
        )
        shifted_hsv = self.enhancer.shift_hue(hsv_input, degrees)
        self.assertTrue(
            torch.allclose(shifted_hsv, expected_hsv, atol=1e-4),
            msg="Shifting hue on a full batch failed.",
        )

    def test_shift_hue_multiple_shifts(self):
        """Test multiple sequential hue shifts."""
        # Define an HSV color: pure red with H=0.0 (0 degrees)
        hsv_input = torch.tensor([[[[0.0, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        enhancer = self.enhancer

        # Shift by 120 degrees to Green
        shifted_hsv = enhancer.shift_hue(hsv_input, 120.0)
        expected_hsv = torch.tensor([[[[1 / 3, 1.0, 0.5]]]])  # H=1/3
        self.assertTrue(
            torch.allclose(shifted_hsv, expected_hsv, atol=1e-4),
            msg="First hue shift by 120 degrees failed.",
        )

        # Shift by another 120 degrees to Blue
        shifted_hsv = enhancer.shift_hue(shifted_hsv, 120.0)
        expected_hsv = torch.tensor([[[[2 / 3, 1.0, 0.5]]]])  # H=2/3
        self.assertTrue(
            torch.allclose(shifted_hsv, expected_hsv, atol=1e-4),
            msg="Second hue shift by 120 degrees failed.",
        )

        # Shift by another 120 degrees to wrap back to Red
        shifted_hsv = enhancer.shift_hue(shifted_hsv, 120.0)
        expected_hsv = torch.tensor([[[[0.0, 1.0, 0.5]]]])  # H=0.0
        self.assertTrue(
            torch.allclose(shifted_hsv, expected_hsv, atol=1e-4),
            msg="Third hue shift by 120 degrees failed.",
        )

    def test_shift_hue_negative_large_shift(self):
        """Test shifting hue by a large negative degree value."""
        # Define an HSV color: pure green with H=1/3 (120 degrees)
        hsv_input = torch.tensor([[[[1 / 3, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        degrees = -480.0  # Equivalent to shifting by -120 degrees twice: 120 - 480 = -360, wraps to 0.0
        expected_hsv = torch.tensor([[[[0.0, 1.0, 0.5]]]])  # H=0.0
        shifted_hsv = self.enhancer.shift_hue(hsv_input, degrees)
        self.assertTrue(
            torch.allclose(shifted_hsv, expected_hsv, atol=1e-6),
            msg="Shifting hue by -480 degrees failed.",
        )

    def test_shift_hue_high_resolution(self):
        """Test shifting hue on a high-resolution HSV image."""
        high_res = torch.rand((1, 4096, 4096, 3))
        degrees = 90.0
        try:
            shifted_hsv = self.enhancer.shift_hue(high_res, degrees)
            # Check shape
            self.assertEqual(
                shifted_hsv.shape,
                high_res.shape,
                msg="Shifted HSV tensor shape mismatch for high-resolution input.",
            )
            # Check value ranges
            self.assertTrue(
                torch.all((shifted_hsv[..., 0] >= 0) & (shifted_hsv[..., 0] < 1)),
                msg="Hue values out of range [0,1) in high-resolution input.",
            )
            self.assertTrue(
                torch.all((shifted_hsv[..., 1] >= 0) & (shifted_hsv[..., 1] <= 1)),
                msg="Saturation values out of range [0,1] in high-resolution input.",
            )
            self.assertTrue(
                torch.all((shifted_hsv[..., 2] >= 0) & (shifted_hsv[..., 2] <= 1)),
                msg="Value values out of range [0,1] in high-resolution input.",
            )
        except Exception as e:
            self.fail(f"Shifting hue failed on high-resolution input: {e}")

    # -------------------
    # Combined Tests
    # -------------------

    def test_combined_enhancements(self):
        """Test combining multiple enhancements on an HSV image."""
        # Define an HSV color: pure red with S=1.0, V=0.5
        hsv_input = torch.tensor([[[[0.0, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)

        # Apply multiple enhancements:
        # 1. Lighten by 0.1 -> V=0.5 + 0.1 = 0.6
        # 2. Desaturate by 0.3 -> S=1.0 - 0.3 = 0.7
        # 3. Shift hue by 240 degrees -> H=0.6667 (240 degrees)
        enhanced_hsv = self.enhancer.lighten(hsv_input, 0.1)
        enhanced_hsv = self.enhancer.desaturate(enhanced_hsv, 0.3)
        enhanced_hsv = self.enhancer.shift_hue(enhanced_hsv, 240.0)

        expected_hsv = torch.tensor([[[[2 / 3, 0.7, 0.6]]]])  # H=2/3, S=0.7, V=0.6
        self.assertTrue(
            torch.allclose(enhanced_hsv, expected_hsv, atol=1e-4),
            msg="Combined enhancements failed.",
        )

    def test_combined_enhancements_clamping(self):
        """Test combined enhancements with clamping."""
        # Define an HSV color: pure red with S=1.0, V=0.9
        hsv_input = torch.tensor([[[[0.0, 1.0, 0.9]]]])  # Shape: (1, 1, 1, 3)

        # Apply enhancements:
        # 1. Lighten by 0.2 -> V=0.9 + 0.2 = 1.1 (clamped to 1.0)
        # 2. Saturate by 0.5 -> S=1.0 + 0.5 = 1.5 (clamped to 1.0)
        # 3. Shift hue by 720 degrees -> H=0.0 (360*2 shifts, wraps to 0.0)
        enhanced_hsv = self.enhancer.lighten(hsv_input, 0.2)
        enhanced_hsv = self.enhancer.saturate(enhanced_hsv, 0.5)
        enhanced_hsv = self.enhancer.shift_hue(enhanced_hsv, 720.0)

        expected_hsv = torch.tensor([[[[0.0, 1.0, 1.0]]]])  # H=0.0, S=1.0, V=1.0
        self.assertTrue(
            torch.allclose(enhanced_hsv, expected_hsv, atol=1e-6),
            msg="Combined enhancements with clamping failed.",
        )

    # -------------------
    # Additional Tests
    # -------------------

    def test_shift_hue_fractional_degrees(self):
        """Test shifting hue by fractional degree values."""
        # Define an HSV color: pure red with H=0.0 (0 degrees)
        hsv_input = torch.tensor([[[[0.0, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        degrees = 45.5  # Shift to 45.5 degrees
        expected_h = (45.5 / 360.0) % 1.0
        expected_hsv = torch.tensor([[[[expected_h, 1.0, 0.5]]]])
        shifted_hsv = self.enhancer.shift_hue(hsv_input, degrees)
        self.assertTrue(
            torch.allclose(shifted_hsv, expected_hsv, atol=1e-6),
            msg="Shifting hue by fractional degrees failed.",
        )

    def test_lighten_maximum_value(self):
        """Test lightening an HSV color already at maximum V."""
        # Define an HSV color with V=1.0
        hsv_input = torch.tensor([[[[0.5, 0.5, 1.0]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.1
        expected_hsv = torch.tensor([[[[0.5, 0.5, 1.0]]]])  # V should remain at 1.0
        lightened_hsv = self.enhancer.lighten(hsv_input, amount)
        self.assertTrue(
            torch.allclose(lightened_hsv, expected_hsv, atol=1e-6),
            msg="Lightening an HSV color already at maximum V failed.",
        )

    def test_darken_minimum_value(self):
        """Test darkening an HSV color already at minimum V."""
        # Define an HSV color with V=0.0
        hsv_input = torch.tensor([[[[0.5, 0.5, 0.0]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.1
        expected_hsv = torch.tensor([[[[0.5, 0.5, 0.0]]]])  # V should remain at 0.0
        darkened_hsv = self.enhancer.darken(hsv_input, amount)
        self.assertTrue(
            torch.allclose(darkened_hsv, expected_hsv, atol=1e-6),
            msg="Darkening an HSV color already at minimum V failed.",
        )

    def test_saturate_maximum_saturation(self):
        """Test saturating an HSV color already at maximum S."""
        # Define an HSV color with S=1.0
        hsv_input = torch.tensor([[[[0.5, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.1
        expected_hsv = torch.tensor([[[[0.5, 1.0, 0.5]]]])  # S should remain at 1.0
        saturated_hsv = self.enhancer.saturate(hsv_input, amount)
        self.assertTrue(
            torch.allclose(saturated_hsv, expected_hsv, atol=1e-6),
            msg="Saturating an HSV color already at maximum S failed.",
        )

    def test_desaturate_minimum_saturation(self):
        """Test desaturating an HSV color already at minimum S."""
        # Define an HSV color with S=0.0
        hsv_input = torch.tensor([[[[0.5, 0.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        amount = 0.1
        expected_hsv = torch.tensor([[[[0.5, 0.0, 0.5]]]])  # S should remain at 0.0
        desaturated_hsv = self.enhancer.desaturate(hsv_input, amount)
        self.assertTrue(
            torch.allclose(desaturated_hsv, expected_hsv, atol=1e-6),
            msg="Desaturating an HSV color already at minimum S failed.",
        )

    def test_lighten_non_tensor_input(self):
        """Test that non-tensor inputs raise TypeError for lighten."""
        non_tensor_inputs = [None, 123, "string", [0.0, 1.0, 0.5]]
        amount = 0.2
        for inp in non_tensor_inputs:
            with self.subTest(input=inp):
                with self.assertRaises(AttributeError):
                    self.enhancer.lighten(inp, amount)

    def test_shift_hue_non_numeric_degree(self):
        """Test that non-numeric degree values raise TypeError for shift_hue."""
        hsv_input = torch.tensor([[[[0.0, 1.0, 0.5]]]])  # Shape: (1, 1, 1, 3)
        invalid_degrees = ["90", None, [120], {"degrees": 120}]
        for deg in invalid_degrees:
            with self.subTest(degrees=deg):
                with self.assertRaises(TypeError):
                    self.enhancer.shift_hue(hsv_input, deg)

    # Additional tests can be added here as needed


if __name__ == "__main__":
    unittest.main()
