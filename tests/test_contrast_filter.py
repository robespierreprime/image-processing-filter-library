"""
Unit tests for ContrastFilter.

Tests parameter validation, contrast adjustment algorithm correctness,
and edge cases for the ContrastFilter class.
"""

import pytest
import numpy as np
from image_processing_library.filters.enhancement.correction_filters import ContrastFilter
from image_processing_library.core.utils import FilterValidationError


class TestContrastFilterParameterValidation:
    """Test parameter validation for ContrastFilter."""
    
    def test_valid_contrast_factor_range(self):
        """Test that valid contrast factor values are accepted."""
        # Test boundary values
        filter_min = ContrastFilter(contrast_factor=0.0)
        assert filter_min.parameters['contrast_factor'] == 0.0
        
        filter_max = ContrastFilter(contrast_factor=3.0)
        assert filter_max.parameters['contrast_factor'] == 3.0
        
        # Test typical values
        filter_normal = ContrastFilter(contrast_factor=1.0)
        assert filter_normal.parameters['contrast_factor'] == 1.0
        
        filter_increase = ContrastFilter(contrast_factor=1.5)
        assert filter_increase.parameters['contrast_factor'] == 1.5
        
        filter_decrease = ContrastFilter(contrast_factor=0.5)
        assert filter_decrease.parameters['contrast_factor'] == 0.5
    
    def test_invalid_contrast_factor_type(self):
        """Test that non-numeric contrast factor values are rejected."""
        with pytest.raises(FilterValidationError, match="Contrast factor must be a numeric value"):
            filter_obj = ContrastFilter(contrast_factor="invalid")
            filter_obj._validate_parameters()
        
        with pytest.raises(FilterValidationError, match="Contrast factor must be a numeric value"):
            filter_obj = ContrastFilter(contrast_factor=None)
            filter_obj._validate_parameters()
        
        with pytest.raises(FilterValidationError, match="Contrast factor must be a numeric value"):
            filter_obj = ContrastFilter(contrast_factor=[1.0])
            filter_obj._validate_parameters()
    
    def test_negative_contrast_factor(self):
        """Test that negative contrast factor values are rejected."""
        with pytest.raises(FilterValidationError, match="Contrast factor must be non-negative"):
            filter_obj = ContrastFilter(contrast_factor=-0.1)
            filter_obj._validate_parameters()
        
        with pytest.raises(FilterValidationError, match="Contrast factor must be non-negative"):
            filter_obj = ContrastFilter(contrast_factor=-1.0)
            filter_obj._validate_parameters()
    
    def test_contrast_factor_out_of_range(self):
        """Test that contrast factor values outside [0.0, 3.0] are rejected."""
        with pytest.raises(FilterValidationError, match="Contrast factor must be in range \\[0.0, 3.0\\]"):
            filter_obj = ContrastFilter(contrast_factor=3.1)
            filter_obj._validate_parameters()
        
        with pytest.raises(FilterValidationError, match="Contrast factor must be in range \\[0.0, 3.0\\]"):
            filter_obj = ContrastFilter(contrast_factor=10.0)
            filter_obj._validate_parameters()
    
    def test_parameter_update_validation(self):
        """Test that parameter validation works when updating parameters."""
        filter_obj = ContrastFilter(contrast_factor=1.0)
        
        # Valid update should work
        filter_obj.set_parameters(contrast_factor=2.0)
        assert filter_obj.parameters['contrast_factor'] == 2.0
        
        # Invalid update should raise error during validation
        filter_obj.set_parameters(contrast_factor=5.0)
        with pytest.raises(FilterValidationError):
            filter_obj._validate_parameters()


class TestContrastFilterInputValidation:
    """Test input data validation for ContrastFilter."""
    
    def test_valid_input_formats(self):
        """Test that valid input formats are accepted."""
        filter_obj = ContrastFilter()
        
        # RGB image (3D array with 3 channels)
        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        assert filter_obj.validate_input(rgb_image) is True
        
        # RGBA image (3D array with 4 channels)
        rgba_image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        assert filter_obj.validate_input(rgba_image) is True
        
        # Grayscale image (2D array)
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        assert filter_obj.validate_input(gray_image) is True
        
        # Float images
        float_rgb = np.random.rand(50, 50, 3).astype(np.float32)
        assert filter_obj.validate_input(float_rgb) is True
    
    def test_invalid_input_types(self):
        """Test that invalid input types are rejected."""
        filter_obj = ContrastFilter()
        
        with pytest.raises(FilterValidationError, match="Input must be a numpy array"):
            filter_obj.validate_input("not an array")
        
        with pytest.raises(FilterValidationError, match="Input must be a numpy array"):
            filter_obj.validate_input([1, 2, 3])
        
        with pytest.raises(FilterValidationError, match="Input must be a numpy array"):
            filter_obj.validate_input(None)
    
    def test_empty_array_rejection(self):
        """Test that empty arrays are rejected."""
        filter_obj = ContrastFilter()
        
        with pytest.raises(FilterValidationError, match="Input array cannot be empty"):
            filter_obj.validate_input(np.array([]))
        
        with pytest.raises(FilterValidationError, match="Input array cannot be empty"):
            filter_obj.validate_input(np.array([]).reshape(0, 0))
    
    def test_invalid_dimensions(self):
        """Test that arrays with invalid dimensions are rejected."""
        filter_obj = ContrastFilter()
        
        # 1D array
        with pytest.raises(FilterValidationError, match="Image data must be 2D or 3D array"):
            filter_obj.validate_input(np.array([1, 2, 3]))
        
        # 4D array
        with pytest.raises(FilterValidationError, match="Image data must be 2D or 3D array"):
            filter_obj.validate_input(np.random.rand(10, 10, 3, 2))
    
    def test_invalid_channel_count(self):
        """Test that 3D arrays with invalid channel counts are rejected."""
        filter_obj = ContrastFilter()
        
        # 1 channel (should be 2D for grayscale)
        with pytest.raises(FilterValidationError, match="Color images must have 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_obj.validate_input(np.random.rand(10, 10, 1))
        
        # 2 channels (invalid)
        with pytest.raises(FilterValidationError, match="Color images must have 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_obj.validate_input(np.random.rand(10, 10, 2))
        
        # 5 channels (invalid)
        with pytest.raises(FilterValidationError, match="Color images must have 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_obj.validate_input(np.random.rand(10, 10, 5))


class TestContrastFilterAlgorithm:
    """Test the mathematical correctness of contrast adjustment algorithm."""
    
    def test_identity_case_contrast_factor_one(self):
        """Test that contrast_factor = 1.0 returns unchanged image."""
        filter_obj = ContrastFilter(contrast_factor=1.0)
        
        # Test with uint8 RGB image
        original = np.array([[[100, 150, 200]]], dtype=np.uint8)
        result = filter_obj.apply(original)
        np.testing.assert_array_equal(result, original)
        
        # Test with float image
        original_float = np.array([[[0.4, 0.6, 0.8]]], dtype=np.float32)
        result_float = filter_obj.apply(original_float)
        np.testing.assert_array_almost_equal(result_float, original_float)
        
        # Test with grayscale
        original_gray = np.array([[100, 150, 200]], dtype=np.uint8)
        result_gray = filter_obj.apply(original_gray)
        np.testing.assert_array_equal(result_gray, original_gray)
    
    def test_contrast_increase_uint8(self):
        """Test contrast increase with uint8 images."""
        filter_obj = ContrastFilter(contrast_factor=2.0)
        
        # Test with known values around midpoint (128)
        # For contrast_factor = 2.0: output = (input - 128) * 2 + 128
        original = np.array([[[64, 128, 192]]], dtype=np.uint8)  # -64, 0, +64 from midpoint
        expected = np.array([[[0, 128, 255]]], dtype=np.uint8)   # -128, 0, +128 from midpoint (clipped)
        
        result = filter_obj.apply(original)
        np.testing.assert_array_equal(result, expected)
    
    def test_contrast_decrease_uint8(self):
        """Test contrast decrease with uint8 images."""
        filter_obj = ContrastFilter(contrast_factor=0.5)
        
        # Test with known values around midpoint (128)
        # For contrast_factor = 0.5: output = (input - 128) * 0.5 + 128
        original = np.array([[[0, 128, 255]]], dtype=np.uint8)    # -128, 0, +127 from midpoint
        expected = np.array([[[64, 128, 191]]], dtype=np.uint8)   # -64, 0, +63.5 from midpoint
        
        result = filter_obj.apply(original)
        np.testing.assert_array_equal(result, expected)
    
    def test_contrast_adjustment_float(self):
        """Test contrast adjustment with float images."""
        filter_obj = ContrastFilter(contrast_factor=2.0)
        
        # Test with known values around midpoint (0.5)
        # For contrast_factor = 2.0: output = (input - 0.5) * 2 + 0.5
        original = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)  # -0.25, 0, +0.25 from midpoint
        expected = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)    # -0.5, 0, +0.5 from midpoint
        
        result = filter_obj.apply(original)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
    
    def test_contrast_adjustment_grayscale(self):
        """Test contrast adjustment with grayscale images."""
        filter_obj = ContrastFilter(contrast_factor=1.5)
        
        # Test with grayscale image
        original = np.array([[100, 128, 156]], dtype=np.uint8)  # -28, 0, +28 from midpoint
        # Expected: (100-128)*1.5+128=86, (128-128)*1.5+128=128, (156-128)*1.5+128=170
        expected = np.array([[86, 128, 170]], dtype=np.uint8)
        
        result = filter_obj.apply(original)
        np.testing.assert_array_equal(result, expected)
    
    def test_contrast_adjustment_rgba(self):
        """Test that RGBA images preserve alpha channel."""
        filter_obj = ContrastFilter(contrast_factor=2.0)
        
        # RGBA image with alpha channel
        original = np.array([[[64, 128, 192, 255]]], dtype=np.uint8)
        result = filter_obj.apply(original)
        
        # Alpha channel should be processed like other channels
        # (255 - 128) * 2 + 128 = 255 (clipped)
        expected = np.array([[[0, 128, 255, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
    
    def test_extreme_contrast_values(self):
        """Test behavior with extreme contrast factor values."""
        # Test with contrast_factor = 0.0 (should flatten to midpoint)
        filter_zero = ContrastFilter(contrast_factor=0.0)
        original = np.array([[[0, 128, 255]]], dtype=np.uint8)
        result_zero = filter_zero.apply(original)
        expected_zero = np.array([[[128, 128, 128]]], dtype=np.uint8)
        np.testing.assert_array_equal(result_zero, expected_zero)
        
        # Test with contrast_factor = 3.0 (maximum allowed)
        filter_max = ContrastFilter(contrast_factor=3.0)
        original = np.array([[[100, 128, 156]]], dtype=np.uint8)
        result_max = filter_max.apply(original)
        # (100-128)*3+128 = 44, (128-128)*3+128 = 128, (156-128)*3+128 = 212
        expected_max = np.array([[[44, 128, 212]]], dtype=np.uint8)
        np.testing.assert_array_equal(result_max, expected_max)
    
    def test_clipping_behavior(self):
        """Test that values are properly clipped to valid ranges."""
        filter_obj = ContrastFilter(contrast_factor=3.0)
        
        # Test values that will exceed valid range after contrast adjustment
        original = np.array([[[0, 128, 255]]], dtype=np.uint8)
        result = filter_obj.apply(original)
        
        # (0-128)*3+128 = -256 -> clipped to 0
        # (128-128)*3+128 = 128 -> unchanged
        # (255-128)*3+128 = 509 -> clipped to 255
        expected = np.array([[[0, 128, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        
        # Test with float values
        filter_float = ContrastFilter(contrast_factor=3.0)
        original_float = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        result_float = filter_float.apply(original_float)
        
        # (0.0-0.5)*3+0.5 = -1.0 -> clipped to 0.0
        # (0.5-0.5)*3+0.5 = 0.5 -> unchanged
        # (1.0-0.5)*3+0.5 = 2.0 -> clipped to 1.0
        expected_float = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result_float, expected_float)


class TestContrastFilterIntegration:
    """Test integration with BaseFilter features."""
    
    def test_filter_metadata(self):
        """Test that filter metadata is correctly set."""
        filter_obj = ContrastFilter()
        
        assert filter_obj.name == "contrast"
        assert filter_obj.category == "enhancement"
        assert hasattr(filter_obj, 'data_type')
        assert hasattr(filter_obj, 'color_format')
    
    def test_parameter_override_in_apply(self):
        """Test that parameters can be overridden in apply() method."""
        filter_obj = ContrastFilter(contrast_factor=1.0)
        
        original = np.array([[[64, 128, 192]]], dtype=np.uint8)
        
        # Apply with different contrast_factor
        result = filter_obj.apply(original, contrast_factor=2.0)
        expected = np.array([[[0, 128, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        
        # Parameter should be updated after apply
        assert filter_obj.parameters['contrast_factor'] == 2.0
    
    def test_progress_tracking(self):
        """Test that progress tracking works."""
        filter_obj = ContrastFilter()
        
        # Create a larger image to ensure progress tracking is called
        large_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Apply filter and check that it completes without error
        result = filter_obj.apply(large_image)
        
        # Verify result has correct shape
        assert result.shape == large_image.shape
        assert result.dtype == large_image.dtype
    
    def test_timing_measurement(self):
        """Test that execution timing is measured."""
        filter_obj = ContrastFilter()
        
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = filter_obj.apply(image)
        
        # Check that result is correct (timing is handled internally)
        assert result.shape == image.shape
        assert result.dtype == image.dtype
    
    def test_memory_usage_tracking(self):
        """Test that memory usage is tracked."""
        filter_obj = ContrastFilter()
        
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = filter_obj.apply(image)
        
        # Verify result is correct
        assert result.shape == image.shape
        assert result.dtype == image.dtype
    
    def test_different_data_types(self):
        """Test filter works with different numpy data types."""
        filter_obj = ContrastFilter(contrast_factor=1.5)
        
        # Test uint8
        uint8_image = np.array([[[100, 128, 156]]], dtype=np.uint8)
        result_uint8 = filter_obj.apply(uint8_image)
        assert result_uint8.dtype == np.uint8
        
        # Test float32
        float32_image = np.array([[[0.4, 0.5, 0.6]]], dtype=np.float32)
        result_float32 = filter_obj.apply(float32_image)
        assert result_float32.dtype == np.float32
        
        # Test float64
        float64_image = np.array([[[0.4, 0.5, 0.6]]], dtype=np.float64)
        result_float64 = filter_obj.apply(float64_image)
        assert result_float64.dtype == np.float64


class TestContrastFilterEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_pixel_image(self):
        """Test with single pixel images."""
        filter_obj = ContrastFilter(contrast_factor=2.0)
        
        # Single pixel RGB
        single_pixel = np.array([[[128, 128, 128]]], dtype=np.uint8)
        result = filter_obj.apply(single_pixel)
        # At midpoint, contrast adjustment should not change the value
        np.testing.assert_array_equal(result, single_pixel)
        
        # Single pixel grayscale
        single_gray = np.array([[128]], dtype=np.uint8)
        result_gray = filter_obj.apply(single_gray)
        np.testing.assert_array_equal(result_gray, single_gray)
    
    def test_uniform_images(self):
        """Test with uniform (single color) images."""
        filter_obj = ContrastFilter(contrast_factor=2.0)
        
        # Uniform image at midpoint
        uniform_mid = np.full((10, 10, 3), 128, dtype=np.uint8)
        result_mid = filter_obj.apply(uniform_mid)
        np.testing.assert_array_equal(result_mid, uniform_mid)
        
        # Uniform image not at midpoint
        uniform_bright = np.full((10, 10, 3), 200, dtype=np.uint8)
        result_bright = filter_obj.apply(uniform_bright)
        # (200-128)*2+128 = 272 -> clipped to 255
        expected_bright = np.full((10, 10, 3), 255, dtype=np.uint8)
        np.testing.assert_array_equal(result_bright, expected_bright)
    
    def test_very_small_images(self):
        """Test with very small images."""
        filter_obj = ContrastFilter(contrast_factor=1.5)
        
        # 1x1 image
        tiny_rgb = np.array([[[100, 128, 156]]], dtype=np.uint8)
        result_tiny = filter_obj.apply(tiny_rgb)
        assert result_tiny.shape == (1, 1, 3)
        
        # 2x2 grayscale
        small_gray = np.array([[100, 156], [128, 200]], dtype=np.uint8)
        result_small = filter_obj.apply(small_gray)
        assert result_small.shape == (2, 2)
    
    def test_near_identity_contrast_factors(self):
        """Test with contrast factors very close to 1.0."""
        # Test values very close to 1.0 (should trigger identity case)
        filter_close = ContrastFilter(contrast_factor=1.0000001)
        
        original = np.array([[[100, 128, 156]]], dtype=np.uint8)
        result = filter_close.apply(original)
        
        # Should be treated as identity case
        np.testing.assert_array_equal(result, original)
    
    def test_parameter_persistence(self):
        """Test that filter parameters persist correctly."""
        filter_obj = ContrastFilter(contrast_factor=2.0)
        
        # Apply filter multiple times
        image1 = np.array([[[64, 128, 192]]], dtype=np.uint8)
        image2 = np.array([[[32, 128, 224]]], dtype=np.uint8)
        
        result1 = filter_obj.apply(image1)
        result2 = filter_obj.apply(image2)
        
        # Parameters should remain the same
        assert filter_obj.parameters['contrast_factor'] == 2.0
        
        # Results should be consistent with the same parameters
        expected1 = np.array([[[0, 128, 255]]], dtype=np.uint8)
        expected2 = np.array([[[0, 128, 255]]], dtype=np.uint8)  # Both clipped
        
        np.testing.assert_array_equal(result1, expected1)
        np.testing.assert_array_equal(result2, expected2)


if __name__ == "__main__":
    pytest.main([__file__])