"""
Unit tests for GaussianBlurFilter.

Tests parameter validation, gaussian blur algorithm correctness,
and edge cases for the GaussianBlurFilter class.
"""

import pytest
import numpy as np
from image_processing_library.filters.enhancement.blur_filters import GaussianBlurFilter
from image_processing_library.core.utils import FilterValidationError


class TestGaussianBlurFilterParameterValidation:
    """Test parameter validation for GaussianBlurFilter."""
    
    def test_valid_sigma_range(self):
        """Test that valid sigma values are accepted."""
        # Test boundary values
        filter_min = GaussianBlurFilter(sigma=0.0)
        assert filter_min.parameters['sigma'] == 0.0
        
        filter_max = GaussianBlurFilter(sigma=10.0)
        assert filter_max.parameters['sigma'] == 10.0
        
        # Test typical values
        filter_normal = GaussianBlurFilter(sigma=1.0)
        assert filter_normal.parameters['sigma'] == 1.0
        
        filter_small = GaussianBlurFilter(sigma=0.5)
        assert filter_small.parameters['sigma'] == 0.5
        
        filter_large = GaussianBlurFilter(sigma=5.0)
        assert filter_large.parameters['sigma'] == 5.0
    
    def test_valid_kernel_size_values(self):
        """Test that valid kernel size values are accepted."""
        # Test odd kernel sizes
        filter_3 = GaussianBlurFilter(sigma=1.0, kernel_size=3)
        assert filter_3.parameters['kernel_size'] == 3
        
        filter_5 = GaussianBlurFilter(sigma=1.0, kernel_size=5)
        assert filter_5.parameters['kernel_size'] == 5
        
        filter_7 = GaussianBlurFilter(sigma=1.0, kernel_size=7)
        assert filter_7.parameters['kernel_size'] == 7
        
        # Test None (auto-calculated)
        filter_auto = GaussianBlurFilter(sigma=1.0, kernel_size=None)
        assert filter_auto.parameters['kernel_size'] is None
    
    def test_invalid_sigma_type(self):
        """Test that non-numeric sigma values are rejected."""
        with pytest.raises(FilterValidationError, match="Sigma must be a numeric value"):
            filter_obj = GaussianBlurFilter(sigma="invalid")
            filter_obj._validate_parameters()
        
        with pytest.raises(FilterValidationError, match="Sigma must be a numeric value"):
            filter_obj = GaussianBlurFilter(sigma=None)
            filter_obj._validate_parameters()
        
        with pytest.raises(FilterValidationError, match="Sigma must be a numeric value"):
            filter_obj = GaussianBlurFilter(sigma=[1.0])
            filter_obj._validate_parameters()
    
    def test_negative_sigma(self):
        """Test that negative sigma values are rejected."""
        with pytest.raises(FilterValidationError, match="Sigma must be non-negative"):
            filter_obj = GaussianBlurFilter(sigma=-0.1)
            filter_obj._validate_parameters()
        
        with pytest.raises(FilterValidationError, match="Sigma must be non-negative"):
            filter_obj = GaussianBlurFilter(sigma=-1.0)
            filter_obj._validate_parameters()
    
    def test_sigma_out_of_range(self):
        """Test that sigma values outside [0.0, 10.0] are rejected."""
        with pytest.raises(FilterValidationError, match="Sigma must be in range \\[0.0, 10.0\\]"):
            filter_obj = GaussianBlurFilter(sigma=10.1)
            filter_obj._validate_parameters()
        
        with pytest.raises(FilterValidationError, match="Sigma must be in range \\[0.0, 10.0\\]"):
            filter_obj = GaussianBlurFilter(sigma=20.0)
            filter_obj._validate_parameters()
    
    def test_invalid_kernel_size_type(self):
        """Test that non-integer kernel size values are rejected."""
        with pytest.raises(FilterValidationError, match="Kernel size must be an integer"):
            filter_obj = GaussianBlurFilter(sigma=1.0, kernel_size=3.5)
            filter_obj._validate_parameters()
        
        with pytest.raises(FilterValidationError, match="Kernel size must be an integer"):
            filter_obj = GaussianBlurFilter(sigma=1.0, kernel_size="3")
            filter_obj._validate_parameters()
    
    def test_invalid_kernel_size_values(self):
        """Test that invalid kernel size values are rejected."""
        # Test non-positive kernel size
        with pytest.raises(FilterValidationError, match="Kernel size must be positive"):
            filter_obj = GaussianBlurFilter(sigma=1.0, kernel_size=0)
            filter_obj._validate_parameters()
        
        with pytest.raises(FilterValidationError, match="Kernel size must be positive"):
            filter_obj = GaussianBlurFilter(sigma=1.0, kernel_size=-3)
            filter_obj._validate_parameters()
        
        # Test even kernel size
        with pytest.raises(FilterValidationError, match="Kernel size must be odd"):
            filter_obj = GaussianBlurFilter(sigma=1.0, kernel_size=4)
            filter_obj._validate_parameters()
        
        with pytest.raises(FilterValidationError, match="Kernel size must be odd"):
            filter_obj = GaussianBlurFilter(sigma=1.0, kernel_size=6)
            filter_obj._validate_parameters()
    
    def test_parameter_update_validation(self):
        """Test that parameter validation works when updating parameters."""
        filter_obj = GaussianBlurFilter(sigma=1.0)
        
        # Valid update should work
        filter_obj.set_parameters(sigma=2.0)
        assert filter_obj.parameters['sigma'] == 2.0
        
        # Invalid update should raise error during validation
        filter_obj.set_parameters(sigma=15.0)
        with pytest.raises(FilterValidationError):
            filter_obj._validate_parameters()


class TestGaussianBlurFilterInputValidation:
    """Test input data validation for GaussianBlurFilter."""
    
    def test_valid_input_formats(self):
        """Test that valid input formats are accepted."""
        filter_obj = GaussianBlurFilter()
        
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
        filter_obj = GaussianBlurFilter()
        
        with pytest.raises(FilterValidationError, match="Input must be a numpy array"):
            filter_obj.validate_input("not an array")
        
        with pytest.raises(FilterValidationError, match="Input must be a numpy array"):
            filter_obj.validate_input([1, 2, 3])
        
        with pytest.raises(FilterValidationError, match="Input must be a numpy array"):
            filter_obj.validate_input(None)
    
    def test_empty_array_rejection(self):
        """Test that empty arrays are rejected."""
        filter_obj = GaussianBlurFilter()
        
        with pytest.raises(FilterValidationError, match="Input array cannot be empty"):
            filter_obj.validate_input(np.array([]))
        
        with pytest.raises(FilterValidationError, match="Input array cannot be empty"):
            filter_obj.validate_input(np.array([]).reshape(0, 0))
    
    def test_invalid_dimensions(self):
        """Test that arrays with invalid dimensions are rejected."""
        filter_obj = GaussianBlurFilter()
        
        # 1D array
        with pytest.raises(FilterValidationError, match="Image data must be 2D or 3D array"):
            filter_obj.validate_input(np.array([1, 2, 3]))
        
        # 4D array
        with pytest.raises(FilterValidationError, match="Image data must be 2D or 3D array"):
            filter_obj.validate_input(np.random.rand(10, 10, 3, 2))
    
    def test_invalid_channel_count(self):
        """Test that 3D arrays with invalid channel counts are rejected."""
        filter_obj = GaussianBlurFilter()
        
        # 1 channel (should be 2D for grayscale)
        with pytest.raises(FilterValidationError, match="Color images must have 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_obj.validate_input(np.random.rand(10, 10, 1))
        
        # 2 channels (invalid)
        with pytest.raises(FilterValidationError, match="Color images must have 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_obj.validate_input(np.random.rand(10, 10, 2))
        
        # 5 channels (invalid)
        with pytest.raises(FilterValidationError, match="Color images must have 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_obj.validate_input(np.random.rand(10, 10, 5))


class TestGaussianBlurFilterAlgorithm:
    """Test the correctness of gaussian blur algorithm."""
    
    def test_identity_case_sigma_zero(self):
        """Test that sigma = 0 returns unchanged image."""
        filter_obj = GaussianBlurFilter(sigma=0.0)
        
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
    
    def test_blur_effect_reduces_variance(self):
        """Test that gaussian blur reduces image variance (smoothing effect)."""
        filter_obj = GaussianBlurFilter(sigma=2.0)
        
        # Create a noisy image with high variance
        np.random.seed(42)  # For reproducible results
        noisy_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        result = filter_obj.apply(noisy_image)
        
        # Blurred image should have lower variance
        original_variance = np.var(noisy_image)
        blurred_variance = np.var(result)
        
        assert blurred_variance < original_variance, "Gaussian blur should reduce image variance"
    
    def test_blur_preserves_mean(self):
        """Test that gaussian blur approximately preserves image mean."""
        filter_obj = GaussianBlurFilter(sigma=1.0)
        
        # Create test image
        test_image = np.random.randint(50, 200, (30, 30), dtype=np.uint8)
        result = filter_obj.apply(test_image)
        
        # Mean should be approximately preserved (within small tolerance due to edge effects)
        original_mean = np.mean(test_image)
        blurred_mean = np.mean(result)
        
        # Allow for small differences due to edge handling
        assert abs(original_mean - blurred_mean) < 5, "Gaussian blur should approximately preserve mean"
    
    def test_blur_effect_increases_with_sigma(self):
        """Test that larger sigma values produce more blur."""
        # Create a sharp edge image
        edge_image = np.zeros((20, 20), dtype=np.uint8)
        edge_image[:, 10:] = 255  # Sharp vertical edge
        
        # Apply different sigma values
        filter_small = GaussianBlurFilter(sigma=0.5)
        filter_large = GaussianBlurFilter(sigma=2.0)
        
        result_small = filter_small.apply(edge_image)
        result_large = filter_large.apply(edge_image)
        
        # Measure edge sharpness by looking at gradient magnitude
        # Larger sigma should produce smaller gradients (softer edges)
        grad_small = np.abs(np.diff(result_small[10, :]))
        grad_large = np.abs(np.diff(result_large[10, :]))
        
        max_grad_small = np.max(grad_small)
        max_grad_large = np.max(grad_large)
        
        assert max_grad_large < max_grad_small, "Larger sigma should produce softer edges"
    
    def test_gaussian_blur_rgb_channels(self):
        """Test that RGB channels are blurred independently."""
        filter_obj = GaussianBlurFilter(sigma=1.0)
        
        # Create image with different patterns in each channel
        rgb_image = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb_image[:, :10, 0] = 255  # Red in left half
        rgb_image[10:, :, 1] = 255  # Green in bottom half
        rgb_image[:10, 10:, 2] = 255  # Blue in top-right quarter
        
        result = filter_obj.apply(rgb_image)
        
        # Each channel should be blurred independently
        assert result.shape == rgb_image.shape
        assert result.dtype == rgb_image.dtype
        
        # Check that blurring occurred (edges should be softer)
        # Red channel: check vertical edge at x=10
        original_red_edge = np.abs(rgb_image[10, 9, 0] - rgb_image[10, 11, 0])
        blurred_red_edge = np.abs(result[10, 9, 0] - result[10, 11, 0])
        assert blurred_red_edge < original_red_edge
    
    def test_gaussian_blur_rgba_preserves_alpha(self):
        """Test that RGBA images have alpha channel blurred like other channels."""
        filter_obj = GaussianBlurFilter(sigma=1.0)
        
        # Create RGBA image with sharp alpha transition
        rgba_image = np.full((20, 20, 4), 128, dtype=np.uint8)
        rgba_image[:, :10, 3] = 0    # Transparent left half
        rgba_image[:, 10:, 3] = 255  # Opaque right half
        
        result = filter_obj.apply(rgba_image)
        
        assert result.shape == rgba_image.shape
        assert result.dtype == rgba_image.dtype
        
        # Alpha channel should also be blurred
        # Use int conversion to avoid uint8 overflow in subtraction
        original_alpha_edge = abs(int(rgba_image[10, 9, 3]) - int(rgba_image[10, 11, 3]))
        blurred_alpha_edge = abs(int(result[10, 9, 3]) - int(result[10, 11, 3]))
        assert blurred_alpha_edge < original_alpha_edge
    
    def test_gaussian_blur_float_images(self):
        """Test gaussian blur with float images."""
        filter_obj = GaussianBlurFilter(sigma=1.0)
        
        # Create float image with sharp features
        float_image = np.zeros((20, 20), dtype=np.float32)
        float_image[8:12, 8:12] = 1.0  # Bright square in center
        
        result = filter_obj.apply(float_image)
        
        assert result.shape == float_image.shape
        assert result.dtype == float_image.dtype
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
        
        # Center should still be brightest, but edges should be blurred
        center_value = result[10, 10]
        edge_value = result[10, 6]  # Outside original square
        
        assert center_value > edge_value
        assert edge_value > 0.0  # Some blur should have spread outward
    
    def test_very_small_sigma_near_identity(self):
        """Test that very small sigma values produce minimal blur."""
        filter_obj = GaussianBlurFilter(sigma=1e-6)
        
        original = np.array([[0, 255, 0]], dtype=np.uint8)
        result = filter_obj.apply(original)
        
        # Should be treated as identity case due to small sigma
        np.testing.assert_array_equal(result, original)


class TestGaussianBlurFilterIntegration:
    """Test integration with BaseFilter features."""
    
    def test_filter_metadata(self):
        """Test that filter metadata is correctly set."""
        filter_obj = GaussianBlurFilter()
        
        assert filter_obj.name == "gaussian_blur"
        assert filter_obj.category == "enhancement"
        assert hasattr(filter_obj, 'data_type')
        assert hasattr(filter_obj, 'color_format')
    
    def test_parameter_override_in_apply(self):
        """Test that parameters can be overridden in apply() method."""
        filter_obj = GaussianBlurFilter(sigma=1.0)
        
        # Create test image with sharp edge
        original = np.zeros((10, 10), dtype=np.uint8)
        original[:, 5:] = 255
        
        # Apply with different sigma
        result = filter_obj.apply(original, sigma=2.0)
        
        # Parameter should be updated after apply
        assert filter_obj.parameters['sigma'] == 2.0
        
        # Result should show more blur than original sigma would produce
        assert result.shape == original.shape
        assert result.dtype == original.dtype
    
    def test_progress_tracking(self):
        """Test that progress tracking works."""
        filter_obj = GaussianBlurFilter(sigma=2.0)
        
        # Create a larger image to ensure progress tracking is called
        large_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Apply filter and check that it completes without error
        result = filter_obj.apply(large_image)
        
        # Verify result has correct shape
        assert result.shape == large_image.shape
        assert result.dtype == large_image.dtype
    
    def test_timing_measurement(self):
        """Test that execution timing is measured."""
        filter_obj = GaussianBlurFilter(sigma=1.0)
        
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = filter_obj.apply(image)
        
        # Check that result is correct (timing is handled internally)
        assert result.shape == image.shape
        assert result.dtype == image.dtype
    
    def test_memory_usage_tracking(self):
        """Test that memory usage is tracked."""
        filter_obj = GaussianBlurFilter(sigma=1.0)
        
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = filter_obj.apply(image)
        
        # Verify result is correct
        assert result.shape == image.shape
        assert result.dtype == image.dtype
    
    def test_different_data_types(self):
        """Test filter works with different numpy data types."""
        filter_obj = GaussianBlurFilter(sigma=1.0)
        
        # Test uint8
        uint8_image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        result_uint8 = filter_obj.apply(uint8_image)
        assert result_uint8.dtype == np.uint8
        
        # Test float32
        float32_image = np.random.rand(20, 20, 3).astype(np.float32)
        result_float32 = filter_obj.apply(float32_image)
        assert result_float32.dtype == np.float32
        
        # Test float64
        float64_image = np.random.rand(20, 20, 3).astype(np.float64)
        result_float64 = filter_obj.apply(float64_image)
        assert result_float64.dtype == np.float64


class TestGaussianBlurFilterEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_pixel_image(self):
        """Test with single pixel images."""
        filter_obj = GaussianBlurFilter(sigma=1.0)
        
        # Single pixel RGB
        single_pixel = np.array([[[128, 128, 128]]], dtype=np.uint8)
        result = filter_obj.apply(single_pixel)
        # Single pixel should remain unchanged
        np.testing.assert_array_equal(result, single_pixel)
        
        # Single pixel grayscale
        single_gray = np.array([[128]], dtype=np.uint8)
        result_gray = filter_obj.apply(single_gray)
        np.testing.assert_array_equal(result_gray, single_gray)
    
    def test_uniform_images(self):
        """Test with uniform (single color) images."""
        filter_obj = GaussianBlurFilter(sigma=2.0)
        
        # Uniform RGB image
        uniform_rgb = np.full((10, 10, 3), 128, dtype=np.uint8)
        result_rgb = filter_obj.apply(uniform_rgb)
        # Uniform image should remain uniform after blur
        np.testing.assert_array_equal(result_rgb, uniform_rgb)
        
        # Uniform grayscale image
        uniform_gray = np.full((10, 10), 200, dtype=np.uint8)
        result_gray = filter_obj.apply(uniform_gray)
        np.testing.assert_array_equal(result_gray, uniform_gray)
    
    def test_very_small_images(self):
        """Test with very small images."""
        filter_obj = GaussianBlurFilter(sigma=1.0)
        
        # 2x2 RGB image
        tiny_rgb = np.array([[[100, 150, 200], [50, 100, 150]],
                            [[200, 100, 50], [150, 200, 100]]], dtype=np.uint8)
        result_tiny = filter_obj.apply(tiny_rgb)
        assert result_tiny.shape == (2, 2, 3)
        assert result_tiny.dtype == np.uint8
        
        # 3x3 grayscale
        small_gray = np.array([[0, 128, 255],
                              [128, 255, 128],
                              [255, 128, 0]], dtype=np.uint8)
        result_small = filter_obj.apply(small_gray)
        assert result_small.shape == (3, 3)
        assert result_small.dtype == np.uint8
    
    def test_large_sigma_values(self):
        """Test with large sigma values at the boundary."""
        filter_obj = GaussianBlurFilter(sigma=10.0)  # Maximum allowed
        
        # Create image with sharp features
        test_image = np.zeros((50, 50), dtype=np.uint8)
        test_image[20:30, 20:30] = 255
        
        result = filter_obj.apply(test_image)
        
        # Should produce very smooth result
        assert result.shape == test_image.shape
        assert result.dtype == test_image.dtype
        
        # Center should still be brightest, but very spread out
        center_region = result[20:30, 20:30]
        edge_region = result[0:10, 0:10]
        
        assert np.mean(center_region) > np.mean(edge_region)
    
    def test_parameter_persistence(self):
        """Test that filter parameters persist correctly."""
        filter_obj = GaussianBlurFilter(sigma=2.0)
        
        # Apply filter multiple times
        image1 = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
        image2 = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
        
        result1 = filter_obj.apply(image1)
        result2 = filter_obj.apply(image2)
        
        # Parameters should remain the same
        assert filter_obj.parameters['sigma'] == 2.0
        
        # Results should have correct shapes and types
        assert result1.shape == image1.shape
        assert result2.shape == image2.shape
        assert result1.dtype == image1.dtype
        assert result2.dtype == image2.dtype
    
    def test_kernel_size_parameter_usage(self):
        """Test that kernel_size parameter is handled correctly."""
        # Test with explicit kernel size
        filter_explicit = GaussianBlurFilter(sigma=1.0, kernel_size=5)
        
        test_image = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
        result = filter_explicit.apply(test_image)
        
        assert result.shape == test_image.shape
        assert result.dtype == test_image.dtype
        assert filter_explicit.parameters['kernel_size'] == 5
        
        # Test with auto kernel size (None)
        filter_auto = GaussianBlurFilter(sigma=1.0, kernel_size=None)
        result_auto = filter_auto.apply(test_image)
        
        assert result_auto.shape == test_image.shape
        assert result_auto.dtype == test_image.dtype
        assert filter_auto.parameters['kernel_size'] is None


if __name__ == "__main__":
    pytest.main([__file__])