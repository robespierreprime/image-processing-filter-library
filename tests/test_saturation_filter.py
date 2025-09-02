"""
Unit tests for SaturationFilter.

Tests parameter validation, HSV color space conversion, saturation adjustment,
and integration with BaseFilter features including progress tracking and timing.
"""

import pytest
import numpy as np
from image_processing_library.filters.enhancement.color_filters import SaturationFilter
from image_processing_library.core.utils import FilterValidationError


class TestSaturationFilter:
    """Test suite for SaturationFilter functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.filter = SaturationFilter()
    
    def test_filter_initialization(self):
        """Test that SaturationFilter initializes with correct metadata."""
        assert self.filter.name == "saturation"
        assert self.filter.category == "enhancement"
        assert self.filter.data_type.value == "image"
        assert self.filter.color_format.value == "rgb"
        
        # Test default parameter
        params = self.filter.get_parameters()
        assert params['saturation_factor'] == 1.0
    
    def test_filter_initialization_with_parameters(self):
        """Test SaturationFilter initialization with custom parameters."""
        filter_custom = SaturationFilter(saturation_factor=2.0)
        params = filter_custom.get_parameters()
        assert params['saturation_factor'] == 2.0
    
    def test_parameter_validation_valid(self):
        """Test parameter validation with valid parameters."""
        # Test valid saturation factors
        valid_factors = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        
        for factor in valid_factors:
            self.filter.set_parameters(saturation_factor=factor)
            self.filter._validate_parameters()  # Should not raise
    
    def test_parameter_validation_invalid(self):
        """Test parameter validation with invalid parameters."""
        # Test negative saturation factor
        self.filter.set_parameters(saturation_factor=-0.1)
        with pytest.raises(FilterValidationError, match="saturation_factor must be in range \\[0.0, 3.0\\]"):
            self.filter._validate_parameters()
        
        # Test saturation factor too high
        self.filter.set_parameters(saturation_factor=3.1)
        with pytest.raises(FilterValidationError, match="saturation_factor must be in range \\[0.0, 3.0\\]"):
            self.filter._validate_parameters()
        
        # Test non-numeric saturation factor
        self.filter.set_parameters(saturation_factor="invalid")
        with pytest.raises(FilterValidationError, match="saturation_factor must be a number"):
            self.filter._validate_parameters()
        
        # Test None saturation factor
        self.filter.set_parameters(saturation_factor=None)
        with pytest.raises(FilterValidationError, match="saturation_factor must be a number"):
            self.filter._validate_parameters()
    
    def test_input_validation_valid_inputs(self):
        """Test input validation with valid inputs."""
        # Test RGB image (3D array with 3 channels)
        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        assert self.filter.validate_input(rgb_image) is True
        
        # Test RGBA image (3D array with 4 channels)
        rgba_image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        assert self.filter.validate_input(rgba_image) is True
        
        # Test grayscale image (2D array)
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        assert self.filter.validate_input(gray_image) is True
        
        # Test float images
        float_rgb = np.random.rand(50, 50, 3).astype(np.float32)
        assert self.filter.validate_input(float_rgb) is True
    
    def test_input_validation_invalid_inputs(self):
        """Test input validation with invalid inputs."""
        # Test non-numpy array
        with pytest.raises(FilterValidationError, match="Input must be a numpy array"):
            self.filter.validate_input([1, 2, 3])
        
        # Test empty array
        with pytest.raises(FilterValidationError, match="Input array cannot be empty"):
            self.filter.validate_input(np.array([]))
        
        # Test wrong dimensions (1D array)
        with pytest.raises(FilterValidationError, match="Image data must be 2D or 3D array"):
            self.filter.validate_input(np.array([1, 2, 3]))
        
        # Test wrong dimensions (4D array)
        with pytest.raises(FilterValidationError, match="Image data must be 2D or 3D array"):
            self.filter.validate_input(np.random.rand(2, 100, 100, 3))
        
        # Test wrong number of channels
        with pytest.raises(FilterValidationError, match="Color images must have 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            self.filter.validate_input(np.random.rand(100, 100, 2))
    
    def test_identity_case_saturation_1_0(self):
        """Test that saturation factor of 1.0 returns unchanged image."""
        # Test with RGB image
        rgb_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        self.filter.set_parameters(saturation_factor=1.0)
        result = self.filter.apply(rgb_image)
        
        np.testing.assert_array_equal(result, rgb_image)
        assert result.dtype == rgb_image.dtype
        assert result.shape == rgb_image.shape
    
    def test_grayscale_image_unchanged(self):
        """Test that grayscale images are returned unchanged."""
        gray_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        self.filter.set_parameters(saturation_factor=2.0)
        result = self.filter.apply(gray_image)
        
        np.testing.assert_array_equal(result, gray_image)
        assert result.dtype == gray_image.dtype
        assert result.shape == gray_image.shape
    
    def test_zero_saturation_grayscale_conversion(self):
        """Test that saturation factor of 0.0 converts to grayscale."""
        # Create a colorful RGB image
        rgb_image = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # Pure red, green, blue
            [[255, 255, 0], [255, 0, 255], [0, 255, 255]]  # Yellow, magenta, cyan
        ], dtype=np.uint8)
        
        self.filter.set_parameters(saturation_factor=0.0)
        result = self.filter.apply(rgb_image)
        
        # With zero saturation, all channels should be equal (grayscale)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                r, g, b = result[i, j]
                assert r == g == b, f"Pixel at ({i}, {j}) is not grayscale: {result[i, j]}"
        
        assert result.dtype == rgb_image.dtype
        assert result.shape == rgb_image.shape
    
    def test_increased_saturation(self):
        """Test that saturation factor > 1.0 increases color intensity."""
        # Create a muted color image
        rgb_image = np.array([
            [[200, 150, 150], [150, 200, 150], [150, 150, 200]],  # Muted red, green, blue
        ], dtype=np.uint8)
        
        self.filter.set_parameters(saturation_factor=2.0)
        result = self.filter.apply(rgb_image)
        
        # The dominant color channel should become more dominant
        # Red pixel should become more red (higher difference between R and G,B)
        original_red_dominance = rgb_image[0, 0, 0] - max(rgb_image[0, 0, 1], rgb_image[0, 0, 2])
        result_red_dominance = result[0, 0, 0] - max(result[0, 0, 1], result[0, 0, 2])
        
        assert result_red_dominance > original_red_dominance
        assert result.dtype == rgb_image.dtype
        assert result.shape == rgb_image.shape
    
    def test_rgba_alpha_preservation(self):
        """Test that alpha channel is preserved in RGBA images."""
        rgba_image = np.array([
            [[255, 0, 0, 128], [0, 255, 0, 64], [0, 0, 255, 192]],
            [[255, 255, 0, 255], [255, 0, 255, 32], [0, 255, 255, 96]]
        ], dtype=np.uint8)
        
        original_alpha = rgba_image[..., 3].copy()
        
        self.filter.set_parameters(saturation_factor=2.0)
        result = self.filter.apply(rgba_image)
        
        # Alpha channel should be unchanged
        np.testing.assert_array_equal(result[..., 3], original_alpha)
        assert result.dtype == rgba_image.dtype
        assert result.shape == rgba_image.shape
    
    def test_float_image_processing(self):
        """Test saturation adjustment with float images."""
        # Create float RGB image in [0, 1] range
        rgb_image = np.array([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # Pure colors
            [[0.8, 0.6, 0.6], [0.6, 0.8, 0.6], [0.6, 0.6, 0.8]]   # Muted colors
        ], dtype=np.float32)
        
        self.filter.set_parameters(saturation_factor=0.5)
        result = self.filter.apply(rgb_image)
        
        # Result should be valid float values in [0, 1] range
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert result.dtype == np.float32
        assert result.shape == rgb_image.shape
        
        # Pure colors should become less saturated
        # The difference between channels should decrease
        original_diff = np.abs(rgb_image[0, 0, 0] - rgb_image[0, 0, 1])
        result_diff = np.abs(result[0, 0, 0] - result[0, 0, 1])
        assert result_diff < original_diff
    
    def test_hsv_conversion_accuracy(self):
        """Test accuracy of HSV color space conversion."""
        # Test with known RGB values and expected HSV conversions
        test_cases = [
            # RGB -> Expected HSV (approximately)
            ([255, 0, 0], [0.0, 1.0, 1.0]),      # Pure red
            ([0, 255, 0], [120/360, 1.0, 1.0]),  # Pure green  
            ([0, 0, 255], [240/360, 1.0, 1.0]),  # Pure blue
            ([255, 255, 255], [0.0, 0.0, 1.0]),  # White
            ([0, 0, 0], [0.0, 0.0, 0.0]),        # Black
            ([128, 128, 128], [0.0, 0.0, 0.5]),  # Gray
        ]
        
        for rgb, expected_hsv in test_cases:
            # Create single pixel image
            rgb_array = np.array([[rgb]], dtype=np.uint8)
            rgb_normalized = rgb_array.astype(np.float32) / 255.0
            
            # Test our HSV conversion
            hsv_result = self.filter._rgb_to_hsv_vectorized(rgb_normalized.reshape(-1, 3))
            
            # Check conversion accuracy (with some tolerance for floating point)
            np.testing.assert_allclose(hsv_result[0], expected_hsv, atol=0.01)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test single pixel image
        single_pixel = np.array([[[128, 64, 192]]], dtype=np.uint8)
        self.filter.set_parameters(saturation_factor=1.5)
        result = self.filter.apply(single_pixel)
        
        assert result.shape == single_pixel.shape
        assert result.dtype == single_pixel.dtype
        
        # Test minimum size grayscale (should be unchanged)
        min_gray = np.array([[128]], dtype=np.uint8)
        result = self.filter.apply(min_gray)
        np.testing.assert_array_equal(result, min_gray)
        
        # Test maximum saturation factor
        rgb_image = np.array([[[100, 50, 150]]], dtype=np.uint8)
        self.filter.set_parameters(saturation_factor=3.0)
        result = self.filter.apply(rgb_image)
        
        assert result.shape == rgb_image.shape
        assert result.dtype == rgb_image.dtype
        # Values should be clamped to valid range
        assert np.all(result >= 0)
        assert np.all(result <= 255)
    
    def test_parameter_override_in_apply(self):
        """Test parameter override through apply method kwargs."""
        rgb_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        
        # Set initial parameter
        self.filter.set_parameters(saturation_factor=1.0)
        
        # Override in apply call
        result = self.filter.apply(rgb_image, saturation_factor=2.0)
        
        # Parameter should be updated
        assert self.filter.get_parameters()['saturation_factor'] == 2.0
        
        # Result should reflect the overridden parameter
        assert result.shape == rgb_image.shape
        assert result.dtype == rgb_image.dtype
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        progress_values = []
        
        def progress_callback(progress):
            progress_values.append(progress)
        
        self.filter.set_progress_callback(progress_callback)
        
        # Apply filter to trigger progress updates
        test_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        self.filter.set_parameters(saturation_factor=1.5)
        self.filter.apply(test_image)
        
        # Check that progress was tracked
        assert len(progress_values) >= 2  # At least start (0.0) and end (1.0)
        assert progress_values[0] == 0.0
        assert progress_values[-1] == 1.0
        assert self.filter.metadata.progress == 1.0
    
    def test_timing_metadata(self):
        """Test execution time tracking."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Apply filter
        self.filter.set_parameters(saturation_factor=1.5)
        self.filter.apply(test_image)
        
        # Check timing metadata
        assert self.filter.metadata.execution_time > 0
        assert self.filter.metadata.error_message is None
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Apply filter
        self.filter.set_parameters(saturation_factor=1.5)
        result = self.filter.apply(test_image)
        
        # Check memory metadata
        assert self.filter.metadata.memory_usage > 0
        assert self.filter.metadata.peak_memory_usage > 0
        assert self.filter.metadata.input_shape == test_image.shape
        assert self.filter.metadata.output_shape == result.shape
        assert self.filter.metadata.memory_efficiency_ratio >= 0
    
    def test_integration_with_base_filter(self):
        """Test integration with BaseFilter features."""
        test_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        # Test parameter management
        initial_params = self.filter.get_parameters()
        assert isinstance(initial_params, dict)
        assert 'saturation_factor' in initial_params
        
        # Test parameter updates
        self.filter.set_parameters(saturation_factor=2.0)
        updated_params = self.filter.get_parameters()
        assert updated_params['saturation_factor'] == 2.0
        
        # Test metadata reset between applications
        self.filter.apply(test_image)
        first_execution_time = self.filter.metadata.execution_time
        
        self.filter.apply(test_image)
        second_execution_time = self.filter.metadata.execution_time
        
        # Each application should have its own timing
        assert first_execution_time > 0
        assert second_execution_time > 0
    
    def test_error_handling(self):
        """Test error handling and metadata recording."""
        # Test with invalid input to trigger error
        with pytest.raises(FilterValidationError):
            self.filter.apply("not an array")
        
        # Test with invalid parameters
        with pytest.raises(FilterValidationError):
            self.filter.apply(np.random.rand(10, 10, 3), saturation_factor=-1.0)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large images."""
        # Create a larger image to test memory management
        large_image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        
        self.filter.set_parameters(saturation_factor=1.5)
        result = self.filter.apply(large_image)
        
        # Verify result correctness
        assert result.shape == large_image.shape
        assert result.dtype == large_image.dtype
        
        # Memory metadata should be populated
        assert self.filter.metadata.memory_usage > 0
        assert self.filter.metadata.peak_memory_usage > 0
    
    def test_vectorized_hsv_conversion_consistency(self):
        """Test that vectorized HSV conversion is consistent."""
        # Create test RGB data
        rgb_data = np.random.randint(0, 256, (100, 3), dtype=np.uint8)
        rgb_normalized = rgb_data.astype(np.float32) / 255.0
        
        # Convert to HSV and back to RGB
        hsv_data = self.filter._rgb_to_hsv_vectorized(rgb_normalized)
        rgb_result = self.filter._hsv_to_rgb_vectorized(hsv_data)
        
        # Should be approximately equal to original (within floating point precision)
        np.testing.assert_allclose(rgb_result, rgb_normalized, atol=1e-6)
    
    def test_saturation_clamping(self):
        """Test that saturation values are properly clamped to [0, 1]."""
        # Create image with high saturation colors
        rgb_image = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Pure red
        
        # Apply very high saturation factor
        self.filter.set_parameters(saturation_factor=3.0)
        result = self.filter.apply(rgb_image)
        
        # Result should still be valid
        assert np.all(result >= 0)
        assert np.all(result <= 255)
        assert result.dtype == rgb_image.dtype