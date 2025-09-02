"""
Unit tests for HueRotationFilter.

Tests parameter validation, HSV color space conversion, hue rotation,
and integration with BaseFilter features including progress tracking and timing.
"""

import pytest
import numpy as np
from image_processing_library.filters.enhancement.color_filters import HueRotationFilter
from image_processing_library.core.utils import FilterValidationError


class TestHueRotationFilter:
    """Test suite for HueRotationFilter functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.filter = HueRotationFilter()
    
    def test_filter_initialization(self):
        """Test that HueRotationFilter initializes with correct metadata."""
        assert self.filter.name == "hue_rotation"
        assert self.filter.category == "enhancement"
        assert self.filter.data_type.value == "image"
        assert self.filter.color_format.value == "rgb"
        
        # Test default parameter
        params = self.filter.get_parameters()
        assert params['rotation_degrees'] == 0.0
    
    def test_filter_initialization_with_parameters(self):
        """Test HueRotationFilter initialization with custom parameters."""
        filter_custom = HueRotationFilter(rotation_degrees=180.0)
        params = filter_custom.get_parameters()
        assert params['rotation_degrees'] == 180.0
    
    def test_parameter_validation_valid(self):
        """Test parameter validation with valid parameters."""
        # Test valid rotation degrees
        valid_rotations = [0.0, 45.0, 90.0, 180.0, 270.0, 360.0]
        
        for rotation in valid_rotations:
            self.filter.set_parameters(rotation_degrees=rotation)
            self.filter._validate_parameters()  # Should not raise
    
    def test_parameter_validation_invalid(self):
        """Test parameter validation with invalid parameters."""
        # Test negative rotation degrees
        self.filter.set_parameters(rotation_degrees=-0.1)
        with pytest.raises(FilterValidationError, match="rotation_degrees must be in range \\[0.0, 360.0\\]"):
            self.filter._validate_parameters()
        
        # Test rotation degrees too high
        self.filter.set_parameters(rotation_degrees=360.1)
        with pytest.raises(FilterValidationError, match="rotation_degrees must be in range \\[0.0, 360.0\\]"):
            self.filter._validate_parameters()
        
        # Test non-numeric rotation degrees
        self.filter.set_parameters(rotation_degrees="invalid")
        with pytest.raises(FilterValidationError, match="rotation_degrees must be a number"):
            self.filter._validate_parameters()
        
        # Test None rotation degrees
        self.filter.set_parameters(rotation_degrees=None)
        with pytest.raises(FilterValidationError, match="rotation_degrees must be a number"):
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
    
    def test_identity_case_rotation_0_degrees(self):
        """Test that rotation of 0 degrees returns unchanged image."""
        # Test with RGB image
        rgb_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        self.filter.set_parameters(rotation_degrees=0.0)
        result = self.filter.apply(rgb_image)
        
        np.testing.assert_array_equal(result, rgb_image)
        assert result.dtype == rgb_image.dtype
        assert result.shape == rgb_image.shape
    
    def test_identity_case_rotation_360_degrees(self):
        """Test that rotation of 360 degrees returns unchanged image."""
        # Test with RGB image
        rgb_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        self.filter.set_parameters(rotation_degrees=360.0)
        result = self.filter.apply(rgb_image)
        
        np.testing.assert_array_equal(result, rgb_image)
        assert result.dtype == rgb_image.dtype
        assert result.shape == rgb_image.shape
    
    def test_grayscale_image_unchanged(self):
        """Test that grayscale images are returned unchanged."""
        gray_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        self.filter.set_parameters(rotation_degrees=180.0)
        result = self.filter.apply(gray_image)
        
        np.testing.assert_array_equal(result, gray_image)
        assert result.dtype == gray_image.dtype
        assert result.shape == gray_image.shape
    
    def test_hue_rotation_180_degrees(self):
        """Test that 180-degree rotation produces complementary colors."""
        # Create primary color pixels
        rgb_image = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # Red, Green, Blue
        ], dtype=np.uint8)
        
        self.filter.set_parameters(rotation_degrees=180.0)
        result = self.filter.apply(rgb_image)
        
        # After 180-degree rotation:
        # Red (hue=0) -> Cyan-ish (hue=180)
        # Green (hue=120) -> Magenta-ish (hue=300)  
        # Blue (hue=240) -> Yellow-ish (hue=60)
        
        # The dominant channel should change for each pixel
        # Red pixel should no longer have red as dominant
        assert result[0, 0, 0] < max(result[0, 0, 1], result[0, 0, 2])
        # Green pixel should no longer have green as dominant
        assert result[0, 1, 1] < max(result[0, 1, 0], result[0, 1, 2])
        # Blue pixel should no longer have blue as dominant
        assert result[0, 2, 2] < max(result[0, 2, 0], result[0, 2, 1])
        
        assert result.dtype == rgb_image.dtype
        assert result.shape == rgb_image.shape
    
    def test_hue_rotation_120_degrees(self):
        """Test 120-degree hue rotation (color wheel shift)."""
        # Pure red should become pure green after 120-degree rotation
        red_pixel = np.array([[[255, 0, 0]]], dtype=np.uint8)
        
        self.filter.set_parameters(rotation_degrees=120.0)
        result = self.filter.apply(red_pixel)
        
        # After 120-degree rotation, red should become green-dominant
        assert result[0, 0, 1] > result[0, 0, 0]  # Green > Red
        assert result[0, 0, 1] > result[0, 0, 2]  # Green > Blue
        
        assert result.dtype == red_pixel.dtype
        assert result.shape == red_pixel.shape
    
    def test_hue_rotation_wraparound(self):
        """Test hue rotation wraparound behavior."""
        # Test that rotation wraps around correctly within valid range
        rgb_image = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Pure red
        
        # 30-degree rotation
        self.filter.set_parameters(rotation_degrees=30.0)
        result_30 = self.filter.apply(rgb_image.copy())
        
        # Test that 0 and 360 degrees give the same result (wraparound)
        self.filter.set_parameters(rotation_degrees=0.0)
        result_0 = self.filter.apply(rgb_image.copy())
        
        self.filter.set_parameters(rotation_degrees=360.0)
        result_360 = self.filter.apply(rgb_image.copy())
        
        # 0 and 360 degrees should give identical results
        np.testing.assert_array_equal(result_0, result_360)
        
        # Both should be identical to original (no rotation)
        np.testing.assert_array_equal(result_0, rgb_image)
        np.testing.assert_array_equal(result_360, rgb_image)
    
    def test_rgba_alpha_preservation(self):
        """Test that alpha channel is preserved in RGBA images."""
        rgba_image = np.array([
            [[255, 0, 0, 128], [0, 255, 0, 64], [0, 0, 255, 192]],
            [[255, 255, 0, 255], [255, 0, 255, 32], [0, 255, 255, 96]]
        ], dtype=np.uint8)
        
        original_alpha = rgba_image[..., 3].copy()
        
        self.filter.set_parameters(rotation_degrees=90.0)
        result = self.filter.apply(rgba_image)
        
        # Alpha channel should be unchanged
        np.testing.assert_array_equal(result[..., 3], original_alpha)
        assert result.dtype == rgba_image.dtype
        assert result.shape == rgba_image.shape
    
    def test_float_image_processing(self):
        """Test hue rotation with float images."""
        # Create float RGB image in [0, 1] range
        rgb_image = np.array([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # Pure colors
            [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8]]   # Tinted colors
        ], dtype=np.float32)
        
        self.filter.set_parameters(rotation_degrees=60.0)
        result = self.filter.apply(rgb_image)
        
        # Result should be valid float values in [0, 1] range
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert result.dtype == np.float32
        assert result.shape == rgb_image.shape
        
        # Colors should have changed (not identical to original)
        assert not np.array_equal(result, rgb_image)
    
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
        self.filter.set_parameters(rotation_degrees=45.0)
        result = self.filter.apply(single_pixel)
        
        assert result.shape == single_pixel.shape
        assert result.dtype == single_pixel.dtype
        
        # Test minimum size grayscale (should be unchanged)
        min_gray = np.array([[128]], dtype=np.uint8)
        result = self.filter.apply(min_gray)
        np.testing.assert_array_equal(result, min_gray)
        
        # Test maximum rotation angle
        rgb_image = np.array([[[100, 50, 150]]], dtype=np.uint8)
        self.filter.set_parameters(rotation_degrees=360.0)
        result = self.filter.apply(rgb_image)
        
        # Should be identical to original (full rotation)
        np.testing.assert_array_equal(result, rgb_image)
    
    def test_parameter_override_in_apply(self):
        """Test parameter override through apply method kwargs."""
        rgb_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        
        # Set initial parameter
        self.filter.set_parameters(rotation_degrees=0.0)
        
        # Override in apply call
        result = self.filter.apply(rgb_image, rotation_degrees=90.0)
        
        # Parameter should be updated
        assert self.filter.get_parameters()['rotation_degrees'] == 90.0
        
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
        self.filter.set_parameters(rotation_degrees=90.0)
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
        self.filter.set_parameters(rotation_degrees=180.0)
        self.filter.apply(test_image)
        
        # Check timing metadata
        assert self.filter.metadata.execution_time > 0
        assert self.filter.metadata.error_message is None
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Apply filter
        self.filter.set_parameters(rotation_degrees=270.0)
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
        assert 'rotation_degrees' in initial_params
        
        # Test parameter updates
        self.filter.set_parameters(rotation_degrees=120.0)
        updated_params = self.filter.get_parameters()
        assert updated_params['rotation_degrees'] == 120.0
        
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
            self.filter.apply(np.random.rand(10, 10, 3), rotation_degrees=-10.0)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large images."""
        # Create a larger image to test memory management
        large_image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        
        self.filter.set_parameters(rotation_degrees=45.0)
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
    
    def test_hue_rotation_specific_angles(self):
        """Test hue rotation at specific meaningful angles."""
        # Test with a known color
        orange_rgb = np.array([[[255, 165, 0]]], dtype=np.uint8)  # Orange
        
        # Test 90-degree rotation
        self.filter.set_parameters(rotation_degrees=90.0)
        result_90 = self.filter.apply(orange_rgb.copy())
        
        # Test 180-degree rotation  
        self.filter.set_parameters(rotation_degrees=180.0)
        result_180 = self.filter.apply(orange_rgb.copy())
        
        # Test 270-degree rotation
        self.filter.set_parameters(rotation_degrees=270.0)
        result_270 = self.filter.apply(orange_rgb.copy())
        
        # All results should be different from original and from each other
        assert not np.array_equal(result_90, orange_rgb)
        assert not np.array_equal(result_180, orange_rgb)
        assert not np.array_equal(result_270, orange_rgb)
        assert not np.array_equal(result_90, result_180)
        assert not np.array_equal(result_180, result_270)
        assert not np.array_equal(result_90, result_270)
        
        # All should have same shape and dtype
        for result in [result_90, result_180, result_270]:
            assert result.shape == orange_rgb.shape
            assert result.dtype == orange_rgb.dtype
    
    def test_hue_rotation_preserves_brightness_and_saturation(self):
        """Test that hue rotation preserves brightness (value) and saturation."""
        # Create a test image with known HSV properties
        rgb_image = np.array([
            [[255, 128, 64], [128, 255, 128], [64, 64, 255]],  # Different colors
        ], dtype=np.uint8)
        
        # Convert to HSV to get original values
        rgb_normalized = rgb_image.astype(np.float32) / 255.0
        original_hsv = self.filter._rgb_to_hsv_vectorized(rgb_normalized.reshape(-1, 3))
        
        # Apply hue rotation
        self.filter.set_parameters(rotation_degrees=60.0)
        result = self.filter.apply(rgb_image)
        
        # Convert result back to HSV
        result_normalized = result.astype(np.float32) / 255.0
        result_hsv = self.filter._rgb_to_hsv_vectorized(result_normalized.reshape(-1, 3))
        
        # Saturation and Value should be preserved (approximately)
        np.testing.assert_allclose(result_hsv[:, 1], original_hsv[:, 1], atol=0.01)  # Saturation
        np.testing.assert_allclose(result_hsv[:, 2], original_hsv[:, 2], atol=0.01)  # Value
        
        # Hue should be different (rotated)
        hue_diff = np.abs(result_hsv[:, 0] - original_hsv[:, 0])
        # Account for wraparound
        hue_diff = np.minimum(hue_diff, 1.0 - hue_diff)
        assert np.all(hue_diff > 0.01)  # Hue should have changed significantly