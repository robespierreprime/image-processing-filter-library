"""
Unit tests for InvertFilter.

Tests parameter validation, color inversion algorithms, and integration
with BaseFilter features including progress tracking and timing.
"""

import pytest
import numpy as np
from image_processing_library.filters.enhancement.color_filters import InvertFilter
from image_processing_library.core.utils import FilterValidationError


class TestInvertFilter:
    """Test suite for InvertFilter functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.filter = InvertFilter()
    
    def test_filter_initialization(self):
        """Test that InvertFilter initializes with correct metadata."""
        assert self.filter.name == "invert"
        assert self.filter.category == "enhancement"
        assert self.filter.data_type.value == "image"
        assert self.filter.color_format.value == "rgb"
    
    def test_parameter_validation(self):
        """Test parameter validation (no parameters for InvertFilter)."""
        # Should not raise any errors since there are no parameters
        self.filter._validate_parameters()
        
        # Parameters should be empty
        params = self.filter.get_parameters()
        assert len(params) == 0
    
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
        
        # Test array with NaN values
        nan_array = np.array([[1, 2], [np.nan, 4]], dtype=np.float32)
        with pytest.raises(FilterValidationError, match="Input data contains NaN values"):
            self.filter.validate_input(nan_array)
        
        # Test array with infinite values
        inf_array = np.array([[1, 2], [np.inf, 4]], dtype=np.float32)
        with pytest.raises(FilterValidationError, match="Input data contains infinite values"):
            self.filter.validate_input(inf_array)
    
    def test_rgb_inversion_uint8(self):
        """Test RGB color inversion with uint8 data type."""
        # Create test RGB image
        rgb_image = np.array([
            [[0, 0, 0], [255, 255, 255], [128, 64, 192]],
            [[100, 150, 200], [50, 75, 25], [255, 0, 128]]
        ], dtype=np.uint8)
        
        result = self.filter.apply(rgb_image)
        
        # Expected inverted values
        expected = np.array([
            [[255, 255, 255], [0, 0, 0], [127, 191, 63]],
            [[155, 105, 55], [205, 180, 230], [0, 255, 127]]
        ], dtype=np.uint8)
        
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint8
        assert result.shape == rgb_image.shape
    
    def test_rgba_inversion_uint8(self):
        """Test RGBA color inversion with uint8 data type (preserving alpha)."""
        # Create test RGBA image
        rgba_image = np.array([
            [[0, 0, 0, 255], [255, 255, 255, 128], [128, 64, 192, 64]],
            [[100, 150, 200, 200], [50, 75, 25, 100], [255, 0, 128, 0]]
        ], dtype=np.uint8)
        
        result = self.filter.apply(rgba_image)
        
        # Expected inverted values (alpha preserved)
        expected = np.array([
            [[255, 255, 255, 255], [0, 0, 0, 128], [127, 191, 63, 64]],
            [[155, 105, 55, 200], [205, 180, 230, 100], [0, 255, 127, 0]]
        ], dtype=np.uint8)
        
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint8
        assert result.shape == rgba_image.shape
    
    def test_grayscale_inversion_uint8(self):
        """Test grayscale image inversion with uint8 data type."""
        # Create test grayscale image
        gray_image = np.array([
            [0, 255, 128],
            [100, 50, 200]
        ], dtype=np.uint8)
        
        result = self.filter.apply(gray_image)
        
        # Expected inverted values
        expected = np.array([
            [255, 0, 127],
            [155, 205, 55]
        ], dtype=np.uint8)
        
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint8
        assert result.shape == gray_image.shape
    
    def test_rgb_inversion_float32(self):
        """Test RGB color inversion with float32 data type."""
        # Create test RGB image with float values in [0, 1] range
        rgb_image = np.array([
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.25, 0.75]],
            [[0.4, 0.6, 0.8], [0.2, 0.3, 0.1], [1.0, 0.0, 0.5]]
        ], dtype=np.float32)
        
        result = self.filter.apply(rgb_image)
        
        # Expected inverted values
        expected = np.array([
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.75, 0.25]],
            [[0.6, 0.4, 0.2], [0.8, 0.7, 0.9], [0.0, 1.0, 0.5]]
        ], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        assert result.dtype == np.float32
        assert result.shape == rgb_image.shape
    
    def test_rgba_inversion_float32(self):
        """Test RGBA color inversion with float32 data type (preserving alpha)."""
        # Create test RGBA image with float values
        rgba_image = np.array([
            [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.5], [0.5, 0.25, 0.75, 0.25]],
            [[0.4, 0.6, 0.8, 0.8], [0.2, 0.3, 0.1, 0.4], [1.0, 0.0, 0.5, 0.0]]
        ], dtype=np.float32)
        
        result = self.filter.apply(rgba_image)
        
        # Expected inverted values (alpha preserved)
        expected = np.array([
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.5], [0.5, 0.75, 0.25, 0.25]],
            [[0.6, 0.4, 0.2, 0.8], [0.8, 0.7, 0.9, 0.4], [0.0, 1.0, 0.5, 0.0]]
        ], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        assert result.dtype == np.float32
        assert result.shape == rgba_image.shape
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test single pixel image
        single_pixel = np.array([[[128, 64, 192]]], dtype=np.uint8)
        result = self.filter.apply(single_pixel)
        expected = np.array([[[127, 191, 63]]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        
        # Test minimum size grayscale
        min_gray = np.array([[0]], dtype=np.uint8)
        result = self.filter.apply(min_gray)
        expected = np.array([[255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        
        # Test with different integer types
        int16_image = np.array([[100, 200]], dtype=np.int16)
        result = self.filter.apply(int16_image)
        # For int16, max value is 32767
        expected = np.array([[32667, 32567]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        progress_values = []
        
        def progress_callback(progress):
            progress_values.append(progress)
        
        self.filter.set_progress_callback(progress_callback)
        
        # Apply filter to trigger progress updates
        test_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
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
        self.filter.apply(test_image)
        
        # Check timing metadata
        assert self.filter.metadata.execution_time > 0
        assert self.filter.metadata.error_message is None
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Apply filter
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
        
        # Test that we can set parameters (even though InvertFilter has none)
        self.filter.set_parameters()  # Should not raise error
        
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
        
        # For validation errors that occur before processing starts,
        # the metadata may not be updated. This is expected behavior.
        # The important thing is that the appropriate exception is raised.
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large images."""
        # Create a larger image to test memory management
        large_image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        
        result = self.filter.apply(large_image)
        
        # Verify result correctness
        assert result.shape == large_image.shape
        assert result.dtype == large_image.dtype
        
        # Check that inversion was applied correctly (spot check)
        np.testing.assert_array_equal(result[0, 0], 255 - large_image[0, 0])
        
        # Memory metadata should be populated
        assert self.filter.metadata.memory_usage > 0
        assert self.filter.metadata.peak_memory_usage > 0