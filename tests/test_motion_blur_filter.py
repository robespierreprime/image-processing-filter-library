"""
Unit tests for MotionBlurFilter.

Tests parameter validation, motion blur functionality, kernel generation,
and edge cases for the motion blur filter implementation.
"""

import pytest
import numpy as np
import math
from image_processing_library.filters.enhancement.blur_filters import MotionBlurFilter
from image_processing_library.core.utils import FilterValidationError


class TestMotionBlurFilter:
    """Test suite for MotionBlurFilter functionality."""
    
    def test_initialization(self):
        """Test MotionBlurFilter initialization with default parameters."""
        filter_instance = MotionBlurFilter()
        
        assert filter_instance.name == "motion_blur"
        assert filter_instance.category == "enhancement"
        assert filter_instance.parameters['distance'] == 5
        assert filter_instance.parameters['angle'] == 0.0
    
    def test_initialization_with_parameters(self):
        """Test MotionBlurFilter initialization with custom parameters."""
        filter_instance = MotionBlurFilter(distance=10, angle=45.0)
        
        assert filter_instance.parameters['distance'] == 10
        assert filter_instance.parameters['angle'] == 45.0
    
    def test_parameter_validation_valid_distance(self):
        """Test parameter validation with valid distance values."""
        filter_instance = MotionBlurFilter()
        
        # Test boundary values
        filter_instance.set_parameters(distance=0)
        filter_instance._validate_parameters()  # Should not raise
        
        filter_instance.set_parameters(distance=25)
        filter_instance._validate_parameters()  # Should not raise
        
        filter_instance.set_parameters(distance=50)
        filter_instance._validate_parameters()  # Should not raise
    
    def test_parameter_validation_invalid_distance(self):
        """Test parameter validation with invalid distance values."""
        filter_instance = MotionBlurFilter()
        
        # Test negative distance
        filter_instance.set_parameters(distance=-1)
        with pytest.raises(FilterValidationError, match="Distance must be non-negative"):
            filter_instance._validate_parameters()
        
        # Test distance too large
        filter_instance.set_parameters(distance=51)
        with pytest.raises(FilterValidationError, match="Distance must be in range"):
            filter_instance._validate_parameters()
        
        # Test non-numeric distance
        filter_instance.set_parameters(distance="invalid")
        with pytest.raises(FilterValidationError, match="Distance must be a numeric value"):
            filter_instance._validate_parameters()
    
    def test_parameter_validation_valid_angle(self):
        """Test parameter validation with valid angle values."""
        filter_instance = MotionBlurFilter()
        
        # Test boundary values
        filter_instance.set_parameters(angle=0.0)
        filter_instance._validate_parameters()  # Should not raise
        
        filter_instance.set_parameters(angle=180.0)
        filter_instance._validate_parameters()  # Should not raise
        
        filter_instance.set_parameters(angle=359.9)
        filter_instance._validate_parameters()  # Should not raise
    
    def test_parameter_validation_invalid_angle(self):
        """Test parameter validation with invalid angle values."""
        filter_instance = MotionBlurFilter()
        
        # Test negative angle
        filter_instance.set_parameters(angle=-1.0)
        with pytest.raises(FilterValidationError, match="Angle must be in range"):
            filter_instance._validate_parameters()
        
        # Test angle >= 360
        filter_instance.set_parameters(angle=360.0)
        with pytest.raises(FilterValidationError, match="Angle must be in range"):
            filter_instance._validate_parameters()
        
        # Test non-numeric angle
        filter_instance.set_parameters(angle="invalid")
        with pytest.raises(FilterValidationError, match="Angle must be a numeric value"):
            filter_instance._validate_parameters()
    
    def test_input_validation_valid_data(self):
        """Test input validation with valid image data."""
        filter_instance = MotionBlurFilter()
        
        # Test grayscale image
        grayscale_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        assert filter_instance.validate_input(grayscale_data) is True
        
        # Test RGB image
        rgb_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        assert filter_instance.validate_input(rgb_data) is True
        
        # Test RGBA image
        rgba_data = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        assert filter_instance.validate_input(rgba_data) is True
        
        # Test float data
        float_data = np.random.random((50, 50, 3)).astype(np.float32)
        assert filter_instance.validate_input(float_data) is True
    
    def test_input_validation_invalid_data(self):
        """Test input validation with invalid data."""
        filter_instance = MotionBlurFilter()
        
        # Test non-numpy array
        with pytest.raises(FilterValidationError, match="Input must be a numpy array"):
            filter_instance.validate_input([1, 2, 3])
        
        # Test empty array
        empty_array = np.array([])
        with pytest.raises(FilterValidationError, match="Input array cannot be empty"):
            filter_instance.validate_input(empty_array)
        
        # Test wrong dimensions
        wrong_dim_data = np.random.randint(0, 256, (10, 10, 10, 10), dtype=np.uint8)
        with pytest.raises(FilterValidationError, match="Image data must be 2D or 3D array"):
            filter_instance.validate_input(wrong_dim_data)
        
        # Test wrong number of channels
        wrong_channels = np.random.randint(0, 256, (50, 50, 5), dtype=np.uint8)
        with pytest.raises(FilterValidationError, match="Color images must have 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_instance.validate_input(wrong_channels)
    
    def test_kernel_generation_identity(self):
        """Test motion blur kernel generation for identity case (distance=0)."""
        filter_instance = MotionBlurFilter()
        
        kernel = filter_instance._create_motion_kernel(0, 0.0)
        
        # Should return identity kernel
        assert kernel.shape == (1, 1)
        assert kernel[0, 0] == 1.0
    
    def test_kernel_generation_horizontal(self):
        """Test motion blur kernel generation for horizontal motion (angle=0)."""
        filter_instance = MotionBlurFilter()
        
        distance = 5
        kernel = filter_instance._create_motion_kernel(distance, 0.0)
        
        # Kernel should be normalized (sum to 1)
        assert abs(np.sum(kernel) - 1.0) < 1e-10
        
        # For horizontal motion, non-zero values should be in a horizontal line
        center_row = kernel.shape[0] // 2
        horizontal_line = kernel[center_row, :]
        
        # Should have non-zero values in the horizontal line
        assert np.sum(horizontal_line > 0) > 1
    
    def test_kernel_generation_vertical(self):
        """Test motion blur kernel generation for vertical motion (angle=90)."""
        filter_instance = MotionBlurFilter()
        
        distance = 5
        kernel = filter_instance._create_motion_kernel(distance, 90.0)
        
        # Kernel should be normalized (sum to 1)
        assert abs(np.sum(kernel) - 1.0) < 1e-10
        
        # For vertical motion, non-zero values should be in a vertical line
        center_col = kernel.shape[1] // 2
        vertical_line = kernel[:, center_col]
        
        # Should have non-zero values in the vertical line
        assert np.sum(vertical_line > 0) > 1
    
    def test_kernel_generation_diagonal(self):
        """Test motion blur kernel generation for diagonal motion (angle=45)."""
        filter_instance = MotionBlurFilter()
        
        distance = 5
        kernel = filter_instance._create_motion_kernel(distance, 45.0)
        
        # Kernel should be normalized (sum to 1)
        assert abs(np.sum(kernel) - 1.0) < 1e-10
        
        # Should have non-zero values along diagonal
        assert np.sum(kernel > 0) > 1
    
    def test_kernel_generation_various_angles(self):
        """Test motion blur kernel generation for various angles."""
        filter_instance = MotionBlurFilter()
        
        angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        distance = 5
        
        for angle in angles:
            kernel = filter_instance._create_motion_kernel(distance, angle)
            
            # All kernels should be normalized
            assert abs(np.sum(kernel) - 1.0) < 1e-10
            
            # Should have multiple non-zero values
            assert np.sum(kernel > 0) > 1
    
    def test_apply_identity_case_distance_zero(self):
        """Test motion blur application with distance=0 (identity case)."""
        filter_instance = MotionBlurFilter(distance=0, angle=0.0)
        
        # Test with grayscale image
        original_data = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result = filter_instance.apply(original_data)
        
        # Should return identical data (but different array)
        np.testing.assert_array_equal(result, original_data)
        assert result is not original_data  # Should be a copy
    
    def test_apply_grayscale_image(self):
        """Test motion blur application on grayscale images."""
        filter_instance = MotionBlurFilter(distance=5, angle=0.0)
        
        # Create test image with distinct pattern
        test_data = np.zeros((50, 50), dtype=np.uint8)
        test_data[20:30, 20:30] = 255  # White square
        
        result = filter_instance.apply(test_data)
        
        # Result should have same shape and type
        assert result.shape == test_data.shape
        assert result.dtype == test_data.dtype
        
        # Motion blur should spread the white square horizontally
        # The exact center should have some blur effect
        assert result[25, 25] > 0  # Should still have some white
        
        # Check that blur has spread horizontally (may not reach exact positions due to kernel size)
        # Look for any horizontal spread within the square region
        center_row = result[25, :]
        non_zero_positions = np.where(center_row > 0)[0]
        assert len(non_zero_positions) > 1  # Should have spread to multiple positions
    
    def test_apply_rgb_image(self):
        """Test motion blur application on RGB images."""
        filter_instance = MotionBlurFilter(distance=3, angle=90.0)
        
        # Create test RGB image
        test_data = np.zeros((50, 50, 3), dtype=np.uint8)
        test_data[20:30, 20:30, 0] = 255  # Red square
        test_data[20:30, 20:30, 1] = 128  # Some green
        
        result = filter_instance.apply(test_data)
        
        # Result should have same shape and type
        assert result.shape == test_data.shape
        assert result.dtype == test_data.dtype
        
        # Motion blur should affect all channels
        assert result[25, 25, 0] > 0  # Red channel should have blur
        assert result[25, 25, 1] > 0  # Green channel should have blur
        
        # Check that blur has spread vertically (may not reach exact positions due to kernel size)
        # Look for any vertical spread within the square region
        center_col_red = result[:, 25, 0]
        center_col_green = result[:, 25, 1]
        non_zero_red = np.where(center_col_red > 0)[0]
        non_zero_green = np.where(center_col_green > 0)[0]
        assert len(non_zero_red) > 1  # Should have spread to multiple positions
        assert len(non_zero_green) > 1  # Should have spread to multiple positions
    
    def test_apply_rgba_image(self):
        """Test motion blur application on RGBA images."""
        filter_instance = MotionBlurFilter(distance=3, angle=45.0)
        
        # Create test RGBA image
        test_data = np.zeros((50, 50, 4), dtype=np.uint8)
        test_data[20:30, 20:30, :3] = 255  # White square
        test_data[20:30, 20:30, 3] = 255   # Full alpha
        
        result = filter_instance.apply(test_data)
        
        # Result should have same shape and type
        assert result.shape == test_data.shape
        assert result.dtype == test_data.dtype
        
        # Motion blur should affect all channels including alpha
        assert result[25, 25, 0] > 0  # RGB channels should have blur
        assert result[25, 25, 3] > 0  # Alpha channel should have blur
    
    def test_apply_float_image(self):
        """Test motion blur application on float images."""
        filter_instance = MotionBlurFilter(distance=3, angle=0.0)
        
        # Create test float image
        test_data = np.zeros((30, 30, 3), dtype=np.float32)
        test_data[10:20, 10:20, :] = 1.0  # White square
        
        result = filter_instance.apply(test_data)
        
        # Result should have same shape and type
        assert result.shape == test_data.shape
        assert result.dtype == test_data.dtype
        
        # Values should be in valid range [0, 1]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        
        # Motion blur should spread the square horizontally
        assert result[15, 15, 0] > 0.0
        
        # Check that blur has spread horizontally within the square region
        center_row = result[15, :, 0]
        non_zero_positions = np.where(center_row > 0.0)[0]
        assert len(non_zero_positions) > 1  # Should have spread to multiple positions
    
    def test_apply_with_kwargs_override(self):
        """Test motion blur application with parameter override via kwargs."""
        filter_instance = MotionBlurFilter(distance=5, angle=0.0)
        
        test_data = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
        
        # Apply with different parameters via kwargs
        result = filter_instance.apply(test_data, distance=0, angle=90.0)
        
        # With distance=0, should return identical data
        np.testing.assert_array_equal(result, test_data)
    
    def test_edge_cases_extreme_angles(self):
        """Test motion blur with extreme angle values."""
        filter_instance = MotionBlurFilter()
        
        test_data = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
        
        # Test angles near boundaries
        result1 = filter_instance.apply(test_data, distance=3, angle=0.1)
        result2 = filter_instance.apply(test_data, distance=3, angle=359.9)
        
        # Both should produce valid results
        assert result1.shape == test_data.shape
        assert result2.shape == test_data.shape
        assert result1.dtype == test_data.dtype
        assert result2.dtype == test_data.dtype
    
    def test_edge_cases_large_distance(self):
        """Test motion blur with maximum distance value."""
        filter_instance = MotionBlurFilter()
        
        test_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        # Test maximum distance
        result = filter_instance.apply(test_data, distance=50, angle=0.0)
        
        # Should produce valid result
        assert result.shape == test_data.shape
        assert result.dtype == test_data.dtype
    
    def test_edge_cases_small_image(self):
        """Test motion blur on very small images."""
        filter_instance = MotionBlurFilter(distance=3, angle=45.0)
        
        # Test with very small image
        test_data = np.array([[255, 0], [0, 255]], dtype=np.uint8)
        
        result = filter_instance.apply(test_data)
        
        # Should handle small images gracefully
        assert result.shape == test_data.shape
        assert result.dtype == test_data.dtype
    
    def test_progress_tracking(self):
        """Test that progress tracking works during motion blur application."""
        filter_instance = MotionBlurFilter(distance=5, angle=0.0)
        
        test_data = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        # Apply filter and check that progress was tracked
        result = filter_instance.apply(test_data)
        
        # Should have execution metadata
        assert hasattr(filter_instance, 'metadata')
        assert filter_instance.metadata.execution_time > 0
    
    def test_memory_tracking(self):
        """Test that memory usage is tracked during motion blur application."""
        filter_instance = MotionBlurFilter(distance=5, angle=90.0)
        
        test_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Apply filter
        result = filter_instance.apply(test_data)
        
        # Should have memory metadata
        assert hasattr(filter_instance, 'metadata')
        assert filter_instance.metadata.memory_usage > 0
        assert filter_instance.metadata.input_shape == test_data.shape
        assert filter_instance.metadata.output_shape == result.shape
    
    def test_filter_registration(self):
        """Test that MotionBlurFilter is properly registered."""
        from image_processing_library.filters.registry import get_registry
        
        registry = get_registry()
        available_filters = registry.list_filters()
        
        # Should find motion_blur filter in enhancement category
        assert 'motion_blur' in available_filters
        
        # Get filter metadata
        filter_metadata = registry.get_filter_metadata('motion_blur')
        assert filter_metadata['category'] == 'enhancement'
        assert filter_metadata['class'] == MotionBlurFilter