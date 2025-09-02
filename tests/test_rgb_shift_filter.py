"""
Unit tests for RGBShiftFilter.

Tests parameter validation, channel shifting algorithms, and edge cases for the RGBShiftFilter class.
"""

import pytest
import numpy as np
from image_processing_library.filters.artistic.rgb_shift_filter import RGBShiftFilter
from image_processing_library.core.utils import FilterValidationError


class TestRGBShiftFilterInitialization:
    """Test RGBShiftFilter initialization and parameter validation."""
    
    def test_default_initialization(self):
        """Test filter initialization with default parameters."""
        filter_instance = RGBShiftFilter()
        
        assert filter_instance.name == "rgb_shift"
        assert filter_instance.category == "artistic"
        assert filter_instance.parameters['red_shift'] == (0, 0)
        assert filter_instance.parameters['green_shift'] == (0, 0)
        assert filter_instance.parameters['blue_shift'] == (0, 0)
        assert filter_instance.parameters['edge_mode'] == "clip"
    
    def test_custom_initialization(self):
        """Test filter initialization with custom parameters."""
        filter_instance = RGBShiftFilter(
            red_shift=(5, -3),
            green_shift=(-2, 4),
            blue_shift=(1, -1),
            edge_mode="wrap"
        )
        
        assert filter_instance.parameters['red_shift'] == (5, -3)
        assert filter_instance.parameters['green_shift'] == (-2, 4)
        assert filter_instance.parameters['blue_shift'] == (1, -1)
        assert filter_instance.parameters['edge_mode'] == "wrap"
    
    def test_parameter_updates(self):
        """Test parameter updates after initialization."""
        filter_instance = RGBShiftFilter()
        
        filter_instance.set_parameters(
            red_shift=(10, 5),
            green_shift=(-5, 10),
            blue_shift=(0, -8),
            edge_mode="reflect"
        )
        
        params = filter_instance.get_parameters()
        assert params['red_shift'] == (10, 5)
        assert params['green_shift'] == (-5, 10)
        assert params['blue_shift'] == (0, -8)
        assert params['edge_mode'] == "reflect"


class TestRGBShiftFilterParameterValidation:
    """Test parameter validation for RGBShiftFilter."""
    
    def test_valid_shift_tuples(self):
        """Test validation accepts valid shift tuples."""
        valid_shifts = [(0, 0), (5, -3), (-10, 15), (100, -50)]
        
        for shift in valid_shifts:
            filter_instance = RGBShiftFilter(
                red_shift=shift,
                green_shift=shift,
                blue_shift=shift
            )
            filter_instance._validate_parameters()  # Should not raise
    
    def test_invalid_shift_tuple_length(self):
        """Test validation rejects shift tuples with wrong length."""
        invalid_shifts = [(5,), (1, 2, 3), (1, 2, 3, 4)]
        
        for shift in invalid_shifts:
            filter_instance = RGBShiftFilter(red_shift=shift)
            
            with pytest.raises(FilterValidationError, match="red_shift must have exactly 2 elements"):
                filter_instance._validate_parameters()
    
    def test_invalid_shift_tuple_type(self):
        """Test validation rejects non-tuple shift values."""
        invalid_shifts = [5, "invalid", {"x": 1, "y": 2}]
        
        for shift in invalid_shifts:
            filter_instance = RGBShiftFilter(green_shift=shift)
            
            with pytest.raises(FilterValidationError, match="green_shift must be a tuple or list"):
                filter_instance._validate_parameters()
    
    def test_invalid_shift_element_types(self):
        """Test validation rejects non-integer shift elements."""
        invalid_shifts = [(1.5, 2), (1, "invalid"), (None, 5)]
        
        for shift in invalid_shifts:
            filter_instance = RGBShiftFilter(blue_shift=shift)
            
            with pytest.raises(FilterValidationError, match="blue_shift elements must be integers"):
                filter_instance._validate_parameters()
    
    def test_valid_edge_modes(self):
        """Test validation accepts valid edge modes."""
        valid_modes = ["clip", "wrap", "reflect"]
        
        for mode in valid_modes:
            filter_instance = RGBShiftFilter(edge_mode=mode)
            filter_instance._validate_parameters()  # Should not raise
    
    def test_invalid_edge_mode_string(self):
        """Test validation rejects invalid edge mode strings."""
        filter_instance = RGBShiftFilter(edge_mode="invalid_mode")
        
        with pytest.raises(FilterValidationError, match="edge_mode must be one of"):
            filter_instance._validate_parameters()
    
    def test_invalid_edge_mode_type(self):
        """Test validation rejects non-string edge modes."""
        filter_instance = RGBShiftFilter()
        filter_instance.set_parameters(edge_mode=123)
        
        with pytest.raises(FilterValidationError, match="edge_mode must be a string"):
            filter_instance._validate_parameters()


class TestRGBShiftFilterInputValidation:
    """Test input validation for RGBShiftFilter."""
    
    def test_valid_rgb_image(self):
        """Test validation accepts valid RGB images."""
        filter_instance = RGBShiftFilter()
        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        assert filter_instance.validate_input(rgb_image) is True
    
    def test_valid_rgba_image(self):
        """Test validation accepts valid RGBA images."""
        filter_instance = RGBShiftFilter()
        rgba_image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        
        assert filter_instance.validate_input(rgba_image) is True
    
    def test_invalid_dimensions(self):
        """Test validation rejects images with invalid dimensions."""
        filter_instance = RGBShiftFilter()
        
        # 1D array
        with pytest.raises(FilterValidationError, match="RGB shift requires 3D array"):
            filter_instance.validate_input(np.array([1, 2, 3]))
        
        # 2D array (grayscale not supported)
        with pytest.raises(FilterValidationError, match="RGB shift requires 3D array"):
            filter_instance.validate_input(np.random.randint(0, 256, (100, 100)))
        
        # 4D array
        with pytest.raises(FilterValidationError, match="RGB shift requires 3D array"):
            filter_instance.validate_input(np.random.randint(0, 256, (10, 10, 10, 3)))
    
    def test_invalid_channels(self):
        """Test validation rejects images with invalid channel counts."""
        filter_instance = RGBShiftFilter()
        
        # 1 channel
        with pytest.raises(FilterValidationError, match="RGB shift requires 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_instance.validate_input(np.random.randint(0, 256, (100, 100, 1)))
        
        # 2 channels
        with pytest.raises(FilterValidationError, match="RGB shift requires 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_instance.validate_input(np.random.randint(0, 256, (100, 100, 2)))
        
        # 5 channels
        with pytest.raises(FilterValidationError, match="RGB shift requires 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_instance.validate_input(np.random.randint(0, 256, (100, 100, 5)))
    
    def test_empty_array(self):
        """Test validation rejects empty arrays."""
        filter_instance = RGBShiftFilter()
        
        with pytest.raises(FilterValidationError, match="Input array cannot be empty"):
            filter_instance.validate_input(np.array([]))
    
    def test_non_numpy_array(self):
        """Test validation rejects non-numpy arrays."""
        filter_instance = RGBShiftFilter()
        
        with pytest.raises(FilterValidationError, match="Input must be a numpy array"):
            filter_instance.validate_input([[[1, 2, 3], [4, 5, 6]]])


class TestRGBShiftFilterChannelShifting:
    """Test RGB channel shifting functionality."""
    
    def test_zero_shift_identity(self):
        """Test that zero shifts produce identical output."""
        filter_instance = RGBShiftFilter(
            red_shift=(0, 0),
            green_shift=(0, 0),
            blue_shift=(0, 0)
        )
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Should be identical to original
        assert np.array_equal(result, original)
    
    def test_single_channel_shift(self):
        """Test shifting only one channel."""
        # Create a test image with a pattern that will show shifting
        original = np.zeros((10, 10, 3), dtype=np.uint8)
        original[2:5, 2:5, 0] = 255  # Red square
        original[..., 1] = 128  # Green channel uniform
        original[..., 2] = 64   # Blue channel uniform
        
        # Shift only red channel
        filter_instance = RGBShiftFilter(
            red_shift=(2, 1),
            green_shift=(0, 0),
            blue_shift=(0, 0)
        )
        
        result = filter_instance.apply(original)
        
        # Green and blue channels should be unchanged
        assert np.array_equal(result[..., 1], original[..., 1])
        assert np.array_equal(result[..., 2], original[..., 2])
        
        # Red channel should be shifted
        assert not np.array_equal(result[..., 0], original[..., 0])
    
    def test_multiple_channel_shifts(self):
        """Test shifting multiple channels independently."""
        # Create a test image with distinct patterns
        original = np.zeros((20, 20, 3), dtype=np.uint8)
        original[5:15, 5:15, 0] = 255  # Red square
        original[2:8, 2:8, 1] = 128    # Green square
        original[12:18, 12:18, 2] = 64 # Blue square
        
        filter_instance = RGBShiftFilter(
            red_shift=(3, 0),
            green_shift=(0, 3),
            blue_shift=(-2, -2)
        )
        
        result = filter_instance.apply(original)
        
        # Result should be different from original
        assert not np.array_equal(result, original)
        
        # Shape and type should be preserved
        assert result.shape == original.shape
        assert result.dtype == original.dtype
    
    def test_rgba_preserves_alpha(self):
        """Test that RGBA images preserve alpha channel."""
        original = np.random.randint(0, 256, (50, 50, 4), dtype=np.uint8)
        original_alpha = original[..., 3].copy()
        
        filter_instance = RGBShiftFilter(
            red_shift=(5, 0),
            green_shift=(0, 5),
            blue_shift=(-3, 3)
        )
        
        result = filter_instance.apply(original)
        
        # Alpha channel should be preserved
        assert np.array_equal(result[..., 3], original_alpha)
        
        # RGB channels should be different (with high probability)
        assert not np.array_equal(result[..., :3], original[..., :3])


class TestRGBShiftFilterEdgeModes:
    """Test different edge handling modes."""
    
    def test_clip_edge_mode(self):
        """Test clip edge mode behavior."""
        # Create a simple gradient image
        original = np.zeros((10, 10, 3), dtype=np.uint8)
        for i in range(10):
            original[i, :, 0] = i * 25  # Red gradient
        
        filter_instance = RGBShiftFilter(
            red_shift=(5, 0),  # Shift right by 5 pixels
            edge_mode="clip"
        )
        
        result = filter_instance.apply(original)
        
        # When shifting right by 5, the leftmost 5 columns should be clipped
        # (they should get the value from the leftmost edge of the original)
        assert np.all(result[:, :5, 0] == original[:, 0:1, 0])  # First 5 columns should be edge value
        
        # Shape should be preserved
        assert result.shape == original.shape
    
    def test_wrap_edge_mode(self):
        """Test wrap edge mode behavior."""
        # Create a test pattern that's easy to verify wrapping
        original = np.zeros((10, 10, 3), dtype=np.uint8)
        original[:, 0, 0] = 255  # First column red
        original[:, -1, 0] = 128  # Last column different red
        
        filter_instance = RGBShiftFilter(
            red_shift=(1, 0),  # Shift right by 1 pixel
            edge_mode="wrap"
        )
        
        result = filter_instance.apply(original)
        
        # Last column should wrap to first column
        assert np.all(result[:, 0, 0] == 128)  # Wrapped from last column
        
        # Shape should be preserved
        assert result.shape == original.shape
    
    def test_reflect_edge_mode(self):
        """Test reflect edge mode behavior."""
        # Create a gradient for reflection testing
        original = np.zeros((5, 5, 3), dtype=np.uint8)
        for i in range(5):
            original[:, i, 0] = i * 50  # Column gradient: 0, 50, 100, 150, 200
        
        filter_instance = RGBShiftFilter(
            red_shift=(-2, 0),  # Shift left by 2 pixels
            edge_mode="reflect"
        )
        
        result = filter_instance.apply(original)
        
        # When shifting left by 2, we need values from columns -2 and -1
        # These should be reflected: -2 -> 2, -1 -> 1
        # So first column should get value from column 2 (100)
        # Second column should get value from column 3 (150)
        assert result[0, 0, 0] == 100  # From column 2
        assert result[0, 1, 0] == 150  # From column 3
        
        # Shape should be preserved
        assert result.shape == original.shape


class TestRGBShiftFilterEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_large_shifts(self):
        """Test behavior with shifts larger than image dimensions."""
        original = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        
        filter_instance = RGBShiftFilter(
            red_shift=(50, 30),  # Larger than image
            green_shift=(-40, -25),
            blue_shift=(15, -35),
            edge_mode="clip"
        )
        
        result = filter_instance.apply(original)
        
        # Should handle gracefully without error
        assert result.shape == original.shape
        assert result.dtype == original.dtype
    
    def test_single_pixel_image(self):
        """Test RGB shift on single pixel images."""
        original = np.array([[[255, 128, 64]]], dtype=np.uint8)
        
        filter_instance = RGBShiftFilter(
            red_shift=(1, 1),
            green_shift=(-1, 0),
            blue_shift=(0, -1)
        )
        
        result = filter_instance.apply(original)
        
        # Should handle single pixel without error
        assert result.shape == original.shape
        assert result.dtype == original.dtype
    
    def test_negative_shifts(self):
        """Test negative shift values."""
        original = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        
        filter_instance = RGBShiftFilter(
            red_shift=(-5, -3),
            green_shift=(-2, -8),
            blue_shift=(-10, -1)
        )
        
        result = filter_instance.apply(original)
        
        # Should handle negative shifts without error
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        assert not np.array_equal(result, original)
    
    def test_mixed_positive_negative_shifts(self):
        """Test combination of positive and negative shifts."""
        original = np.random.randint(0, 256, (25, 25, 3), dtype=np.uint8)
        
        filter_instance = RGBShiftFilter(
            red_shift=(5, -3),
            green_shift=(-2, 4),
            blue_shift=(0, -7)
        )
        
        result = filter_instance.apply(original)
        
        # Should handle mixed shifts without error
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        assert not np.array_equal(result, original)
    
    def test_parameter_override_in_apply(self):
        """Test parameter override during apply call."""
        filter_instance = RGBShiftFilter(
            red_shift=(1, 1),
            green_shift=(1, 1),
            blue_shift=(1, 1)
        )
        original = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        
        # Override parameters in apply call
        result = filter_instance.apply(
            original,
            red_shift=(5, 0),
            green_shift=(0, 5),
            blue_shift=(-3, -3),
            edge_mode="wrap"
        )
        
        # Should use overridden parameters
        assert filter_instance.parameters['red_shift'] == (5, 0)
        assert filter_instance.parameters['green_shift'] == (0, 5)
        assert filter_instance.parameters['blue_shift'] == (-3, -3)
        assert filter_instance.parameters['edge_mode'] == "wrap"


class TestRGBShiftFilterDataTypes:
    """Test RGB shift with different data types."""
    
    def test_uint8_images(self):
        """Test RGB shift with uint8 images."""
        original = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        
        filter_instance = RGBShiftFilter(
            red_shift=(3, 2),
            green_shift=(-1, 4),
            blue_shift=(2, -3)
        )
        
        result = filter_instance.apply(original)
        
        assert result.dtype == np.uint8
        assert result.shape == original.shape
        assert np.all(result >= 0)
        assert np.all(result <= 255)
    
    def test_float32_images(self):
        """Test RGB shift with float32 images."""
        original = np.random.random((30, 30, 3)).astype(np.float32)
        
        filter_instance = RGBShiftFilter(
            red_shift=(2, -1),
            green_shift=(-3, 2),
            blue_shift=(1, 3)
        )
        
        result = filter_instance.apply(original)
        
        assert result.dtype == np.float32
        assert result.shape == original.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
    
    def test_float64_images(self):
        """Test RGB shift with float64 images."""
        original = np.random.random((30, 30, 3)).astype(np.float64)
        
        filter_instance = RGBShiftFilter(
            red_shift=(4, 1),
            green_shift=(0, -2),
            blue_shift=(-1, 4)
        )
        
        result = filter_instance.apply(original)
        
        assert result.dtype == np.float64
        assert result.shape == original.shape


class TestRGBShiftFilterIntegration:
    """Test integration with BaseFilter features."""
    
    def test_progress_tracking(self):
        """Test that progress tracking works correctly."""
        filter_instance = RGBShiftFilter(
            red_shift=(5, 0),
            green_shift=(0, 5),
            blue_shift=(-3, 3)
        )
        progress_values = []
        
        def progress_callback(progress):
            progress_values.append(progress)
        
        filter_instance.set_progress_callback(progress_callback)
        
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        filter_instance.apply(original)
        
        # Should have recorded progress values
        assert len(progress_values) > 0
        assert 0.0 in progress_values  # Should start at 0
        assert 1.0 in progress_values  # Should end at 1
        
        # Should have intermediate progress values for channel processing
        intermediate_values = [p for p in progress_values if 0.0 < p < 1.0]
        assert len(intermediate_values) > 0
    
    def test_metadata_recording(self):
        """Test that metadata is properly recorded."""
        filter_instance = RGBShiftFilter(
            red_shift=(3, -2),
            green_shift=(-1, 4),
            blue_shift=(2, 0)
        )
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Check metadata was recorded
        assert filter_instance.metadata.execution_time > 0
        assert filter_instance.metadata.input_shape == original.shape
        assert filter_instance.metadata.output_shape == result.shape
        assert filter_instance.metadata.progress == 1.0
    
    def test_memory_efficiency(self):
        """Test memory efficiency features."""
        filter_instance = RGBShiftFilter(
            red_shift=(2, 1),
            green_shift=(1, -2),
            blue_shift=(-1, 1)
        )
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Check memory tracking
        assert filter_instance.metadata.memory_usage > 0
        assert filter_instance.metadata.peak_memory_usage > 0
    
    def test_filter_registry_integration(self):
        """Test that filter is properly registered."""
        # This test verifies the @register_filter decorator works
        filter_instance = RGBShiftFilter()
        
        # Basic properties should be set correctly for registry
        assert filter_instance.name == "rgb_shift"
        assert filter_instance.category == "artistic"
        assert hasattr(filter_instance, 'apply')
        assert callable(filter_instance.apply)


class TestRGBShiftFilterAccuracy:
    """Test accuracy of RGB shift operations."""
    
    def test_shift_accuracy_simple_pattern(self):
        """Test shift accuracy with a simple test pattern."""
        # Create a test pattern with a single white pixel
        original = np.zeros((10, 10, 3), dtype=np.uint8)
        original[5, 5, 0] = 255  # White pixel in red channel at (5, 5)
        
        filter_instance = RGBShiftFilter(
            red_shift=(2, 1),  # Should move to (7, 6)
            green_shift=(0, 0),
            blue_shift=(0, 0),
            edge_mode="clip"
        )
        
        result = filter_instance.apply(original)
        
        # Original position should be black (or clipped edge value)
        # New position should have the white pixel
        assert result[6, 7, 0] == 255  # Shifted position (y+1, x+2)
        assert result[5, 5, 0] == 0    # Original position should be black
        
        # Other channels should be unchanged
        assert np.array_equal(result[..., 1], original[..., 1])
        assert np.array_equal(result[..., 2], original[..., 2])
    
    def test_boundary_pixel_handling(self):
        """Test handling of pixels at image boundaries."""
        # Create image with distinct edge values
        original = np.zeros((10, 10, 3), dtype=np.uint8)
        original[0, :, 0] = 255  # Top edge red
        original[-1, :, 0] = 128  # Bottom edge red
        original[:, 0, 1] = 64   # Left edge green
        original[:, -1, 1] = 32  # Right edge green
        
        filter_instance = RGBShiftFilter(
            red_shift=(0, -1),  # Shift up
            green_shift=(-1, 0), # Shift left
            blue_shift=(0, 0),
            edge_mode="clip"
        )
        
        result = filter_instance.apply(original)
        
        # When shifting up by 1, the top row should get values from row 1 (which is 0 in our test)
        # The bottom row should get values from the last row (clipped)
        assert np.all(result[-1, :, 0] == 128)  # Bottom row should remain red
        
        # When shifting left by 1, the left column should get values from column 1 (which is 0 in our test)
        # The right column should get values from the last column (clipped)
        assert np.all(result[:, -1, 1] == 32)   # Right column should remain green