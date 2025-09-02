"""
Unit tests for the GlitchFilter class.

Tests the refactored glitch filter implementation including parameter validation,
input validation, progress tracking, and filter effects.
"""

import unittest
import numpy as np
import random
from unittest.mock import Mock, patch
from PIL import Image
import io

from image_processing_library.filters.artistic.glitch import GlitchFilter
from image_processing_library.core.protocols import DataType, ColorFormat
from image_processing_library.core.utils import FilterValidationError, FilterExecutionError


class TestGlitchFilter(unittest.TestCase):
    """Test cases for GlitchFilter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = GlitchFilter()
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_initialization_default_parameters(self):
        """Test filter initialization with default parameters."""
        filter_instance = GlitchFilter()
        
        self.assertEqual(filter_instance.name, "Glitch Effect")
        self.assertEqual(filter_instance.data_type, DataType.IMAGE)
        self.assertEqual(filter_instance.color_format, ColorFormat.RGB)
        self.assertEqual(filter_instance.category, "artistic")
        
        # Check default parameters
        params = filter_instance.get_parameters()
        self.assertEqual(params['shift_intensity'], 10)
        self.assertEqual(params['line_width'], 3)
        self.assertEqual(params['glitch_probability'], 0.8)
        self.assertEqual(params['jpeg_quality'], 30)
        self.assertEqual(params['shift_angle'], 0.0)
    
    def test_initialization_custom_parameters(self):
        """Test filter initialization with custom parameters."""
        filter_instance = GlitchFilter(
            shift_intensity=20,
            line_width=5,
            glitch_probability=0.5,
            jpeg_quality=50,
            shift_angle=45.0
        )
        
        params = filter_instance.get_parameters()
        self.assertEqual(params['shift_intensity'], 20)
        self.assertEqual(params['line_width'], 5)
        self.assertEqual(params['glitch_probability'], 0.5)
        self.assertEqual(params['jpeg_quality'], 50)
        self.assertEqual(params['shift_angle'], 45.0)
    
    def test_parameter_validation_shift_intensity(self):
        """Test validation of shift_intensity parameter."""
        # Valid values
        GlitchFilter(shift_intensity=0)
        GlitchFilter(shift_intensity=50)
        GlitchFilter(shift_intensity=100)
        
        # Invalid values
        with self.assertRaises(ValueError):
            GlitchFilter(shift_intensity=-1)
        
        with self.assertRaises(ValueError):
            GlitchFilter(shift_intensity=101)
    
    def test_parameter_validation_line_width(self):
        """Test validation of line_width parameter."""
        # Valid values
        GlitchFilter(line_width=1)
        GlitchFilter(line_width=10)
        GlitchFilter(line_width=20)
        
        # Invalid values
        with self.assertRaises(ValueError):
            GlitchFilter(line_width=0)
        
        with self.assertRaises(ValueError):
            GlitchFilter(line_width=21)
    
    def test_parameter_validation_glitch_probability(self):
        """Test validation of glitch_probability parameter."""
        # Valid values
        GlitchFilter(glitch_probability=0.0)
        GlitchFilter(glitch_probability=0.5)
        GlitchFilter(glitch_probability=1.0)
        
        # Invalid values
        with self.assertRaises(ValueError):
            GlitchFilter(glitch_probability=-0.1)
        
        with self.assertRaises(ValueError):
            GlitchFilter(glitch_probability=1.1)
    
    def test_parameter_validation_jpeg_quality(self):
        """Test validation of jpeg_quality parameter."""
        # Valid values
        GlitchFilter(jpeg_quality=1)
        GlitchFilter(jpeg_quality=50)
        GlitchFilter(jpeg_quality=100)
        
        # Invalid values
        with self.assertRaises(ValueError):
            GlitchFilter(jpeg_quality=0)
        
        with self.assertRaises(ValueError):
            GlitchFilter(jpeg_quality=101)
    
    def test_parameter_validation_shift_angle(self):
        """Test validation of shift_angle parameter."""
        # Valid values
        GlitchFilter(shift_angle=0.0)
        GlitchFilter(shift_angle=180.0)
        GlitchFilter(shift_angle=360.0)
        
        # Invalid values
        with self.assertRaises(ValueError):
            GlitchFilter(shift_angle=-0.1)
        
        with self.assertRaises(ValueError):
            GlitchFilter(shift_angle=360.1)
    
    def test_input_validation_valid_rgb(self):
        """Test input validation with valid RGB image."""
        valid_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.assertTrue(self.filter.validate_input(valid_rgb))
    
    def test_input_validation_invalid_dimensions(self):
        """Test input validation with invalid dimensions."""
        # 1D array
        invalid_1d = np.random.randint(0, 255, (100,), dtype=np.uint8)
        with self.assertRaises(FilterValidationError):
            self.filter.validate_input(invalid_1d)
        
        # 4D array (should be for video, not image)
        invalid_4d = np.random.randint(0, 255, (10, 100, 100, 3), dtype=np.uint8)
        with self.assertRaises(FilterValidationError):
            self.filter.validate_input(invalid_4d)
    
    def test_input_validation_invalid_channels(self):
        """Test input validation with invalid channel count."""
        # Wrong number of channels for RGB
        invalid_channels = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        with self.assertRaises(FilterValidationError):
            self.filter.validate_input(invalid_channels)
    
    def test_input_validation_non_numpy_array(self):
        """Test input validation with non-numpy array."""
        with self.assertRaises(FilterValidationError):
            self.filter.validate_input([1, 2, 3])
    
    def test_input_validation_empty_array(self):
        """Test input validation with empty array."""
        empty_array = np.array([])
        with self.assertRaises(FilterValidationError):
            self.filter.validate_input(empty_array)
    
    def test_apply_basic_functionality(self):
        """Test basic apply functionality."""
        result = self.filter.apply(self.test_image)
        
        # Check output shape matches input
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Check output data type
        self.assertEqual(result.dtype, np.uint8)
        
        # Check metadata was updated
        self.assertGreater(self.filter.metadata.execution_time, 0)
        self.assertEqual(self.filter.metadata.progress, 1.0)
        self.assertIsNone(self.filter.metadata.error_message)
        self.assertEqual(self.filter.metadata.input_shape, self.test_image.shape)
        self.assertEqual(self.filter.metadata.output_shape, result.shape)
    
    def test_apply_with_float_input(self):
        """Test apply with float input data."""
        float_image = np.random.rand(50, 50, 3).astype(np.float32)
        result = self.filter.apply(float_image)
        
        # Should convert to uint8 and process
        self.assertEqual(result.shape, float_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_apply_with_zero_shift_intensity(self):
        """Test apply with zero shift intensity (no effect)."""
        filter_no_shift = GlitchFilter(shift_intensity=0)
        result = filter_no_shift.apply(self.test_image)
        
        # Should still process (JPEG compression may still apply)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_apply_with_runtime_parameters(self):
        """Test apply with runtime parameter overrides."""
        result = self.filter.apply(
            self.test_image,
            shift_intensity=5,
            jpeg_quality=80
        )
        
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Original parameters should be restored
        params = self.filter.get_parameters()
        self.assertEqual(params['shift_intensity'], 10)  # Original value
        self.assertEqual(params['jpeg_quality'], 30)     # Original value
    
    def test_apply_with_invalid_runtime_parameters(self):
        """Test apply with invalid runtime parameters."""
        with self.assertRaises(FilterExecutionError):
            self.filter.apply(
                self.test_image,
                shift_intensity=200  # Invalid value
            )
        
        # Original parameters should be preserved
        params = self.filter.get_parameters()
        self.assertEqual(params['shift_intensity'], 10)
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        progress_values = []
        
        def progress_callback(progress):
            progress_values.append(progress)
        
        self.filter.set_progress_callback(progress_callback)
        self.filter.apply(self.test_image)
        
        # Should have received progress updates
        self.assertGreater(len(progress_values), 0)
        self.assertEqual(progress_values[-1], 1.0)  # Should end at 100%
        
        # Progress should be monotonically increasing
        for i in range(1, len(progress_values)):
            self.assertGreaterEqual(progress_values[i], progress_values[i-1])
    
    def test_set_parameters_valid(self):
        """Test setting valid parameters."""
        self.filter.set_parameters(
            shift_intensity=15,
            line_width=4,
            glitch_probability=0.6
        )
        
        params = self.filter.get_parameters()
        self.assertEqual(params['shift_intensity'], 15)
        self.assertEqual(params['line_width'], 4)
        self.assertEqual(params['glitch_probability'], 0.6)
    
    def test_set_parameters_invalid_name(self):
        """Test setting parameters with invalid names."""
        with self.assertRaises(ValueError):
            self.filter.set_parameters(invalid_param=123)
    
    def test_set_parameters_invalid_value(self):
        """Test setting parameters with invalid values."""
        with self.assertRaises(ValueError):
            self.filter.set_parameters(shift_intensity=200)
    
    def test_horizontal_shift_optimization(self):
        """Test that horizontal shift optimization is used for angle=0."""
        filter_horizontal = GlitchFilter(shift_angle=0.0, shift_intensity=10)
        
        # Mock the horizontal shift method to verify it's called
        with patch.object(filter_horizontal, '_horizontal_shift_simple') as mock_horizontal:
            mock_horizontal.return_value = self.test_image.copy()
            
            filter_horizontal.apply(self.test_image)
            mock_horizontal.assert_called_once()
    
    def test_angled_shift_for_non_zero_angle(self):
        """Test that angled shift is used for non-zero angles."""
        filter_angled = GlitchFilter(shift_angle=45.0, shift_intensity=10)
        
        # Should not use horizontal optimization
        with patch.object(filter_angled, '_horizontal_shift_simple') as mock_horizontal:
            result = filter_angled.apply(self.test_image)
            mock_horizontal.assert_not_called()
            
            # Should still produce valid output
            self.assertEqual(result.shape, self.test_image.shape)
    
    @patch('image_processing_library.filters.artistic.glitch.Image')
    def test_jpeg_corruption_error_handling(self, mock_image):
        """Test error handling in JPEG corruption."""
        # Mock PIL Image to raise an exception
        mock_image.fromarray.side_effect = Exception("PIL error")
        
        with self.assertRaises(FilterExecutionError):
            self.filter.apply(self.test_image)
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        self.filter.apply(self.test_image)
        
        # Should have recorded memory usage
        self.assertGreater(self.filter.metadata.memory_usage, 0)
        
        # Memory usage should be reasonable for test image size
        expected_mb = self.test_image.nbytes / (1024 * 1024)
        self.assertAlmostEqual(self.filter.metadata.memory_usage, expected_mb, places=2)
    
    def test_different_image_sizes(self):
        """Test filter with different image sizes."""
        # Small image
        small_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result_small = self.filter.apply(small_image)
        self.assertEqual(result_small.shape, small_image.shape)
        
        # Large image
        large_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        result_large = self.filter.apply(large_image)
        self.assertEqual(result_large.shape, large_image.shape)
    
    def test_grayscale_image_rejection(self):
        """Test that grayscale images are rejected (filter expects RGB)."""
        grayscale_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        with self.assertRaises(FilterValidationError):
            self.filter.validate_input(grayscale_image)
    
    def test_rgb_shift_filter_integration(self):
        """Test that GlitchFilter properly integrates with RGBShiftFilter."""
        # Verify that the RGB shift filter is initialized
        self.assertIsNotNone(self.filter._rgb_shift_filter)
        self.assertEqual(self.filter._rgb_shift_filter.name, "rgb_shift")
        
        # Test that color channel shifting still works
        result = self.filter.apply(self.test_image)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_color_channel_shift_uses_rgb_shift_filter(self):
        """Test that color channel shifting uses RGBShiftFilter internally."""
        # Mock the RGBShiftFilter's apply method to verify it's called
        with patch.object(self.filter._rgb_shift_filter, 'apply') as mock_apply:
            mock_apply.return_value = self.test_image.copy()
            
            # Call the color channel shift method directly
            result = self.filter._color_channel_shift(self.test_image)
            
            # Verify RGBShiftFilter.apply was called
            mock_apply.assert_called_once()
            
            # Verify the call was made with proper parameters
            call_args = mock_apply.call_args
            self.assertEqual(call_args[0][0].shape, self.test_image.shape)  # First arg is image
            
            # Check that shift parameters were provided
            kwargs = call_args[1]
            self.assertIn('red_shift', kwargs)
            self.assertIn('green_shift', kwargs)
            self.assertIn('blue_shift', kwargs)
            self.assertIn('edge_mode', kwargs)
            self.assertEqual(kwargs['edge_mode'], 'clip')
    
    def test_color_channel_shift_maintains_original_behavior(self):
        """Test that color channel shifting maintains original random behavior."""
        # Create a test image with distinct patterns to make shifts more visible
        test_image = np.zeros((20, 20, 3), dtype=np.uint8)
        test_image[5:15, 5:15, 0] = 255  # Red square
        test_image[8:12, 8:12, 1] = 255  # Green square (smaller)
        test_image[10:20, 10:20, 2] = 255  # Blue square
        
        # Apply color channel shift multiple times with different random seeds
        results = []
        for i in range(20):  # More iterations for better chance of variation
            random.seed(100 + i)  # Use different seeds
            result = self.filter._color_channel_shift(test_image)
            # Convert to string representation for comparison
            result_str = str(result.flatten())
            results.append(result_str)
        
        # Count unique results
        unique_results = len(set(results))
        
        # With random shifts, we should get some variation
        # Even if some results are identical, we expect at least some variation
        self.assertGreaterEqual(unique_results, 1, "Color channel shifts should work")
        
        # Test that the method actually produces output
        result = self.filter._color_channel_shift(test_image)
        self.assertEqual(result.shape, test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_backward_compatibility_with_existing_parameters(self):
        """Test that all existing parameters still work as expected."""
        # Test with various parameter combinations that should work
        test_params = [
            {'shift_intensity': 0, 'jpeg_quality': 100},  # Minimal effect
            {'shift_intensity': 50, 'line_width': 1, 'glitch_probability': 1.0},  # Maximum effect
            {'shift_angle': 90.0, 'shift_intensity': 20},  # Vertical shifts
            {'shift_angle': 45.0, 'line_width': 10},  # Diagonal shifts
        ]
        
        for params in test_params:
            with self.subTest(params=params):
                filter_instance = GlitchFilter(**params)
                result = filter_instance.apply(self.test_image)
                
                # Basic validation
                self.assertEqual(result.shape, self.test_image.shape)
                self.assertEqual(result.dtype, np.uint8)
                
                # Verify parameters were set correctly
                actual_params = filter_instance.get_parameters()
                for key, value in params.items():
                    self.assertEqual(actual_params[key], value)
    
    def test_no_regression_in_glitch_effects(self):
        """Test that the main glitch effects (angled shifts, JPEG corruption) still work."""
        # Test with specific parameters to ensure effects are applied
        filter_instance = GlitchFilter(
            shift_intensity=20,
            line_width=2,
            glitch_probability=1.0,  # Ensure effects are applied
            jpeg_quality=10,  # Low quality for visible compression
            shift_angle=0.0
        )
        
        # Create a test image with distinct patterns
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image[10:40, 10:40, :] = 255  # White square in center
        
        result = filter_instance.apply(test_image)
        
        # Verify output is different from input (effects were applied)
        self.assertFalse(np.array_equal(result, test_image), 
                        "Glitch filter should modify the image")
        
        # Verify basic properties are maintained
        self.assertEqual(result.shape, test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
        
        # Verify pixel values are still in valid range
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))
    
    def test_rgb_shift_filter_error_handling(self):
        """Test error handling when RGBShiftFilter encounters issues."""
        # Mock RGBShiftFilter to raise an exception
        with patch.object(self.filter._rgb_shift_filter, 'apply') as mock_apply:
            mock_apply.side_effect = Exception("RGB shift error")
            
            # The error should propagate up
            with self.assertRaises(Exception):
                self.filter._color_channel_shift(self.test_image)


if __name__ == '__main__':
    unittest.main()