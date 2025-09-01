"""
Unit tests for the PrintSimulationFilter class.

Tests the refactored print simulation filter implementation including parameter validation,
input validation, progress tracking, and print effect simulation.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
from PIL import Image

from image_processing_library.filters.artistic.print_simulation import PrintSimulationFilter
from image_processing_library.core.protocols import DataType, ColorFormat
from image_processing_library.core.utils import FilterValidationError, FilterExecutionError


class TestPrintSimulationFilter(unittest.TestCase):
    """Test cases for PrintSimulationFilter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = PrintSimulationFilter()
        self.test_image_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_image_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    def test_initialization_default_parameters(self):
        """Test filter initialization with default parameters."""
        filter_instance = PrintSimulationFilter()
        
        self.assertEqual(filter_instance.name, "Print Simulation")
        self.assertEqual(filter_instance.data_type, DataType.IMAGE)
        self.assertEqual(filter_instance.color_format, ColorFormat.RGB)
        self.assertEqual(filter_instance.category, "artistic")
        
        # Check default parameters
        params = filter_instance.get_parameters()
        self.assertEqual(params['band_intensity'], 20)
        self.assertEqual(params['band_frequency'], 30)
        self.assertEqual(params['noise_level'], 10)
        self.assertEqual(params['contrast_factor'], 0.85)
    
    def test_initialization_custom_parameters(self):
        """Test filter initialization with custom parameters."""
        filter_instance = PrintSimulationFilter(
            band_intensity=40,
            band_frequency=20,
            noise_level=15,
            contrast_factor=0.7
        )
        
        params = filter_instance.get_parameters()
        self.assertEqual(params['band_intensity'], 40)
        self.assertEqual(params['band_frequency'], 20)
        self.assertEqual(params['noise_level'], 15)
        self.assertEqual(params['contrast_factor'], 0.7)
    
    def test_parameter_validation_band_intensity(self):
        """Test validation of band_intensity parameter."""
        # Valid values
        PrintSimulationFilter(band_intensity=0)
        PrintSimulationFilter(band_intensity=50)
        PrintSimulationFilter(band_intensity=100)
        
        # Invalid values
        with self.assertRaises(ValueError):
            PrintSimulationFilter(band_intensity=-1)
        
        with self.assertRaises(ValueError):
            PrintSimulationFilter(band_intensity=101)
    
    def test_parameter_validation_band_frequency(self):
        """Test validation of band_frequency parameter."""
        # Valid values
        PrintSimulationFilter(band_frequency=5)
        PrintSimulationFilter(band_frequency=50)
        PrintSimulationFilter(band_frequency=100)
        
        # Invalid values
        with self.assertRaises(ValueError):
            PrintSimulationFilter(band_frequency=4)
        
        with self.assertRaises(ValueError):
            PrintSimulationFilter(band_frequency=101)
    
    def test_parameter_validation_noise_level(self):
        """Test validation of noise_level parameter."""
        # Valid values
        PrintSimulationFilter(noise_level=0)
        PrintSimulationFilter(noise_level=25)
        PrintSimulationFilter(noise_level=50)
        
        # Invalid values
        with self.assertRaises(ValueError):
            PrintSimulationFilter(noise_level=-1)
        
        with self.assertRaises(ValueError):
            PrintSimulationFilter(noise_level=51)
    
    def test_parameter_validation_contrast_factor(self):
        """Test validation of contrast_factor parameter."""
        # Valid values
        PrintSimulationFilter(contrast_factor=0.1)
        PrintSimulationFilter(contrast_factor=0.5)
        PrintSimulationFilter(contrast_factor=1.0)
        
        # Invalid values
        with self.assertRaises(ValueError):
            PrintSimulationFilter(contrast_factor=0.09)
        
        with self.assertRaises(ValueError):
            PrintSimulationFilter(contrast_factor=1.1)
    
    def test_input_validation_valid_rgb(self):
        """Test input validation with valid RGB image."""
        valid_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.assertTrue(self.filter.validate_input(valid_rgb))
    
    def test_input_validation_valid_grayscale(self):
        """Test input validation with valid grayscale image."""
        valid_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.assertTrue(self.filter.validate_input(valid_gray))
    
    def test_input_validation_valid_rgba(self):
        """Test input validation with valid RGBA image."""
        valid_rgba = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        self.assertTrue(self.filter.validate_input(valid_rgba))
    
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
        # Invalid number of channels
        invalid_channels = np.random.randint(0, 255, (100, 100, 5), dtype=np.uint8)
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
    
    def test_apply_rgb_image(self):
        """Test applying filter to RGB image."""
        result = self.filter.apply(self.test_image_rgb)
        
        # Check output shape matches input
        self.assertEqual(result.shape, self.test_image_rgb.shape)
        
        # Check output data type
        self.assertEqual(result.dtype, np.uint8)
        
        # Check metadata was updated
        self.assertGreater(self.filter.metadata.execution_time, 0)
        self.assertEqual(self.filter.metadata.progress, 1.0)
        self.assertIsNone(self.filter.metadata.error_message)
        self.assertEqual(self.filter.metadata.input_shape, self.test_image_rgb.shape)
        self.assertEqual(self.filter.metadata.output_shape, result.shape)
    
    def test_apply_grayscale_image(self):
        """Test applying filter to grayscale image."""
        result = self.filter.apply(self.test_image_gray)
        
        # Check output shape matches input
        self.assertEqual(result.shape, self.test_image_gray.shape)
        
        # Check output data type
        self.assertEqual(result.dtype, np.uint8)
        
        # Check metadata was updated
        self.assertGreater(self.filter.metadata.execution_time, 0)
        self.assertEqual(self.filter.metadata.progress, 1.0)
    
    def test_apply_rgba_image(self):
        """Test applying filter to RGBA image."""
        rgba_image = np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8)
        result = self.filter.apply(rgba_image)
        
        # Check output shape matches input
        self.assertEqual(result.shape, rgba_image.shape)
        
        # Check alpha channel is preserved
        np.testing.assert_array_equal(result[:, :, 3], rgba_image[:, :, 3])
    
    def test_apply_with_float_input(self):
        """Test apply with float input data."""
        float_image = np.random.rand(50, 50, 3).astype(np.float32)
        result = self.filter.apply(float_image)
        
        # Should convert to uint8 and process
        self.assertEqual(result.shape, float_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_apply_with_zero_effects(self):
        """Test apply with zero effect parameters."""
        filter_no_effects = PrintSimulationFilter(
            band_intensity=0,
            noise_level=0,
            contrast_factor=1.0
        )
        result = filter_no_effects.apply(self.test_image_rgb)
        
        # Should still process but with minimal changes
        self.assertEqual(result.shape, self.test_image_rgb.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_apply_with_runtime_parameters(self):
        """Test apply with runtime parameter overrides."""
        result = self.filter.apply(
            self.test_image_rgb,
            band_intensity=5,
            noise_level=5
        )
        
        self.assertEqual(result.shape, self.test_image_rgb.shape)
        
        # Original parameters should be restored
        params = self.filter.get_parameters()
        self.assertEqual(params['band_intensity'], 20)  # Original value
        self.assertEqual(params['noise_level'], 10)     # Original value
    
    def test_apply_with_invalid_runtime_parameters(self):
        """Test apply with invalid runtime parameters."""
        with self.assertRaises(FilterExecutionError):
            self.filter.apply(
                self.test_image_rgb,
                band_intensity=200  # Invalid value
            )
        
        # Original parameters should be preserved
        params = self.filter.get_parameters()
        self.assertEqual(params['band_intensity'], 20)
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        progress_values = []
        
        def progress_callback(progress):
            progress_values.append(progress)
        
        self.filter.set_progress_callback(progress_callback)
        self.filter.apply(self.test_image_rgb)
        
        # Should have received progress updates
        self.assertGreater(len(progress_values), 0)
        self.assertEqual(progress_values[-1], 1.0)  # Should end at 100%
        
        # Progress should be monotonically increasing
        for i in range(1, len(progress_values)):
            self.assertGreaterEqual(progress_values[i], progress_values[i-1])
    
    def test_set_parameters_valid(self):
        """Test setting valid parameters."""
        self.filter.set_parameters(
            band_intensity=30,
            band_frequency=25,
            noise_level=15
        )
        
        params = self.filter.get_parameters()
        self.assertEqual(params['band_intensity'], 30)
        self.assertEqual(params['band_frequency'], 25)
        self.assertEqual(params['noise_level'], 15)
    
    def test_set_parameters_invalid_name(self):
        """Test setting parameters with invalid names."""
        with self.assertRaises(ValueError):
            self.filter.set_parameters(invalid_param=123)
    
    def test_set_parameters_invalid_value(self):
        """Test setting parameters with invalid values."""
        with self.assertRaises(ValueError):
            self.filter.set_parameters(band_intensity=200)
    
    def test_horizontal_bands_effect(self):
        """Test horizontal bands effect specifically."""
        # Create filter with only band effect
        filter_bands = PrintSimulationFilter(
            band_intensity=50,
            band_frequency=10,
            noise_level=0,
            contrast_factor=1.0
        )
        
        # Create uniform test image to see banding effect
        uniform_image = np.full((100, 100), 128, dtype=np.uint8)
        result = filter_bands.apply(uniform_image)
        
        # Should have variations due to banding
        self.assertNotEqual(np.std(result), 0)
    
    def test_noise_effect(self):
        """Test noise effect specifically."""
        # Create filter with only noise effect
        filter_noise = PrintSimulationFilter(
            band_intensity=0,
            noise_level=20,
            contrast_factor=1.0
        )
        
        # Create uniform test image to see noise effect
        uniform_image = np.full((100, 100), 128, dtype=np.uint8)
        result = filter_noise.apply(uniform_image)
        
        # Should have variations due to noise
        self.assertGreater(np.std(result), 0)
    
    def test_contrast_degradation_effect(self):
        """Test contrast degradation effect."""
        # Create high contrast test image
        high_contrast = np.zeros((100, 100), dtype=np.uint8)
        high_contrast[:50, :] = 255  # Half white, half black
        
        # Apply contrast degradation
        filter_contrast = PrintSimulationFilter(
            band_intensity=0,
            noise_level=0,
            contrast_factor=0.5
        )
        
        result = filter_contrast.apply(high_contrast)
        
        # Contrast should be reduced (values closer to middle gray)
        self.assertLess(np.max(result), 255)
        self.assertGreater(np.min(result), 0)
    
    @patch('image_processing_library.filters.artistic.print_simulation.Image')
    def test_pil_processing_error_handling(self, mock_image):
        """Test error handling in PIL processing."""
        # Mock PIL Image to raise an exception
        mock_image.fromarray.side_effect = Exception("PIL error")
        
        with self.assertRaises(FilterExecutionError):
            self.filter.apply(self.test_image_rgb)
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        self.filter.apply(self.test_image_rgb)
        
        # Should have recorded memory usage
        self.assertGreater(self.filter.metadata.memory_usage, 0)
        
        # Memory usage should be reasonable for test image size
        expected_mb = self.test_image_rgb.nbytes / (1024 * 1024)
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
    
    def test_grayscale_3d_input(self):
        """Test with 3D grayscale input (H, W, 1)."""
        grayscale_3d = np.random.randint(0, 255, (50, 50, 1), dtype=np.uint8)
        result = self.filter.apply(grayscale_3d)
        
        # Should maintain shape
        self.assertEqual(result.shape, grayscale_3d.shape)


if __name__ == '__main__':
    unittest.main()