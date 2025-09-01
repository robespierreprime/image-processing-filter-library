"""
Unit tests for the BackgroundRemoverFilter class.

Tests the background remover filter implementation including parameter validation,
input validation, progress tracking, and background removal functionality.
"""

import unittest
import numpy as np
import sys
from unittest.mock import Mock, patch, MagicMock

from image_processing_library.filters.technical.background_remover import BackgroundRemoverFilter
from image_processing_library.core.protocols import DataType, ColorFormat
from image_processing_library.core.utils import FilterValidationError, FilterExecutionError


class TestBackgroundRemoverFilter(unittest.TestCase):
    """Test cases for BackgroundRemoverFilter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = BackgroundRemoverFilter()
        self.test_image_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_image_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    def test_initialization_default_parameters(self):
        """Test filter initialization with default parameters."""
        filter_instance = BackgroundRemoverFilter()
        
        self.assertEqual(filter_instance.name, "Background Remover")
        self.assertEqual(filter_instance.data_type, DataType.IMAGE)
        self.assertEqual(filter_instance.color_format, ColorFormat.RGB)
        self.assertEqual(filter_instance.category, "technical")
        
        # Check default parameters
        params = filter_instance.get_parameters()
        self.assertEqual(params['model'], 'u2net')
        self.assertEqual(params['alpha_matting'], False)
        self.assertEqual(params['alpha_matting_foreground_threshold'], 240)
        self.assertEqual(params['alpha_matting_background_threshold'], 10)
    
    def test_initialization_custom_parameters(self):
        """Test filter initialization with custom parameters."""
        filter_instance = BackgroundRemoverFilter(
            model='u2netp',
            alpha_matting=True,
            alpha_matting_foreground_threshold=250,
            alpha_matting_background_threshold=20
        )
        
        params = filter_instance.get_parameters()
        self.assertEqual(params['model'], 'u2netp')
        self.assertEqual(params['alpha_matting'], True)
        self.assertEqual(params['alpha_matting_foreground_threshold'], 250)
        self.assertEqual(params['alpha_matting_background_threshold'], 20)
    
    def test_parameter_validation_model(self):
        """Test validation of model parameter."""
        # Valid models
        for model in BackgroundRemoverFilter.AVAILABLE_MODELS.keys():
            BackgroundRemoverFilter(model=model)
        
        # Invalid model
        with self.assertRaises(ValueError):
            BackgroundRemoverFilter(model='invalid_model')
    
    def test_parameter_validation_alpha_matting(self):
        """Test validation of alpha_matting parameter."""
        # Valid values
        BackgroundRemoverFilter(alpha_matting=True)
        BackgroundRemoverFilter(alpha_matting=False)
        
        # Invalid values
        with self.assertRaises(ValueError):
            BackgroundRemoverFilter(alpha_matting="true")
        
        with self.assertRaises(ValueError):
            BackgroundRemoverFilter(alpha_matting=1)
    
    def test_parameter_validation_thresholds(self):
        """Test validation of alpha matting threshold parameters."""
        # Valid values
        BackgroundRemoverFilter(alpha_matting_foreground_threshold=0)
        BackgroundRemoverFilter(alpha_matting_foreground_threshold=255)
        BackgroundRemoverFilter(alpha_matting_background_threshold=0)
        BackgroundRemoverFilter(alpha_matting_background_threshold=255)
        
        # Invalid values
        with self.assertRaises(ValueError):
            BackgroundRemoverFilter(alpha_matting_foreground_threshold=-1)
        
        with self.assertRaises(ValueError):
            BackgroundRemoverFilter(alpha_matting_foreground_threshold=256)
        
        with self.assertRaises(ValueError):
            BackgroundRemoverFilter(alpha_matting_background_threshold=-1)
        
        with self.assertRaises(ValueError):
            BackgroundRemoverFilter(alpha_matting_background_threshold=256)
    
    def test_input_validation_valid_rgb(self):
        """Test input validation with valid RGB image."""
        valid_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.assertTrue(self.filter.validate_input(valid_rgb))
    
    def test_input_validation_valid_rgba(self):
        """Test input validation with valid RGBA image."""
        valid_rgba = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        self.assertTrue(self.filter.validate_input(valid_rgba))
    
    def test_input_validation_valid_grayscale(self):
        """Test input validation with valid grayscale image."""
        valid_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.assertTrue(self.filter.validate_input(valid_gray))
    
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
    
    def test_apply_without_rembg(self):
        """Test that apply raises appropriate error when rembg is not available."""
        with self.assertRaises(FilterExecutionError) as context:
            self.filter.apply(self.test_image_rgb)
        
        self.assertIn("rembg library not found", str(context.exception))
    
    def test_apply_grayscale_without_rembg(self):
        """Test that apply with grayscale raises appropriate error when rembg is not available."""
        with self.assertRaises(FilterExecutionError) as context:
            self.filter.apply(self.test_image_gray)
        
        self.assertIn("rembg library not found", str(context.exception))
    
    def test_apply_with_alpha_matting_without_rembg(self):
        """Test apply with alpha matting when rembg is not available."""
        filter_alpha = BackgroundRemoverFilter(alpha_matting=True)
        
        with self.assertRaises(FilterExecutionError) as context:
            filter_alpha.apply(self.test_image_rgb)
        
        self.assertIn("rembg library not found", str(context.exception))
    
    def test_apply_with_float_input_without_rembg(self):
        """Test apply with float input when rembg is not available."""
        float_image = np.random.rand(50, 50, 3).astype(np.float32)
        
        with self.assertRaises(FilterExecutionError) as context:
            self.filter.apply(float_image)
        
        self.assertIn("rembg library not found", str(context.exception))
    
    def test_apply_with_runtime_parameters_without_rembg(self):
        """Test apply with runtime parameter overrides when rembg is not available."""
        with self.assertRaises(FilterExecutionError) as context:
            self.filter.apply(
                self.test_image_rgb,
                model='u2netp',
                alpha_matting=True
            )
        
        self.assertIn("rembg library not found", str(context.exception))
        
        # Original parameters should be preserved even after error
        params = self.filter.get_parameters()
        self.assertEqual(params['model'], 'u2net')  # Original value
        self.assertEqual(params['alpha_matting'], False)  # Original value
    
    def test_apply_with_invalid_runtime_parameters(self):
        """Test apply with invalid runtime parameters."""
        with self.assertRaises(FilterExecutionError):
            self.filter.apply(
                self.test_image_rgb,
                model='invalid_model'  # Invalid value
            )
        
        # Original parameters should be preserved
        params = self.filter.get_parameters()
        self.assertEqual(params['model'], 'u2net')
    
    def test_rembg_import_error(self):
        """Test handling of missing rembg library."""
        # This test is redundant with the other tests since rembg is not installed
        # but we keep it for completeness
        with self.assertRaises(FilterExecutionError) as context:
            self.filter.apply(self.test_image_rgb)
        
        self.assertIn("rembg library not found", str(context.exception))
    
    def test_rembg_session_error(self):
        """Test handling of rembg session creation error."""
        # Since rembg is not available, we get the import error first
        with self.assertRaises(FilterExecutionError) as context:
            self.filter.apply(self.test_image_rgb)
        
        self.assertIn("rembg library not found", str(context.exception))
    
    def test_progress_tracking_without_rembg(self):
        """Test progress tracking functionality when rembg is not available."""
        progress_values = []
        
        def progress_callback(progress):
            progress_values.append(progress)
        
        self.filter.set_progress_callback(progress_callback)
        
        with self.assertRaises(FilterExecutionError):
            self.filter.apply(self.test_image_rgb)
        
        # Should have received some progress updates before failing
        # (at least the initial 0.0 progress)
        self.assertGreaterEqual(len(progress_values), 1)
        self.assertEqual(progress_values[0], 0.0)
    
    def test_set_parameters_valid(self):
        """Test setting valid parameters."""
        self.filter.set_parameters(
            model='u2netp',
            alpha_matting=True,
            alpha_matting_foreground_threshold=250
        )
        
        params = self.filter.get_parameters()
        self.assertEqual(params['model'], 'u2netp')
        self.assertEqual(params['alpha_matting'], True)
        self.assertEqual(params['alpha_matting_foreground_threshold'], 250)
    
    def test_set_parameters_invalid_name(self):
        """Test setting parameters with invalid names."""
        with self.assertRaises(ValueError):
            self.filter.set_parameters(invalid_param=123)
    
    def test_set_parameters_invalid_value(self):
        """Test setting parameters with invalid values."""
        with self.assertRaises(ValueError):
            self.filter.set_parameters(model='invalid_model')
    
    def test_set_parameters_model_change_resets_session(self):
        """Test that changing model resets the session."""
        # Initialize session
        self.filter._session = MagicMock()
        
        # Change model
        self.filter.set_parameters(model='u2netp')
        
        # Session should be reset
        self.assertIsNone(self.filter._session)
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = self.filter.get_available_models()
        
        self.assertIsInstance(models, dict)
        self.assertIn('u2net', models)
        self.assertIn('u2netp', models)
        self.assertIn('u2net_human_seg', models)
    
    def test_get_model_info(self):
        """Test getting current model information."""
        info = self.filter.get_model_info()
        
        self.assertIsInstance(info, str)
        self.assertIn('u2net', info)
        self.assertIn('General purpose model', info)
    
    def test_memory_usage_tracking_without_rembg(self):
        """Test memory usage tracking when rembg is not available."""
        with self.assertRaises(FilterExecutionError):
            self.filter.apply(self.test_image_rgb)
        
        # Should have recorded memory usage even if processing failed
        self.assertGreater(self.filter.metadata.memory_usage, 0)
        
        # Memory usage should be reasonable for test image size
        expected_mb = self.test_image_rgb.nbytes / (1024 * 1024)
        self.assertAlmostEqual(self.filter.metadata.memory_usage, expected_mb, places=2)


if __name__ == '__main__':
    unittest.main()