"""
Unit tests for base filter functionality.

Tests the FilterMetadata dataclass, BaseFilter class, and input validation system.
"""

import unittest
import numpy as np
from unittest.mock import Mock
import time

from image_processing_library.core import (
    BaseFilter,
    FilterMetadata,
    DataType,
    ColorFormat,
    FilterValidationError,
)


class TestFilterMetadata(unittest.TestCase):
    """Test cases for FilterMetadata dataclass."""

    def test_default_values(self):
        """Test FilterMetadata default initialization."""
        metadata = FilterMetadata()

        self.assertEqual(metadata.execution_time, 0.0)
        self.assertEqual(metadata.progress, 0.0)
        self.assertIsNone(metadata.error_message)
        self.assertEqual(metadata.memory_usage, 0.0)
        self.assertIsNone(metadata.input_shape)
        self.assertIsNone(metadata.output_shape)

    def test_custom_values(self):
        """Test FilterMetadata with custom values."""
        metadata = FilterMetadata(
            execution_time=1.5,
            progress=0.75,
            error_message="Test error",
            memory_usage=100.0,
            input_shape=(100, 100, 3),
            output_shape=(100, 100, 3),
        )

        self.assertEqual(metadata.execution_time, 1.5)
        self.assertEqual(metadata.progress, 0.75)
        self.assertEqual(metadata.error_message, "Test error")
        self.assertEqual(metadata.memory_usage, 100.0)
        self.assertEqual(metadata.input_shape, (100, 100, 3))
        self.assertEqual(metadata.output_shape, (100, 100, 3))


class TestBaseFilter(unittest.TestCase):
    """Test cases for BaseFilter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.filter = BaseFilter(
            name="test_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test",
            param1="value1",
            param2=42,
        )

    def test_initialization(self):
        """Test BaseFilter initialization."""
        self.assertEqual(self.filter.name, "test_filter")
        self.assertEqual(self.filter.data_type, DataType.IMAGE)
        self.assertEqual(self.filter.color_format, ColorFormat.RGB)
        self.assertEqual(self.filter.category, "test")
        self.assertEqual(self.filter.parameters["param1"], "value1")
        self.assertEqual(self.filter.parameters["param2"], 42)
        self.assertIsInstance(self.filter.metadata, FilterMetadata)

    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        progress_values = []

        def progress_callback(progress):
            progress_values.append(progress)

        self.filter.set_progress_callback(progress_callback)

        # Test progress updates
        self.filter._update_progress(0.5)
        self.assertEqual(self.filter.metadata.progress, 0.5)
        self.assertEqual(progress_values, [0.5])

        # Test progress clamping
        self.filter._update_progress(1.5)  # Should clamp to 1.0
        self.assertEqual(self.filter.metadata.progress, 1.0)

        self.filter._update_progress(-0.5)  # Should clamp to 0.0
        self.assertEqual(self.filter.metadata.progress, 0.0)

    def test_parameter_management(self):
        """Test parameter get/set functionality."""
        # Test get_parameters
        params = self.filter.get_parameters()
        self.assertEqual(params["param1"], "value1")
        self.assertEqual(params["param2"], 42)

        # Ensure it's a copy
        params["param1"] = "modified"
        self.assertEqual(self.filter.parameters["param1"], "value1")

        # Test set_parameters
        self.filter.set_parameters(param1="new_value", param3="new_param")
        self.assertEqual(self.filter.parameters["param1"], "new_value")
        self.assertEqual(self.filter.parameters["param2"], 42)
        self.assertEqual(self.filter.parameters["param3"], "new_param")

    def test_execution_time_measurement(self):
        """Test execution time measurement."""

        def test_function():
            time.sleep(0.01)  # Small delay
            return "result"

        result = self.filter._measure_execution_time(test_function)

        self.assertEqual(result, "result")
        self.assertGreater(self.filter.metadata.execution_time, 0)
        self.assertIsNone(self.filter.metadata.error_message)

    def test_execution_time_with_exception(self):
        """Test execution time measurement with exception."""

        def failing_function():
            time.sleep(0.01)
            raise ValueError("Test error")

        with self.assertRaises(ValueError):
            self.filter._measure_execution_time(failing_function)

        self.assertGreater(self.filter.metadata.execution_time, 0)
        self.assertEqual(self.filter.metadata.error_message, "Test error")

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        # Test with different array sizes
        small_array = np.zeros((10, 10, 3), dtype=np.uint8)
        large_array = np.zeros((1000, 1000, 3), dtype=np.float64)

        small_memory = self.filter._estimate_memory_usage(small_array)
        large_memory = self.filter._estimate_memory_usage(large_array)

        self.assertGreater(large_memory, small_memory)
        self.assertAlmostEqual(small_memory, 0.0003, places=4)  # ~300 bytes

    def test_shape_recording(self):
        """Test input/output shape recording."""
        input_data = np.zeros((100, 100, 3))
        output_data = np.zeros((50, 50, 3))

        self.filter._record_shapes(input_data, output_data)

        self.assertEqual(self.filter.metadata.input_shape, (100, 100, 3))
        self.assertEqual(self.filter.metadata.output_shape, (50, 50, 3))


class TestInputValidation(unittest.TestCase):
    """Test cases for input validation system."""

    def setUp(self):
        """Set up test fixtures."""
        self.rgb_filter = BaseFilter(
            name="rgb_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test",
        )

        self.rgba_filter = BaseFilter(
            name="rgba_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGBA,
            category="test",
        )

        self.grayscale_filter = BaseFilter(
            name="grayscale_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.GRAYSCALE,
            category="test",
        )

        self.video_filter = BaseFilter(
            name="video_filter",
            data_type=DataType.VIDEO,
            color_format=ColorFormat.RGB,
            category="test",
        )

        self.grayscale_video_filter = BaseFilter(
            name="grayscale_video_filter",
            data_type=DataType.VIDEO,
            color_format=ColorFormat.GRAYSCALE,
            category="test",
        )

    def test_valid_rgb_image(self):
        """Test validation of valid RGB image."""
        valid_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.assertTrue(self.rgb_filter.validate_input(valid_rgb))

    def test_valid_rgba_image(self):
        """Test validation of valid RGBA image."""
        valid_rgba = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        self.assertTrue(self.rgba_filter.validate_input(valid_rgba))

    def test_valid_grayscale_image(self):
        """Test validation of valid grayscale image."""
        valid_grayscale = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.assertTrue(self.grayscale_filter.validate_input(valid_grayscale))

    def test_valid_video(self):
        """Test validation of valid video data."""
        valid_video = np.random.randint(0, 255, (10, 100, 100, 3), dtype=np.uint8)
        self.assertTrue(self.video_filter.validate_input(valid_video))

    def test_invalid_input_type(self):
        """Test validation with non-numpy input."""
        with self.assertRaises(FilterValidationError) as cm:
            self.rgb_filter.validate_input([1, 2, 3])
        self.assertIn("must be a numpy array", str(cm.exception))

    def test_empty_array(self):
        """Test validation with empty array."""
        empty_array = np.array([])
        with self.assertRaises(FilterValidationError) as cm:
            self.rgb_filter.validate_input(empty_array)
        self.assertIn("cannot be empty", str(cm.exception))

    def test_wrong_dimensions_for_image(self):
        """Test validation with wrong dimensions for image."""
        # 1D array for image
        invalid_1d = np.random.randint(0, 255, (100,), dtype=np.uint8)
        with self.assertRaises(FilterValidationError) as cm:
            self.rgb_filter.validate_input(invalid_1d)
        self.assertIn("must be 2D or 3D array", str(cm.exception))

        # 4D array for image
        invalid_4d = np.random.randint(0, 255, (10, 100, 100, 3), dtype=np.uint8)
        with self.assertRaises(FilterValidationError) as cm:
            self.rgb_filter.validate_input(invalid_4d)
        self.assertIn("must be 2D or 3D array", str(cm.exception))

    def test_wrong_dimensions_for_video(self):
        """Test validation with wrong dimensions for video."""
        # 3D array for video
        invalid_3d = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with self.assertRaises(FilterValidationError) as cm:
            self.video_filter.validate_input(invalid_3d)
        self.assertIn("must be 4D array", str(cm.exception))

    def test_wrong_channels_for_rgb(self):
        """Test validation with wrong number of channels for RGB."""
        # 4 channels for RGB filter
        invalid_rgb = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        with self.assertRaises(FilterValidationError) as cm:
            self.rgb_filter.validate_input(invalid_rgb)
        self.assertIn("RGB format requires 3 channels", str(cm.exception))

    def test_wrong_channels_for_rgba(self):
        """Test validation with wrong number of channels for RGBA."""
        # 3 channels for RGBA filter
        invalid_rgba = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with self.assertRaises(FilterValidationError) as cm:
            self.rgba_filter.validate_input(invalid_rgba)
        self.assertIn("RGBA format requires 4 channels", str(cm.exception))

    def test_wrong_dimensions_for_grayscale(self):
        """Test validation with wrong dimensions for grayscale."""
        # 3D array for grayscale image
        invalid_grayscale = np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8)
        with self.assertRaises(FilterValidationError) as cm:
            self.grayscale_filter.validate_input(invalid_grayscale)
        self.assertIn("Grayscale image format requires 2D array", str(cm.exception))

    def test_non_numeric_data(self):
        """Test validation with non-numeric data."""
        string_array = np.array([["a", "b"], ["c", "d"]])
        with self.assertRaises(FilterValidationError) as cm:
            self.rgb_filter.validate_input(string_array)
        self.assertIn("Data must be numeric", str(cm.exception))

    def test_nan_values(self):
        """Test validation with NaN values."""
        nan_array = np.array([[1.0, 2.0], [np.nan, 4.0]])
        with self.assertRaises(FilterValidationError) as cm:
            self.grayscale_filter.validate_input(nan_array)
        self.assertIn("contains NaN values", str(cm.exception))

    def test_infinite_values(self):
        """Test validation with infinite values."""
        inf_array = np.array([[1.0, 2.0], [np.inf, 4.0]])
        with self.assertRaises(FilterValidationError) as cm:
            self.grayscale_filter.validate_input(inf_array)
        self.assertIn("contains infinite values", str(cm.exception))

    def test_uint8_range_validation(self):
        """Test validation of uint8 value ranges."""
        # Valid uint8 range
        valid_uint8 = np.array([[0, 128, 255]], dtype=np.uint8)
        self.assertTrue(self.grayscale_filter.validate_input(valid_uint8))

        # Invalid uint8 range (this shouldn't happen with proper uint8, but test the logic)
        # We'll create an array that violates the range check
        invalid_array = np.array(
            [[-1, 256]], dtype=np.int16
        )  # Use int16 to allow invalid values
        grayscale_int16_filter = BaseFilter(
            name="test",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.GRAYSCALE,
            category="test",
        )
        # This should pass since we're not enforcing uint8 range for int16
        self.assertTrue(grayscale_int16_filter.validate_input(invalid_array))

    def test_float_data(self):
        """Test validation with float data."""
        # Valid float data in [0, 1] range
        valid_float = np.random.rand(100, 100).astype(np.float32)
        self.assertTrue(self.grayscale_filter.validate_input(valid_float))

        # Float data outside [0, 1] range (should still pass with warning)
        out_of_range_float = np.array([[2.0, -1.0]], dtype=np.float32)
        self.assertTrue(self.grayscale_filter.validate_input(out_of_range_float))

    def test_grayscale_video_validation(self):
        """Test validation for grayscale video data."""
        # Valid 3D grayscale video (frames, height, width)
        valid_3d_video = np.random.rand(10, 100, 100).astype(np.float32)
        self.assertTrue(self.grayscale_video_filter.validate_input(valid_3d_video))
        
        # Valid 4D grayscale video with single channel (frames, height, width, 1)
        valid_4d_video = np.random.rand(10, 100, 100, 1).astype(np.float32)
        self.assertTrue(self.grayscale_video_filter.validate_input(valid_4d_video))
        
        # Invalid 2D array for video
        invalid_2d = np.random.rand(100, 100).astype(np.float32)
        with self.assertRaises(FilterValidationError):
            self.grayscale_video_filter.validate_input(invalid_2d)

    def test_edge_case_dimensions(self):
        """Test validation with edge case dimensions."""
        # Very small valid arrays
        tiny_rgb = np.ones((1, 1, 3), dtype=np.uint8)
        self.assertTrue(self.rgb_filter.validate_input(tiny_rgb))
        
        tiny_grayscale = np.ones((1, 1), dtype=np.uint8)
        self.assertTrue(self.grayscale_filter.validate_input(tiny_grayscale))
        
        # Single frame video
        single_frame_video = np.ones((1, 10, 10, 3), dtype=np.uint8)
        self.assertTrue(self.video_filter.validate_input(single_frame_video))

    def test_different_numeric_dtypes(self):
        """Test validation with different numeric data types."""
        test_shape = (10, 10, 3)
        
        # Test various integer types
        for dtype in [np.int8, np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]:
            test_array = np.ones(test_shape, dtype=dtype)
            self.assertTrue(self.rgb_filter.validate_input(test_array))
        
        # Test various float types
        for dtype in [np.float16, np.float32, np.float64]:
            test_array = np.ones(test_shape, dtype=dtype) * 0.5
            self.assertTrue(self.rgb_filter.validate_input(test_array))


class TestFilterProtocolCompliance(unittest.TestCase):
    """Test cases for FilterProtocol compliance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = BaseFilter(
            name="protocol_test_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test",
            param1="value1",
            param2=42
        )
    
    def test_protocol_attributes_exist(self):
        """Test that all required protocol attributes exist."""
        self.assertTrue(hasattr(self.filter, 'name'))
        self.assertTrue(hasattr(self.filter, 'data_type'))
        self.assertTrue(hasattr(self.filter, 'color_format'))
        self.assertTrue(hasattr(self.filter, 'category'))
    
    def test_protocol_attributes_types(self):
        """Test that protocol attributes have correct types."""
        self.assertIsInstance(self.filter.name, str)
        self.assertIsInstance(self.filter.data_type, DataType)
        self.assertIsInstance(self.filter.color_format, ColorFormat)
        self.assertIsInstance(self.filter.category, str)
    
    def test_protocol_methods_exist(self):
        """Test that all required protocol methods exist."""
        self.assertTrue(hasattr(self.filter, 'apply'))
        self.assertTrue(hasattr(self.filter, 'get_parameters'))
        self.assertTrue(hasattr(self.filter, 'set_parameters'))
        self.assertTrue(hasattr(self.filter, 'validate_input'))
        
        # Check methods are callable
        self.assertTrue(callable(self.filter.get_parameters))
        self.assertTrue(callable(self.filter.set_parameters))
        self.assertTrue(callable(self.filter.validate_input))
    
    def test_protocol_method_signatures(self):
        """Test that protocol methods have correct signatures."""
        # Test get_parameters returns dict
        params = self.filter.get_parameters()
        self.assertIsInstance(params, dict)
        
        # Test set_parameters accepts kwargs
        self.filter.set_parameters(new_param="new_value")
        
        # Test validate_input accepts numpy array and returns bool
        test_data = np.ones((10, 10, 3))
        result = self.filter.validate_input(test_data)
        self.assertIsInstance(result, bool)


class TestMemoryManagement(unittest.TestCase):
    """Test cases for memory management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = BaseFilter(
            name="memory_test_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test"
        )
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        # Test with different array sizes and types
        small_uint8 = np.zeros((10, 10, 3), dtype=np.uint8)
        large_float64 = np.zeros((1000, 1000, 3), dtype=np.float64)
        
        small_memory = self.filter._estimate_memory_usage(small_uint8)
        large_memory = self.filter._estimate_memory_usage(large_float64)
        
        # Large array should use more memory
        self.assertGreater(large_memory, small_memory)
        
        # Check reasonable memory estimates
        self.assertGreater(small_memory, 0)
        self.assertGreater(large_memory, 0)
    
    def test_inplace_decision(self):
        """Test in-place processing decision logic."""
        # Small array - might not need in-place
        small_array = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Large array - should recommend in-place
        large_array = np.zeros((2000, 2000, 3), dtype=np.float64)
        
        # The decision depends on available memory, so we just test the method works
        small_decision = self.filter._should_use_inplace(small_array)
        large_decision = self.filter._should_use_inplace(large_array)
        
        self.assertIsInstance(small_decision, bool)
        self.assertIsInstance(large_decision, bool)
    
    def test_memory_requirements_check(self):
        """Test memory requirements checking."""
        test_array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test without copy (should pass)
        result_no_copy = self.filter._check_memory_requirements(test_array, creates_copy=False)
        self.assertIsInstance(result_no_copy, bool)
        
        # Test with copy (should pass for reasonable size)
        result_with_copy = self.filter._check_memory_requirements(test_array, creates_copy=True)
        self.assertIsInstance(result_with_copy, bool)
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        input_data = np.ones((50, 50, 3), dtype=np.uint8)
        output_data = np.ones((50, 50, 3), dtype=np.uint8) * 2
        
        # Track memory usage
        self.filter._track_memory_usage(input_data, output_data, used_inplace=False)
        
        # Check metadata was updated
        self.assertGreater(self.filter.metadata.memory_usage, 0)
        self.assertGreater(self.filter.metadata.peak_memory_usage, 0)
        self.assertGreaterEqual(self.filter.metadata.memory_efficiency_ratio, 0)
        self.assertFalse(self.filter.metadata.used_inplace_processing)
        
        # Test with in-place processing
        self.filter._track_memory_usage(input_data, output_data, used_inplace=True)
        self.assertTrue(self.filter.metadata.used_inplace_processing)
    
    def test_shape_recording(self):
        """Test input/output shape recording."""
        input_data = np.zeros((100, 100, 3))
        output_data = np.zeros((50, 50, 3))
        
        self.filter._record_shapes(input_data, output_data)
        
        self.assertEqual(self.filter.metadata.input_shape, (100, 100, 3))
        self.assertEqual(self.filter.metadata.output_shape, (50, 50, 3))
    
    def test_chunked_processing_decision(self):
        """Test chunked processing decision logic."""
        # Small array - should not need chunked processing
        small_array = np.zeros((10, 10, 3), dtype=np.uint8)
        small_decision = self.filter._should_use_chunked_processing(small_array, memory_threshold_mb=1.0)
        
        # Large array - might need chunked processing
        large_array = np.zeros((1000, 1000, 3), dtype=np.float64)
        large_decision = self.filter._should_use_chunked_processing(large_array, memory_threshold_mb=1.0)
        
        self.assertIsInstance(small_decision, bool)
        self.assertIsInstance(large_decision, bool)


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = BaseFilter(
            name="error_test_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test"
        )
    
    def test_execution_time_with_success(self):
        """Test execution time measurement with successful function."""
        def successful_function():
            time.sleep(0.01)
            return "success"
        
        result = self.filter._measure_execution_time(successful_function)
        
        self.assertEqual(result, "success")
        self.assertGreater(self.filter.metadata.execution_time, 0)
        self.assertIsNone(self.filter.metadata.error_message)
    
    def test_execution_time_with_exception(self):
        """Test execution time measurement with exception."""
        def failing_function():
            time.sleep(0.01)
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            self.filter._measure_execution_time(failing_function)
        
        self.assertGreater(self.filter.metadata.execution_time, 0)
        self.assertEqual(self.filter.metadata.error_message, "Test error")
    
    def test_progress_clamping(self):
        """Test that progress values are properly clamped."""
        # Test progress > 1.0
        self.filter._update_progress(1.5)
        self.assertEqual(self.filter.metadata.progress, 1.0)
        
        # Test progress < 0.0
        self.filter._update_progress(-0.5)
        self.assertEqual(self.filter.metadata.progress, 0.0)
        
        # Test valid progress
        self.filter._update_progress(0.7)
        self.assertEqual(self.filter.metadata.progress, 0.7)


if __name__ == "__main__":
    unittest.main()
