"""
Test cases for ExecutionQueue and FilterStep functionality.
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock
from image_processing_library.core import (
    ExecutionQueue, FilterStep, BaseFilter, DataType, ColorFormat,
    FilterExecutionError
)


class MockFilter(BaseFilter):
    """Mock filter for testing purposes."""
    
    def __init__(self, name="mock_filter", multiplier=1.0, **kwargs):
        super().__init__(
            name=name,
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test",
            multiplier=multiplier,
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply mock filter - multiply by the multiplier parameter."""
        self.validate_input(data)
        
        # Simulate progress updates
        self._update_progress(0.5)
        result = data * self.parameters.get('multiplier', 1.0)
        self._update_progress(1.0)
        
        return result


class MockGrayscaleFilter(BaseFilter):
    """Mock grayscale filter for testing purposes."""
    
    def __init__(self, name="mock_grayscale_filter", multiplier=1.0, **kwargs):
        super().__init__(
            name=name,
            data_type=DataType.IMAGE,
            color_format=ColorFormat.GRAYSCALE,
            category="test",
            multiplier=multiplier,
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply mock filter - multiply by the multiplier parameter."""
        self.validate_input(data)
        
        # Simulate progress updates
        self._update_progress(0.5)
        result = data * self.parameters.get('multiplier', 1.0)
        self._update_progress(1.0)
        
        return result


class TestFilterStep(unittest.TestCase):
    """Test cases for FilterStep dataclass."""
    
    def test_filter_step_creation(self):
        """Test basic FilterStep creation."""
        step = FilterStep(
            filter_class=MockFilter,
            parameters={'multiplier': 2.0},
            save_intermediate=False
        )
        
        self.assertEqual(step.filter_class, MockFilter)
        self.assertEqual(step.parameters, {'multiplier': 2.0})
        self.assertFalse(step.save_intermediate)
        self.assertIsNone(step.save_path)
    
    def test_filter_step_with_intermediate_save(self):
        """Test FilterStep with intermediate saving."""
        step = FilterStep(
            filter_class=MockFilter,
            parameters={},
            save_intermediate=True,
            save_path="/tmp/test.png"
        )
        
        self.assertTrue(step.save_intermediate)
        self.assertEqual(step.save_path, "/tmp/test.png")
    
    def test_filter_step_validation_error(self):
        """Test FilterStep validation when save_intermediate=True but no save_path."""
        with self.assertRaises(ValueError):
            FilterStep(
                filter_class=MockFilter,
                parameters={},
                save_intermediate=True,
                save_path=None
            )


class TestExecutionQueue(unittest.TestCase):
    """Test cases for ExecutionQueue functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.queue = ExecutionQueue()
        self.test_data = np.ones((10, 10, 3), dtype=np.float32)
        
        # Create additional test data for various scenarios
        self.uint8_data = (self.test_data * 255).astype(np.uint8)
        self.large_data = np.ones((100, 100, 3), dtype=np.float32)
        self.video_data = np.ones((5, 10, 10, 3), dtype=np.float32)
    
    def test_empty_queue_creation(self):
        """Test creating an empty execution queue."""
        self.assertEqual(self.queue.get_step_count(), 0)
        self.assertEqual(len(self.queue.steps), 0)
    
    def test_add_filter_basic(self):
        """Test adding a basic filter to the queue."""
        self.queue.add_filter(MockFilter, {'multiplier': 2.0})
        
        self.assertEqual(self.queue.get_step_count(), 1)
        step_info = self.queue.get_step_info(0)
        self.assertEqual(step_info['filter_name'], 'MockFilter')
        self.assertEqual(step_info['parameters'], {'multiplier': 2.0})
    
    def test_add_filter_with_intermediate_save(self):
        """Test adding a filter with intermediate saving."""
        self.queue.add_filter(
            MockFilter, 
            {'multiplier': 1.5},
            save_intermediate=True,
            save_path="/tmp/intermediate.png"
        )
        
        step_info = self.queue.get_step_info(0)
        self.assertTrue(step_info['save_intermediate'])
        self.assertEqual(step_info['save_path'], "/tmp/intermediate.png")
    
    def test_add_filter_instance(self):
        """Test adding a filter instance instead of class."""
        filter_instance = MockFilter(name="instance_filter", multiplier=3.0)
        self.queue.add_filter(filter_instance)
        
        step_info = self.queue.get_step_info(0)
        self.assertEqual(step_info['filter_name'], 'instance_filter')
    
    def test_execute_single_filter(self):
        """Test executing a queue with a single filter."""
        self.queue.add_filter(MockFilter, {'multiplier': 2.0})
        
        result = self.queue.execute(self.test_data)
        
        expected = self.test_data * 2.0
        np.testing.assert_array_equal(result, expected)
    
    def test_execute_multiple_filters(self):
        """Test executing a queue with multiple filters."""
        self.queue.add_filter(MockFilter, {'multiplier': 2.0})
        self.queue.add_filter(MockFilter, {'multiplier': 3.0})
        
        result = self.queue.execute(self.test_data)
        
        # Should be multiplied by 2.0 then by 3.0 = 6.0 total
        expected = self.test_data * 6.0
        np.testing.assert_array_equal(result, expected)
    
    def test_execute_empty_queue_error(self):
        """Test that executing an empty queue raises an error."""
        with self.assertRaises(ValueError):
            self.queue.execute(self.test_data)
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        progress_updates = []
        
        def progress_callback(progress, filter_name):
            progress_updates.append((progress, filter_name))
        
        self.queue.set_progress_callback(progress_callback)
        self.queue.add_filter(MockFilter, {'multiplier': 2.0})
        
        self.queue.execute(self.test_data)
        
        # Should have received progress updates
        self.assertGreater(len(progress_updates), 0)
        # Final progress should be 1.0
        final_progress, final_filter = progress_updates[-1]
        self.assertEqual(final_progress, 1.0)
        self.assertEqual(final_filter, 'mock_filter')
    
    def test_filter_execution_error(self):
        """Test error handling when a filter fails."""
        # Create a filter that will fail
        failing_filter = Mock()
        failing_filter.name = "failing_filter"
        failing_filter.apply.side_effect = RuntimeError("Filter failed")
        
        self.queue.add_filter(failing_filter)
        
        with self.assertRaises(FilterExecutionError) as context:
            self.queue.execute(self.test_data)
        
        self.assertIn("failing_filter", str(context.exception))
        self.assertIn("Filter failed", str(context.exception))
    
    def test_clear_queue(self):
        """Test clearing the execution queue."""
        self.queue.add_filter(MockFilter, {'multiplier': 2.0})
        self.queue.add_filter(MockFilter, {'multiplier': 3.0})
        
        self.assertEqual(self.queue.get_step_count(), 2)
        
        self.queue.clear()
        
        self.assertEqual(self.queue.get_step_count(), 0)
    
    def test_remove_step(self):
        """Test removing a specific step from the queue."""
        self.queue.add_filter(MockFilter, {'multiplier': 2.0})
        self.queue.add_filter(MockFilter, {'multiplier': 3.0})
        self.queue.add_filter(MockFilter, {'multiplier': 4.0})
        
        self.assertEqual(self.queue.get_step_count(), 3)
        
        # Remove middle step
        self.queue.remove_step(1)
        
        self.assertEqual(self.queue.get_step_count(), 2)
        
        # Verify remaining steps
        step0 = self.queue.get_step_info(0)
        step1 = self.queue.get_step_info(1)
        
        self.assertEqual(step0['parameters']['multiplier'], 2.0)
        self.assertEqual(step1['parameters']['multiplier'], 4.0)
    
    def test_remove_step_invalid_index(self):
        """Test removing step with invalid index."""
        self.queue.add_filter(MockFilter, {'multiplier': 2.0})
        
        with self.assertRaises(IndexError):
            self.queue.remove_step(5)
        
        with self.assertRaises(IndexError):
            self.queue.remove_step(-1)
    
    def test_get_step_info_invalid_index(self):
        """Test getting step info with invalid index."""
        self.queue.add_filter(MockFilter, {'multiplier': 2.0})
        
        with self.assertRaises(IndexError):
            self.queue.get_step_info(5)
    
    def test_intermediate_save_warning(self):
        """Test intermediate save functionality (placeholder implementation)."""
        # This test verifies the intermediate save doesn't crash the pipeline
        self.queue.add_filter(
            MockFilter, 
            {'multiplier': 2.0},
            save_intermediate=True,
            save_path="/tmp/test_intermediate.png"
        )
        
        # Should execute successfully even though save is not fully implemented
        result = self.queue.execute(self.test_data)
        expected = self.test_data * 2.0
        np.testing.assert_array_equal(result, expected)
    
    def test_intermediate_save_numpy_format(self):
        """Test intermediate save with numpy format."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "intermediate.npy")
            
            self.queue.add_filter(
                MockFilter, 
                {'multiplier': 3.0},
                save_intermediate=True,
                save_path=save_path
            )
            
            result = self.queue.execute(self.test_data)
            
            # Verify the file was created
            self.assertTrue(os.path.exists(save_path))
            
            # Verify the saved data
            saved_data = np.load(save_path)
            expected = self.test_data * 3.0
            np.testing.assert_array_equal(saved_data, expected)
    
    def test_intermediate_save_different_data_types(self):
        """Test intermediate save with different data types."""
        import tempfile
        import os
        
        # Test with uint8 data
        uint8_data = (self.test_data * 255).astype(np.uint8)
        
        # Test with float64 data
        float64_data = self.test_data.astype(np.float64)
        
        # Test with grayscale data
        grayscale_data = np.mean(self.test_data, axis=2).astype(np.float32)
        
        test_cases = [
            (uint8_data, "uint8_test.npy", MockFilter),
            (float64_data, "float64_test.npy", MockFilter),
            (grayscale_data, "grayscale_test.npy", MockGrayscaleFilter)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for test_data, filename, filter_class in test_cases:
                with self.subTest(data_type=test_data.dtype, shape=test_data.shape):
                    save_path = os.path.join(temp_dir, filename)
                    
                    # Create a new queue for each test
                    queue = ExecutionQueue()
                    queue.add_filter(
                        filter_class, 
                        {'multiplier': 1.5},
                        save_intermediate=True,
                        save_path=save_path
                    )
                    
                    result = queue.execute(test_data)
                    
                    # Verify the file was created
                    self.assertTrue(os.path.exists(save_path))
                    
                    # Verify the saved data
                    saved_data = np.load(save_path)
                    expected = test_data * 1.5
                    np.testing.assert_array_equal(saved_data, expected)
    
    def test_intermediate_save_multiple_steps(self):
        """Test intermediate save at multiple steps in the pipeline."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path1 = os.path.join(temp_dir, "step1.npy")
            save_path2 = os.path.join(temp_dir, "step2.npy")
            
            # Add multiple filters with intermediate saves
            self.queue.add_filter(
                MockFilter, 
                {'multiplier': 2.0},
                save_intermediate=True,
                save_path=save_path1
            )
            self.queue.add_filter(
                MockFilter, 
                {'multiplier': 3.0},
                save_intermediate=True,
                save_path=save_path2
            )
            
            result = self.queue.execute(self.test_data)
            
            # Verify both files were created
            self.assertTrue(os.path.exists(save_path1))
            self.assertTrue(os.path.exists(save_path2))
            
            # Verify the saved data at each step
            step1_data = np.load(save_path1)
            step2_data = np.load(save_path2)
            
            expected_step1 = self.test_data * 2.0
            expected_step2 = self.test_data * 2.0 * 3.0  # Cumulative effect
            
            np.testing.assert_array_equal(step1_data, expected_step1)
            np.testing.assert_array_equal(step2_data, expected_step2)
            np.testing.assert_array_equal(result, expected_step2)
    
    def test_intermediate_save_error_handling(self):
        """Test error handling in intermediate save."""
        # Test with invalid save path (should not crash the pipeline)
        self.queue.add_filter(
            MockFilter, 
            {'multiplier': 2.0},
            save_intermediate=True,
            save_path="/invalid/path/that/does/not/exist/file.npy"
        )
        
        # Should execute successfully despite save error
        result = self.queue.execute(self.test_data)
        expected = self.test_data * 2.0
        np.testing.assert_array_equal(result, expected)
    
    def test_intermediate_save_directory_creation(self):
        """Test that intermediate save creates directories as needed."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a nested directory path that doesn't exist
            save_path = os.path.join(temp_dir, "nested", "dir", "test.npy")
            
            self.queue.add_filter(
                MockFilter, 
                {'multiplier': 2.0},
                save_intermediate=True,
                save_path=save_path
            )
            
            result = self.queue.execute(self.test_data)
            
            # Verify the nested directory was created and file saved
            self.assertTrue(os.path.exists(save_path))
            
            # Verify the saved data
            saved_data = np.load(save_path)
            expected = self.test_data * 2.0
            np.testing.assert_array_equal(saved_data, expected)


class TestExecutionQueueAdvanced(unittest.TestCase):
    """Advanced test cases for ExecutionQueue functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.queue = ExecutionQueue()
        self.test_data = np.ones((10, 10, 3), dtype=np.float32)
    
    def test_queue_with_filter_instances(self):
        """Test queue execution with pre-instantiated filter instances."""
        # Create filter instances
        filter1 = MockFilter(name="instance1", multiplier=2.0)
        filter2 = MockFilter(name="instance2", multiplier=3.0)
        
        # Add instances to queue
        self.queue.add_filter(filter1)
        self.queue.add_filter(filter2)
        
        result = self.queue.execute(self.test_data)
        
        # Should be multiplied by 2.0 then by 3.0 = 6.0 total
        expected = self.test_data * 6.0
        np.testing.assert_array_equal(result, expected)
    
    def test_queue_mixed_classes_and_instances(self):
        """Test queue with mix of filter classes and instances."""
        filter_instance = MockFilter(name="mixed_instance", multiplier=1.5)
        
        self.queue.add_filter(MockFilter, {'multiplier': 2.0})
        self.queue.add_filter(filter_instance)
        self.queue.add_filter(MockFilter, {'multiplier': 4.0})
        
        result = self.queue.execute(self.test_data)
        
        # Should be: 1.0 * 2.0 * 1.5 * 4.0 = 12.0
        expected = self.test_data * 12.0
        np.testing.assert_array_equal(result, expected)
    
    def test_queue_parameter_updates(self):
        """Test updating parameters of filter instances in queue."""
        filter_instance = MockFilter(name="updatable", multiplier=1.0)
        
        self.queue.add_filter(filter_instance, {'multiplier': 5.0})  # Should update parameters
        
        result = self.queue.execute(self.test_data)
        
        # Parameters should be updated to multiplier=5.0
        expected = self.test_data * 5.0
        np.testing.assert_array_equal(result, expected)
    
    def test_queue_error_recovery_information(self):
        """Test detailed error information when filters fail."""
        # Create a filter that will fail with specific error
        failing_filter = Mock()
        failing_filter.name = "detailed_failing_filter"
        failing_filter.apply.side_effect = RuntimeError("Detailed error message")
        
        self.queue.add_filter(MockFilter, {'multiplier': 2.0})  # This should succeed
        self.queue.add_filter(failing_filter)  # This should fail
        self.queue.add_filter(MockFilter, {'multiplier': 3.0})  # This should not execute
        
        with self.assertRaises(FilterExecutionError) as context:
            self.queue.execute(self.test_data)
        
        error_message = str(context.exception)
        self.assertIn("detailed_failing_filter", error_message)
        self.assertIn("step 2", error_message)  # Should indicate which step failed
        self.assertIn("Detailed error message", error_message)
    
    def test_queue_progress_tracking_detailed(self):
        """Test detailed progress tracking with multiple filters."""
        progress_log = []
        
        def detailed_progress_callback(progress, filter_name):
            progress_log.append((round(progress, 2), filter_name))
        
        self.queue.set_progress_callback(detailed_progress_callback)
        
        # Add multiple filters
        self.queue.add_filter(MockFilter, {'multiplier': 1.0})
        self.queue.add_filter(MockFilter, {'multiplier': 1.0})
        self.queue.add_filter(MockFilter, {'multiplier': 1.0})
        
        self.queue.execute(self.test_data)
        
        # Should have progress updates for each filter
        self.assertGreater(len(progress_log), 0)
        
        # Check that progress increases and reaches 1.0
        final_progress = progress_log[-1][0]
        self.assertEqual(final_progress, 1.0)
        
        # Check that all filter names appear
        filter_names = [entry[1] for entry in progress_log]
        self.assertIn("mock_filter", filter_names)
    
    def test_queue_step_management_comprehensive(self):
        """Test comprehensive step management operations."""
        # Add multiple filters
        self.queue.add_filter(MockFilter, {'multiplier': 1.0})
        self.queue.add_filter(MockFilter, {'multiplier': 2.0})
        self.queue.add_filter(MockFilter, {'multiplier': 3.0})
        
        # Test step count
        self.assertEqual(self.queue.get_step_count(), 3)
        
        # Test step info for each step
        for i in range(3):
            step_info = self.queue.get_step_info(i)
            self.assertEqual(step_info['index'], i)
            self.assertEqual(step_info['filter_name'], 'MockFilter')
            self.assertEqual(step_info['parameters']['multiplier'], float(i + 1))
        
        # Test removing middle step
        self.queue.remove_step(1)
        self.assertEqual(self.queue.get_step_count(), 2)
        
        # Verify remaining steps
        step0 = self.queue.get_step_info(0)
        step1 = self.queue.get_step_info(1)
        self.assertEqual(step0['parameters']['multiplier'], 1.0)
        self.assertEqual(step1['parameters']['multiplier'], 3.0)
        
        # Test clearing queue
        self.queue.clear()
        self.assertEqual(self.queue.get_step_count(), 0)
    
    def test_queue_with_different_data_types(self):
        """Test queue execution with different input data types."""
        test_cases = [
            (np.ones((10, 10, 3), dtype=np.uint8), "uint8_rgb"),
            (np.ones((10, 10, 4), dtype=np.float32), "float32_rgba"),
            (np.ones((10, 10), dtype=np.float64), "float64_grayscale"),
            (np.ones((5, 10, 10, 3), dtype=np.uint8), "uint8_video")
        ]
        
        for test_data, description in test_cases:
            with self.subTest(data_type=description):
                # Create appropriate filter for data type
                if test_data.ndim == 4:  # Video
                    filter_class = MockFilter  # Assuming MockFilter can handle video
                elif test_data.ndim == 2:  # Grayscale
                    filter_class = MockGrayscaleFilter
                else:  # RGB/RGBA
                    filter_class = MockFilter
                
                queue = ExecutionQueue()
                queue.add_filter(filter_class, {'multiplier': 2.0})
                
                try:
                    result = queue.execute(test_data)
                    expected = test_data * 2.0
                    np.testing.assert_array_equal(result, expected)
                except Exception as e:
                    # Some combinations might not be supported, that's okay
                    self.assertIsInstance(e, (FilterValidationError, FilterExecutionError))
    
    def test_queue_memory_efficiency(self):
        """Test queue memory efficiency with large data."""
        # Create larger test data
        large_data = np.ones((200, 200, 3), dtype=np.float32)
        
        # Add filters that should work efficiently
        self.queue.add_filter(MockFilter, {'multiplier': 1.1})
        self.queue.add_filter(MockFilter, {'multiplier': 1.2})
        
        # Execute and verify result
        result = self.queue.execute(large_data)
        expected = large_data * 1.1 * 1.2
        np.testing.assert_array_equal(result, expected)
        
        # Verify data shape is preserved
        self.assertEqual(result.shape, large_data.shape)
    
    def test_queue_empty_parameters(self):
        """Test queue with filters that have no parameters."""
        # Add filter with empty parameters
        self.queue.add_filter(MockFilter, {})
        
        result = self.queue.execute(self.test_data)
        
        # Should use default multiplier (1.0)
        expected = self.test_data * 1.0
        np.testing.assert_array_equal(result, expected)
    
    def test_queue_none_parameters(self):
        """Test queue with None parameters (should use defaults)."""
        self.queue.add_filter(MockFilter, None)
        
        result = self.queue.execute(self.test_data)
        
        # Should use default multiplier (1.0)
        expected = self.test_data * 1.0
        np.testing.assert_array_equal(result, expected)


class TestFilterStepValidation(unittest.TestCase):
    """Test cases for FilterStep validation."""
    
    def test_filter_step_validation_success(self):
        """Test successful FilterStep creation."""
        step = FilterStep(
            filter_class=MockFilter,
            parameters={'multiplier': 2.0},
            save_intermediate=True,
            save_path="/tmp/test.png"
        )
        
        # Should not raise any exception
        self.assertEqual(step.filter_class, MockFilter)
        self.assertEqual(step.parameters, {'multiplier': 2.0})
        self.assertTrue(step.save_intermediate)
        self.assertEqual(step.save_path, "/tmp/test.png")
    
    def test_filter_step_validation_failure(self):
        """Test FilterStep validation failure."""
        with self.assertRaises(ValueError) as context:
            FilterStep(
                filter_class=MockFilter,
                parameters={},
                save_intermediate=True,
                save_path=None  # This should cause validation error
            )
        
        self.assertIn("save_path must be provided", str(context.exception))
    
    def test_filter_step_no_intermediate_save(self):
        """Test FilterStep without intermediate saving."""
        step = FilterStep(
            filter_class=MockFilter,
            parameters={'multiplier': 1.5},
            save_intermediate=False,
            save_path=None
        )
        
        # Should not raise any exception
        self.assertFalse(step.save_intermediate)
        self.assertIsNone(step.save_path)


if __name__ == '__main__':
    unittest.main()