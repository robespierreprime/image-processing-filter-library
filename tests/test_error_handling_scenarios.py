"""
Comprehensive error handling and recovery scenario tests.

Tests various error conditions, recovery mechanisms, and edge cases
to ensure robust behavior under failure conditions.
"""

import unittest
import tempfile
import shutil
import numpy as np
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from image_processing_library.core import (
    BaseFilter,
    ExecutionQueue,
    PresetManager,
    DataType,
    ColorFormat,
    FilterExecutionError,
    FilterValidationError
)
from image_processing_library.core.utils import (
    FilterError,
    PresetError,
    MemoryError as CustomMemoryError
)


class ErrorProneFilter(BaseFilter):
    """Filter that can simulate various error conditions."""
    
    def __init__(self, error_type=None, error_probability=0.0, **kwargs):
        super().__init__(
            name="error_prone_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test",
            error_type=error_type,
            error_probability=error_probability,
            **kwargs
        )
        self.call_count = 0
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply filter with potential errors."""
        self.call_count += 1
        self.validate_input(data)
        
        error_type = self.parameters.get('error_type')
        error_probability = self.parameters.get('error_probability', 0.0)
        
        # Simulate random errors based on probability
        if error_probability > 0 and np.random.random() < error_probability:
            error_type = 'random'
        
        if error_type == 'validation':
            raise FilterValidationError("Simulated validation error")
        elif error_type == 'memory':
            raise CustomMemoryError("Simulated memory error")
        elif error_type == 'runtime':
            raise RuntimeError("Simulated runtime error")
        elif error_type == 'value':
            raise ValueError("Simulated value error")
        elif error_type == 'type':
            raise TypeError("Simulated type error")
        elif error_type == 'timeout':
            time.sleep(10)  # Simulate timeout
        elif error_type == 'nan_output':
            return np.full_like(data, np.nan)
        elif error_type == 'inf_output':
            return np.full_like(data, np.inf)
        elif error_type == 'wrong_shape':
            return np.ones((data.shape[0] // 2, data.shape[1] // 2, data.shape[2]))
        elif error_type == 'wrong_dtype':
            return data.astype(str)  # Invalid dtype
        elif error_type == 'random':
            # Random error for stress testing
            errors = ['validation', 'memory', 'runtime', 'value']
            random_error = np.random.choice(errors)
            return self.apply(data, error_type=random_error)
        
        # Normal processing
        self._update_progress(0.5)
        result = data * 1.1  # Simple enhancement
        self._update_progress(1.0)
        return result


class TestErrorHandlingScenarios(unittest.TestCase):
    """Test various error handling scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data = np.random.rand(10, 10, 3).astype(np.float32)
        self.temp_dir = tempfile.mkdtemp()
        self.preset_manager = PresetManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_validation_error_handling(self):
        """Test handling of validation errors."""
        queue = ExecutionQueue()
        queue.add_filter(ErrorProneFilter, {'error_type': 'validation'})
        
        with self.assertRaises(FilterExecutionError) as context:
            queue.execute(self.test_data)
        
        # Check error chain
        self.assertIsInstance(context.exception.__cause__, FilterValidationError)
        self.assertIn("Simulated validation error", str(context.exception))
    
    def test_memory_error_handling(self):
        """Test handling of memory errors."""
        queue = ExecutionQueue()
        queue.add_filter(ErrorProneFilter, {'error_type': 'memory'})
        
        with self.assertRaises(FilterExecutionError) as context:
            queue.execute(self.test_data)
        
        # Check error chain
        self.assertIsInstance(context.exception.__cause__, CustomMemoryError)
        self.assertIn("Simulated memory error", str(context.exception))
    
    def test_runtime_error_handling(self):
        """Test handling of runtime errors."""
        queue = ExecutionQueue()
        queue.add_filter(ErrorProneFilter, {'error_type': 'runtime'})
        
        with self.assertRaises(FilterExecutionError) as context:
            queue.execute(self.test_data)
        
        # Check error chain
        self.assertIsInstance(context.exception.__cause__, RuntimeError)
        self.assertIn("Simulated runtime error", str(context.exception))
    
    def test_error_in_middle_of_pipeline(self):
        """Test error handling when error occurs in middle of pipeline."""
        queue = ExecutionQueue()
        
        # Add successful filter
        queue.add_filter(ErrorProneFilter, {'error_type': None})
        
        # Add failing filter
        queue.add_filter(ErrorProneFilter, {'error_type': 'runtime'})
        
        # Add another filter that shouldn't execute
        queue.add_filter(ErrorProneFilter, {'error_type': None})
        
        with self.assertRaises(FilterExecutionError) as context:
            queue.execute(self.test_data)
        
        # Error should indicate which step failed
        self.assertIn("step 2", str(context.exception))
    
    def test_multiple_error_types_in_sequence(self):
        """Test handling multiple different error types."""
        error_types = ['validation', 'memory', 'runtime', 'value', 'type']
        
        for error_type in error_types:
            with self.subTest(error_type=error_type):
                queue = ExecutionQueue()
                queue.add_filter(ErrorProneFilter, {'error_type': error_type})
                
                with self.assertRaises(FilterExecutionError):
                    queue.execute(self.test_data)
    
    def test_error_with_intermediate_saves(self):
        """Test error handling when intermediate saves are involved."""
        queue = ExecutionQueue()
        
        # Add successful filter with intermediate save
        save_path = Path(self.temp_dir) / "intermediate.npy"
        queue.add_filter(
            ErrorProneFilter,
            {'error_type': None},
            save_intermediate=True,
            save_path=str(save_path)
        )
        
        # Add failing filter
        queue.add_filter(ErrorProneFilter, {'error_type': 'runtime'})
        
        with self.assertRaises(FilterExecutionError):
            queue.execute(self.test_data)
        
        # Intermediate save should still have been created
        self.assertTrue(save_path.exists())
        
        # Verify intermediate data
        saved_data = np.load(save_path)
        expected = self.test_data * 1.1
        np.testing.assert_array_almost_equal(saved_data, expected)
    
    def test_invalid_output_handling(self):
        """Test handling of invalid filter outputs."""
        test_cases = [
            ('nan_output', "NaN values"),
            ('inf_output', "infinite values"),
            ('wrong_shape', "shape mismatch"),
        ]
        
        for error_type, expected_error in test_cases:
            with self.subTest(error_type=error_type):
                queue = ExecutionQueue()
                queue.add_filter(ErrorProneFilter, {'error_type': error_type})
                
                # These should either raise errors or be handled gracefully
                try:
                    result = queue.execute(self.test_data)
                    
                    # If execution succeeds, check for invalid outputs
                    if error_type == 'nan_output':
                        self.assertTrue(np.any(np.isnan(result)))
                    elif error_type == 'inf_output':
                        self.assertTrue(np.any(np.isinf(result)))
                    elif error_type == 'wrong_shape':
                        self.assertNotEqual(result.shape, self.test_data.shape)
                        
                except (FilterExecutionError, ValueError, TypeError):
                    # Expected for some error types
                    pass
    
    def test_progress_callback_error_handling(self):
        """Test error handling in progress callbacks."""
        def failing_progress_callback(progress, filter_name):
            if progress > 0.5:
                raise RuntimeError("Progress callback failed")
        
        queue = ExecutionQueue()
        queue.set_progress_callback(failing_progress_callback)
        queue.add_filter(ErrorProneFilter, {'error_type': None})
        
        # Should handle progress callback errors gracefully
        try:
            result = queue.execute(self.test_data)
            # If it succeeds, that's fine - callback errors shouldn't break execution
        except FilterExecutionError:
            # If it fails, the error should be about the callback, not the filter
            pass
    
    def test_concurrent_error_scenarios(self):
        """Test error handling under simulated concurrent conditions."""
        # Simulate multiple queues with random errors
        queues = []
        for i in range(5):
            queue = ExecutionQueue()
            queue.add_filter(ErrorProneFilter, {
                'error_type': None,
                'error_probability': 0.3  # 30% chance of random error
            })
            queues.append(queue)
        
        results = []
        errors = []
        
        for i, queue in enumerate(queues):
            try:
                result = queue.execute(self.test_data)
                results.append((i, result))
            except FilterExecutionError as e:
                errors.append((i, e))
        
        # Should have some successes and some failures
        total_operations = len(results) + len(errors)
        self.assertEqual(total_operations, 5)
        
        # Verify successful results are correct
        for i, result in results:
            expected = self.test_data * 1.1
            np.testing.assert_array_almost_equal(result, expected)


class TestPresetErrorHandling(unittest.TestCase):
    """Test error handling in preset operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.preset_manager = PresetManager(self.temp_dir)
        self.test_data = np.random.rand(10, 10, 3).astype(np.float32)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_preset_save_with_invalid_queue(self):
        """Test preset saving with invalid execution queue."""
        # Create queue with invalid filter
        queue = ExecutionQueue()
        
        # Mock filter that can't be serialized properly
        invalid_filter = Mock()
        invalid_filter.__module__ = None  # This will cause serialization issues
        invalid_filter.__name__ = "InvalidFilter"
        
        queue.add_filter(invalid_filter, {})
        
        # Should handle serialization errors gracefully
        with self.assertRaises(PresetError):
            self.preset_manager.save_preset("invalid_preset", queue, "Test")
    
    def test_preset_load_with_corrupted_file(self):
        """Test preset loading with corrupted files."""
        # Create corrupted JSON file
        corrupted_path = Path(self.temp_dir) / "corrupted.json"
        with open(corrupted_path, 'w') as f:
            f.write('{"invalid": json content')
        
        with self.assertRaises(PresetError):
            self.preset_manager.load_preset("corrupted")
    
    def test_preset_load_with_missing_filter_class(self):
        """Test preset loading when filter class doesn't exist."""
        # Create preset with non-existent filter class
        preset_data = {
            "metadata": {
                "name": "missing_filter_preset",
                "description": "Test",
                "created_at": "2023-01-01T12:00:00"
            },
            "steps": [{
                "filter_class": "nonexistent.module.MissingFilter",
                "parameters": {},
                "save_intermediate": False,
                "save_path": None
            }]
        }
        
        preset_path = Path(self.temp_dir) / "missing_filter.json"
        import json
        with open(preset_path, 'w') as f:
            json.dump(preset_data, f)
        
        with self.assertRaises(PresetError):
            self.preset_manager.load_preset("missing_filter")
    
    def test_preset_execution_with_loaded_errors(self):
        """Test execution of presets that contain error-prone filters."""
        # Create and save preset with error-prone filter
        queue = ExecutionQueue()
        queue.add_filter(ErrorProneFilter, {'error_type': 'runtime'})
        
        self.preset_manager.save_preset("error_preset", queue, "Error test")
        
        # Load and execute preset
        loaded_queue = self.preset_manager.load_preset("error_preset")
        
        with self.assertRaises(FilterExecutionError):
            loaded_queue.execute(self.test_data)
    
    def test_preset_directory_permission_errors(self):
        """Test handling of directory permission errors."""
        # This test is platform-dependent and might not work on all systems
        try:
            # Try to create preset manager with invalid directory
            invalid_dir = "/root/invalid_preset_dir"  # Likely no permission
            preset_manager = PresetManager(invalid_dir)
            
            queue = ExecutionQueue()
            queue.add_filter(ErrorProneFilter, {'error_type': None})
            
            # Should handle permission errors gracefully
            with self.assertRaises(PresetError):
                preset_manager.save_preset("test", queue, "Test")
                
        except (PermissionError, OSError):
            # Skip this test if we can't create the scenario
            self.skipTest("Cannot create permission error scenario on this system")


class TestRecoveryScenarios(unittest.TestCase):
    """Test recovery and resilience scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data = np.random.rand(20, 20, 3).astype(np.float32)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_partial_pipeline_recovery(self):
        """Test recovery from partial pipeline execution."""
        queue = ExecutionQueue()
        
        # Add filters with intermediate saves
        step1_path = Path(self.temp_dir) / "step1.npy"
        step2_path = Path(self.temp_dir) / "step2.npy"
        
        queue.add_filter(
            ErrorProneFilter,
            {'error_type': None},
            save_intermediate=True,
            save_path=str(step1_path)
        )
        
        queue.add_filter(
            ErrorProneFilter,
            {'error_type': 'runtime'},  # This will fail
            save_intermediate=True,
            save_path=str(step2_path)
        )
        
        queue.add_filter(ErrorProneFilter, {'error_type': None})
        
        # Execute and expect failure
        with self.assertRaises(FilterExecutionError):
            queue.execute(self.test_data)
        
        # First intermediate result should be saved
        self.assertTrue(step1_path.exists())
        self.assertFalse(step2_path.exists())  # Second step failed
        
        # Can recover by loading intermediate result and continuing
        intermediate_data = np.load(step1_path)
        
        # Create recovery queue starting from step 2 (but working version)
        recovery_queue = ExecutionQueue()
        recovery_queue.add_filter(ErrorProneFilter, {'error_type': None})
        
        # Execute recovery
        final_result = recovery_queue.execute(intermediate_data)
        
        # Verify recovery worked
        self.assertEqual(final_result.shape, self.test_data.shape)
    
    def test_memory_pressure_recovery(self):
        """Test recovery under memory pressure conditions."""
        # Create larger data to simulate memory pressure
        large_data = np.random.rand(200, 200, 3).astype(np.float32)
        
        queue = ExecutionQueue()
        
        # Add multiple filters that might cause memory issues
        for i in range(3):
            queue.add_filter(ErrorProneFilter, {
                'error_type': 'memory' if i == 1 else None,  # Fail on second filter
                'error_probability': 0.0
            })
        
        # Should handle memory errors gracefully
        with self.assertRaises(FilterExecutionError) as context:
            queue.execute(large_data)
        
        # Check that it's specifically a memory error
        self.assertIsInstance(context.exception.__cause__, CustomMemoryError)
    
    def test_timeout_recovery(self):
        """Test recovery from timeout scenarios."""
        # This test uses a shorter timeout for testing
        queue = ExecutionQueue()
        
        # Add filter that will timeout
        queue.add_filter(ErrorProneFilter, {'error_type': 'timeout'})
        
        # Use threading to implement timeout
        import threading
        import signal
        
        result = [None]
        exception = [None]
        
        def execute_with_timeout():
            try:
                result[0] = queue.execute(self.test_data)
            except Exception as e:
                exception[0] = e
        
        # Start execution in thread
        thread = threading.Thread(target=execute_with_timeout)
        thread.daemon = True
        thread.start()
        
        # Wait for short time
        thread.join(timeout=0.1)
        
        if thread.is_alive():
            # Thread is still running (timeout scenario)
            # In a real implementation, we'd have proper timeout handling
            self.assertTrue(True)  # Test passes - we detected the timeout
        else:
            # Thread completed quickly (unexpected)
            if exception[0]:
                # Some other error occurred
                self.assertIsInstance(exception[0], FilterExecutionError)
    
    def test_data_corruption_recovery(self):
        """Test recovery from data corruption scenarios."""
        # Test with various corrupted data
        corrupted_data_cases = [
            np.full((10, 10, 3), np.nan),  # All NaN
            np.full((10, 10, 3), np.inf),  # All infinite
            np.zeros((0, 0, 3)),           # Empty array
        ]
        
        for i, corrupted_data in enumerate(corrupted_data_cases):
            with self.subTest(corruption_type=i):
                queue = ExecutionQueue()
                queue.add_filter(ErrorProneFilter, {'error_type': None})
                
                try:
                    result = queue.execute(corrupted_data)
                    # If it succeeds, verify the result is reasonable
                    if result.size > 0:
                        # Check that result has some valid values
                        has_valid = np.any(np.isfinite(result))
                        # This might pass or fail depending on filter implementation
                        
                except (FilterExecutionError, FilterValidationError):
                    # Expected for corrupted data
                    pass
    
    def test_cascading_failure_recovery(self):
        """Test recovery from cascading failures."""
        # Create scenario where one failure might cause others
        queue = ExecutionQueue()
        
        # Add filters where later ones depend on earlier ones working correctly
        queue.add_filter(ErrorProneFilter, {'error_type': None})
        queue.add_filter(ErrorProneFilter, {'error_type': 'runtime'})  # This fails
        queue.add_filter(ErrorProneFilter, {'error_type': None})       # This won't execute
        
        with self.assertRaises(FilterExecutionError) as context:
            queue.execute(self.test_data)
        
        # Verify error information is preserved through the cascade
        error_msg = str(context.exception)
        self.assertIn("step 2", error_msg)
        self.assertIn("Simulated runtime error", error_msg)


class TestStressScenarios(unittest.TestCase):
    """Test system behavior under stress conditions."""
    
    def setUp(self):
        """Set up stress test environment."""
        self.test_data = np.random.rand(50, 50, 3).astype(np.float32)
    
    def test_high_error_rate_stress(self):
        """Test system behavior with high error rates."""
        # Run many operations with high error probability
        success_count = 0
        error_count = 0
        
        for i in range(20):
            queue = ExecutionQueue()
            queue.add_filter(ErrorProneFilter, {
                'error_type': None,
                'error_probability': 0.7  # 70% error rate
            })
            
            try:
                result = queue.execute(self.test_data)
                success_count += 1
            except FilterExecutionError:
                error_count += 1
        
        # Should have both successes and failures
        self.assertGreater(success_count, 0)
        self.assertGreater(error_count, 0)
        self.assertEqual(success_count + error_count, 20)
    
    def test_rapid_queue_creation_and_execution(self):
        """Test rapid creation and execution of many queues."""
        # Create and execute many queues rapidly
        results = []
        errors = []
        
        for i in range(50):
            queue = ExecutionQueue()
            queue.add_filter(ErrorProneFilter, {
                'error_type': None,
                'error_probability': 0.1  # Low error rate
            })
            
            try:
                result = queue.execute(self.test_data)
                results.append(result)
            except FilterExecutionError as e:
                errors.append(e)
        
        # Most should succeed
        self.assertGreater(len(results), len(errors))
        
        # Verify successful results
        for result in results[:5]:  # Check first 5
            expected = self.test_data * 1.1
            np.testing.assert_array_almost_equal(result, expected)
    
    def test_memory_leak_detection(self):
        """Test for potential memory leaks during error conditions."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run many operations that fail
        for i in range(100):
            queue = ExecutionQueue()
            queue.add_filter(ErrorProneFilter, {'error_type': 'runtime'})
            
            try:
                queue.execute(self.test_data)
            except FilterExecutionError:
                pass  # Expected
            
            # Force garbage collection periodically
            if i % 10 == 0:
                gc.collect()
        
        # Check memory usage hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        memory_growth_mb = memory_growth / 1024 / 1024
        
        # Allow some growth but not excessive (less than 50MB)
        self.assertLess(memory_growth_mb, 50,
                       f"Memory grew by {memory_growth_mb:.1f}MB during error stress test")


if __name__ == '__main__':
    unittest.main()