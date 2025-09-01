"""
Integration tests for complete filter workflows.

Tests end-to-end functionality including filter pipeline execution,
preset creation and execution workflows, error handling and recovery,
and performance benchmarks.
"""

import unittest
import tempfile
import shutil
import numpy as np
import time
from pathlib import Path
from unittest.mock import Mock, patch

from image_processing_library.core import (
    BaseFilter,
    ExecutionQueue,
    PresetManager,
    DataType,
    ColorFormat,
    FilterExecutionError,
    FilterValidationError
)


class IntegrationTestFilter(BaseFilter):
    """Test filter for integration testing."""
    
    def __init__(self, name="integration_filter", operation="multiply", 
                 value=1.0, delay=0.0, **kwargs):
        super().__init__(
            name=name,
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="integration_test",
            operation=operation,
            value=value,
            delay=delay,
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the filter with configurable operations."""
        self.validate_input(data)
        
        # Simulate processing time if delay is specified
        if self.parameters.get('delay', 0) > 0:
            time.sleep(self.parameters['delay'])
        
        self._update_progress(0.3)
        
        operation = self.parameters.get('operation', 'multiply')
        value = self.parameters.get('value', 1.0)
        
        if operation == 'multiply':
            result = data * value
        elif operation == 'add':
            result = data + value
        elif operation == 'subtract':
            result = data - value
        elif operation == 'divide':
            result = data / value if value != 0 else data
        elif operation == 'clip':
            result = np.clip(data, 0, value)
        elif operation == 'normalize':
            result = (data - data.min()) / (data.max() - data.min()) if data.max() > data.min() else data
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        self._update_progress(0.8)
        
        # Record shapes and track memory
        self._record_shapes(data, result)
        self._track_memory_usage(data, result, used_inplace=(operation in ['clip']))
        
        self._update_progress(1.0)
        return result


class FailingFilter(BaseFilter):
    """Filter that fails for testing error handling."""
    
    def __init__(self, failure_mode="exception", **kwargs):
        super().__init__(
            name="failing_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test",
            failure_mode=failure_mode,
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply filter that fails in various ways."""
        self.validate_input(data)
        
        failure_mode = self.parameters.get('failure_mode', 'exception')
        
        if failure_mode == 'exception':
            raise RuntimeError("Intentional filter failure")
        elif failure_mode == 'validation':
            raise FilterValidationError("Validation failure")
        elif failure_mode == 'memory':
            raise MemoryError("Out of memory")
        elif failure_mode == 'timeout':
            time.sleep(10)  # Simulate timeout
            return data
        else:
            return data


class TestEndToEndWorkflows(unittest.TestCase):
    """Test complete end-to-end filter workflows."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.preset_manager = PresetManager(self.temp_dir)
        
        # Create test data of various sizes
        self.small_data = np.random.rand(10, 10, 3).astype(np.float32)
        self.medium_data = np.random.rand(100, 100, 3).astype(np.float32)
        self.large_data = np.random.rand(500, 500, 3).astype(np.float32)
        
        # Create uint8 data
        self.uint8_data = (self.small_data * 255).astype(np.uint8)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_simple_pipeline_execution(self):
        """Test simple filter pipeline execution."""
        queue = ExecutionQueue()
        
        # Add filters to create a processing pipeline
        queue.add_filter(IntegrationTestFilter, {
            'name': 'enhance',
            'operation': 'multiply',
            'value': 1.2
        })
        
        queue.add_filter(IntegrationTestFilter, {
            'name': 'normalize',
            'operation': 'normalize'
        })
        
        queue.add_filter(IntegrationTestFilter, {
            'name': 'clip',
            'operation': 'clip',
            'value': 1.0
        })
        
        # Execute pipeline
        result = queue.execute(self.medium_data)
        
        # Verify result properties
        self.assertEqual(result.shape, self.medium_data.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1.0))
    
    def test_complex_pipeline_with_intermediate_saves(self):
        """Test complex pipeline with intermediate result saving."""
        queue = ExecutionQueue()
        
        # Create temporary paths for intermediate saves
        step1_path = Path(self.temp_dir) / "step1.npy"
        step2_path = Path(self.temp_dir) / "step2.npy"
        
        # Add filters with intermediate saves
        queue.add_filter(
            IntegrationTestFilter,
            {'operation': 'multiply', 'value': 2.0},
            save_intermediate=True,
            save_path=str(step1_path)
        )
        
        queue.add_filter(
            IntegrationTestFilter,
            {'operation': 'add', 'value': 0.1},
            save_intermediate=True,
            save_path=str(step2_path)
        )
        
        queue.add_filter(
            IntegrationTestFilter,
            {'operation': 'normalize'}
        )
        
        # Execute pipeline
        result = queue.execute(self.small_data)
        
        # Verify intermediate files were created
        self.assertTrue(step1_path.exists())
        self.assertTrue(step2_path.exists())
        
        # Verify intermediate results
        step1_data = np.load(step1_path)
        step2_data = np.load(step2_path)
        
        # Step 1: multiply by 2.0
        expected_step1 = self.small_data * 2.0
        np.testing.assert_array_almost_equal(step1_data, expected_step1)
        
        # Step 2: add 0.1
        expected_step2 = expected_step1 + 0.1
        np.testing.assert_array_almost_equal(step2_data, expected_step2)
        
        # Final result should be normalized
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1.0))
    
    def test_preset_creation_and_execution_workflow(self):
        """Test complete preset creation and execution workflow."""
        # Create a complex processing pipeline
        original_queue = ExecutionQueue()
        
        original_queue.add_filter(IntegrationTestFilter, {
            'name': 'enhance_contrast',
            'operation': 'multiply',
            'value': 1.5
        })
        
        original_queue.add_filter(IntegrationTestFilter, {
            'name': 'brightness_adjust',
            'operation': 'add',
            'value': 0.1
        })
        
        original_queue.add_filter(IntegrationTestFilter, {
            'name': 'final_normalize',
            'operation': 'normalize'
        })
        
        # Save as preset
        preset_path = self.preset_manager.save_preset(
            name="enhancement_pipeline",
            execution_queue=original_queue,
            description="Image enhancement pipeline with contrast and brightness adjustment",
            author="Integration Test Suite"
        )
        
        # Load preset and create new queue
        loaded_queue = self.preset_manager.load_preset("enhancement_pipeline")
        
        # Execute both original and loaded queues
        original_result = original_queue.execute(self.medium_data)
        loaded_result = loaded_queue.execute(self.medium_data)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(original_result, loaded_result)
        
        # Verify preset file structure
        self.assertTrue(Path(preset_path).exists())
        
        # Verify preset can be listed
        presets = self.preset_manager.list_presets()
        preset_names = [p.name for p in presets]
        self.assertIn("enhancement_pipeline", preset_names)
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios."""
        queue = ExecutionQueue()
        
        # Add successful filter
        queue.add_filter(IntegrationTestFilter, {
            'operation': 'multiply',
            'value': 2.0
        })
        
        # Add failing filter
        queue.add_filter(FailingFilter, {
            'failure_mode': 'exception'
        })
        
        # Add another filter that won't execute due to failure
        queue.add_filter(IntegrationTestFilter, {
            'operation': 'normalize'
        })
        
        # Execution should fail with detailed error information
        with self.assertRaises(FilterExecutionError) as context:
            queue.execute(self.small_data)
        
        error_message = str(context.exception)
        self.assertIn("failing_filter", error_message)
        self.assertIn("step 2", error_message)
        self.assertIn("Intentional filter failure", error_message)
    
    def test_different_error_types_handling(self):
        """Test handling of different types of errors."""
        error_scenarios = [
            ('validation', FilterValidationError),
            ('memory', MemoryError),
            ('exception', RuntimeError)
        ]
        
        for failure_mode, expected_base_error in error_scenarios:
            with self.subTest(failure_mode=failure_mode):
                queue = ExecutionQueue()
                queue.add_filter(FailingFilter, {'failure_mode': failure_mode})
                
                with self.assertRaises(FilterExecutionError) as context:
                    queue.execute(self.small_data)
                
                # Check that the original error is preserved in the chain
                self.assertIsInstance(context.exception.__cause__, expected_base_error)
    
    def test_progress_tracking_integration(self):
        """Test progress tracking across complete workflows."""
        progress_log = []
        
        def progress_callback(progress, filter_name):
            progress_log.append((round(progress, 2), filter_name))
        
        queue = ExecutionQueue()
        queue.set_progress_callback(progress_callback)
        
        # Add filters with delays to make progress visible
        queue.add_filter(IntegrationTestFilter, {
            'name': 'step1',
            'operation': 'multiply',
            'value': 1.1,
            'delay': 0.01
        })
        
        queue.add_filter(IntegrationTestFilter, {
            'name': 'step2',
            'operation': 'add',
            'value': 0.05,
            'delay': 0.01
        })
        
        queue.add_filter(IntegrationTestFilter, {
            'name': 'step3',
            'operation': 'normalize',
            'delay': 0.01
        })
        
        # Execute and track progress
        result = queue.execute(self.small_data)
        
        # Verify progress tracking
        self.assertGreater(len(progress_log), 0)
        
        # Check final progress is 1.0
        final_progress = progress_log[-1][0]
        self.assertEqual(final_progress, 1.0)
        
        # Check that all filter names appear
        filter_names = [entry[1] for entry in progress_log]
        self.assertIn("step1", filter_names)
        self.assertIn("step2", filter_names)
        self.assertIn("step3", filter_names)
    
    def test_memory_efficiency_workflow(self):
        """Test memory-efficient processing of large data."""
        # Create a pipeline that processes large data
        queue = ExecutionQueue()
        
        queue.add_filter(IntegrationTestFilter, {
            'operation': 'multiply',
            'value': 1.1
        })
        
        queue.add_filter(IntegrationTestFilter, {
            'operation': 'clip',
            'value': 1.0
        })
        
        # Execute with large data
        result = queue.execute(self.large_data)
        
        # Verify result
        self.assertEqual(result.shape, self.large_data.shape)
        self.assertTrue(np.all(result <= 1.0))
        
        # Result should be reasonable (not all zeros or ones)
        self.assertGreater(np.mean(result), 0.1)
        self.assertLess(np.mean(result), 0.9)
    
    def test_different_data_types_workflow(self):
        """Test workflows with different input data types."""
        test_cases = [
            (self.uint8_data, "uint8"),
            (self.small_data.astype(np.float64), "float64"),
            (self.small_data.astype(np.float16), "float16")
        ]
        
        for test_data, dtype_name in test_cases:
            with self.subTest(dtype=dtype_name):
                queue = ExecutionQueue()
                
                queue.add_filter(IntegrationTestFilter, {
                    'operation': 'multiply',
                    'value': 1.2
                })
                
                queue.add_filter(IntegrationTestFilter, {
                    'operation': 'normalize'
                })
                
                # Execute pipeline
                result = queue.execute(test_data)
                
                # Verify basic properties
                self.assertEqual(result.shape, test_data.shape)
                self.assertTrue(np.all(np.isfinite(result)))
    
    def test_preset_workflow_with_complex_parameters(self):
        """Test preset workflow with complex parameter structures."""
        queue = ExecutionQueue()
        
        # Add filter with complex parameters
        complex_params = {
            'operation': 'multiply',
            'value': 1.5,
            'metadata': {
                'author': 'test',
                'version': '1.0',
                'settings': {
                    'quality': 'high',
                    'optimization': True,
                    'parameters': [1.0, 2.0, 3.0]
                }
            }
        }
        
        queue.add_filter(IntegrationTestFilter, complex_params)
        
        # Save and load preset
        self.preset_manager.save_preset(
            name="complex_preset",
            execution_queue=queue,
            description="Preset with complex parameters"
        )
        
        loaded_queue = self.preset_manager.load_preset("complex_preset")
        
        # Execute both queues
        original_result = queue.execute(self.small_data)
        loaded_result = loaded_queue.execute(self.small_data)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(original_result, loaded_result)
        
        # Verify complex parameters were preserved
        loaded_params = loaded_queue.steps[0].parameters
        self.assertEqual(loaded_params['metadata']['author'], 'test')
        self.assertEqual(loaded_params['metadata']['settings']['quality'], 'high')
        self.assertTrue(loaded_params['metadata']['settings']['optimization'])
        self.assertEqual(loaded_params['metadata']['settings']['parameters'], [1.0, 2.0, 3.0])


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for large file processing."""
    
    def setUp(self):
        """Set up performance test environment."""
        # Create test data of various sizes
        self.small_data = np.random.rand(100, 100, 3).astype(np.float32)
        self.medium_data = np.random.rand(500, 500, 3).astype(np.float32)
        self.large_data = np.random.rand(1000, 1000, 3).astype(np.float32)
        
        # Performance thresholds (in seconds)
        self.small_threshold = 0.1
        self.medium_threshold = 1.0
        self.large_threshold = 5.0
    
    def test_single_filter_performance(self):
        """Test performance of single filter operations."""
        test_cases = [
            (self.small_data, self.small_threshold, "small"),
            (self.medium_data, self.medium_threshold, "medium"),
            (self.large_data, self.large_threshold, "large")
        ]
        
        for test_data, threshold, size_name in test_cases:
            with self.subTest(size=size_name):
                filter_instance = IntegrationTestFilter(
                    operation='multiply',
                    value=1.2
                )
                
                start_time = time.time()
                result = filter_instance.apply(test_data)
                execution_time = time.time() - start_time
                
                # Check performance
                self.assertLess(execution_time, threshold,
                              f"{size_name} data processing took {execution_time:.3f}s, "
                              f"expected < {threshold}s")
                
                # Verify result correctness
                expected = test_data * 1.2
                np.testing.assert_array_almost_equal(result, expected)
    
    def test_pipeline_performance(self):
        """Test performance of filter pipelines."""
        queue = ExecutionQueue()
        
        # Add multiple filters to create a realistic pipeline
        queue.add_filter(IntegrationTestFilter, {'operation': 'multiply', 'value': 1.1})
        queue.add_filter(IntegrationTestFilter, {'operation': 'add', 'value': 0.05})
        queue.add_filter(IntegrationTestFilter, {'operation': 'clip', 'value': 1.0})
        queue.add_filter(IntegrationTestFilter, {'operation': 'normalize'})
        
        # Test with medium-sized data
        start_time = time.time()
        result = queue.execute(self.medium_data)
        execution_time = time.time() - start_time
        
        # Pipeline should complete within reasonable time
        pipeline_threshold = self.medium_threshold * 2  # Allow 2x time for 4 filters
        self.assertLess(execution_time, pipeline_threshold,
                       f"Pipeline processing took {execution_time:.3f}s, "
                       f"expected < {pipeline_threshold}s")
        
        # Verify result properties
        self.assertEqual(result.shape, self.medium_data.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1.0))
    
    def test_memory_usage_efficiency(self):
        """Test memory usage efficiency during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and execute pipeline with large data
        queue = ExecutionQueue()
        queue.add_filter(IntegrationTestFilter, {'operation': 'multiply', 'value': 1.1})
        queue.add_filter(IntegrationTestFilter, {'operation': 'normalize'})
        
        result = queue.execute(self.large_data)
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        # Memory increase should be reasonable (less than 10x data size)
        data_size_mb = self.large_data.nbytes / 1024 / 1024
        max_expected_increase = data_size_mb * 10  # Allow 10x for processing overhead
        
        self.assertLess(memory_increase, max_expected_increase,
                       f"Memory usage increased by {memory_increase:.1f}MB, "
                       f"expected < {max_expected_increase:.1f}MB")
    
    def test_concurrent_processing_simulation(self):
        """Test performance under simulated concurrent processing."""
        # Create multiple queues to simulate concurrent processing
        queues = []
        for i in range(3):
            queue = ExecutionQueue()
            queue.add_filter(IntegrationTestFilter, {
                'name': f'concurrent_filter_{i}',
                'operation': 'multiply',
                'value': 1.0 + i * 0.1
            })
            queues.append(queue)
        
        # Execute all queues and measure total time
        start_time = time.time()
        results = []
        
        for queue in queues:
            result = queue.execute(self.medium_data)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Total time should be reasonable
        expected_max_time = self.medium_threshold * len(queues) * 1.5  # Allow 50% overhead
        self.assertLess(total_time, expected_max_time,
                       f"Concurrent processing took {total_time:.3f}s, "
                       f"expected < {expected_max_time:.3f}s")
        
        # Verify all results are correct
        for i, result in enumerate(results):
            expected = self.medium_data * (1.0 + i * 0.1)
            np.testing.assert_array_almost_equal(result, expected)


if __name__ == '__main__':
    unittest.main()