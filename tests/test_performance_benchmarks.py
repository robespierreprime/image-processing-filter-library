"""
Performance benchmark tests for large file processing.

Tests system performance under various load conditions, memory usage,
processing speed, and scalability characteristics.
"""

import unittest
import time
import numpy as np
import tempfile
import shutil
from pathlib import Path
import gc
from unittest.mock import patch

from image_processing_library.core import (
    BaseFilter,
    ExecutionQueue,
    PresetManager,
    DataType,
    ColorFormat
)


class BenchmarkFilter(BaseFilter):
    """Filter designed for performance benchmarking."""
    
    def __init__(self, operation="enhance", intensity=1.0, complexity="low", **kwargs):
        super().__init__(
            name="benchmark_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="benchmark",
            operation=operation,
            intensity=intensity,
            complexity=complexity,
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply filter with configurable computational complexity."""
        self.validate_input(data)
        
        operation = self.parameters.get('operation', 'enhance')
        intensity = self.parameters.get('intensity', 1.0)
        complexity = self.parameters.get('complexity', 'low')
        
        self._update_progress(0.1)
        
        # Different complexity levels for benchmarking
        if complexity == 'low':
            # Simple element-wise operations
            result = data * intensity
            
        elif complexity == 'medium':
            # More complex operations with some loops
            result = data.copy()
            for i in range(3):  # Process each channel
                result[:, :, i] = result[:, :, i] * intensity + 0.1 * np.sin(result[:, :, i] * 10)
            
        elif complexity == 'high':
            # Complex operations with convolutions and multiple passes
            result = data.copy()
            
            # Simulate complex processing
            for iteration in range(3):
                self._update_progress(0.1 + 0.7 * iteration / 3)
                
                # Gaussian-like blur simulation
                kernel_size = 5
                for i in range(result.shape[2]):
                    channel = result[:, :, i]
                    
                    # Simple convolution simulation
                    padded = np.pad(channel, kernel_size//2, mode='edge')
                    for y in range(channel.shape[0]):
                        for x in range(channel.shape[1]):
                            window = padded[y:y+kernel_size, x:x+kernel_size]
                            channel[y, x] = np.mean(window) * intensity
                    
                    result[:, :, i] = channel
        
        elif complexity == 'extreme':
            # Very complex operations for stress testing
            result = data.copy()
            
            # Multiple complex operations
            for iteration in range(5):
                self._update_progress(0.1 + 0.8 * iteration / 5)
                
                # Complex mathematical operations
                result = result * intensity
                result = np.sin(result * np.pi) * 0.5 + 0.5
                
                # Simulate expensive operations
                for i in range(result.shape[2]):
                    channel = result[:, :, i]
                    # Matrix operations
                    mean_val = np.mean(channel)
                    std_val = np.std(channel)
                    result[:, :, i] = (channel - mean_val) / (std_val + 1e-8) * 0.2 + mean_val
        
        else:
            result = data * intensity
        
        # Record performance metadata
        self._record_shapes(data, result)
        self._track_memory_usage(data, result, used_inplace=False)
        
        self._update_progress(1.0)
        return result


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""
    
    def setUp(self):
        """Set up benchmark environment."""
        # Create test data of various sizes
        self.tiny_data = np.random.rand(32, 32, 3).astype(np.float32)
        self.small_data = np.random.rand(128, 128, 3).astype(np.float32)
        self.medium_data = np.random.rand(512, 512, 3).astype(np.float32)
        self.large_data = np.random.rand(1024, 1024, 3).astype(np.float32)
        self.huge_data = np.random.rand(2048, 2048, 3).astype(np.float32)
        
        # Performance thresholds (in seconds)
        self.tiny_threshold = 0.01
        self.small_threshold = 0.05
        self.medium_threshold = 0.5
        self.large_threshold = 2.0
        self.huge_threshold = 10.0
        
        # Memory thresholds (in MB)
        self.memory_threshold_multiplier = 5.0  # Allow 5x data size for processing
    
    def _measure_performance(self, func, *args, **kwargs):
        """Measure execution time and memory usage of a function."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - baseline_memory
        
        return result, execution_time, memory_usage
    
    def test_single_filter_performance_scaling(self):
        """Test performance scaling of single filter with different data sizes."""
        test_cases = [
            (self.tiny_data, self.tiny_threshold, "tiny"),
            (self.small_data, self.small_threshold, "small"),
            (self.medium_data, self.medium_threshold, "medium"),
            (self.large_data, self.large_threshold, "large"),
        ]
        
        for test_data, threshold, size_name in test_cases:
            with self.subTest(size=size_name):
                filter_instance = BenchmarkFilter(
                    operation='enhance',
                    intensity=1.2,
                    complexity='low'
                )
                
                result, exec_time, memory_usage = self._measure_performance(
                    filter_instance.apply, test_data
                )
                
                # Check performance
                self.assertLess(exec_time, threshold,
                              f"{size_name} data processing took {exec_time:.3f}s, "
                              f"expected < {threshold}s")
                
                # Check memory usage is reasonable
                data_size_mb = test_data.nbytes / 1024 / 1024
                max_memory = data_size_mb * self.memory_threshold_multiplier
                self.assertLess(memory_usage, max_memory,
                              f"{size_name} data used {memory_usage:.1f}MB, "
                              f"expected < {max_memory:.1f}MB")
                
                # Verify result correctness
                expected = test_data * 1.2
                np.testing.assert_array_almost_equal(result, expected)
    
    def test_complexity_scaling_performance(self):
        """Test performance scaling with different computational complexities."""
        complexities = ['low', 'medium', 'high']
        base_threshold = 0.1
        
        for i, complexity in enumerate(complexities):
            with self.subTest(complexity=complexity):
                filter_instance = BenchmarkFilter(
                    operation='enhance',
                    intensity=1.1,
                    complexity=complexity
                )
                
                result, exec_time, memory_usage = self._measure_performance(
                    filter_instance.apply, self.small_data
                )
                
                # Higher complexity should take more time (with some tolerance)
                expected_threshold = base_threshold * (2 ** i)  # Exponential scaling
                self.assertLess(exec_time, expected_threshold,
                              f"{complexity} complexity took {exec_time:.3f}s, "
                              f"expected < {expected_threshold:.3f}s")
                
                # Verify result is valid
                self.assertEqual(result.shape, self.small_data.shape)
                self.assertTrue(np.all(np.isfinite(result)))
    
    def test_pipeline_performance_scaling(self):
        """Test performance scaling of filter pipelines."""
        pipeline_sizes = [1, 2, 4, 8]
        base_threshold = 0.1
        
        for num_filters in pipeline_sizes:
            with self.subTest(pipeline_size=num_filters):
                queue = ExecutionQueue()
                
                # Add multiple filters to pipeline
                for i in range(num_filters):
                    queue.add_filter(BenchmarkFilter, {
                        'operation': 'enhance',
                        'intensity': 1.0 + i * 0.01,  # Slight variation
                        'complexity': 'low'
                    })
                
                result, exec_time, memory_usage = self._measure_performance(
                    queue.execute, self.medium_data
                )
                
                # Pipeline time should scale roughly linearly
                expected_threshold = base_threshold * num_filters * 2  # Allow 2x overhead
                self.assertLess(exec_time, expected_threshold,
                              f"Pipeline with {num_filters} filters took {exec_time:.3f}s, "
                              f"expected < {expected_threshold:.3f}s")
                
                # Verify result
                self.assertEqual(result.shape, self.medium_data.shape)
    
    def test_memory_efficiency_large_data(self):
        """Test memory efficiency with large data processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large data
        filter_instance = BenchmarkFilter(
            operation='enhance',
            intensity=1.1,
            complexity='medium'
        )
        
        result = filter_instance.apply(self.large_data)
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        # Memory increase should be reasonable
        data_size_mb = self.large_data.nbytes / 1024 / 1024
        max_expected_increase = data_size_mb * 3  # Allow 3x for processing
        
        self.assertLess(memory_increase, max_expected_increase,
                       f"Memory increased by {memory_increase:.1f}MB, "
                       f"expected < {max_expected_increase:.1f}MB")
        
        # Clean up and verify memory is released
        del result
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_retained = final_memory - baseline_memory
        
        # Should release most memory (allow 20% retention)
        max_retained = memory_increase * 0.2
        self.assertLess(memory_retained, max_retained,
                       f"Retained {memory_retained:.1f}MB after cleanup, "
                       f"expected < {max_retained:.1f}MB")
    
    def test_concurrent_processing_performance(self):
        """Test performance under concurrent processing simulation."""
        import threading
        import queue as thread_queue
        
        num_threads = 4
        results_queue = thread_queue.Queue()
        
        def process_data(thread_id):
            """Process data in a separate thread."""
            try:
                filter_instance = BenchmarkFilter(
                    operation='enhance',
                    intensity=1.0 + thread_id * 0.1,
                    complexity='low'
                )
                
                start_time = time.time()
                result = filter_instance.apply(self.medium_data)
                exec_time = time.time() - start_time
                
                results_queue.put((thread_id, result, exec_time, None))
                
            except Exception as e:
                results_queue.put((thread_id, None, None, e))
        
        # Start all threads
        threads = []
        overall_start = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=process_data, args=(i,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        overall_time = time.time() - overall_start
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify all threads completed successfully
        self.assertEqual(len(results), num_threads)
        
        for thread_id, result, exec_time, error in results:
            self.assertIsNone(error, f"Thread {thread_id} failed: {error}")
            self.assertIsNotNone(result)
            self.assertIsNotNone(exec_time)
            
            # Individual thread performance should be reasonable
            self.assertLess(exec_time, 2.0, f"Thread {thread_id} took {exec_time:.3f}s")
        
        # Overall time should be less than sequential processing
        sequential_estimate = self.medium_threshold * num_threads
        self.assertLess(overall_time, sequential_estimate,
                       f"Concurrent processing took {overall_time:.3f}s, "
                       f"sequential estimate: {sequential_estimate:.3f}s")
    
    def test_preset_performance_overhead(self):
        """Test performance overhead of preset save/load operations."""
        temp_dir = tempfile.mkdtemp()
        try:
            preset_manager = PresetManager(temp_dir)
            
            # Create complex pipeline
            queue = ExecutionQueue()
            for i in range(5):
                queue.add_filter(BenchmarkFilter, {
                    'operation': 'enhance',
                    'intensity': 1.0 + i * 0.05,
                    'complexity': 'low'
                })
            
            # Measure preset save performance
            save_start = time.time()
            preset_path = preset_manager.save_preset(
                "performance_test",
                queue,
                "Performance test preset"
            )
            save_time = time.time() - save_start
            
            # Preset save should be fast
            self.assertLess(save_time, 0.1, f"Preset save took {save_time:.3f}s")
            
            # Measure preset load performance
            load_start = time.time()
            loaded_queue = preset_manager.load_preset("performance_test")
            load_time = time.time() - load_start
            
            # Preset load should be fast
            self.assertLess(load_time, 0.1, f"Preset load took {load_time:.3f}s")
            
            # Measure execution performance difference
            original_result, original_time, _ = self._measure_performance(
                queue.execute, self.medium_data
            )
            
            loaded_result, loaded_time, _ = self._measure_performance(
                loaded_queue.execute, self.medium_data
            )
            
            # Execution times should be similar (within 10%)
            time_difference = abs(loaded_time - original_time)
            max_difference = max(original_time, loaded_time) * 0.1
            self.assertLess(time_difference, max_difference,
                           f"Execution time difference: {time_difference:.3f}s")
            
            # Results should be identical
            np.testing.assert_array_almost_equal(original_result, loaded_result)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many operations
        num_iterations = 50
        
        for i in range(num_iterations):
            filter_instance = BenchmarkFilter(
                operation='enhance',
                intensity=1.1,
                complexity='low'
            )
            
            result = filter_instance.apply(self.small_data)
            
            # Explicitly delete result to help garbage collection
            del result
            del filter_instance
            
            # Force garbage collection every 10 iterations
            if i % 10 == 0:
                gc.collect()
        
        # Final garbage collection
        gc.collect()
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - baseline_memory
        
        # Memory growth should be minimal (less than 10MB)
        self.assertLess(memory_growth, 10,
                       f"Memory grew by {memory_growth:.1f}MB after {num_iterations} iterations")
    
    def test_large_pipeline_performance(self):
        """Test performance of very large processing pipelines."""
        # Create a large pipeline
        queue = ExecutionQueue()
        
        num_filters = 20
        for i in range(num_filters):
            queue.add_filter(BenchmarkFilter, {
                'operation': 'enhance',
                'intensity': 1.0 + i * 0.001,  # Very small increments
                'complexity': 'low'
            })
        
        # Test with medium data
        result, exec_time, memory_usage = self._measure_performance(
            queue.execute, self.medium_data
        )
        
        # Large pipeline should still complete in reasonable time
        max_time = self.medium_threshold * num_filters * 0.5  # Allow 0.5x per filter
        self.assertLess(exec_time, max_time,
                       f"Large pipeline took {exec_time:.3f}s, expected < {max_time:.3f}s")
        
        # Memory usage should be reasonable
        data_size_mb = self.medium_data.nbytes / 1024 / 1024
        max_memory = data_size_mb * 10  # Allow 10x for large pipeline
        self.assertLess(memory_usage, max_memory,
                       f"Large pipeline used {memory_usage:.1f}MB, expected < {max_memory:.1f}MB")
        
        # Verify result
        self.assertEqual(result.shape, self.medium_data.shape)
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_extreme_data_size_handling(self):
        """Test handling of extremely large data (if system can handle it)."""
        try:
            # Only run if we have enough memory
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            if available_memory_gb < 4:  # Need at least 4GB available
                self.skipTest("Insufficient memory for extreme data size test")
            
            # Test with huge data
            filter_instance = BenchmarkFilter(
                operation='enhance',
                intensity=1.05,
                complexity='low'
            )
            
            result, exec_time, memory_usage = self._measure_performance(
                filter_instance.apply, self.huge_data
            )
            
            # Should complete within reasonable time
            self.assertLess(exec_time, self.huge_threshold,
                           f"Huge data processing took {exec_time:.3f}s, "
                           f"expected < {self.huge_threshold}s")
            
            # Verify result
            self.assertEqual(result.shape, self.huge_data.shape)
            
        except MemoryError:
            self.skipTest("System cannot handle extreme data size")


class TestScalabilityBenchmarks(unittest.TestCase):
    """Test scalability characteristics of the system."""
    
    def setUp(self):
        """Set up scalability test environment."""
        self.base_data = np.random.rand(100, 100, 3).astype(np.float32)
    
    def test_data_size_scalability(self):
        """Test how performance scales with data size."""
        sizes = [64, 128, 256, 512]
        times = []
        
        for size in sizes:
            test_data = np.random.rand(size, size, 3).astype(np.float32)
            
            filter_instance = BenchmarkFilter(
                operation='enhance',
                intensity=1.1,
                complexity='low'
            )
            
            start_time = time.time()
            result = filter_instance.apply(test_data)
            exec_time = time.time() - start_time
            
            times.append(exec_time)
            
            # Verify result
            self.assertEqual(result.shape, test_data.shape)
        
        # Check that scaling is reasonable (not exponential)
        # Time should roughly scale with data size (quadratic for 2D data)
        for i in range(1, len(times)):
            size_ratio = (sizes[i] / sizes[i-1]) ** 2  # Quadratic scaling
            time_ratio = times[i] / times[i-1]
            
            # Allow up to 3x the expected scaling factor
            max_ratio = size_ratio * 3
            self.assertLess(time_ratio, max_ratio,
                           f"Time scaling from {sizes[i-1]} to {sizes[i]}: "
                           f"{time_ratio:.2f}x, expected < {max_ratio:.2f}x")
    
    def test_pipeline_length_scalability(self):
        """Test how performance scales with pipeline length."""
        pipeline_lengths = [1, 2, 4, 8, 16]
        times = []
        
        for length in pipeline_lengths:
            queue = ExecutionQueue()
            
            for i in range(length):
                queue.add_filter(BenchmarkFilter, {
                    'operation': 'enhance',
                    'intensity': 1.01,  # Very small change
                    'complexity': 'low'
                })
            
            start_time = time.time()
            result = queue.execute(self.base_data)
            exec_time = time.time() - start_time
            
            times.append(exec_time)
            
            # Verify result
            self.assertEqual(result.shape, self.base_data.shape)
        
        # Check that scaling is roughly linear
        for i in range(1, len(times)):
            length_ratio = pipeline_lengths[i] / pipeline_lengths[i-1]
            time_ratio = times[i] / times[i-1]
            
            # Allow up to 2x the expected linear scaling
            max_ratio = length_ratio * 2
            self.assertLess(time_ratio, max_ratio,
                           f"Time scaling from {pipeline_lengths[i-1]} to {pipeline_lengths[i]} filters: "
                           f"{time_ratio:.2f}x, expected < {max_ratio:.2f}x")
    
    def test_complexity_scalability(self):
        """Test how performance scales with computational complexity."""
        complexities = ['low', 'medium', 'high']
        times = []
        
        for complexity in complexities:
            filter_instance = BenchmarkFilter(
                operation='enhance',
                intensity=1.1,
                complexity=complexity
            )
            
            start_time = time.time()
            result = filter_instance.apply(self.base_data)
            exec_time = time.time() - start_time
            
            times.append(exec_time)
            
            # Verify result
            self.assertEqual(result.shape, self.base_data.shape)
        
        # Each complexity level should take more time than the previous
        for i in range(1, len(times)):
            self.assertGreater(times[i], times[i-1] * 0.8,  # Allow some variance
                              f"{complexities[i]} should take more time than {complexities[i-1]}")


if __name__ == '__main__':
    unittest.main()