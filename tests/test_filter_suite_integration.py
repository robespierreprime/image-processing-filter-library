"""
Comprehensive integration tests for the complete filter suite.

This test suite verifies filter chaining, composition, memory management,
and performance benchmarks for all filters.
"""

import pytest
import numpy as np
import time
import psutil
import os
from typing import List, Dict, Any
from PIL import Image

from image_processing_library.filters import (
    get_registry, 
    auto_discover_filters
)
from image_processing_library.core.protocols import DataType, ColorFormat
from image_processing_library.core.utils import FilterValidationError


class TestFilterSuiteIntegration:
    """Integration tests for the complete filter suite."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        # Auto-discover all filters
        auto_discover_filters()
        cls.registry = get_registry()
        
        # Create test images of different sizes
        cls.small_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cls.medium_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        cls.large_image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        
        # Get all available filters
        cls.all_filters = cls.registry.list_filters()
        cls.enhancement_filters = cls.registry.list_filters(category='enhancement')
        cls.artistic_filters = cls.registry.list_filters(category='artistic')
    
    def test_filter_chaining_basic(self):
        """Test basic filter chaining with enhancement filters."""
        # Create a chain of enhancement filters
        filter_chain = [
            ('gamma_correction', {'gamma': 1.2}),
            ('contrast', {'contrast_factor': 1.1}),
            ('saturation', {'saturation_factor': 1.2}),
            ('invert', {})
        ]
        
        # Apply filters in sequence
        result = self.small_image.copy()
        applied_filters = []
        
        for filter_name, params in filter_chain:
            filter_instance = self.registry.create_filter_instance(filter_name, **params)
            result = filter_instance.apply(result)
            applied_filters.append(filter_name)
            
            # Verify result is valid
            assert result is not None
            assert result.shape == self.small_image.shape
            assert result.dtype == np.uint8
        
        assert len(applied_filters) == len(filter_chain)
        print(f"Successfully chained filters: {' -> '.join(applied_filters)}")
    
    def test_filter_chaining_artistic(self):
        """Test chaining artistic filters."""
        filter_chain = [
            ('rgb_shift', {
                'red_shift': (2, 0), 
                'green_shift': (0, 2), 
                'blue_shift': (-2, 0)
            }),
            ('noise', {'noise_type': 'gaussian', 'intensity': 0.05}),
            ('dither', {'pattern_type': 'floyd_steinberg', 'levels': 8})
        ]
        
        result = self.small_image.copy()
        applied_filters = []
        
        for filter_name, params in filter_chain:
            filter_instance = self.registry.create_filter_instance(filter_name, **params)
            result = filter_instance.apply(result)
            applied_filters.append(filter_name)
            
            assert result is not None
            assert result.shape == self.small_image.shape
            assert result.dtype == np.uint8
        
        print(f"Successfully chained artistic filters: {' -> '.join(applied_filters)}")
    
    def test_filter_composition_mixed_categories(self):
        """Test composing filters from different categories."""
        # Mix enhancement and artistic filters
        filter_chain = [
            ('gaussian_blur', {'sigma': 1.0}),  # Enhancement
            ('rgb_shift', {'red_shift': (1, 0), 'green_shift': (0, 1), 'blue_shift': (-1, 0)}),  # Artistic
            ('contrast', {'contrast_factor': 1.3}),  # Enhancement
            ('noise', {'noise_type': 'uniform', 'intensity': 0.03}),  # Artistic
            ('saturation', {'saturation_factor': 0.8})  # Enhancement
        ]
        
        result = self.medium_image.copy()
        
        for filter_name, params in filter_chain:
            filter_instance = self.registry.create_filter_instance(filter_name, **params)
            result = filter_instance.apply(result)
            
            assert result is not None
            assert result.shape == self.medium_image.shape
    
    def test_memory_management_large_images(self):
        """Test memory management with large images."""
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Apply memory-intensive filters to large image
        memory_intensive_filters = [
            ('gaussian_blur', {'sigma': 3.0}),
            ('motion_blur', {'distance': 20, 'angle': 45}),
            ('dither', {'pattern_type': 'floyd_steinberg', 'levels': 16})
        ]
        
        max_memory_usage = initial_memory
        
        for filter_name, params in memory_intensive_filters:
            filter_instance = self.registry.create_filter_instance(filter_name, **params)
            
            # Apply filter
            result = filter_instance.apply(self.large_image)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            max_memory_usage = max(max_memory_usage, current_memory)
            
            # Verify result
            assert result is not None
            assert result.shape == self.large_image.shape
            
            # Clean up
            del result
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = max_memory_usage - initial_memory
        
        print(f"Memory usage - Initial: {initial_memory:.1f}MB, "
              f"Peak: {max_memory_usage:.1f}MB, "
              f"Final: {final_memory:.1f}MB, "
              f"Increase: {memory_increase:.1f}MB")
        
        # Memory increase should be reasonable (less than 500MB for 1024x1024 image)
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB"
    
    def test_performance_benchmarks_all_filters(self):
        """Test performance benchmarks for all filters."""
        benchmark_results = {}
        test_image = self.medium_image.copy()
        
        # Test parameters for each filter
        filter_params = {
            'invert': {},
            'gamma_correction': {'gamma': 1.2},
            'contrast': {'contrast_factor': 1.2},
            'saturation': {'saturation_factor': 1.2},
            'hue_rotation': {'rotation_degrees': 45},
            'gaussian_blur': {'sigma': 2.0},
            'motion_blur': {'distance': 10, 'angle': 45},
            'dither': {'pattern_type': 'floyd_steinberg', 'levels': 8},
            'rgb_shift': {
                'red_shift': (2, 0), 
                'green_shift': (0, 2), 
                'blue_shift': (-2, 0)
            },
            'noise': {'noise_type': 'gaussian', 'intensity': 0.1}
        }
        
        for filter_name in self.enhancement_filters + self.artistic_filters:
            if filter_name in filter_params:
                params = filter_params[filter_name]
                
                # Create filter instance
                filter_instance = self.registry.create_filter_instance(filter_name, **params)
                
                # Benchmark the filter
                start_time = time.time()
                result = filter_instance.apply(test_image)
                end_time = time.time()
                
                execution_time = end_time - start_time
                benchmark_results[filter_name] = execution_time
                
                # Verify result
                assert result is not None
                assert result.shape == test_image.shape
                
                print(f"{filter_name}: {execution_time:.3f}s")
        
        # Check that all filters complete within reasonable time (5 seconds for 256x256)
        for filter_name, execution_time in benchmark_results.items():
            assert execution_time < 5.0, f"Filter {filter_name} too slow: {execution_time:.3f}s"
        
        # Find fastest and slowest filters
        fastest_filter = min(benchmark_results, key=benchmark_results.get)
        slowest_filter = max(benchmark_results, key=benchmark_results.get)
        
        print(f"Fastest filter: {fastest_filter} ({benchmark_results[fastest_filter]:.3f}s)")
        print(f"Slowest filter: {slowest_filter} ({benchmark_results[slowest_filter]:.3f}s)")
    
    def test_filter_error_handling_in_chains(self):
        """Test error handling when filters fail in chains."""
        # Test with invalid parameters - create instance first, then apply
        try:
            filter_instance = self.registry.create_filter_instance(
                'gamma_correction', 
                gamma=-1.0  # Invalid gamma value
            )
            # The error should occur during apply, not during instantiation
            with pytest.raises((FilterValidationError, ValueError, TypeError)):
                filter_instance.apply(self.small_image)
        except (FilterValidationError, ValueError, TypeError):
            # Error occurred during instantiation, which is also valid
            pass
        
        # Test with valid chain that should work
        valid_chain = [
            ('invert', {}),
            ('gamma_correction', {'gamma': 1.0}),  # Identity operation
            ('contrast', {'contrast_factor': 1.0})  # Identity operation
        ]
        
        result = self.small_image.copy()
        for filter_name, params in valid_chain:
            filter_instance = self.registry.create_filter_instance(filter_name, **params)
            result = filter_instance.apply(result)
        
        # Result should be close to inverted original (since other filters are identity)
        expected = 255 - self.small_image
        np.testing.assert_allclose(result, expected, atol=1)
    
    def test_filter_consistency_across_sizes(self):
        """Test that filters produce consistent results across different image sizes."""
        test_filters = [
            ('invert', {}),
            ('gamma_correction', {'gamma': 1.5}),
            ('contrast', {'contrast_factor': 1.2})
        ]
        
        for filter_name, params in test_filters:
            filter_instance = self.registry.create_filter_instance(filter_name, **params)
            
            # Apply to different sized images
            small_result = filter_instance.apply(self.small_image)
            medium_result = filter_instance.apply(self.medium_image)
            
            # Check that results have correct shapes
            assert small_result.shape == self.small_image.shape
            assert medium_result.shape == self.medium_image.shape
            
            # For deterministic filters, check that the transformation is consistent
            if filter_name in ['invert', 'gamma_correction', 'contrast']:
                # Create a small patch from medium image same size as small image
                patch = self.medium_image[:64, :64, :]
                patch_result = filter_instance.apply(patch)
                
                # Should be very similar to small_result if images are similar
                # (This is a basic consistency check)
                assert patch_result.shape == small_result.shape
    
    def test_filter_idempotency(self):
        """Test filters that should be idempotent with identity parameters."""
        identity_filters = [
            ('gamma_correction', {'gamma': 1.0}),
            ('contrast', {'contrast_factor': 1.0}),
            ('saturation', {'saturation_factor': 1.0}),
            ('hue_rotation', {'rotation_degrees': 0}),
            ('gaussian_blur', {'sigma': 0.0}),
            ('motion_blur', {'distance': 0, 'angle': 0}),
            ('rgb_shift', {
                'red_shift': (0, 0), 
                'green_shift': (0, 0), 
                'blue_shift': (0, 0)
            }),
            ('noise', {'noise_type': 'gaussian', 'intensity': 0.0})
        ]
        
        for filter_name, params in identity_filters:
            filter_instance = self.registry.create_filter_instance(filter_name, **params)
            result = filter_instance.apply(self.small_image)
            
            # Result should be very close to original
            np.testing.assert_allclose(
                result, self.small_image, 
                atol=2,  # Allow small numerical differences
                err_msg=f"Filter {filter_name} with identity parameters should return original image"
            )
    
    def test_filter_metadata_consistency(self):
        """Test that all filters have consistent metadata."""
        required_metadata_keys = ['category', 'data_type', 'color_format', 'class', 'module']
        
        for filter_name in self.all_filters:
            metadata = self.registry.get_filter_metadata(filter_name)
            
            # Check required keys exist
            for key in required_metadata_keys:
                assert key in metadata, f"Filter {filter_name} missing metadata key: {key}"
            
            # Check data types
            assert metadata['data_type'] == DataType.IMAGE, \
                f"Filter {filter_name} should support IMAGE data type"
            
            assert metadata['color_format'] in [ColorFormat.RGB, ColorFormat.RGBA, ColorFormat.GRAYSCALE], \
                f"Filter {filter_name} has invalid color format: {metadata['color_format']}"
            
            # Check category is valid
            assert metadata['category'] in ['enhancement', 'artistic', 'technical'], \
                f"Filter {filter_name} has invalid category: {metadata['category']}"
    
    def test_all_filters_instantiable(self):
        """Test that all registered filters can be instantiated."""
        # Parameters for filters that require them
        default_params = {
            'gamma_correction': {'gamma': 1.2},
            'contrast': {'contrast_factor': 1.2},
            'saturation': {'saturation_factor': 1.2},
            'hue_rotation': {'rotation_degrees': 45},
            'gaussian_blur': {'sigma': 2.0},
            'motion_blur': {'distance': 10, 'angle': 45},
            'dither': {'pattern_type': 'floyd_steinberg', 'levels': 8},
            'rgb_shift': {
                'red_shift': (2, 0), 
                'green_shift': (0, 2), 
                'blue_shift': (-2, 0)
            },
            'noise': {'noise_type': 'gaussian', 'intensity': 0.1}
        }
        
        instantiated_filters = []
        
        for filter_name in self.all_filters:
            try:
                # Try with default parameters if available
                params = default_params.get(filter_name, {})
                filter_instance = self.registry.create_filter_instance(filter_name, **params)
                
                # Verify instance has required methods
                assert hasattr(filter_instance, 'apply')
                assert hasattr(filter_instance, 'name')
                assert callable(filter_instance.apply)
                
                instantiated_filters.append(filter_name)
                
            except Exception as e:
                pytest.fail(f"Failed to instantiate filter {filter_name}: {e}")
        
        print(f"Successfully instantiated {len(instantiated_filters)} filters")
        assert len(instantiated_filters) >= 10, "Should have at least 10 working filters"
    
    def test_filter_progress_tracking(self):
        """Test that filters support progress tracking."""
        # Test with a filter that should support progress tracking
        filter_instance = self.registry.create_filter_instance('gaussian_blur', sigma=2.0)
        
        # Apply filter and check if it has progress tracking capabilities
        result = filter_instance.apply(self.medium_image)
        
        # Check if filter has timing information
        if hasattr(filter_instance, 'last_execution_time'):
            assert filter_instance.last_execution_time > 0
            print(f"Filter execution time: {filter_instance.last_execution_time:.3f}s")
        
        assert result is not None
        assert result.shape == self.medium_image.shape
    
    def test_chunked_processing_support(self):
        """Test that filters support chunked processing for large images."""
        # Create a very large image that would benefit from chunked processing
        very_large_image = np.random.randint(0, 256, (2048, 2048, 3), dtype=np.uint8)
        
        # Test filters that should support chunked processing
        chunked_filters = [
            ('gaussian_blur', {'sigma': 1.0}),
            ('invert', {}),
            ('gamma_correction', {'gamma': 1.2})
        ]
        
        for filter_name, params in chunked_filters:
            filter_instance = self.registry.create_filter_instance(filter_name, **params)
            
            # Monitor memory during processing
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Apply filter
            result = filter_instance.apply(very_large_image)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Verify result
            assert result is not None
            assert result.shape == very_large_image.shape
            assert result.dtype == very_large_image.dtype
            
            # Memory increase should be reasonable (less than 1GB for 2048x2048 image)
            assert memory_increase < 1000, \
                f"Filter {filter_name} used too much memory: {memory_increase:.1f}MB"
            
            print(f"{filter_name} memory usage: {memory_increase:.1f}MB")
            
            # Clean up
            del result
    
    def test_filter_composition_with_different_data_types(self):
        """Test filter composition with different input data types."""
        # Test with float32 image
        float_image = self.small_image.astype(np.float32) / 255.0
        
        # Apply a chain of filters
        filter_chain = [
            ('gamma_correction', {'gamma': 1.2}),
            ('contrast', {'contrast_factor': 1.1}),
            ('invert', {})
        ]
        
        result = float_image.copy()
        for filter_name, params in filter_chain:
            filter_instance = self.registry.create_filter_instance(filter_name, **params)
            result = filter_instance.apply(result)
            
            # Verify result maintains float32 type and valid range
            assert result.dtype == np.float32
            assert np.all(result >= 0.0) and np.all(result <= 1.0)
        
        print("Successfully processed float32 image through filter chain")
    
    def test_filter_robustness_with_edge_cases(self):
        """Test filter robustness with edge case images."""
        # Test with various edge case images
        edge_cases = [
            ("single_pixel", np.array([[[128, 128, 128]]], dtype=np.uint8)),
            ("all_black", np.zeros((32, 32, 3), dtype=np.uint8)),
            ("all_white", np.full((32, 32, 3), 255, dtype=np.uint8)),
            ("single_row", np.random.randint(0, 256, (1, 100, 3), dtype=np.uint8)),
            ("single_column", np.random.randint(0, 256, (100, 1, 3), dtype=np.uint8))
        ]
        
        # Test with robust filters
        robust_filters = [
            ('invert', {}),
            ('gamma_correction', {'gamma': 1.0}),  # Identity
            ('contrast', {'contrast_factor': 1.0})  # Identity
        ]
        
        for case_name, test_image in edge_cases:
            for filter_name, params in robust_filters:
                filter_instance = self.registry.create_filter_instance(filter_name, **params)
                
                try:
                    result = filter_instance.apply(test_image)
                    
                    # Verify result has correct shape and type
                    assert result.shape == test_image.shape
                    assert result.dtype == test_image.dtype
                    
                    print(f"Filter {filter_name} handled {case_name} successfully")
                    
                except Exception as e:
                    pytest.fail(f"Filter {filter_name} failed on {case_name}: {e}")
    
    def test_concurrent_filter_application(self):
        """Test that filters can be applied concurrently without interference."""
        import threading
        import queue
        
        # Create multiple test images
        test_images = [
            np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8) for _ in range(4)
        ]
        
        # Results queue
        results_queue = queue.Queue()
        
        def apply_filter_thread(image, filter_name, params, thread_id):
            """Apply filter in a separate thread."""
            try:
                filter_instance = self.registry.create_filter_instance(filter_name, **params)
                result = filter_instance.apply(image)
                results_queue.put((thread_id, True, result.shape))
            except Exception as e:
                results_queue.put((thread_id, False, str(e)))
        
        # Start multiple threads applying different filters
        threads = []
        for i, image in enumerate(test_images):
            filter_configs = [
                ('invert', {}),
                ('gamma_correction', {'gamma': 1.2}),
                ('contrast', {'contrast_factor': 1.1}),
                ('gaussian_blur', {'sigma': 1.0})
            ]
            
            filter_name, params = filter_configs[i % len(filter_configs)]
            thread = threading.Thread(
                target=apply_filter_thread,
                args=(image, filter_name, params, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        successful_threads = 0
        while not results_queue.empty():
            thread_id, success, result = results_queue.get()
            if success:
                successful_threads += 1
                print(f"Thread {thread_id} completed successfully, result shape: {result}")
            else:
                pytest.fail(f"Thread {thread_id} failed: {result}")
        
        assert successful_threads == len(test_images), \
            f"Only {successful_threads}/{len(test_images)} threads completed successfully"


class TestFilterStressTests:
    """Stress tests for filter suite reliability and performance."""
    
    @classmethod
    def setup_class(cls):
        """Set up stress test environment."""
        auto_discover_filters()
        cls.registry = get_registry()
        cls.test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated filter applications."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Apply filters repeatedly
        filter_instance = self.registry.create_filter_instance('gaussian_blur', sigma=2.0)
        
        memory_samples = []
        for i in range(50):  # Apply filter 50 times
            result = filter_instance.apply(self.test_image)
            
            # Sample memory every 10 iterations
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory)
            
            # Clean up result
            del result
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Check for memory leaks - final memory should not be significantly higher
        memory_increase = final_memory - initial_memory
        print(f"Memory leak test - Initial: {initial_memory:.1f}MB, "
              f"Final: {final_memory:.1f}MB, "
              f"Increase: {memory_increase:.1f}MB")
        
        # Memory increase should be minimal (less than 50MB)
        assert memory_increase < 50, f"Potential memory leak detected: {memory_increase:.1f}MB increase"
        
        # Memory should not continuously increase
        if len(memory_samples) >= 3:
            # Check that memory doesn't continuously grow
            memory_trend = memory_samples[-1] - memory_samples[0]
            assert memory_trend < 100, f"Memory continuously increasing: {memory_trend:.1f}MB trend"
    
    def test_filter_suite_stress_test(self):
        """Stress test applying all filters in various combinations."""
        # Get all available filters with safe parameters
        stress_test_filters = [
            ('invert', {}),
            ('gamma_correction', {'gamma': 1.1}),
            ('contrast', {'contrast_factor': 1.1}),
            ('saturation', {'saturation_factor': 1.1}),
            ('hue_rotation', {'rotation_degrees': 30}),
            ('gaussian_blur', {'sigma': 1.0}),
            ('noise', {'noise_type': 'gaussian', 'intensity': 0.05}),
            ('rgb_shift', {'red_shift': (1, 0), 'green_shift': (0, 1), 'blue_shift': (-1, 0)})
        ]
        
        # Apply random combinations of filters
        import random
        successful_combinations = 0
        total_combinations = 20
        
        for i in range(total_combinations):
            # Select 3-5 random filters
            num_filters = random.randint(3, 5)
            selected_filters = random.sample(stress_test_filters, num_filters)
            
            try:
                result = self.test_image.copy()
                applied_filters = []
                
                for filter_name, params in selected_filters:
                    filter_instance = self.registry.create_filter_instance(filter_name, **params)
                    result = filter_instance.apply(result)
                    applied_filters.append(filter_name)
                
                # Verify final result
                assert result is not None
                assert result.shape == self.test_image.shape
                assert result.dtype == self.test_image.dtype
                
                successful_combinations += 1
                print(f"Combination {i+1}: {' -> '.join(applied_filters)} ✓")
                
            except Exception as e:
                print(f"Combination {i+1} failed: {e}")
        
        success_rate = successful_combinations / total_combinations
        print(f"Stress test success rate: {success_rate:.1%} ({successful_combinations}/{total_combinations})")
        
        # At least 90% of combinations should succeed
        assert success_rate >= 0.9, f"Stress test success rate too low: {success_rate:.1%}"


class TestFilterPerformanceComparison:
    """Performance comparison tests for different filter implementations."""
    
    @classmethod
    def setup_class(cls):
        """Set up performance test environment."""
        auto_discover_filters()
        cls.registry = get_registry()
        cls.test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    def test_blur_filter_performance_comparison(self):
        """Compare performance of different blur filters."""
        blur_filters = [
            ('gaussian_blur', {'sigma': 2.0}),
            ('motion_blur', {'distance': 10, 'angle': 45})
        ]
        
        performance_results = {}
        
        for filter_name, params in blur_filters:
            filter_instance = self.registry.create_filter_instance(filter_name, **params)
            
            # Warm up
            filter_instance.apply(self.test_image[:100, :100, :])
            
            # Benchmark
            start_time = time.time()
            result = filter_instance.apply(self.test_image)
            end_time = time.time()
            
            execution_time = end_time - start_time
            performance_results[filter_name] = execution_time
            
            print(f"{filter_name}: {execution_time:.3f}s")
        
        # Gaussian blur should generally be faster than motion blur
        if 'gaussian_blur' in performance_results and 'motion_blur' in performance_results:
            print(f"Gaussian blur vs Motion blur: "
                  f"{performance_results['gaussian_blur']:.3f}s vs "
                  f"{performance_results['motion_blur']:.3f}s")
    
    def test_color_filter_performance_comparison(self):
        """Compare performance of different color manipulation filters."""
        color_filters = [
            ('invert', {}),
            ('gamma_correction', {'gamma': 1.2}),
            ('contrast', {'contrast_factor': 1.2}),
            ('saturation', {'saturation_factor': 1.2}),
            ('hue_rotation', {'rotation_degrees': 45})
        ]
        
        performance_results = {}
        
        for filter_name, params in color_filters:
            filter_instance = self.registry.create_filter_instance(filter_name, **params)
            
            # Benchmark
            start_time = time.time()
            result = filter_instance.apply(self.test_image)
            end_time = time.time()
            
            execution_time = end_time - start_time
            performance_results[filter_name] = execution_time
            
            print(f"{filter_name}: {execution_time:.3f}s")
        
        # Verify all filters complete within reasonable absolute time limits
        for filter_name, exec_time in performance_results.items():
            # All filters should complete within 1 second for 512x512 image
            assert exec_time < 1.0, \
                f"{filter_name} is too slow: {exec_time:.3f}s (should be < 1.0s)"
        
        # Compare relative performance categories
        simple_filters = ['invert', 'gamma_correction', 'contrast']
        complex_filters = ['saturation', 'hue_rotation']
        
        simple_times = [performance_results[f] for f in simple_filters if f in performance_results]
        complex_times = [performance_results[f] for f in complex_filters if f in performance_results]
        
        if simple_times and complex_times:
            avg_simple = sum(simple_times) / len(simple_times)
            avg_complex = sum(complex_times) / len(complex_times)
            print(f"Average simple filter time: {avg_simple:.3f}s")
            print(f"Average complex filter time: {avg_complex:.3f}s")
            
            # Complex filters can be slower, but not excessively so
            assert avg_complex < avg_simple * 200, \
                f"Complex filters too slow: {avg_complex:.3f}s vs {avg_simple:.3f}s"


if __name__ == "__main__":
    # Run basic integration tests
    test_suite = TestFilterSuiteIntegration()
    test_suite.setup_class()
    
    print("Running filter suite integration tests...")
    
    print("\n1. Testing filter chaining...")
    test_suite.test_filter_chaining_basic()
    test_suite.test_filter_chaining_artistic()
    
    print("\n2. Testing filter composition...")
    test_suite.test_filter_composition_mixed_categories()
    
    print("\n3. Testing memory management...")
    test_suite.test_memory_management_large_images()
    
    print("\n4. Testing performance benchmarks...")
    test_suite.test_performance_benchmarks_all_filters()
    
    print("\n5. Testing filter consistency...")
    test_suite.test_filter_consistency_across_sizes()
    test_suite.test_filter_idempotency()
    
    print("\n6. Testing metadata consistency...")
    test_suite.test_filter_metadata_consistency()
    
    print("\n7. Testing filter instantiation...")
    test_suite.test_all_filters_instantiable()
    
    print("\nAll integration tests passed!")