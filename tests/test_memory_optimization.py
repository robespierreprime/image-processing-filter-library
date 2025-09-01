"""
Performance tests for memory optimization features.

This module contains tests for memory management utilities,
in-place vs copy decision logic, and memory cleanup functionality.
"""

import unittest
import numpy as np
import psutil
import gc
from unittest.mock import patch, MagicMock
from image_processing_library.core.utils import MemoryManager, MemoryError
from image_processing_library.core.base_filter import BaseFilter, FilterMetadata
from image_processing_library.core.protocols import DataType, ColorFormat


class TestMemoryManager(unittest.TestCase):
    """Test cases for MemoryManager utility class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.small_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.large_array = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        self.huge_array = np.random.randint(0, 255, (5000, 5000, 3), dtype=np.uint8)
    
    def test_get_array_memory_usage(self):
        """Test memory usage calculation for arrays."""
        # Test small array
        small_memory = MemoryManager.get_array_memory_usage(self.small_array)
        expected_small = self.small_array.nbytes / (1024 * 1024)
        self.assertAlmostEqual(small_memory, expected_small, places=2)
        
        # Test large array
        large_memory = MemoryManager.get_array_memory_usage(self.large_array)
        expected_large = self.large_array.nbytes / (1024 * 1024)
        self.assertAlmostEqual(large_memory, expected_large, places=2)
        
        # Verify large array is actually larger
        self.assertGreater(large_memory, small_memory)
    
    def test_get_available_memory(self):
        """Test available memory detection."""
        available_memory = MemoryManager.get_available_memory()
        
        # Should return a positive number
        self.assertGreater(available_memory, 0)
        
        # Should be reasonable (less than total system memory)
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        self.assertLess(available_memory, total_memory)
    
    def test_should_use_inplace_small_array(self):
        """Test in-place decision for small arrays."""
        # Small arrays should not use in-place by default
        should_inplace = MemoryManager.should_use_inplace(self.small_array)
        self.assertFalse(should_inplace)
    
    def test_should_use_inplace_large_array(self):
        """Test in-place decision for large arrays."""
        # Large arrays should use in-place
        should_inplace = MemoryManager.should_use_inplace(self.large_array)
        self.assertTrue(should_inplace)
    
    def test_should_use_inplace_low_memory(self):
        """Test in-place decision when memory is low."""
        # Mock very low available memory (less than 2x array size)
        array_size = MemoryManager.get_array_memory_usage(self.small_array)
        low_memory = array_size * 1.5  # Less than 2x array size
        with patch.object(MemoryManager, 'get_available_memory', return_value=low_memory):
            # Even small arrays should use in-place when memory is low
            should_inplace = MemoryManager.should_use_inplace(self.small_array)
            self.assertTrue(should_inplace)
    
    def test_check_memory_requirements_sufficient(self):
        """Test memory requirement check with sufficient memory."""
        # Should pass for small arrays
        result = MemoryManager.check_memory_requirements(self.small_array)
        self.assertTrue(result)
    
    def test_check_memory_requirements_insufficient(self):
        """Test memory requirement check with insufficient memory."""
        # Mock very low available memory
        with patch.object(MemoryManager, 'get_available_memory', return_value=1.0):
            with self.assertRaises(MemoryError):
                MemoryManager.check_memory_requirements(self.large_array)
    
    def test_check_memory_requirements_warning(self):
        """Test memory requirement check with warning for large arrays."""
        with patch('warnings.warn') as mock_warn:
            MemoryManager.check_memory_requirements(self.huge_array)
            mock_warn.assert_called_once()
    
    def test_get_memory_efficiency_ratio(self):
        """Test memory efficiency ratio calculation."""
        # Same size arrays should have ratio of 1.0
        ratio = MemoryManager.get_memory_efficiency_ratio(self.small_array, self.small_array)
        self.assertAlmostEqual(ratio, 1.0, places=2)
        
        # Larger output should have ratio > 1.0
        ratio = MemoryManager.get_memory_efficiency_ratio(self.small_array, self.large_array)
        self.assertGreater(ratio, 1.0)
        
        # Smaller output should have ratio < 1.0
        ratio = MemoryManager.get_memory_efficiency_ratio(self.large_array, self.small_array)
        self.assertLess(ratio, 1.0)
    
    def test_estimate_peak_memory(self):
        """Test peak memory estimation."""
        # Without copy
        peak_no_copy = MemoryManager.estimate_peak_memory(self.small_array, creates_copy=False)
        base_memory = MemoryManager.get_array_memory_usage(self.small_array)
        self.assertAlmostEqual(peak_no_copy, base_memory, places=2)
        
        # With copy
        peak_with_copy = MemoryManager.estimate_peak_memory(self.small_array, creates_copy=True)
        self.assertAlmostEqual(peak_with_copy, base_memory * 2, places=2)
        
        # With intermediate arrays
        peak_with_intermediate = MemoryManager.estimate_peak_memory(
            self.small_array, creates_copy=True, intermediate_arrays=2
        )
        expected = base_memory * 4  # input + output + 2 intermediate
        self.assertAlmostEqual(peak_with_intermediate, expected, places=2)
    
    def test_cleanup_arrays(self):
        """Test array cleanup functionality."""
        # Create arrays to clean up
        array1 = np.random.rand(1000, 1000)
        array2 = np.random.rand(1000, 1000)
        
        # This should not raise an exception
        MemoryManager.cleanup_arrays(array1, array2)
        
        # Test with invalid objects (should handle gracefully)
        MemoryManager.cleanup_arrays(None, "not_an_array")


class TestBaseFilterMemoryOptimization(unittest.TestCase):
    """Test cases for BaseFilter memory optimization features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = BaseFilter(
            name="test_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test"
        )
        self.small_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.large_array = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
    
    def test_should_use_inplace(self):
        """Test in-place decision in BaseFilter."""
        # Small array should not use in-place
        should_inplace = self.filter._should_use_inplace(self.small_array)
        self.assertFalse(should_inplace)
        
        # Large array should use in-place
        should_inplace = self.filter._should_use_inplace(self.large_array)
        self.assertTrue(should_inplace)
    
    def test_check_memory_requirements(self):
        """Test memory requirement checking in BaseFilter."""
        # Should pass for small arrays
        result = self.filter._check_memory_requirements(self.small_array)
        self.assertTrue(result)
        
        # Test with insufficient memory
        with patch.object(MemoryManager, 'get_available_memory', return_value=1.0):
            with self.assertRaises(MemoryError):
                self.filter._check_memory_requirements(self.large_array)
    
    def test_track_memory_usage(self):
        """Test memory usage tracking."""
        output_array = self.small_array.copy()
        
        # Track memory usage
        self.filter._track_memory_usage(self.small_array, output_array, used_inplace=False)
        
        # Check metadata was updated
        self.assertGreater(self.filter.metadata.memory_usage, 0)
        self.assertAlmostEqual(self.filter.metadata.memory_efficiency_ratio, 1.0, places=2)
        self.assertFalse(self.filter.metadata.used_inplace_processing)
        self.assertGreater(self.filter.metadata.peak_memory_usage, 0)
    
    def test_track_memory_usage_inplace(self):
        """Test memory usage tracking for in-place operations."""
        output_array = self.small_array  # Same reference for in-place
        
        # Track memory usage for in-place operation
        self.filter._track_memory_usage(self.small_array, output_array, used_inplace=True)
        
        # Check metadata
        self.assertTrue(self.filter.metadata.used_inplace_processing)
        self.assertAlmostEqual(self.filter.metadata.memory_efficiency_ratio, 1.0, places=2)
    
    def test_cleanup_memory(self):
        """Test memory cleanup in BaseFilter."""
        array1 = np.random.rand(100, 100)
        array2 = np.random.rand(100, 100)
        
        # Should not raise an exception
        self.filter._cleanup_memory(array1, array2)


class TestMemoryOptimizationIntegration(unittest.TestCase):
    """Integration tests for memory optimization features."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete filter for testing
        class TestFilter(BaseFilter):
            def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
                # Check memory requirements
                self._check_memory_requirements(data, creates_copy=True)
                
                # Decide on in-place vs copy
                use_inplace = self._should_use_inplace(data)
                
                if use_inplace:
                    # Simulate in-place operation (convert to float first for safe operations)
                    if data.dtype == np.uint8:
                        result = data.astype(np.float32)
                        result *= 0.8
                        result = result.astype(np.uint8)
                    else:
                        result = data
                        result *= 0.8
                else:
                    # Create copy
                    result = data.copy()
                    result = (result * 0.8).astype(data.dtype)
                
                # Track memory usage
                self._track_memory_usage(data, result, used_inplace=use_inplace)
                
                return result
        
        self.filter = TestFilter(
            name="test_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test"
        )
    
    def test_small_array_processing(self):
        """Test processing of small arrays (should use copy)."""
        small_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = self.filter.apply(small_array)
        
        # Should have used copy (not in-place)
        self.assertFalse(self.filter.metadata.used_inplace_processing)
        self.assertGreater(self.filter.metadata.memory_usage, 0)
        self.assertGreater(self.filter.metadata.peak_memory_usage, 0)
    
    def test_large_array_processing(self):
        """Test processing of large arrays (should use in-place)."""
        large_array = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        
        result = self.filter.apply(large_array)
        
        # Should have used in-place processing
        self.assertTrue(self.filter.metadata.used_inplace_processing)
        self.assertGreater(self.filter.metadata.memory_usage, 0)
    
    def test_memory_insufficient_error(self):
        """Test error handling when memory is insufficient."""
        large_array = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        
        # Mock very low available memory
        with patch.object(MemoryManager, 'get_available_memory', return_value=1.0):
            with self.assertRaises(MemoryError):
                self.filter.apply(large_array)


class TestMemoryOptimizationPerformance(unittest.TestCase):
    """Performance tests for memory optimization."""
    
    def test_memory_tracking_overhead(self):
        """Test that memory tracking doesn't add significant overhead."""
        import time
        
        # Create test data
        test_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        # Test without memory tracking
        start_time = time.time()
        for _ in range(10):
            result = test_array.copy()
        no_tracking_time = time.time() - start_time
        
        # Test with memory tracking
        filter_instance = BaseFilter(
            name="test", data_type=DataType.IMAGE, 
            color_format=ColorFormat.RGB, category="test"
        )
        
        start_time = time.time()
        for _ in range(10):
            result = test_array.copy()
            filter_instance._track_memory_usage(test_array, result)
        with_tracking_time = time.time() - start_time
        
        # Memory tracking should not add more than 50% overhead
        overhead_ratio = with_tracking_time / no_tracking_time
        self.assertLess(overhead_ratio, 1.5, 
                       f"Memory tracking overhead too high: {overhead_ratio:.2f}x")
    
    def test_inplace_vs_copy_performance(self):
        """Test performance difference between in-place and copy operations."""
        import time
        
        # Create large test array
        large_array = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        
        # Test copy operation
        start_time = time.time()
        copy_result = large_array.copy()
        copy_result = (copy_result * 0.8).astype(np.uint8)
        copy_time = time.time() - start_time
        
        # Test in-place operation
        inplace_array = large_array.astype(np.float32)  # Convert to float for in-place ops
        start_time = time.time()
        inplace_array *= 0.8
        inplace_time = time.time() - start_time
        
        # In-place should be faster (at least 20% faster)
        self.assertLess(inplace_time, copy_time * 0.8,
                       f"In-place not significantly faster: {inplace_time:.3f}s vs {copy_time:.3f}s")


if __name__ == '__main__':
    unittest.main()