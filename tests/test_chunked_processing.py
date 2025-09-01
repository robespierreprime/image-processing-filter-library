"""
Tests for chunked processing functionality.

This module contains tests for the ChunkedProcessor class and
chunked processing integration in BaseFilter.
"""

import unittest
import numpy as np
import time
from unittest.mock import patch, MagicMock
from image_processing_library.core.utils import ChunkedProcessor, MemoryManager
from image_processing_library.core.base_filter import BaseFilter
from image_processing_library.core.protocols import DataType, ColorFormat


class TestChunkedProcessor(unittest.TestCase):
    """Test cases for ChunkedProcessor utility class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.small_2d = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.large_2d = np.random.randint(0, 255, (2000, 2000), dtype=np.uint8)
        self.small_3d = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.large_3d = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        self.video_4d = np.random.randint(0, 255, (10, 500, 500, 3), dtype=np.uint8)
    
    def test_calculate_chunk_size_2d(self):
        """Test chunk size calculation for 2D arrays."""
        chunk_size = ChunkedProcessor.calculate_chunk_size(self.large_2d.shape)
        
        # Should return 2D chunk size
        self.assertEqual(len(chunk_size), 2)
        self.assertGreater(chunk_size[0], 0)
        self.assertGreater(chunk_size[1], 0)
        
        # Chunk should be smaller than or equal to original
        self.assertLessEqual(chunk_size[0], self.large_2d.shape[0])
        self.assertLessEqual(chunk_size[1], self.large_2d.shape[1])
    
    def test_calculate_chunk_size_3d(self):
        """Test chunk size calculation for 3D arrays."""
        chunk_size = ChunkedProcessor.calculate_chunk_size(self.large_3d.shape)
        
        # Should return 3D chunk size
        self.assertEqual(len(chunk_size), 3)
        self.assertGreater(chunk_size[0], 0)
        
        # Width and channels should remain the same
        self.assertEqual(chunk_size[1], self.large_3d.shape[1])
        self.assertEqual(chunk_size[2], self.large_3d.shape[2])
    
    def test_calculate_chunk_size_4d(self):
        """Test chunk size calculation for 4D arrays (video)."""
        chunk_size = ChunkedProcessor.calculate_chunk_size(self.video_4d.shape)
        
        # Should return 4D chunk size
        self.assertEqual(len(chunk_size), 4)
        self.assertGreater(chunk_size[0], 0)
        
        # Spatial dimensions should remain the same
        self.assertEqual(chunk_size[1], self.video_4d.shape[1])
        self.assertEqual(chunk_size[2], self.video_4d.shape[2])
        self.assertEqual(chunk_size[3], self.video_4d.shape[3])
    
    def test_calculate_chunk_size_with_memory_limit(self):
        """Test chunk size calculation with memory constraints."""
        # Test with very low memory limit
        chunk_size = ChunkedProcessor.calculate_chunk_size(
            self.large_2d.shape, available_memory_mb=10.0, target_memory_mb=5.0
        )
        
        # Should produce smaller chunks
        self.assertLess(chunk_size[0] * chunk_size[1], 
                       self.large_2d.shape[0] * self.large_2d.shape[1])
    
    def test_process_in_chunks_2d(self):
        """Test chunked processing for 2D arrays."""
        def simple_filter(chunk):
            return (chunk * 0.8).astype(np.uint8)
        
        # Process with chunking
        result = ChunkedProcessor.process_in_chunks(self.large_2d, simple_filter)
        
        # Result should have same shape
        self.assertEqual(result.shape, self.large_2d.shape)
        
        # Result should be different from input (filtered)
        self.assertFalse(np.array_equal(result, self.large_2d))
        
        # Compare with non-chunked processing
        expected = simple_filter(self.large_2d)
        np.testing.assert_array_equal(result, expected)
    
    def test_process_in_chunks_3d(self):
        """Test chunked processing for 3D arrays."""
        def simple_filter(chunk):
            return (chunk * 0.9).astype(np.uint8)
        
        # Process with chunking
        result = ChunkedProcessor.process_in_chunks(self.large_3d, simple_filter)
        
        # Result should have same shape
        self.assertEqual(result.shape, self.large_3d.shape)
        
        # Compare with non-chunked processing
        expected = simple_filter(self.large_3d)
        np.testing.assert_array_equal(result, expected)
    
    def test_process_in_chunks_4d(self):
        """Test chunked processing for 4D arrays (video)."""
        def simple_filter(chunk):
            return (chunk * 0.7).astype(np.uint8)
        
        # Process with chunking
        result = ChunkedProcessor.process_in_chunks(self.video_4d, simple_filter)
        
        # Result should have same shape
        self.assertEqual(result.shape, self.video_4d.shape)
        
        # Compare with non-chunked processing
        expected = simple_filter(self.video_4d)
        np.testing.assert_array_equal(result, expected)
    
    def test_process_in_chunks_small_array(self):
        """Test that small arrays bypass chunking."""
        def simple_filter(chunk):
            return (chunk * 0.5).astype(np.uint8)
        
        # Mock the process function to track calls
        mock_filter = MagicMock(side_effect=simple_filter)
        
        # Process small array
        result = ChunkedProcessor.process_in_chunks(self.small_2d, mock_filter)
        
        # Should be called only once (no chunking)
        mock_filter.assert_called_once()
        
        # Result should be correct
        expected = simple_filter(self.small_2d)
        np.testing.assert_array_equal(result, expected)
    
    def test_process_in_chunks_progress_callback(self):
        """Test progress callback during chunked processing."""
        progress_values = []
        
        def progress_callback(progress):
            progress_values.append(progress)
        
        def simple_filter(chunk):
            return chunk
        
        # Process with progress tracking
        ChunkedProcessor.process_in_chunks(
            self.large_2d, simple_filter, progress_callback=progress_callback
        )
        
        # Should have received progress updates
        self.assertGreater(len(progress_values), 0)
        
        # Progress should be between 0 and 1
        for progress in progress_values:
            self.assertGreaterEqual(progress, 0.0)
            self.assertLessEqual(progress, 1.0)
        
        # Final progress should be 1.0
        self.assertEqual(progress_values[-1], 1.0)
    
    def test_process_in_chunks_custom_chunk_size(self):
        """Test chunked processing with custom chunk size."""
        def simple_filter(chunk):
            return chunk
        
        # Use custom chunk size
        custom_chunk_size = (100, 100)
        result = ChunkedProcessor.process_in_chunks(
            self.large_2d, simple_filter, chunk_size=custom_chunk_size
        )
        
        # Should work correctly
        self.assertEqual(result.shape, self.large_2d.shape)
        np.testing.assert_array_equal(result, self.large_2d)


class TestBaseFilterChunkedProcessing(unittest.TestCase):
    """Test cases for chunked processing integration in BaseFilter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete filter for testing
        class TestChunkedFilter(BaseFilter):
            def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
                # Check if chunked processing should be used
                if self._should_use_chunked_processing(data):
                    # Define chunk processing function
                    def chunk_process(chunk):
                        return (chunk * 0.8).astype(chunk.dtype)
                    
                    # Apply chunked processing
                    result = self._apply_chunked(data, chunk_process)
                else:
                    # Regular processing
                    result = (data * 0.8).astype(data.dtype)
                
                # Track memory usage
                self._track_memory_usage(data, result, used_inplace=False)
                
                return result
        
        self.filter = TestChunkedFilter(
            name="test_chunked_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test"
        )
        
        self.small_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.large_array = np.random.randint(0, 255, (3000, 3000, 3), dtype=np.uint8)
    
    def test_should_use_chunked_processing_small(self):
        """Test chunked processing decision for small arrays."""
        should_chunk = self.filter._should_use_chunked_processing(self.small_array)
        self.assertFalse(should_chunk)
    
    def test_should_use_chunked_processing_large(self):
        """Test chunked processing decision for large arrays."""
        should_chunk = self.filter._should_use_chunked_processing(self.large_array)
        self.assertTrue(should_chunk)
    
    def test_should_use_chunked_processing_low_memory(self):
        """Test chunked processing decision with low memory."""
        # Calculate array size and set low memory
        array_size = MemoryManager.get_array_memory_usage(self.small_array)
        low_memory = array_size * 2.5  # Less than 3x array size
        with patch.object(MemoryManager, 'get_available_memory', return_value=low_memory):
            should_chunk = self.filter._should_use_chunked_processing(self.small_array)
            self.assertTrue(should_chunk)
    
    def test_apply_chunked_processing(self):
        """Test applying filter with chunked processing."""
        def chunk_process(chunk):
            return (chunk * 0.5).astype(chunk.dtype)
        
        result = self.filter._apply_chunked(self.large_array, chunk_process)
        
        # Result should have same shape
        self.assertEqual(result.shape, self.large_array.shape)
        
        # Result should be different (filtered)
        self.assertFalse(np.array_equal(result, self.large_array))
    
    def test_chunked_filter_integration(self):
        """Test complete filter with chunked processing integration."""
        # Apply filter to large array (should use chunking)
        result_large = self.filter.apply(self.large_array)
        
        # Apply filter to small array (should not use chunking)
        result_small = self.filter.apply(self.small_array)
        
        # Both should work correctly
        self.assertEqual(result_large.shape, self.large_array.shape)
        self.assertEqual(result_small.shape, self.small_array.shape)
        
        # Results should be filtered
        self.assertFalse(np.array_equal(result_large, self.large_array))
        self.assertFalse(np.array_equal(result_small, self.small_array))


class TestChunkedProcessingPerformance(unittest.TestCase):
    """Performance tests for chunked processing."""
    
    def test_chunked_vs_regular_processing_correctness(self):
        """Test that chunked processing produces same results as regular processing."""
        # Create test data
        test_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        def test_filter(data):
            return (data * 0.8 + 20).astype(np.uint8)
        
        # Regular processing
        regular_result = test_filter(test_array)
        
        # Chunked processing
        chunked_result = ChunkedProcessor.process_in_chunks(test_array, test_filter)
        
        # Results should be identical
        np.testing.assert_array_equal(regular_result, chunked_result)
    
    def test_chunked_processing_memory_efficiency(self):
        """Test that chunked processing uses less peak memory."""
        # This test is more conceptual - in practice, we'd need memory profiling tools
        # For now, we just verify that chunked processing completes successfully
        # with large arrays that might cause memory issues with regular processing
        
        large_array = np.random.randint(0, 255, (3000, 3000, 3), dtype=np.uint8)
        
        def memory_intensive_filter(chunk):
            # Simulate memory-intensive operation
            temp = chunk.astype(np.float64)  # Double memory usage
            temp = temp * 0.8 + 20
            return temp.astype(np.uint8)
        
        # This should complete without memory errors
        result = ChunkedProcessor.process_in_chunks(large_array, memory_intensive_filter)
        
        self.assertEqual(result.shape, large_array.shape)
    
    def test_chunked_processing_progress_tracking(self):
        """Test that progress tracking works correctly during chunked processing."""
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress)
        
        def slow_filter(chunk):
            # Simulate slow processing
            time.sleep(0.01)
            return chunk
        
        test_array = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        
        # Force small chunk size to ensure multiple chunks
        small_chunk_size = (200, 2000, 3)
        ChunkedProcessor.process_in_chunks(
            test_array, slow_filter, chunk_size=small_chunk_size, progress_callback=progress_callback
        )
        
        # Should have multiple progress updates
        self.assertGreater(len(progress_updates), 1)
        
        # Progress should be monotonically increasing
        for i in range(1, len(progress_updates)):
            self.assertGreaterEqual(progress_updates[i], progress_updates[i-1])
        
        # Final progress should be 1.0
        self.assertEqual(progress_updates[-1], 1.0)


if __name__ == '__main__':
    unittest.main()