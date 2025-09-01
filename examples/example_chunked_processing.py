#!/usr/bin/env python3
"""
Example demonstrating chunked processing for large images and videos.

This script shows how to use the ChunkedProcessor and BaseFilter
chunked processing capabilities for memory-efficient processing
of large arrays.
"""

import numpy as np
import time
from image_processing_library.core import (
    BaseFilter, DataType, ColorFormat, ChunkedProcessor, MemoryManager
)


class LargeImageFilter(BaseFilter):
    """Example filter that demonstrates chunked processing for large images."""
    
    def __init__(self, brightness_factor=1.2, **kwargs):
        super().__init__(
            name="Large Image Brightness Filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="enhancement",
            brightness_factor=brightness_factor,
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply brightness adjustment with automatic chunked processing."""
        self.validate_input(data)
        
        # Get brightness factor from parameters
        brightness_factor = self.parameters.get('brightness_factor', 1.2)
        
        def process_chunk(chunk):
            """Process a single chunk of the image."""
            # Convert to float for processing
            chunk_float = chunk.astype(np.float32)
            
            # Apply brightness adjustment
            chunk_float *= brightness_factor
            
            # Clamp values and convert back
            chunk_float = np.clip(chunk_float, 0, 255)
            return chunk_float.astype(np.uint8)
        
        # Check if chunked processing should be used
        if self._should_use_chunked_processing(data):
            print(f"Using chunked processing for large array: {data.shape}")
            result = self._apply_chunked(data, process_chunk)
        else:
            print(f"Using regular processing for array: {data.shape}")
            result = process_chunk(data)
        
        # Track memory usage
        self._track_memory_usage(data, result, used_inplace=False)
        
        return result


class VideoFrameFilter(BaseFilter):
    """Example filter for processing video frames with chunked processing."""
    
    def __init__(self, contrast_factor=1.1, **kwargs):
        super().__init__(
            name="Video Contrast Filter",
            data_type=DataType.VIDEO,
            color_format=ColorFormat.RGB,
            category="enhancement",
            contrast_factor=contrast_factor,
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply contrast adjustment to video frames."""
        self.validate_input(data)
        
        contrast_factor = self.parameters.get('contrast_factor', 1.1)
        
        def process_frame_chunk(chunk):
            """Process a chunk of video frames."""
            # Convert to float for processing
            chunk_float = chunk.astype(np.float32)
            
            # Apply contrast adjustment (simple version)
            chunk_float = (chunk_float - 128) * contrast_factor + 128
            
            # Clamp values and convert back
            chunk_float = np.clip(chunk_float, 0, 255)
            return chunk_float.astype(np.uint8)
        
        # Always use chunked processing for video to handle frame-by-frame
        print(f"Processing video with shape: {data.shape}")
        result = self._apply_chunked(data, process_frame_chunk)
        
        # Track memory usage
        self._track_memory_usage(data, result, used_inplace=False)
        
        return result


def demonstrate_memory_management():
    """Demonstrate memory management utilities."""
    print("=== Memory Management Demo ===")
    
    # Create test arrays of different sizes
    small_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    medium_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    large_array = np.random.randint(0, 255, (3000, 3000, 3), dtype=np.uint8)
    
    arrays = [
        ("Small", small_array),
        ("Medium", medium_array),
        ("Large", large_array)
    ]
    
    for name, array in arrays:
        memory_mb = MemoryManager.get_array_memory_usage(array)
        should_inplace = MemoryManager.should_use_inplace(array)
        
        print(f"{name} array ({array.shape}): {memory_mb:.2f} MB")
        print(f"  Should use in-place: {should_inplace}")
        print(f"  Available memory: {MemoryManager.get_available_memory():.1f} MB")
        print()


def demonstrate_chunked_processing():
    """Demonstrate chunked processing with different array sizes."""
    print("=== Chunked Processing Demo ===")
    
    # Create filter
    filter_instance = LargeImageFilter(brightness_factor=1.3)
    
    # Test with different array sizes
    arrays = [
        ("Small image", np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)),
        ("Medium image", np.random.randint(0, 255, (1500, 1500, 3), dtype=np.uint8)),
        ("Large image", np.random.randint(0, 255, (3000, 3000, 3), dtype=np.uint8)),
    ]
    
    for name, array in arrays:
        print(f"\nProcessing {name}: {array.shape}")
        print(f"Memory usage: {MemoryManager.get_array_memory_usage(array):.2f} MB")
        
        # Set up progress tracking
        def progress_callback(progress):
            if progress in [0.25, 0.5, 0.75, 1.0]:
                print(f"  Progress: {progress*100:.0f}%")
        
        filter_instance.set_progress_callback(progress_callback)
        
        # Process the array
        start_time = time.time()
        result = filter_instance.apply(array)
        end_time = time.time()
        
        print(f"  Processing time: {end_time - start_time:.2f} seconds")
        print(f"  Used in-place: {filter_instance.metadata.used_inplace_processing}")
        print(f"  Peak memory: {filter_instance.metadata.peak_memory_usage:.2f} MB")
        print(f"  Memory efficiency: {filter_instance.metadata.memory_efficiency_ratio:.2f}")


def demonstrate_video_processing():
    """Demonstrate video processing with chunked processing."""
    print("\n=== Video Processing Demo ===")
    
    # Create a sample video (10 frames, 1000x1000, RGB)
    video_data = np.random.randint(0, 255, (10, 1000, 1000, 3), dtype=np.uint8)
    
    print(f"Video shape: {video_data.shape}")
    print(f"Video memory: {MemoryManager.get_array_memory_usage(video_data):.2f} MB")
    
    # Create video filter
    video_filter = VideoFrameFilter(contrast_factor=1.2)
    
    # Set up progress tracking
    def video_progress_callback(progress):
        print(f"  Video processing progress: {progress*100:.0f}%")
    
    video_filter.set_progress_callback(video_progress_callback)
    
    # Process the video
    start_time = time.time()
    result_video = video_filter.apply(video_data)
    end_time = time.time()
    
    print(f"Video processing time: {end_time - start_time:.2f} seconds")
    print(f"Result shape: {result_video.shape}")
    print(f"Peak memory usage: {video_filter.metadata.peak_memory_usage:.2f} MB")


def demonstrate_manual_chunked_processing():
    """Demonstrate manual use of ChunkedProcessor."""
    print("\n=== Manual Chunked Processing Demo ===")
    
    # Create a large array
    large_array = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
    
    print(f"Array shape: {large_array.shape}")
    print(f"Array memory: {MemoryManager.get_array_memory_usage(large_array):.2f} MB")
    
    # Calculate optimal chunk size
    chunk_size = ChunkedProcessor.calculate_chunk_size(large_array.shape)
    print(f"Calculated chunk size: {chunk_size}")
    
    # Define a processing function
    def edge_detection_filter(chunk):
        """Simple edge detection filter."""
        # Convert to grayscale first
        if len(chunk.shape) == 3:
            gray = np.mean(chunk, axis=-1, keepdims=True)
            gray = np.repeat(gray, 3, axis=-1)
        else:
            gray = chunk
        
        # Simple edge detection (difference with shifted version)
        edges = np.abs(gray[1:, :] - gray[:-1, :])
        
        # Pad to maintain shape
        if len(edges.shape) == 3:
            padded = np.pad(edges, ((0, 1), (0, 0), (0, 0)), mode='edge')
        else:
            padded = np.pad(edges, ((0, 1), (0, 0)), mode='edge')
        
        return padded.astype(np.uint8)
    
    # Set up progress tracking
    progress_values = []
    def manual_progress_callback(progress):
        progress_values.append(progress)
        if len(progress_values) % 5 == 0 or progress == 1.0:
            print(f"  Manual processing progress: {progress*100:.0f}%")
    
    # Process using ChunkedProcessor directly
    start_time = time.time()
    result = ChunkedProcessor.process_in_chunks(
        large_array, 
        edge_detection_filter,
        chunk_size=chunk_size,
        progress_callback=manual_progress_callback
    )
    end_time = time.time()
    
    print(f"Manual chunked processing time: {end_time - start_time:.2f} seconds")
    print(f"Total progress updates: {len(progress_values)}")
    print(f"Result shape: {result.shape}")


if __name__ == "__main__":
    print("Image Processing Library - Chunked Processing Examples")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demonstrate_memory_management()
        demonstrate_chunked_processing()
        demonstrate_video_processing()
        demonstrate_manual_chunked_processing()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()