"""
Common utilities for the image processing library.

This module contains shared utility functions, helper classes,
custom exception definitions, and memory management utilities.
"""

import gc
import psutil
import numpy as np
from typing import Optional, Tuple, Callable
import warnings


class FilterError(Exception):
    """Base exception for filter operations."""
    pass


class FilterValidationError(FilterError):
    """Raised when filter input validation fails."""
    pass


class FilterExecutionError(FilterError):
    """Raised when filter execution fails."""
    pass


class PresetError(FilterError):
    """Raised when preset operations fail."""
    pass


class UnsupportedFormatError(FilterError):
    """Raised when unsupported data format is encountered."""
    pass


class MemoryError(FilterError):
    """Raised when memory-related operations fail."""
    pass


class MemoryManager:
    """
    Memory management utilities for efficient array processing.
    
    This class provides utilities for tracking memory usage, making
    in-place vs copy decisions, and cleaning up large arrays.
    """
    
    # Memory thresholds in MB
    INPLACE_THRESHOLD_MB = 10   # Use in-place for arrays larger than 10MB
    WARNING_THRESHOLD_MB = 50   # Warn when processing arrays larger than 50MB
    
    @staticmethod
    def get_available_memory() -> float:
        """
        Get available system memory in MB.
        
        Returns:
            Available memory in megabytes
        """
        return psutil.virtual_memory().available / (1024 * 1024)
    
    @staticmethod
    def get_array_memory_usage(array: np.ndarray) -> float:
        """
        Calculate memory usage of numpy array in MB.
        
        Args:
            array: Numpy array to calculate memory for
            
        Returns:
            Memory usage in megabytes
        """
        return array.nbytes / (1024 * 1024)
    
    @staticmethod
    def should_use_inplace(array: np.ndarray, available_memory: Optional[float] = None) -> bool:
        """
        Determine if in-place processing should be used based on memory efficiency.
        
        Args:
            array: Input array to process
            available_memory: Available memory in MB (auto-detected if None)
            
        Returns:
            True if in-place processing is recommended
        """
        array_size_mb = MemoryManager.get_array_memory_usage(array)
        
        if available_memory is None:
            available_memory = MemoryManager.get_available_memory()
        
        # Use in-place if:
        # 1. Array is larger than threshold
        # 2. Available memory is less than 2x array size (to avoid copy)
        return (array_size_mb > MemoryManager.INPLACE_THRESHOLD_MB or 
                available_memory < array_size_mb * 2)
    
    @staticmethod
    def check_memory_requirements(array: np.ndarray, operation_multiplier: float = 2.0) -> bool:
        """
        Check if there's enough memory for the operation.
        
        Args:
            array: Input array
            operation_multiplier: Memory multiplier for the operation (default 2.0 for copy)
            
        Returns:
            True if there's enough memory
            
        Raises:
            MemoryError: If insufficient memory is available
        """
        array_size_mb = MemoryManager.get_array_memory_usage(array)
        required_memory = array_size_mb * operation_multiplier
        available_memory = MemoryManager.get_available_memory()
        
        if array_size_mb > MemoryManager.WARNING_THRESHOLD_MB:
            warnings.warn(
                f"Processing large array ({array_size_mb:.1f} MB). "
                f"Consider using chunked processing for better memory efficiency.",
                UserWarning
            )
        
        if required_memory > available_memory:
            raise MemoryError(
                f"Insufficient memory for operation. Required: {required_memory:.1f} MB, "
                f"Available: {available_memory:.1f} MB"
            )
        
        return True
    
    @staticmethod
    def cleanup_arrays(*arrays: np.ndarray) -> None:
        """
        Explicitly clean up large numpy arrays and force garbage collection.
        
        Args:
            *arrays: Arrays to clean up
        """
        for array in arrays:
            if hasattr(array, '__array_interface__'):
                # Clear the array data
                try:
                    del array
                except:
                    pass
        
        # Force garbage collection
        gc.collect()
    
    @staticmethod
    def get_memory_efficiency_ratio(input_array: np.ndarray, output_array: np.ndarray) -> float:
        """
        Calculate memory efficiency ratio (output size / input size).
        
        Args:
            input_array: Input array
            output_array: Output array
            
        Returns:
            Memory efficiency ratio
        """
        input_size = MemoryManager.get_array_memory_usage(input_array)
        output_size = MemoryManager.get_array_memory_usage(output_array)
        
        if input_size == 0:
            return 1.0
        
        return output_size / input_size
    
    @staticmethod
    def estimate_peak_memory(input_array: np.ndarray, creates_copy: bool = True, 
                           intermediate_arrays: int = 0) -> float:
        """
        Estimate peak memory usage during filter operation.
        
        Args:
            input_array: Input array
            creates_copy: Whether the operation creates a copy
            intermediate_arrays: Number of intermediate arrays created
            
        Returns:
            Estimated peak memory usage in MB
        """
        base_memory = MemoryManager.get_array_memory_usage(input_array)
        
        # Account for copy if needed
        if creates_copy:
            peak_memory = base_memory * 2  # Input + output
        else:
            peak_memory = base_memory
        
        # Account for intermediate arrays
        peak_memory += base_memory * intermediate_arrays
        
        return peak_memory


class ChunkedProcessor:
    """
    Chunked processing utilities for handling large images and videos.
    
    This class provides utilities for processing large arrays in chunks
    to manage memory usage and provide progress tracking.
    """
    
    @staticmethod
    def calculate_chunk_size(array_shape: tuple, available_memory_mb: Optional[float] = None,
                           target_memory_mb: float = 100.0) -> tuple:
        """
        Calculate optimal chunk size based on available memory.
        
        Args:
            array_shape: Shape of the array to process
            available_memory_mb: Available memory in MB (auto-detected if None)
            target_memory_mb: Target memory usage per chunk in MB
            
        Returns:
            Tuple of chunk dimensions
        """
        if available_memory_mb is None:
            available_memory_mb = MemoryManager.get_available_memory()
        
        # Use smaller of available memory or target memory
        chunk_memory_mb = min(available_memory_mb * 0.3, target_memory_mb)  # Use 30% of available
        
        # Calculate bytes per element
        bytes_per_element = 1  # Assume uint8, adjust if needed
        if len(array_shape) >= 3:
            bytes_per_element = array_shape[-1]  # Account for channels
        
        # Calculate total elements that fit in chunk memory
        chunk_memory_bytes = chunk_memory_mb * 1024 * 1024
        elements_per_chunk = chunk_memory_bytes // bytes_per_element
        
        if len(array_shape) == 2:  # 2D image (grayscale)
            # Calculate square chunk size
            chunk_side = int(np.sqrt(elements_per_chunk))
            return (min(chunk_side, array_shape[0]), min(chunk_side, array_shape[1]))
        
        elif len(array_shape) == 3:  # 3D image (color) or video frame
            # Keep full width and channels, chunk by height
            elements_per_row = array_shape[1] * array_shape[2]
            chunk_height = max(1, elements_per_chunk // elements_per_row)
            return (min(chunk_height, array_shape[0]), array_shape[1], array_shape[2])
        
        elif len(array_shape) == 4:  # 4D video
            # Keep spatial dimensions, chunk by frames
            elements_per_frame = array_shape[1] * array_shape[2] * array_shape[3]
            chunk_frames = max(1, elements_per_chunk // elements_per_frame)
            return (min(chunk_frames, array_shape[0]), array_shape[1], array_shape[2], array_shape[3])
        
        else:
            # Default: chunk along first dimension
            elements_per_slice = np.prod(array_shape[1:])
            chunk_size = max(1, elements_per_chunk // elements_per_slice)
            return (min(chunk_size, array_shape[0]),) + array_shape[1:]
    
    @staticmethod
    def process_in_chunks(array: np.ndarray, process_func: Callable[[np.ndarray], np.ndarray],
                         chunk_size: Optional[tuple] = None,
                         progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """
        Process array in chunks to manage memory usage.
        
        Args:
            array: Input array to process
            process_func: Function to apply to each chunk
            chunk_size: Size of chunks (auto-calculated if None)
            progress_callback: Callback for progress updates
            
        Returns:
            Processed array
        """
        if chunk_size is None:
            chunk_size = ChunkedProcessor.calculate_chunk_size(array.shape)
        
        # Check if chunking is actually needed
        array_memory_mb = MemoryManager.get_array_memory_usage(array)
        if array_memory_mb < 10:  # Don't chunk small arrays
            if progress_callback:
                progress_callback(1.0)
            return process_func(array)
        
        # Initialize output array
        output_array = np.empty_like(array)
        
        if len(array.shape) == 2:  # 2D image
            return ChunkedProcessor._process_2d_chunks(
                array, output_array, process_func, chunk_size, progress_callback
            )
        elif len(array.shape) == 3:  # 3D image or video frame
            return ChunkedProcessor._process_3d_chunks(
                array, output_array, process_func, chunk_size, progress_callback
            )
        elif len(array.shape) == 4:  # 4D video
            return ChunkedProcessor._process_4d_chunks(
                array, output_array, process_func, chunk_size, progress_callback
            )
        else:
            raise ValueError(f"Unsupported array dimensions: {len(array.shape)}")
    
    @staticmethod
    def _process_2d_chunks(array: np.ndarray, output_array: np.ndarray,
                          process_func: Callable, chunk_size: tuple,
                          progress_callback: Optional[Callable[[float], None]]) -> np.ndarray:
        """Process 2D array in chunks."""
        chunk_h, chunk_w = chunk_size
        total_chunks = ((array.shape[0] + chunk_h - 1) // chunk_h) * \
                      ((array.shape[1] + chunk_w - 1) // chunk_w)
        processed_chunks = 0
        
        for i in range(0, array.shape[0], chunk_h):
            for j in range(0, array.shape[1], chunk_w):
                # Extract chunk
                end_i = min(i + chunk_h, array.shape[0])
                end_j = min(j + chunk_w, array.shape[1])
                chunk = array[i:end_i, j:end_j]
                
                # Process chunk
                processed_chunk = process_func(chunk)
                
                # Store result
                output_array[i:end_i, j:end_j] = processed_chunk
                
                # Update progress
                processed_chunks += 1
                if progress_callback:
                    progress_callback(processed_chunks / total_chunks)
        
        return output_array
    
    @staticmethod
    def _process_3d_chunks(array: np.ndarray, output_array: np.ndarray,
                          process_func: Callable, chunk_size: tuple,
                          progress_callback: Optional[Callable[[float], None]]) -> np.ndarray:
        """Process 3D array in chunks."""
        chunk_h = chunk_size[0]
        total_chunks = (array.shape[0] + chunk_h - 1) // chunk_h
        processed_chunks = 0
        
        for i in range(0, array.shape[0], chunk_h):
            # Extract chunk
            end_i = min(i + chunk_h, array.shape[0])
            chunk = array[i:end_i]
            
            # Process chunk
            processed_chunk = process_func(chunk)
            
            # Store result
            output_array[i:end_i] = processed_chunk
            
            # Update progress
            processed_chunks += 1
            if progress_callback:
                progress_callback(processed_chunks / total_chunks)
        
        return output_array
    
    @staticmethod
    def _process_4d_chunks(array: np.ndarray, output_array: np.ndarray,
                          process_func: Callable, chunk_size: tuple,
                          progress_callback: Optional[Callable[[float], None]]) -> np.ndarray:
        """Process 4D array (video) in chunks."""
        chunk_frames = chunk_size[0]
        total_chunks = (array.shape[0] + chunk_frames - 1) // chunk_frames
        processed_chunks = 0
        
        for i in range(0, array.shape[0], chunk_frames):
            # Extract chunk
            end_i = min(i + chunk_frames, array.shape[0])
            chunk = array[i:end_i]
            
            # Process chunk
            processed_chunk = process_func(chunk)
            
            # Store result
            output_array[i:end_i] = processed_chunk
            
            # Update progress
            processed_chunks += 1
            if progress_callback:
                progress_callback(processed_chunks / total_chunks)
        
        return output_array