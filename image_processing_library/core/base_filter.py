"""
Base filter implementation providing common functionality.

This module contains the BaseFilter class and FilterMetadata dataclass
that provide common functionality for all filters including progress tracking,
timing, error handling, and parameter management.

Note: This implementation was developed with assistance from Claude (Anthropic's AI assistant).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional
import time
import numpy as np
from .protocols import DataType, ColorFormat
from .utils import FilterValidationError, MemoryManager, MemoryError, ChunkedProcessor


@dataclass
class FilterMetadata:
    """
    Metadata for filter operations tracking execution details.
    
    This dataclass stores execution tracking information including
    timing, progress, error details, and memory usage for filter operations.
    """
    execution_time: float = 0.0
    progress: float = 0.0
    error_message: Optional[str] = None
    memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    memory_efficiency_ratio: float = 0.0
    used_inplace_processing: bool = False


class BaseFilter:
    """
    Base implementation providing common filter functionality.
    
    This class provides shared functionality for all filters including:
    - Progress tracking with callbacks
    - Execution time measurement
    - Error handling and metadata tracking
    - Parameter management
    - Input validation framework
    
    Filters can inherit from this class to get common functionality
    while still conforming to the FilterProtocol interface.
    """
    
    def __init__(self, name: str, data_type: DataType, color_format: ColorFormat, 
                 category: str, **parameters):
        """
        Initialize the base filter with common properties.
        
        Args:
            name: Human-readable name for the filter
            data_type: Type of data this filter processes (IMAGE or VIDEO)
            color_format: Required color format (RGB, RGBA, GRAYSCALE)
            category: Filter category for organization
            **parameters: Filter-specific parameters
        """
        self.name = name
        self.data_type = data_type
        self.color_format = color_format
        self.category = category
        self.parameters = parameters
        self.metadata = FilterMetadata()
        self._progress_callback: Optional[Callable[[float], None]] = None
    
    def set_progress_callback(self, callback: Callable[[float], None]) -> None:
        """
        Set callback function for progress updates.
        
        Args:
            callback: Function that accepts progress value (0.0 to 1.0)
        """
        self._progress_callback = callback
    
    def _update_progress(self, progress: float) -> None:
        """
        Update progress and call callback if set.
        
        Args:
            progress: Progress value between 0.0 and 1.0
        """
        self.metadata.progress = max(0.0, min(1.0, progress))
        if self._progress_callback:
            self._progress_callback(self.metadata.progress)
    
    def _measure_execution_time(self, func: Callable) -> Any:
        """
        Measure execution time of a function and update metadata.
        
        Args:
            func: Function to execute and measure
            
        Returns:
            Result of the function execution
            
        Raises:
            Any exception raised by the function, with timing recorded
        """
        start_time = time.time()
        try:
            result = func()
            self.metadata.execution_time = time.time() - start_time
            self.metadata.error_message = None
            return result
        except Exception as e:
            self.metadata.execution_time = time.time() - start_time
            self.metadata.error_message = str(e)
            raise
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current filter parameters.
        
        Returns:
            Copy of current parameter dictionary
        """
        return self.parameters.copy()
    
    def set_parameters(self, **kwargs) -> None:
        """
        Update filter parameters.
        
        Args:
            **kwargs: Parameter names and values to update
        """
        self.parameters.update(kwargs)
    
    def _record_shapes(self, input_data: np.ndarray, output_data: np.ndarray) -> None:
        """
        Record input and output shapes in metadata.
        
        Args:
            input_data: Input numpy array
            output_data: Output numpy array
        """
        self.metadata.input_shape = input_data.shape
        self.metadata.output_shape = output_data.shape
    
    def _estimate_memory_usage(self, data: np.ndarray) -> float:
        """
        Estimate memory usage for numpy array in MB.
        
        Args:
            data: Numpy array to estimate memory for
            
        Returns:
            Estimated memory usage in megabytes
        """
        return MemoryManager.get_array_memory_usage(data)
    
    def _should_use_inplace(self, data: np.ndarray) -> bool:
        """
        Determine if in-place processing should be used for memory efficiency.
        
        Args:
            data: Input array to process
            
        Returns:
            True if in-place processing is recommended
        """
        return MemoryManager.should_use_inplace(data)
    
    def _check_memory_requirements(self, data: np.ndarray, creates_copy: bool = True) -> bool:
        """
        Check if there's enough memory for the filter operation.
        
        Args:
            data: Input array
            creates_copy: Whether the operation creates a copy
            
        Returns:
            True if there's enough memory
            
        Raises:
            MemoryError: If insufficient memory is available
        """
        multiplier = 2.0 if creates_copy else 1.0
        return MemoryManager.check_memory_requirements(data, multiplier)
    
    def _track_memory_usage(self, input_data: np.ndarray, output_data: np.ndarray, 
                           used_inplace: bool = False) -> None:
        """
        Track memory usage and efficiency metrics.
        
        Args:
            input_data: Input array
            output_data: Output array
            used_inplace: Whether in-place processing was used
        """
        self.metadata.memory_usage = self._estimate_memory_usage(output_data)
        self.metadata.memory_efficiency_ratio = MemoryManager.get_memory_efficiency_ratio(
            input_data, output_data
        )
        self.metadata.used_inplace_processing = used_inplace
        
        # Estimate peak memory usage
        self.metadata.peak_memory_usage = MemoryManager.estimate_peak_memory(
            input_data, creates_copy=not used_inplace
        )
    
    def _cleanup_memory(self, *arrays: np.ndarray) -> None:
        """
        Clean up memory used by large arrays.
        
        Args:
            *arrays: Arrays to clean up
        """
        MemoryManager.cleanup_arrays(*arrays)
    
    def _should_use_chunked_processing(self, data: np.ndarray, 
                                     memory_threshold_mb: float = 10.0) -> bool:
        """
        Determine if chunked processing should be used for large arrays.
        
        Args:
            data: Input array to process
            memory_threshold_mb: Memory threshold in MB for chunked processing
            
        Returns:
            True if chunked processing is recommended
        """
        array_memory_mb = MemoryManager.get_array_memory_usage(data)
        available_memory_mb = MemoryManager.get_available_memory()
        
        # Use chunked processing if:
        # 1. Array is larger than threshold
        # 2. Available memory is less than 3x array size (for safe processing)
        return (array_memory_mb > memory_threshold_mb or 
                available_memory_mb < array_memory_mb * 3)
    
    def _apply_chunked(self, data: np.ndarray, chunk_process_func: Callable[[np.ndarray], np.ndarray],
                      chunk_size: Optional[tuple] = None) -> np.ndarray:
        """
        Apply filter using chunked processing for memory efficiency.
        
        Args:
            data: Input array to process
            chunk_process_func: Function to apply to each chunk
            chunk_size: Size of chunks (auto-calculated if None)
            
        Returns:
            Processed array
        """
        def chunk_progress_callback(progress: float):
            # Scale progress to filter's progress range
            self._update_progress(progress)
        
        return ChunkedProcessor.process_in_chunks(
            data, chunk_process_func, chunk_size, chunk_progress_callback
        )
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data format and dimensions.
        
        Performs comprehensive validation of input data including:
        - Type checking (must be numpy array)
        - Data type and range validation
        - Dimension validation based on data type
        - Color format validation
        
        Args:
            data: Input numpy array to validate
            
        Returns:
            True if input is valid
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Check if input is numpy array
        if not isinstance(data, np.ndarray):
            raise FilterValidationError("Input must be a numpy array")
        
        # Check for empty array
        if data.size == 0:
            raise FilterValidationError("Input array cannot be empty")
        
        # Validate data type and range first (before checking dimensions/channels)
        self._validate_data_range(data)
        
        # Validate dimensions based on data type
        if self.data_type == DataType.IMAGE:
            if data.ndim not in [2, 3]:
                raise FilterValidationError(
                    f"Image data must be 2D or 3D array, got {data.ndim}D"
                )
        elif self.data_type == DataType.VIDEO:
            if data.ndim != 4:
                raise FilterValidationError(
                    f"Video data must be 4D array (frames, height, width, channels), got {data.ndim}D"
                )
        
        # Validate color format requirements
        self._validate_color_format(data)
        
        return True
    
    def _validate_color_format(self, data: np.ndarray) -> None:
        """
        Validate color format requirements.
        
        Args:
            data: Input numpy array to validate
            
        Raises:
            FilterValidationError: If color format requirements are not met
        """
        if self.color_format == ColorFormat.RGB:
            if data.ndim < 3 or data.shape[-1] != 3:
                raise FilterValidationError(
                    f"RGB format requires 3 channels, got shape {data.shape}"
                )
        elif self.color_format == ColorFormat.RGBA:
            if data.ndim < 3 or data.shape[-1] != 4:
                raise FilterValidationError(
                    f"RGBA format requires 4 channels, got shape {data.shape}"
                )
        elif self.color_format == ColorFormat.GRAYSCALE:
            if self.data_type == DataType.IMAGE and data.ndim != 2:
                raise FilterValidationError(
                    f"Grayscale image format requires 2D array, got {data.ndim}D"
                )
            elif self.data_type == DataType.VIDEO and (data.ndim != 3 and data.ndim != 4):
                raise FilterValidationError(
                    f"Grayscale video format requires 3D or 4D array, got {data.ndim}D"
                )
    
    def _validate_data_range(self, data: np.ndarray) -> None:
        """
        Validate data type and value ranges.
        
        Args:
            data: Input numpy array to validate
            
        Raises:
            FilterValidationError: If data type or range is invalid
        """
        # Check for valid numeric data types
        if not np.issubdtype(data.dtype, np.number):
            raise FilterValidationError(
                f"Data must be numeric, got dtype {data.dtype}"
            )
        
        # Check for NaN or infinite values
        if np.any(np.isnan(data)):
            raise FilterValidationError("Input data contains NaN values")
        
        if np.any(np.isinf(data)):
            raise FilterValidationError("Input data contains infinite values")
        
        # Validate value ranges for common data types
        if data.dtype == np.uint8:
            if np.any(data < 0) or np.any(data > 255):
                raise FilterValidationError(
                    "uint8 data must be in range [0, 255]"
                )
        elif data.dtype in [np.float32, np.float64]:
            # Common convention: float images in [0, 1] range
            if np.any(data < 0) or np.any(data > 1):
                # Issue warning but don't fail - some filters may work with different ranges
                pass
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply the filter to input data.
        
        This is a default implementation that subclasses should override.
        The base implementation performs validation and returns the input unchanged.
        
        Args:
            data: Input numpy array containing image or video data
            **kwargs: Additional filter-specific parameters
            
        Returns:
            Processed numpy array with filter applied
            
        Raises:
            FilterValidationError: If input data is invalid
            NotImplementedError: If subclass doesn't override this method
        """
        self.validate_input(data)
        
        # Default implementation - subclasses should override
        # This allows the base class to satisfy the protocol while requiring
        # concrete implementations to provide actual functionality
        raise NotImplementedError(
            f"Filter '{self.name}' must implement the apply() method"
        )