"""
Image correction filters for brightness and contrast adjustments.

This module contains filters for image correction operations:
- GammaCorrectionFilter: Applies gamma correction for brightness adjustment
- ContrastFilter: Adjusts image contrast by scaling pixel values around midpoint
"""

import numpy as np
from ...core.base_filter import BaseFilter
from ...core.protocols import DataType, ColorFormat
from ...core.utils import FilterValidationError
from ..registry import register_filter


@register_filter(category="enhancement")
class GammaCorrectionFilter(BaseFilter):
    """
    Filter that applies gamma correction for brightness adjustment.
    
    Gamma correction applies a power law transformation to adjust image brightness:
    output = (input / max_value) ^ (1/gamma) * max_value
    
    - gamma < 1.0: Makes image brighter
    - gamma = 1.0: No change (identity transformation)
    - gamma > 1.0: Makes image darker
    
    The filter supports RGB, RGBA, and GRAYSCALE color formats.
    """
    
    def __init__(self, gamma: float = 1.0):
        """
        Initialize the GammaCorrectionFilter.
        
        Args:
            gamma: Gamma value for correction (0.1-3.0 range, 1.0 = no change)
        """
        super().__init__(
            name="gamma_correction",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,  # Will handle RGB, RGBA, and GRAYSCALE
            category="enhancement",
            gamma=gamma
        )
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """
        Validate filter parameters.
        
        Ensures gamma value is within acceptable range (0.1-3.0).
        
        Raises:
            FilterValidationError: If gamma value is invalid
        """
        gamma = self.parameters.get('gamma', 1.0)
        
        if not isinstance(gamma, (int, float)):
            raise FilterValidationError("Gamma must be a numeric value")
        
        if gamma <= 0:
            raise FilterValidationError("Gamma must be greater than 0")
        
        if gamma < 0.1 or gamma > 3.0:
            raise FilterValidationError(
                f"Gamma must be in range [0.1, 3.0], got {gamma}"
            )
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data for GammaCorrectionFilter.
        
        Extends BaseFilter validation to ensure the input is compatible
        with gamma correction operations.
        
        Args:
            data: Input numpy array to validate
            
        Returns:
            True if input is valid
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Check basic requirements first
        if not isinstance(data, np.ndarray):
            raise FilterValidationError("Input must be a numpy array")
        
        if data.size == 0:
            raise FilterValidationError("Input array cannot be empty")
        
        # Validate data type and range
        self._validate_data_range(data)
        
        # Check dimensions for image data
        if data.ndim not in [2, 3]:
            raise FilterValidationError(
                f"Image data must be 2D or 3D array, got {data.ndim}D"
            )
        
        # Validate color format - accept RGB, RGBA, or grayscale
        if data.ndim == 3:
            channels = data.shape[-1]
            if channels not in [3, 4]:
                raise FilterValidationError(
                    f"Color images must have 3 (RGB) or 4 (RGBA) channels, got {channels}"
                )
        elif data.ndim == 2:
            # Grayscale is acceptable for gamma correction
            pass
        
        return True
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply gamma correction to the input image.
        
        Applies the power law transformation:
        output = (input / max_value) ^ (1/gamma) * max_value
        
        For gamma = 1.0, returns the original image unchanged (identity case).
        
        Args:
            data: Input numpy array containing image data
            **kwargs: Additional parameters (can override gamma)
            
        Returns:
            Numpy array with gamma correction applied
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Validate input data
        self.validate_input(data)
        
        # Update parameters with any kwargs
        if kwargs:
            self.set_parameters(**kwargs)
        
        self._validate_parameters()
        
        gamma = self.parameters.get('gamma', 1.0)
        
        def gamma_correction_operation():
            # Handle identity case (gamma = 1.0)
            if abs(gamma - 1.0) < 1e-6:
                # Return copy to maintain consistent behavior
                return data.copy()
            
            # Determine if we should use in-place processing
            use_inplace = self._should_use_inplace(data)
            
            if use_inplace:
                # Process in-place for memory efficiency
                result = data.astype(np.float64)
            else:
                # Create a copy for processing
                result = data.astype(np.float64)
            
            # Apply gamma correction based on data type
            if data.dtype == np.uint8:
                # For uint8: normalize to [0,1], apply gamma, scale back to [0,255]
                result = result / 255.0
                result = np.power(result, 1.0 / gamma)
                result = result * 255.0
                # Clip to valid range and convert back to uint8
                result = np.clip(result, 0, 255).astype(np.uint8)
                
            elif data.dtype in [np.float32, np.float64]:
                # For float: assume [0,1] range, apply gamma directly
                # Clip to ensure values stay in valid range
                result = np.clip(result, 0.0, 1.0)
                result = np.power(result, 1.0 / gamma)
                result = np.clip(result, 0.0, 1.0)
                # Convert back to original float type
                result = result.astype(data.dtype)
                
            else:
                # For other data types, determine max value and normalize
                if np.issubdtype(data.dtype, np.integer):
                    max_val = float(np.iinfo(data.dtype).max)
                else:
                    # For other float types, assume [0,1] range
                    max_val = 1.0
                
                # Normalize, apply gamma, and scale back
                result = result / max_val
                result = np.power(result, 1.0 / gamma)
                result = result * max_val
                
                # Clip to valid range and convert back to original type
                if np.issubdtype(data.dtype, np.integer):
                    result = np.clip(result, 0, max_val).astype(data.dtype)
                else:
                    result = np.clip(result, 0.0, max_val).astype(data.dtype)
            
            return result
        
        # Execute with timing and progress tracking
        self._update_progress(0.0)
        result = self._measure_execution_time(gamma_correction_operation)
        self._update_progress(1.0)
        
        # Record metadata
        self._record_shapes(data, result)
        self._track_memory_usage(data, result, used_inplace=np.shares_memory(data, result))
        
        return result


@register_filter(category="enhancement")
class ContrastFilter(BaseFilter):
    """
    Filter that adjusts image contrast by scaling pixel values around a midpoint.
    
    Contrast adjustment scales pixel values around the midpoint (128 for uint8):
    output = (input - midpoint) * contrast_factor + midpoint
    
    - contrast_factor < 1.0: Reduces contrast
    - contrast_factor = 1.0: No change (identity transformation)
    - contrast_factor > 1.0: Increases contrast
    
    The filter supports RGB, RGBA, and GRAYSCALE color formats.
    """
    
    def __init__(self, contrast_factor: float = 1.0):
        """
        Initialize the ContrastFilter.
        
        Args:
            contrast_factor: Contrast multiplier (0.0-3.0 range, 1.0 = no change)
        """
        super().__init__(
            name="contrast",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,  # Will handle RGB, RGBA, and GRAYSCALE
            category="enhancement",
            contrast_factor=contrast_factor
        )
    
    def _validate_parameters(self) -> None:
        """
        Validate filter parameters.
        
        Ensures contrast_factor value is within acceptable range (0.0-3.0).
        
        Raises:
            FilterValidationError: If contrast_factor value is invalid
        """
        contrast_factor = self.parameters.get('contrast_factor', 1.0)
        
        if not isinstance(contrast_factor, (int, float)):
            raise FilterValidationError("Contrast factor must be a numeric value")
        
        if contrast_factor < 0:
            raise FilterValidationError("Contrast factor must be non-negative")
        
        if contrast_factor < 0.0 or contrast_factor > 3.0:
            raise FilterValidationError(
                f"Contrast factor must be in range [0.0, 3.0], got {contrast_factor}"
            )
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data for ContrastFilter.
        
        Extends BaseFilter validation to ensure the input is compatible
        with contrast adjustment operations.
        
        Args:
            data: Input numpy array to validate
            
        Returns:
            True if input is valid
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Check basic requirements first
        if not isinstance(data, np.ndarray):
            raise FilterValidationError("Input must be a numpy array")
        
        if data.size == 0:
            raise FilterValidationError("Input array cannot be empty")
        
        # Validate data type and range
        self._validate_data_range(data)
        
        # Check dimensions for image data
        if data.ndim not in [2, 3]:
            raise FilterValidationError(
                f"Image data must be 2D or 3D array, got {data.ndim}D"
            )
        
        # Validate color format - accept RGB, RGBA, or grayscale
        if data.ndim == 3:
            channels = data.shape[-1]
            if channels not in [3, 4]:
                raise FilterValidationError(
                    f"Color images must have 3 (RGB) or 4 (RGBA) channels, got {channels}"
                )
        elif data.ndim == 2:
            # Grayscale is acceptable for contrast adjustment
            pass
        
        return True
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply contrast adjustment to the input image.
        
        Applies contrast scaling around the midpoint:
        output = (input - midpoint) * contrast_factor + midpoint
        
        For contrast_factor = 1.0, returns the original image unchanged (identity case).
        
        Args:
            data: Input numpy array containing image data
            **kwargs: Additional parameters (can override contrast_factor)
            
        Returns:
            Numpy array with contrast adjustment applied
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Validate input data
        self.validate_input(data)
        
        # Update parameters with any kwargs
        if kwargs:
            self.set_parameters(**kwargs)
        
        self._validate_parameters()
        
        contrast_factor = self.parameters.get('contrast_factor', 1.0)
        
        def contrast_adjustment_operation():
            # Handle identity case (contrast_factor = 1.0)
            if abs(contrast_factor - 1.0) < 1e-6:
                # Return copy to maintain consistent behavior
                return data.copy()
            
            # Determine if we should use in-place processing
            use_inplace = self._should_use_inplace(data)
            
            if use_inplace:
                # Process in-place for memory efficiency
                result = data.astype(np.float64)
            else:
                # Create a copy for processing
                result = data.astype(np.float64)
            
            # Apply contrast adjustment based on data type
            if data.dtype == np.uint8:
                # For uint8: midpoint is 128, scale around it
                midpoint = 128.0
                result = (result - midpoint) * contrast_factor + midpoint
                # Clip to valid range and convert back to uint8
                result = np.clip(result, 0, 255).astype(np.uint8)
                
            elif data.dtype in [np.float32, np.float64]:
                # For float: assume [0,1] range, midpoint is 0.5
                midpoint = 0.5
                result = (result - midpoint) * contrast_factor + midpoint
                # Clip to valid range and convert back to original float type
                result = np.clip(result, 0.0, 1.0).astype(data.dtype)
                
            else:
                # For other data types, determine midpoint and apply scaling
                if np.issubdtype(data.dtype, np.integer):
                    max_val = float(np.iinfo(data.dtype).max)
                    midpoint = max_val / 2.0
                else:
                    # For other float types, assume [0,1] range
                    max_val = 1.0
                    midpoint = 0.5
                
                # Apply contrast scaling around midpoint
                result = (result - midpoint) * contrast_factor + midpoint
                
                # Clip to valid range and convert back to original type
                if np.issubdtype(data.dtype, np.integer):
                    result = np.clip(result, 0, max_val).astype(data.dtype)
                else:
                    result = np.clip(result, 0.0, max_val).astype(data.dtype)
            
            return result
        
        # Execute with timing and progress tracking
        self._update_progress(0.0)
        result = self._measure_execution_time(contrast_adjustment_operation)
        self._update_progress(1.0)
        
        # Record metadata
        self._record_shapes(data, result)
        self._track_memory_usage(data, result, used_inplace=np.shares_memory(data, result))
        
        return result