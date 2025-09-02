"""
Color manipulation filters for basic color adjustments.

This module contains filters for fundamental color operations:
- InvertFilter: Inverts RGB color values
- HueRotationFilter: Rotates hue values in HSV color space
- SaturationFilter: Adjusts color saturation
"""

import numpy as np
import colorsys
from ...core.base_filter import BaseFilter
from ...core.protocols import DataType, ColorFormat
from ...core.utils import FilterValidationError
from ..registry import register_filter


@register_filter(category="enhancement")
class InvertFilter(BaseFilter):
    """
    Filter that inverts RGB color values.
    
    This filter inverts all RGB color values by applying the transformation:
    output = 255 - input (for uint8 images) or output = 1.0 - input (for float images).
    Alpha channels are preserved unchanged for RGBA images.
    
    The filter has no parameters and works with both RGB and RGBA color formats.
    """
    
    def __init__(self):
        """
        Initialize the InvertFilter.
        
        This filter requires no parameters and supports both RGB and RGBA formats.
        """
        super().__init__(
            name="invert",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,  # Will handle both RGB and RGBA
            category="enhancement"
        )
    
    def _validate_parameters(self) -> None:
        """
        Validate filter parameters.
        
        Since InvertFilter has no parameters, this method performs no validation
        but is included for consistency with the BaseFilter interface.
        """
        # No parameters to validate for InvertFilter
        pass
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data for InvertFilter.
        
        Extends BaseFilter validation to ensure the input is compatible
        with color inversion operations.
        
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
            # Grayscale is acceptable for inversion
            pass
        
        return True
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply color inversion to the input image.
        
        Inverts RGB color values using the transformation:
        - For uint8 images: output = 255 - input
        - For float images: output = max_value - input (where max_value is 1.0 for [0,1] range)
        
        Alpha channels in RGBA images are preserved unchanged.
        
        Args:
            data: Input numpy array containing image data
            **kwargs: Additional parameters (unused for InvertFilter)
            
        Returns:
            Numpy array with inverted colors
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Validate input data
        self.validate_input(data)
        self._validate_parameters()
        
        def invert_operation():
            # Determine if we should use in-place processing
            use_inplace = self._should_use_inplace(data)
            
            if use_inplace:
                # Process in-place for memory efficiency
                result = data
            else:
                # Create a copy for processing
                result = data.copy()
            
            # Apply inversion based on data type
            if data.dtype == np.uint8:
                # For uint8: invert using 255 - value
                if data.ndim == 3 and data.shape[-1] == 4:
                    # RGBA: preserve alpha channel
                    result[..., :3] = 255 - result[..., :3]
                else:
                    # RGB or grayscale: invert all channels
                    result = 255 - result
            elif data.dtype in [np.float32, np.float64]:
                # For float: assume [0, 1] range and invert using 1.0 - value
                if data.ndim == 3 and data.shape[-1] == 4:
                    # RGBA: preserve alpha channel
                    result[..., :3] = 1.0 - result[..., :3]
                else:
                    # RGB or grayscale: invert all channels
                    result = 1.0 - result
            else:
                # For other data types, try to determine max value and invert
                if np.issubdtype(data.dtype, np.integer):
                    max_val = np.iinfo(data.dtype).max
                else:
                    # For other float types, assume [0, 1] range
                    max_val = 1.0
                
                if data.ndim == 3 and data.shape[-1] == 4:
                    # RGBA: preserve alpha channel
                    result[..., :3] = max_val - result[..., :3]
                else:
                    # RGB or grayscale: invert all channels
                    result = max_val - result
            
            return result
        
        # Execute with timing and progress tracking
        self._update_progress(0.0)
        result = self._measure_execution_time(invert_operation)
        self._update_progress(1.0)
        
        # Record metadata
        self._record_shapes(data, result)
        self._track_memory_usage(data, result, used_inplace=np.shares_memory(data, result))
        
        return result


@register_filter(category="enhancement")
class SaturationFilter(BaseFilter):
    """
    Filter that adjusts color saturation in HSV color space.
    
    This filter adjusts the saturation component of colors by converting RGB to HSV,
    multiplying the saturation channel by the specified factor, and converting back to RGB.
    Alpha channels are preserved unchanged for RGBA images.
    
    Parameters:
        saturation_factor (float): Saturation multiplier (0.0-3.0)
                                 - 0.0 = grayscale (no saturation)
                                 - 1.0 = original saturation (no change)
                                 - >1.0 = increased saturation
    """
    
    def __init__(self, saturation_factor: float = 1.0):
        """
        Initialize the SaturationFilter.
        
        Args:
            saturation_factor: Saturation multiplier (0.0-3.0, default 1.0)
        """
        super().__init__(
            name="saturation",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,  # Will handle both RGB and RGBA
            category="enhancement",
            saturation_factor=saturation_factor
        )
    
    def _validate_parameters(self) -> None:
        """
        Validate filter parameters.
        
        Raises:
            FilterValidationError: If saturation_factor is outside valid range
        """
        saturation_factor = self.parameters.get('saturation_factor', 1.0)
        
        if not isinstance(saturation_factor, (int, float)):
            raise FilterValidationError(
                f"saturation_factor must be a number, got {type(saturation_factor).__name__}"
            )
        
        if saturation_factor < 0.0 or saturation_factor > 3.0:
            raise FilterValidationError(
                f"saturation_factor must be in range [0.0, 3.0], got {saturation_factor}"
            )
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data for SaturationFilter.
        
        Extends BaseFilter validation to ensure the input is compatible
        with HSV color space operations.
        
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
            # Grayscale images have no saturation to adjust - will return unchanged
            pass
        
        return True
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply saturation adjustment to the input image.
        
        Adjusts color saturation by converting RGB to HSV, multiplying the saturation
        channel by the specified factor, and converting back to RGB.
        
        Args:
            data: Input numpy array containing image data
            **kwargs: Additional parameters (can override saturation_factor)
            
        Returns:
            Numpy array with adjusted saturation
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Update parameters with any kwargs
        if kwargs:
            self.set_parameters(**kwargs)
        
        # Validate input data and parameters
        self.validate_input(data)
        self._validate_parameters()
        
        saturation_factor = self.parameters.get('saturation_factor', 1.0)
        
        def saturation_operation():
            # Handle identity case (no change needed)
            if saturation_factor == 1.0:
                return data.copy() if not self._should_use_inplace(data) else data
            
            # Handle grayscale images (no saturation to adjust)
            if data.ndim == 2:
                return data.copy() if not self._should_use_inplace(data) else data
            
            # Determine if we should use in-place processing
            use_inplace = self._should_use_inplace(data)
            
            if use_inplace:
                # Process in-place for memory efficiency
                result = data
            else:
                # Create a copy for processing
                result = data.copy()
            
            # Extract RGB channels (preserve alpha if present)
            has_alpha = data.ndim == 3 and data.shape[-1] == 4
            if has_alpha:
                rgb_data = result[..., :3]
                alpha_channel = result[..., 3:4]  # Keep as 3D for broadcasting
            else:
                rgb_data = result
            
            # Convert to HSV and adjust saturation
            self._adjust_saturation_hsv(rgb_data, saturation_factor)
            
            return result
        
        # Execute with timing and progress tracking
        self._update_progress(0.0)
        result = self._measure_execution_time(saturation_operation)
        self._update_progress(1.0)
        
        # Record metadata
        self._record_shapes(data, result)
        self._track_memory_usage(data, result, used_inplace=np.shares_memory(data, result))
        
        return result
    
    def _adjust_saturation_hsv(self, rgb_data: np.ndarray, saturation_factor: float) -> None:
        """
        Adjust saturation by converting to HSV and modifying the S channel.
        
        This method modifies the input array in-place for memory efficiency.
        Uses a vectorized approach for better performance.
        
        Args:
            rgb_data: RGB data array to modify (shape: [..., 3])
            saturation_factor: Factor to multiply saturation by
        """
        # Get original shape and flatten for processing
        original_shape = rgb_data.shape
        flat_rgb = rgb_data.reshape(-1, 3)
        
        # Normalize to [0, 1] range for HSV conversion
        if rgb_data.dtype == np.uint8:
            rgb_normalized = flat_rgb.astype(np.float32) / 255.0
        else:
            rgb_normalized = flat_rgb.astype(np.float32)
            # Ensure values are in [0, 1] range
            rgb_normalized = np.clip(rgb_normalized, 0.0, 1.0)
        
        # Vectorized RGB to HSV conversion
        hsv_data = self._rgb_to_hsv_vectorized(rgb_normalized)
        
        # Adjust saturation channel and clamp to [0, 1]
        hsv_data[:, 1] = np.clip(hsv_data[:, 1] * saturation_factor, 0.0, 1.0)
        
        # Convert back to RGB
        rgb_result = self._hsv_to_rgb_vectorized(hsv_data)
        
        # Convert back to original data type and range
        if rgb_data.dtype == np.uint8:
            rgb_result = (rgb_result * 255.0).astype(np.uint8)
        else:
            rgb_result = rgb_result.astype(rgb_data.dtype)
        
        # Reshape back to original shape and update the input array
        rgb_data[...] = rgb_result.reshape(original_shape)
    
    def _rgb_to_hsv_vectorized(self, rgb: np.ndarray) -> np.ndarray:
        """
        Vectorized RGB to HSV conversion.
        
        Args:
            rgb: RGB array of shape (N, 3) with values in [0, 1]
            
        Returns:
            HSV array of shape (N, 3) with values in [0, 1]
        """
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        # Value channel
        v = max_val
        
        # Saturation channel
        s = np.where(max_val != 0, diff / max_val, 0)
        
        # Hue channel
        h = np.zeros_like(max_val)
        
        # Avoid division by zero
        mask = diff != 0
        
        # Red is max
        red_max = mask & (max_val == r)
        h[red_max] = (60 * ((g[red_max] - b[red_max]) / diff[red_max]) + 360) % 360
        
        # Green is max
        green_max = mask & (max_val == g)
        h[green_max] = (60 * ((b[green_max] - r[green_max]) / diff[green_max]) + 120) % 360
        
        # Blue is max
        blue_max = mask & (max_val == b)
        h[blue_max] = (60 * ((r[blue_max] - g[blue_max]) / diff[blue_max]) + 240) % 360
        
        # Normalize hue to [0, 1]
        h = h / 360.0
        
        return np.column_stack([h, s, v])
    
    def _hsv_to_rgb_vectorized(self, hsv: np.ndarray) -> np.ndarray:
        """
        Vectorized HSV to RGB conversion.
        
        Args:
            hsv: HSV array of shape (N, 3) with values in [0, 1]
            
        Returns:
            RGB array of shape (N, 3) with values in [0, 1]
        """
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        
        # Convert hue from [0, 1] to [0, 360]
        h = h * 360.0
        
        c = v * s
        x = c * (1 - np.abs((h / 60) % 2 - 1))
        m = v - c
        
        # Initialize RGB arrays
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)
        
        # Determine RGB values based on hue sector
        sector_0 = (0 <= h) & (h < 60)
        sector_1 = (60 <= h) & (h < 120)
        sector_2 = (120 <= h) & (h < 180)
        sector_3 = (180 <= h) & (h < 240)
        sector_4 = (240 <= h) & (h < 300)
        sector_5 = (300 <= h) & (h < 360)
        
        r[sector_0] = c[sector_0]
        g[sector_0] = x[sector_0]
        b[sector_0] = 0
        
        r[sector_1] = x[sector_1]
        g[sector_1] = c[sector_1]
        b[sector_1] = 0
        
        r[sector_2] = 0
        g[sector_2] = c[sector_2]
        b[sector_2] = x[sector_2]
        
        r[sector_3] = 0
        g[sector_3] = x[sector_3]
        b[sector_3] = c[sector_3]
        
        r[sector_4] = x[sector_4]
        g[sector_4] = 0
        b[sector_4] = c[sector_4]
        
        r[sector_5] = c[sector_5]
        g[sector_5] = 0
        b[sector_5] = x[sector_5]
        
        # Add the offset
        r += m
        g += m
        b += m
        
        return np.column_stack([r, g, b])


@register_filter(category="enhancement")
class HueRotationFilter(BaseFilter):
    """
    Filter that rotates hue values in HSV color space.
    
    This filter rotates the hue component of colors by converting RGB to HSV,
    adding the specified rotation angle to the hue channel (with wraparound),
    and converting back to RGB. Alpha channels are preserved unchanged for RGBA images.
    
    Parameters:
        rotation_degrees (float): Hue rotation angle in degrees (0-360)
                                - 0 = no rotation (no change)
                                - 180 = opposite colors
                                - 360 = full rotation (same as 0)
    """
    
    def __init__(self, rotation_degrees: float = 0.0):
        """
        Initialize the HueRotationFilter.
        
        Args:
            rotation_degrees: Hue rotation angle in degrees (0-360, default 0.0)
        """
        super().__init__(
            name="hue_rotation",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,  # Will handle both RGB and RGBA
            category="enhancement",
            rotation_degrees=rotation_degrees
        )
    
    def _validate_parameters(self) -> None:
        """
        Validate filter parameters.
        
        Raises:
            FilterValidationError: If rotation_degrees is outside valid range
        """
        rotation_degrees = self.parameters.get('rotation_degrees', 0.0)
        
        if not isinstance(rotation_degrees, (int, float)):
            raise FilterValidationError(
                f"rotation_degrees must be a number, got {type(rotation_degrees).__name__}"
            )
        
        if rotation_degrees < 0.0 or rotation_degrees > 360.0:
            raise FilterValidationError(
                f"rotation_degrees must be in range [0.0, 360.0], got {rotation_degrees}"
            )
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data for HueRotationFilter.
        
        Extends BaseFilter validation to ensure the input is compatible
        with HSV color space operations.
        
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
            # Grayscale images have no hue to rotate - will return unchanged
            pass
        
        return True
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply hue rotation to the input image.
        
        Rotates hue values by converting RGB to HSV, adding the rotation angle
        to the hue channel (with wraparound), and converting back to RGB.
        
        Args:
            data: Input numpy array containing image data
            **kwargs: Additional parameters (can override rotation_degrees)
            
        Returns:
            Numpy array with rotated hue values
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Update parameters with any kwargs
        if kwargs:
            self.set_parameters(**kwargs)
        
        # Validate input data and parameters
        self.validate_input(data)
        self._validate_parameters()
        
        rotation_degrees = self.parameters.get('rotation_degrees', 0.0)
        
        def hue_rotation_operation():
            # Handle identity case (no change needed)
            if rotation_degrees == 0.0 or rotation_degrees == 360.0:
                return data.copy() if not self._should_use_inplace(data) else data
            
            # Handle grayscale images (no hue to rotate)
            if data.ndim == 2:
                return data.copy() if not self._should_use_inplace(data) else data
            
            # Determine if we should use in-place processing
            use_inplace = self._should_use_inplace(data)
            
            if use_inplace:
                # Process in-place for memory efficiency
                result = data
            else:
                # Create a copy for processing
                result = data.copy()
            
            # Extract RGB channels (preserve alpha if present)
            has_alpha = data.ndim == 3 and data.shape[-1] == 4
            if has_alpha:
                rgb_data = result[..., :3]
                alpha_channel = result[..., 3:4]  # Keep as 3D for broadcasting
            else:
                rgb_data = result
            
            # Convert to HSV and rotate hue
            self._rotate_hue_hsv(rgb_data, rotation_degrees)
            
            return result
        
        # Execute with timing and progress tracking
        self._update_progress(0.0)
        result = self._measure_execution_time(hue_rotation_operation)
        self._update_progress(1.0)
        
        # Record metadata
        self._record_shapes(data, result)
        self._track_memory_usage(data, result, used_inplace=np.shares_memory(data, result))
        
        return result
    
    def _rotate_hue_hsv(self, rgb_data: np.ndarray, rotation_degrees: float) -> None:
        """
        Rotate hue by converting to HSV and modifying the H channel.
        
        This method modifies the input array in-place for memory efficiency.
        Uses a vectorized approach for better performance.
        
        Args:
            rgb_data: RGB data array to modify (shape: [..., 3])
            rotation_degrees: Degrees to rotate hue by (0-360)
        """
        # Get original shape and flatten for processing
        original_shape = rgb_data.shape
        flat_rgb = rgb_data.reshape(-1, 3)
        
        # Normalize to [0, 1] range for HSV conversion
        if rgb_data.dtype == np.uint8:
            rgb_normalized = flat_rgb.astype(np.float32) / 255.0
        else:
            rgb_normalized = flat_rgb.astype(np.float32)
            # Ensure values are in [0, 1] range
            rgb_normalized = np.clip(rgb_normalized, 0.0, 1.0)
        
        # Vectorized RGB to HSV conversion
        hsv_data = self._rgb_to_hsv_vectorized(rgb_normalized)
        
        # Rotate hue channel with wraparound
        # Convert rotation from degrees to [0, 1] range
        rotation_normalized = rotation_degrees / 360.0
        hsv_data[:, 0] = (hsv_data[:, 0] + rotation_normalized) % 1.0
        
        # Convert back to RGB
        rgb_result = self._hsv_to_rgb_vectorized(hsv_data)
        
        # Convert back to original data type and range
        if rgb_data.dtype == np.uint8:
            rgb_result = (rgb_result * 255.0).astype(np.uint8)
        else:
            rgb_result = rgb_result.astype(rgb_data.dtype)
        
        # Reshape back to original shape and update the input array
        rgb_data[...] = rgb_result.reshape(original_shape)
    
    def _rgb_to_hsv_vectorized(self, rgb: np.ndarray) -> np.ndarray:
        """
        Vectorized RGB to HSV conversion.
        
        Args:
            rgb: RGB array of shape (N, 3) with values in [0, 1]
            
        Returns:
            HSV array of shape (N, 3) with values in [0, 1]
        """
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        # Value channel
        v = max_val
        
        # Saturation channel
        s = np.where(max_val != 0, diff / max_val, 0)
        
        # Hue channel
        h = np.zeros_like(max_val)
        
        # Avoid division by zero
        mask = diff != 0
        
        # Red is max
        red_max = mask & (max_val == r)
        h[red_max] = (60 * ((g[red_max] - b[red_max]) / diff[red_max]) + 360) % 360
        
        # Green is max
        green_max = mask & (max_val == g)
        h[green_max] = (60 * ((b[green_max] - r[green_max]) / diff[green_max]) + 120) % 360
        
        # Blue is max
        blue_max = mask & (max_val == b)
        h[blue_max] = (60 * ((r[blue_max] - g[blue_max]) / diff[blue_max]) + 240) % 360
        
        # Normalize hue to [0, 1]
        h = h / 360.0
        
        return np.column_stack([h, s, v])
    
    def _hsv_to_rgb_vectorized(self, hsv: np.ndarray) -> np.ndarray:
        """
        Vectorized HSV to RGB conversion.
        
        Args:
            hsv: HSV array of shape (N, 3) with values in [0, 1]
            
        Returns:
            RGB array of shape (N, 3) with values in [0, 1]
        """
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        
        # Convert hue from [0, 1] to [0, 360]
        h = h * 360.0
        
        c = v * s
        x = c * (1 - np.abs((h / 60) % 2 - 1))
        m = v - c
        
        # Initialize RGB arrays
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)
        
        # Determine RGB values based on hue sector
        sector_0 = (0 <= h) & (h < 60)
        sector_1 = (60 <= h) & (h < 120)
        sector_2 = (120 <= h) & (h < 180)
        sector_3 = (180 <= h) & (h < 240)
        sector_4 = (240 <= h) & (h < 300)
        sector_5 = (300 <= h) & (h < 360)
        
        r[sector_0] = c[sector_0]
        g[sector_0] = x[sector_0]
        b[sector_0] = 0
        
        r[sector_1] = x[sector_1]
        g[sector_1] = c[sector_1]
        b[sector_1] = 0
        
        r[sector_2] = 0
        g[sector_2] = c[sector_2]
        b[sector_2] = x[sector_2]
        
        r[sector_3] = 0
        g[sector_3] = x[sector_3]
        b[sector_3] = c[sector_3]
        
        r[sector_4] = x[sector_4]
        g[sector_4] = 0
        b[sector_4] = c[sector_4]
        
        r[sector_5] = c[sector_5]
        g[sector_5] = 0
        b[sector_5] = x[sector_5]
        
        # Add the offset
        r += m
        g += m
        b += m
        
        return np.column_stack([r, g, b])