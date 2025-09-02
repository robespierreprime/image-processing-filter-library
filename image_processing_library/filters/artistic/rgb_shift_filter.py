"""
RGB shift filter for creating chromatic aberration and glitch effects.

This module contains the RGBShiftFilter class which shifts individual RGB color
channels by specified pixel amounts to create chromatic aberration effects.
"""

import numpy as np
from ...core.base_filter import BaseFilter
from ...core.protocols import DataType, ColorFormat
from ...core.utils import FilterValidationError
from ..registry import register_filter


@register_filter(category="artistic")
class RGBShiftFilter(BaseFilter):
    """
    Filter that shifts individual RGB color channels independently.
    
    This filter creates chromatic aberration effects by translating each RGB channel
    by different amounts. This can simulate lens aberrations, create glitch effects,
    or add artistic color separation.
    
    Parameters:
        red_shift (tuple[int, int]): (x, y) pixel shift for red channel
        green_shift (tuple[int, int]): (x, y) pixel shift for green channel  
        blue_shift (tuple[int, int]): (x, y) pixel shift for blue channel
        edge_mode (str): How to handle pixels shifted outside image boundaries
                        - "clip": Use edge pixel values (default)
                        - "wrap": Wrap around to opposite edge
                        - "reflect": Mirror at edges
    """
    
    def __init__(self, red_shift: tuple[int, int] = (0, 0), 
                 green_shift: tuple[int, int] = (0, 0),
                 blue_shift: tuple[int, int] = (0, 0),
                 edge_mode: str = "clip"):
        """
        Initialize the RGBShiftFilter.
        
        Args:
            red_shift: (x, y) pixel shift for red channel (default (0, 0))
            green_shift: (x, y) pixel shift for green channel (default (0, 0))
            blue_shift: (x, y) pixel shift for blue channel (default (0, 0))
            edge_mode: Edge handling mode ("clip", "wrap", "reflect", default "clip")
        """
        super().__init__(
            name="rgb_shift",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,  # Will handle RGB and RGBA
            category="artistic",
            red_shift=red_shift,
            green_shift=green_shift,
            blue_shift=blue_shift,
            edge_mode=edge_mode
        )
    
    def _validate_parameters(self) -> None:
        """
        Validate filter parameters.
        
        Raises:
            FilterValidationError: If parameters are outside valid ranges or invalid types
        """
        red_shift = self.parameters.get('red_shift', (0, 0))
        green_shift = self.parameters.get('green_shift', (0, 0))
        blue_shift = self.parameters.get('blue_shift', (0, 0))
        edge_mode = self.parameters.get('edge_mode', 'clip')
        
        # Validate shift tuples
        for shift_name, shift_value in [('red_shift', red_shift), 
                                       ('green_shift', green_shift), 
                                       ('blue_shift', blue_shift)]:
            if not isinstance(shift_value, (tuple, list)):
                raise FilterValidationError(
                    f"{shift_name} must be a tuple or list, got {type(shift_value).__name__}"
                )
            
            if len(shift_value) != 2:
                raise FilterValidationError(
                    f"{shift_name} must have exactly 2 elements (x, y), got {len(shift_value)}"
                )
            
            x_shift, y_shift = shift_value
            if not isinstance(x_shift, int) or not isinstance(y_shift, int):
                raise FilterValidationError(
                    f"{shift_name} elements must be integers, got ({type(x_shift).__name__}, {type(y_shift).__name__})"
                )
        
        # Validate edge_mode
        valid_edge_modes = ['clip', 'wrap', 'reflect']
        if not isinstance(edge_mode, str):
            raise FilterValidationError(
                f"edge_mode must be a string, got {type(edge_mode).__name__}"
            )
        if edge_mode not in valid_edge_modes:
            raise FilterValidationError(
                f"edge_mode must be one of {valid_edge_modes}, got '{edge_mode}'"
            )
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data for RGBShiftFilter.
        
        Extends BaseFilter validation to ensure the input is compatible
        with RGB channel shifting operations.
        
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
        if data.ndim != 3:
            raise FilterValidationError(
                f"RGB shift requires 3D array (height, width, channels), got {data.ndim}D"
            )
        
        # Validate color format - must have RGB or RGBA channels
        channels = data.shape[-1]
        if channels not in [3, 4]:
            raise FilterValidationError(
                f"RGB shift requires 3 (RGB) or 4 (RGBA) channels, got {channels}"
            )
        
        return True
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply RGB channel shifting to the input image.
        
        Shifts each RGB channel independently by the specified amounts.
        
        Args:
            data: Input numpy array containing RGB or RGBA image data
            **kwargs: Additional parameters (can override shift values and edge_mode)
            
        Returns:
            Numpy array with RGB channels shifted
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Update parameters with any kwargs
        if kwargs:
            self.set_parameters(**kwargs)
        
        # Validate input data and parameters
        self.validate_input(data)
        self._validate_parameters()
        
        red_shift = self.parameters.get('red_shift', (0, 0))
        green_shift = self.parameters.get('green_shift', (0, 0))
        blue_shift = self.parameters.get('blue_shift', (0, 0))
        
        def shift_operation():
            # Handle identity case (no change needed)
            if (red_shift == (0, 0) and green_shift == (0, 0) and 
                blue_shift == (0, 0)):
                return data.copy() if not self._should_use_inplace(data) else data
            
            # Determine if we should use in-place processing
            use_inplace = self._should_use_inplace(data)
            
            if use_inplace:
                # Process in-place for memory efficiency
                result = data
            else:
                # Create a copy for processing
                result = data.copy()
            
            # Apply channel shifting
            self._apply_channel_shifts(result, red_shift, green_shift, blue_shift)
            
            return result
        
        # Execute with timing and progress tracking
        self._update_progress(0.0)
        result = self._measure_execution_time(shift_operation)
        self._update_progress(1.0)
        
        # Record metadata
        self._record_shapes(data, result)
        self._track_memory_usage(data, result, used_inplace=np.shares_memory(data, result))
        
        return result
    
    def _apply_channel_shifts(self, data: np.ndarray, red_shift: tuple[int, int],
                             green_shift: tuple[int, int], blue_shift: tuple[int, int]) -> None:
        """
        Apply independent shifts to RGB channels in-place.
        
        Args:
            data: Image data array to modify in-place (height, width, channels)
            red_shift: (x, y) shift for red channel
            green_shift: (x, y) shift for green channel  
            blue_shift: (x, y) shift for blue channel
        """
        edge_mode = self.parameters.get('edge_mode', 'clip')
        height, width = data.shape[:2]
        has_alpha = data.shape[-1] == 4
        
        # Store original channels before shifting
        original_red = data[..., 0].copy()
        original_green = data[..., 1].copy()
        original_blue = data[..., 2].copy()
        
        # Apply shifts to each channel
        self._update_progress(0.2)
        data[..., 0] = self._shift_channel(original_red, red_shift, edge_mode)
        
        self._update_progress(0.5)
        data[..., 1] = self._shift_channel(original_green, green_shift, edge_mode)
        
        self._update_progress(0.8)
        data[..., 2] = self._shift_channel(original_blue, blue_shift, edge_mode)
        
        # Alpha channel remains unchanged if present
        # (no need to modify data[..., 3] as it's already preserved)
    
    def _shift_channel(self, channel: np.ndarray, shift: tuple[int, int], 
                      edge_mode: str) -> np.ndarray:
        """
        Shift a single channel by the specified amount.
        
        Args:
            channel: 2D array representing a single color channel
            shift: (x, y) shift amounts in pixels
            edge_mode: How to handle edge pixels ("clip", "wrap", "reflect")
            
        Returns:
            Shifted channel array
        """
        x_shift, y_shift = shift
        
        # Handle no-shift case
        if x_shift == 0 and y_shift == 0:
            return channel
        
        height, width = channel.shape
        shifted = np.zeros_like(channel)
        
        # Create coordinate grids for the shifted positions
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Apply the shift (we need to shift the source coordinates in the opposite direction)
        new_y = y_coords - y_shift
        new_x = x_coords - x_shift
        
        # Handle edge cases based on edge_mode
        if edge_mode == "clip":
            # Clamp coordinates to valid range
            new_y = np.clip(new_y, 0, height - 1)
            new_x = np.clip(new_x, 0, width - 1)
        elif edge_mode == "wrap":
            # Wrap coordinates around
            new_y = new_y % height
            new_x = new_x % width
        elif edge_mode == "reflect":
            # Reflect coordinates at boundaries
            new_y = self._reflect_coordinates(new_y, height)
            new_x = self._reflect_coordinates(new_x, width)
        
        # Sample from the original channel at the new coordinates
        shifted = channel[new_y, new_x]
        
        return shifted
    
    def _reflect_coordinates(self, coords: np.ndarray, max_val: int) -> np.ndarray:
        """
        Reflect coordinates at boundaries.
        
        Args:
            coords: Array of coordinates to reflect
            max_val: Maximum valid coordinate value (exclusive)
            
        Returns:
            Reflected coordinates
        """
        # Handle negative coordinates
        coords = np.abs(coords)
        
        # Handle coordinates >= max_val by reflecting
        period = 2 * max_val
        coords = coords % period
        
        # Reflect coordinates in the second half of the period
        mask = coords >= max_val
        coords[mask] = period - coords[mask] - 1
        
        return coords