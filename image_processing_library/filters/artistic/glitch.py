"""
Glitch effect filter for creating digital artifacts and distortions.

This module provides a glitch effect filter that creates digital artifacts
through pixel shifting, color channel manipulation, and JPEG compression.

Note: This implementation was developed with assistance from Claude.
"""

import numpy as np
import random
import io
from PIL import Image
from typing import Dict, Any, Optional

from ...core.base_filter import BaseFilter
from ...core.protocols import DataType, ColorFormat
from ...core.utils import FilterExecutionError
from ..registry import register_filter
from .rgb_shift_filter import RGBShiftFilter


@register_filter(category="artistic")
class GlitchFilter(BaseFilter):
    """
    Digital glitch effect filter creating pixel shifts and compression artifacts.
    
    This filter applies various glitch effects including:
    - Angled pixel shifting along lines
    - Color channel shifting for chromatic aberration
    - JPEG compression artifacts
    
    The filter supports customizable parameters for controlling the intensity
    and characteristics of the glitch effects.
    """
    
    def __init__(self, 
                 shift_intensity: int = None,
                 line_width: int = 3,
                 glitch_probability: float = None,
                 jpeg_quality: int = 30,
                 shift_angle: float = 0.0,
                 # Legacy parameter names for backward compatibility
                 intensity: float = None,
                 shift_amount: int = None,
                 corruption_probability: float = None,
                 **kwargs):
        """
        Initialize the glitch filter with effect parameters.
        
        Args:
            shift_intensity: Maximum pixel shift distance (0-100)
            line_width: Width of glitch lines in pixels (1-20)
            glitch_probability: Probability of applying glitch to each line (0.0-1.0)
            jpeg_quality: JPEG compression quality for artifacts (1-100)
            shift_angle: Angle of shift lines in degrees (0-360)
            
            # Legacy parameters (for backward compatibility):
            intensity: Alternative to shift_intensity (0.0-1.0, scaled to 0-100)
            shift_amount: Alternative to shift_intensity (0-100)
            corruption_probability: Alternative to glitch_probability (0.0-1.0)
            
            **kwargs: Additional parameters passed to BaseFilter
        """
        
        # Handle parameter mapping for backward compatibility
        final_shift_intensity = self._resolve_shift_intensity(
            shift_intensity, intensity, shift_amount
        )
        final_glitch_probability = self._resolve_glitch_probability(
            glitch_probability, corruption_probability
        )
        super().__init__(
            name="Glitch Effect",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="artistic",
            shift_intensity=final_shift_intensity,
            line_width=line_width,
            glitch_probability=final_glitch_probability,
            jpeg_quality=jpeg_quality,
            shift_angle=shift_angle,
            **kwargs
        )
        
        # Validate parameters
        self._validate_parameters()
        
        # Initialize RGB shift filter for color channel shifting
        self._rgb_shift_filter = RGBShiftFilter()
    
    def _resolve_shift_intensity(self, shift_intensity, intensity, shift_amount):
        """Resolve shift_intensity from multiple possible parameter sources."""
        # Priority: shift_intensity > shift_amount > intensity > default
        if shift_intensity is not None:
            return shift_intensity
        elif shift_amount is not None:
            return shift_amount
        elif intensity is not None:
            # Convert intensity (0.0-1.0) to shift_intensity (0-100)
            return int(intensity * 100)
        else:
            return 10  # Default value
    
    def _resolve_glitch_probability(self, glitch_probability, corruption_probability):
        """Resolve glitch_probability from multiple possible parameter sources."""
        # Priority: glitch_probability > corruption_probability > default
        if glitch_probability is not None:
            return glitch_probability
        elif corruption_probability is not None:
            return corruption_probability
        else:
            return 0.8  # Default value
    
    def _validate_parameters(self) -> None:
        """Validate filter parameters are within acceptable ranges."""
        params = self.parameters
        
        if not (0 <= params['shift_intensity'] <= 100):
            raise ValueError("shift_intensity must be between 0 and 100")
        
        if not (1 <= params['line_width'] <= 20):
            raise ValueError("line_width must be between 1 and 20")
        
        if not (0.0 <= params['glitch_probability'] <= 1.0):
            raise ValueError("glitch_probability must be between 0.0 and 1.0")
        
        if not (1 <= params['jpeg_quality'] <= 100):
            raise ValueError("jpeg_quality must be between 1 and 100")
        
        if not (0.0 <= params['shift_angle'] <= 360.0):
            raise ValueError("shift_angle must be between 0.0 and 360.0")
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply glitch effects to the input image data.
        
        Args:
            data: Input image as numpy array (H, W, 3) with RGB values 0-255
            **kwargs: Additional parameters (merged with instance parameters)
            
        Returns:
            Processed image array with glitch effects applied
            
        Raises:
            FilterValidationError: If input data is invalid
            FilterExecutionError: If processing fails
        """
        # Validate input
        self.validate_input(data)
        
        # Update parameters with any provided kwargs
        old_params = None
        if kwargs:
            old_params = self.parameters.copy()
            try:
                self.set_parameters(**kwargs)
            except ValueError as e:
                # Restore old parameters if validation fails
                self.parameters = old_params
                raise FilterExecutionError(f"Invalid parameters: {e}")
        
        # Record memory usage
        self.metadata.memory_usage = self._estimate_memory_usage(data)
        
        def process():
            self._update_progress(0.0)
            
            # Ensure data is in correct format (uint8, 0-255 range)
            if data.dtype != np.uint8:
                if data.dtype in [np.float32, np.float64]:
                    # Assume float data is in [0, 1] range
                    processed_data = (data * 255).astype(np.uint8)
                else:
                    processed_data = data.astype(np.uint8)
            else:
                processed_data = data.copy()
            
            self._update_progress(0.1)
            
            # Apply angled shift glitch
            processed_data = self._angled_shift_glitch(processed_data)
            self._update_progress(0.5)
            
            # Apply color channel shift
            processed_data = self._color_channel_shift(processed_data)
            self._update_progress(0.8)
            
            # Apply JPEG compression artifacts
            processed_data = self._jpeg_corruption(processed_data)
            self._update_progress(1.0)
            
            return processed_data
        
        try:
            # Execute with timing measurement
            result = self._measure_execution_time(process)
            
            # Record shapes
            self._record_shapes(data, result)
            
            return result
        finally:
            # Restore original parameters if they were changed
            if old_params is not None:
                self.parameters = old_params
    
    def _angled_shift_glitch(self, image_array: np.ndarray) -> np.ndarray:
        """Apply pixel shifting along angled lines."""
        result = image_array.copy()
        height, width = image_array.shape[:2]
        
        shift_intensity = self.parameters['shift_intensity']
        line_width = self.parameters['line_width']
        glitch_probability = self.parameters['glitch_probability']
        shift_angle = self.parameters['shift_angle']
        
        # Skip if no shift intensity
        if shift_intensity == 0:
            return result
        
        # Convert angle to radians
        angle_rad = np.radians(shift_angle)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        # If angle is close to 0, use optimized horizontal shifts
        if abs(shift_angle) < 0.1:
            return self._horizontal_shift_simple(image_array)
        
        # Ensure minimum line width
        effective_line_width = max(1, line_width)
        
        # For angled shifts, process along oriented lines
        for line_idx in range(0, max(height, width), effective_line_width):
            if random.random() < glitch_probability:
                shift_amount = random.randint(-shift_intensity, shift_intensity)
                
                # Skip if no actual shift
                if shift_amount == 0:
                    continue
                
                # Calculate line positions
                for pos in range(max(height, width)):
                    # Starting position on the line
                    if abs(dx) > abs(dy):  # More horizontal than vertical
                        x = pos
                        y = (
                            int(line_idx + (pos - width // 2) * dy / dx)
                            if dx != 0
                            else line_idx
                        )
                    else:  # More vertical than horizontal
                        y = pos
                        x = (
                            int(line_idx + (pos - height // 2) * dx / dy)
                            if dy != 0
                            else line_idx
                        )
                    
                    # Calculate shifted position
                    shift_x = int(x + shift_amount * dx)
                    shift_y = int(y + shift_amount * dy)
                    
                    # Apply shift if both positions are valid
                    if (
                        0 <= x < width
                        and 0 <= y < height
                        and 0 <= shift_x < width
                        and 0 <= shift_y < height
                    ):
                        # Apply line thickness
                        for thickness in range(effective_line_width):
                            thick_offset = thickness - effective_line_width // 2
                            
                            # Perpendicular direction for thickness
                            perp_x = int(x - thick_offset * dy)
                            perp_y = int(y + thick_offset * dx)
                            perp_shift_x = int(shift_x - thick_offset * dy)
                            perp_shift_y = int(shift_y + thick_offset * dx)
                            
                            if (
                                0 <= perp_x < width
                                and 0 <= perp_y < height
                                and 0 <= perp_shift_x < width
                                and 0 <= perp_shift_y < height
                            ):
                                result[perp_shift_y, perp_shift_x] = image_array[
                                    perp_y, perp_x
                                ]
        
        return result
    
    def _horizontal_shift_simple(self, image_array: np.ndarray) -> np.ndarray:
        """Optimized horizontal shifting when angle is close to 0."""
        result = image_array.copy()
        height, width = image_array.shape[:2]
        
        shift_intensity = self.parameters['shift_intensity']
        line_width = self.parameters['line_width']
        glitch_probability = self.parameters['glitch_probability']
        
        # Skip if no shift intensity
        if shift_intensity == 0:
            return result
        
        # Ensure minimum line width
        effective_line_width = max(1, line_width)
        
        # Vectorized approach for better performance
        for y in range(0, height, effective_line_width):
            if random.random() < glitch_probability:
                shift = random.randint(-shift_intensity, shift_intensity)
                
                # Skip if no actual shift
                if shift == 0:
                    continue
                
                # Process multiple rows at once
                end_row = min(y + effective_line_width, height)
                rows_to_process = slice(y, end_row)
                
                if shift > 0:
                    result[rows_to_process, shift:] = image_array[
                        rows_to_process, :-shift
                    ]
                elif shift < 0:
                    result[rows_to_process, :shift] = image_array[
                        rows_to_process, -shift:
                    ]
        
        return result
    
    def _color_channel_shift(self, image_array: np.ndarray) -> np.ndarray:
        """Shift individual color channels for chromatic aberration effect using RGBShiftFilter."""
        # Generate random shifts for each channel (maintaining original behavior)
        red_shift = (0, 0)
        green_shift = (0, 0)
        blue_shift = (0, 0)
        
        # 60% chance per channel to apply shift (original behavior)
        if random.random() < 0.6:
            red_shift = (random.randint(-3, 3), 0)
        
        if random.random() < 0.6:
            green_shift = (random.randint(-3, 3), 0)
        
        if random.random() < 0.6:
            blue_shift = (random.randint(-3, 3), 0)
        
        # Use RGBShiftFilter to apply the shifts
        return self._rgb_shift_filter.apply(
            image_array,
            red_shift=red_shift,
            green_shift=green_shift,
            blue_shift=blue_shift,
            edge_mode="clip"  # Use clip mode to match original behavior
        )
    
    def _jpeg_corruption(self, image_array: np.ndarray) -> np.ndarray:
        """Apply JPEG compression artifacts."""
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_array.astype(np.uint8))
            
            # Apply JPEG compression
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self.parameters['jpeg_quality'])
            buffer.seek(0)
            compressed_image = Image.open(buffer)
            
            # Convert back to numpy array
            return np.array(compressed_image)
            
        except Exception as e:
            raise FilterExecutionError(f"JPEG corruption failed: {e}")
    
    def set_parameters(self, **kwargs) -> None:
        """
        Update filter parameters with validation.
        
        Args:
            **kwargs: Parameter names and values to update
            
        Raises:
            ValueError: If invalid parameter names or values provided
        """
        # Validate parameter names (including legacy names)
        valid_params = {
            'shift_intensity', 'line_width', 'glitch_probability', 
            'jpeg_quality', 'shift_angle',
            # Legacy parameter names
            'intensity', 'shift_amount', 'corruption_probability'
        }
        
        invalid_params = set(kwargs.keys()) - valid_params
        if invalid_params:
            raise ValueError(f"Invalid parameters: {invalid_params}")
        
        # Handle parameter mapping for legacy names
        mapped_kwargs = {}
        
        # Extract legacy parameters
        intensity = kwargs.pop('intensity', None)
        shift_amount = kwargs.pop('shift_amount', None)
        corruption_probability = kwargs.pop('corruption_probability', None)
        
        # Map legacy parameters to current parameters
        if intensity is not None or shift_amount is not None:
            current_shift_intensity = kwargs.get('shift_intensity', self.parameters.get('shift_intensity'))
            mapped_kwargs['shift_intensity'] = self._resolve_shift_intensity(
                kwargs.get('shift_intensity'), intensity, shift_amount
            )
        
        if corruption_probability is not None:
            current_glitch_probability = kwargs.get('glitch_probability', self.parameters.get('glitch_probability'))
            mapped_kwargs['glitch_probability'] = self._resolve_glitch_probability(
                kwargs.get('glitch_probability'), corruption_probability
            )
        
        # Merge mapped parameters with remaining kwargs
        final_kwargs = {**kwargs, **mapped_kwargs}
        
        # Update parameters
        super().set_parameters(**final_kwargs)
        
        # Validate new parameter values
        self._validate_parameters()