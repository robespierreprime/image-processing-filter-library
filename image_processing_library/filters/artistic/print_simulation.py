"""
Print simulation filter for creating printing artifacts and defects.

This module provides a filter that simulates various printing defects including
horizontal banding, noise, and contrast degradation commonly found in
printed materials.
"""

import numpy as np
import random
from PIL import Image, ImageEnhance
from typing import Dict, Any

from ...core.base_filter import BaseFilter
from ...core.protocols import DataType, ColorFormat
from ...core.utils import FilterExecutionError
from ..registry import register_filter
from .noise_filter import NoiseFilter


@register_filter(category="artistic")
class PrintSimulationFilter(BaseFilter):
    """
    Print simulation filter creating realistic printing artifacts.
    
    This filter simulates common printing defects including:
    - Horizontal banding from print head inconsistencies
    - Random noise from paper texture and ink variations
    - Contrast degradation from printing process limitations
    
    The filter supports both grayscale and color images, converting
    color images to grayscale for processing then converting back.
    """
    
    def __init__(self,
                 band_intensity: int = None,
                 band_frequency: int = 30,
                 noise_level: int = None,
                 contrast_factor: float = 0.85,
                 # Legacy/documented parameter names
                 dot_gain: float = None,
                 paper_texture: float = None,
                 ink_bleeding: float = None,
                 **kwargs):
        """
        Initialize the print simulation filter with defect parameters.
        
        Args:
            band_intensity: Intensity of horizontal bands (0-100)
            band_frequency: Spacing between bands in pixels (5-100)
            noise_level: Level of random noise (0-50)
            contrast_factor: Contrast reduction factor (0.1-1.0)
            
            # Documented parameters (mapped to internal parameters):
            dot_gain: Dot gain simulation intensity (0.0-0.5) -> maps to band_intensity
            paper_texture: Paper texture simulation intensity (0.0-1.0) -> maps to noise_level
            ink_bleeding: Ink bleeding effect intensity (0.0-1.0) -> affects contrast_factor
            
            **kwargs: Additional parameters passed to BaseFilter
        """
        
        # Handle parameter mapping for documented API
        final_band_intensity = self._resolve_band_intensity(band_intensity, dot_gain)
        final_noise_level = self._resolve_noise_level(noise_level, paper_texture)
        final_contrast_factor = self._resolve_contrast_factor(contrast_factor, ink_bleeding)
        super().__init__(
            name="Print Simulation",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,  # Accepts RGB but can process grayscale
            category="artistic",
            band_intensity=final_band_intensity,
            band_frequency=band_frequency,
            noise_level=final_noise_level,
            contrast_factor=final_contrast_factor,
            **kwargs
        )
        
        # Initialize internal NoiseFilter for noise generation
        self._noise_filter = NoiseFilter(noise_type="uniform", intensity=0.0)
        
        # Validate parameters
        self._validate_parameters()
    
    def _resolve_band_intensity(self, band_intensity, dot_gain):
        """Resolve band_intensity from multiple possible parameter sources."""
        if band_intensity is not None:
            return band_intensity
        elif dot_gain is not None:
            # Convert dot_gain (0.0-0.5) to band_intensity (0-100)
            return int(dot_gain * 200)  # Scale 0.5 -> 100
        else:
            return 20  # Default value
    
    def _resolve_noise_level(self, noise_level, paper_texture):
        """Resolve noise_level from multiple possible parameter sources."""
        if noise_level is not None:
            return noise_level
        elif paper_texture is not None:
            # Convert paper_texture (0.0-1.0) to noise_level (0-50)
            return int(paper_texture * 50)
        else:
            return 10  # Default value
    
    def _resolve_contrast_factor(self, contrast_factor, ink_bleeding):
        """Resolve contrast_factor from multiple possible parameter sources."""
        if contrast_factor != 0.85:  # If explicitly set (not default)
            return contrast_factor
        elif ink_bleeding is not None:
            # Convert ink_bleeding (0.0-1.0) to contrast reduction
            # Higher ink_bleeding = lower contrast
            return 1.0 - (ink_bleeding * 0.3)  # Max reduction of 0.3
        else:
            return 0.85  # Default value
    
    def _validate_parameters(self) -> None:
        """Validate filter parameters are within acceptable ranges."""
        params = self.parameters
        
        if not (0 <= params['band_intensity'] <= 100):
            raise ValueError("band_intensity must be between 0 and 100")
        
        if not (5 <= params['band_frequency'] <= 100):
            raise ValueError("band_frequency must be between 5 and 100")
        
        if not (0 <= params['noise_level'] <= 50):
            raise ValueError("noise_level must be between 0 and 50")
        
        if not (0.1 <= params['contrast_factor'] <= 1.0):
            raise ValueError("contrast_factor must be between 0.1 and 1.0")
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data with support for both RGB and grayscale.
        
        Args:
            data: Input image as numpy array
            
        Returns:
            True if input is valid
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Check basic requirements
        if not isinstance(data, np.ndarray):
            from ...core.utils import FilterValidationError
            raise FilterValidationError("Input must be a numpy array")
        
        if data.size == 0:
            from ...core.utils import FilterValidationError
            raise FilterValidationError("Input array cannot be empty")
        
        # Check dimensions - support both 2D (grayscale) and 3D (RGB)
        if data.ndim not in [2, 3]:
            from ...core.utils import FilterValidationError
            raise FilterValidationError(
                f"Image data must be 2D or 3D array, got {data.ndim}D"
            )
        
        # If 3D, check channel count
        if data.ndim == 3 and data.shape[-1] not in [1, 3, 4]:
            from ...core.utils import FilterValidationError
            raise FilterValidationError(
                f"3D image must have 1, 3, or 4 channels, got {data.shape[-1]}"
            )
        
        # Validate data range
        self._validate_data_range(data)
        
        return True
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply print simulation effects to the input image data.
        
        Args:
            data: Input image as numpy array (H, W) or (H, W, C)
            **kwargs: Additional parameters (merged with instance parameters)
            
        Returns:
            Processed image array with print simulation effects applied
            
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
            
            # Determine if input is color or grayscale
            is_color = data.ndim == 3 and data.shape[-1] >= 3
            
            # Convert to appropriate format for processing
            if data.dtype != np.uint8:
                if data.dtype in [np.float32, np.float64]:
                    # Assume float data is in [0, 1] range
                    processed_data = (data * 255).astype(np.uint8)
                else:
                    processed_data = data.astype(np.uint8)
            else:
                processed_data = data.copy()
            
            self._update_progress(0.1)
            
            if is_color:
                # Process color image by converting to grayscale, processing, then back
                result = self._process_color_image(processed_data)
            else:
                # Process grayscale image directly
                result = self._process_grayscale_image(processed_data)
            
            self._update_progress(1.0)
            
            return result
        
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
    
    def _process_color_image(self, image_array: np.ndarray) -> np.ndarray:
        """Process color image by converting to grayscale and back."""
        try:
            # Convert to PIL Image for easier processing
            pil_image = Image.fromarray(image_array)
            
            # Convert to grayscale
            grayscale_image = pil_image.convert("L")
            self._update_progress(0.2)
            
            # Apply print effects to grayscale
            processed_grayscale = self._apply_print_effects_pil(grayscale_image)
            self._update_progress(0.8)
            
            # Convert back to RGB by duplicating grayscale across channels
            processed_array = np.array(processed_grayscale)
            if image_array.shape[-1] == 3:
                result = np.stack([processed_array] * 3, axis=-1)
            elif image_array.shape[-1] == 4:
                # Preserve alpha channel
                result = np.stack([processed_array] * 3 + [image_array[:, :, 3]], axis=-1)
            else:
                result = processed_array
            
            return result
            
        except Exception as e:
            raise FilterExecutionError(f"Color image processing failed: {e}")
    
    def _process_grayscale_image(self, image_array: np.ndarray) -> np.ndarray:
        """Process grayscale image directly."""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image_array.squeeze())
            
            # Apply print effects
            processed_image = self._apply_print_effects_pil(pil_image)
            
            # Convert back to numpy array with original shape
            result = np.array(processed_image)
            if image_array.ndim == 3:
                result = result[:, :, np.newaxis]
            
            return result
            
        except Exception as e:
            raise FilterExecutionError(f"Grayscale image processing failed: {e}")
    
    def _apply_print_effects_pil(self, pil_image: Image.Image) -> Image.Image:
        """Apply print effects to PIL Image."""
        # Ensure image is in grayscale mode
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        
        # Apply horizontal bands
        processed_image = self._add_horizontal_bands(pil_image)
        self._update_progress(0.4)
        
        # Add noise
        processed_image = self._add_noise(processed_image)
        self._update_progress(0.6)
        
        # Degrade contrast
        processed_image = self._degrade_contrast(processed_image)
        self._update_progress(0.8)
        
        return processed_image
    
    def _add_horizontal_bands(self, image: Image.Image) -> Image.Image:
        """Add horizontal bands with intensity variations."""
        band_intensity = self.parameters['band_intensity']
        band_frequency = self.parameters['band_frequency']
        
        if band_intensity == 0:
            return image
        
        # Convert to numpy array for processing
        arr = np.array(image, dtype=np.int16)
        
        # Add bands at regular intervals
        for y in range(0, arr.shape[0], band_frequency):
            intensity_variation = random.randint(-band_intensity, band_intensity)
            end_y = min(y + 1, arr.shape[0])
            arr[y:end_y, :] += intensity_variation
        
        # Clip values and convert back
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    
    def _add_noise(self, image: Image.Image) -> Image.Image:
        """Add random noise to simulate paper texture and ink variations using NoiseFilter."""
        noise_level = self.parameters['noise_level']
        
        if noise_level == 0:
            return image
        
        # Convert to numpy array
        arr = np.array(image, dtype=np.uint8)
        
        # Convert noise_level (0-50) to NoiseFilter intensity (0.0-1.0)
        # Scale the noise level to match the original behavior
        noise_intensity = min(noise_level / 50.0, 1.0)  # Map 0-50 to 0.0-1.0
        
        # Use NoiseFilter with uniform noise to match original behavior
        # The original implementation used uniform random integers in range [-noise_level, noise_level]
        # NoiseFilter's uniform noise with appropriate intensity should produce similar results
        try:
            # Apply noise using the internal NoiseFilter
            noisy_arr = self._noise_filter.apply(
                arr, 
                noise_type="uniform", 
                intensity=noise_intensity
            )
            
            return Image.fromarray(noisy_arr)
            
        except Exception as e:
            # Fallback to original implementation if NoiseFilter fails
            # This ensures backward compatibility
            arr_int16 = arr.astype(np.int16)
            noise = np.random.randint(-noise_level, noise_level + 1, arr.shape)
            arr_int16 += noise
            arr_clipped = np.clip(arr_int16, 0, 255).astype(np.uint8)
            return Image.fromarray(arr_clipped)
    
    def _degrade_contrast(self, image: Image.Image) -> Image.Image:
        """Reduce contrast to simulate printing limitations."""
        contrast_factor = self.parameters['contrast_factor']
        
        if contrast_factor >= 1.0:
            return image
        
        # Use PIL's contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(contrast_factor)
    
    def set_parameters(self, **kwargs) -> None:
        """
        Update filter parameters with validation.
        
        Args:
            **kwargs: Parameter names and values to update
            
        Raises:
            ValueError: If invalid parameter names or values provided
        """
        # Validate parameter names (including documented API names)
        valid_params = {
            'band_intensity', 'band_frequency', 'noise_level', 'contrast_factor',
            # Documented API parameter names
            'dot_gain', 'paper_texture', 'ink_bleeding'
        }
        
        invalid_params = set(kwargs.keys()) - valid_params
        if invalid_params:
            raise ValueError(f"Invalid parameters: {invalid_params}")
        
        # Handle parameter mapping for documented API names
        mapped_kwargs = {}
        
        # Extract documented API parameters
        dot_gain = kwargs.pop('dot_gain', None)
        paper_texture = kwargs.pop('paper_texture', None)
        ink_bleeding = kwargs.pop('ink_bleeding', None)
        
        # Map documented parameters to internal parameters
        if dot_gain is not None:
            mapped_kwargs['band_intensity'] = self._resolve_band_intensity(
                kwargs.get('band_intensity'), dot_gain
            )
        
        if paper_texture is not None:
            mapped_kwargs['noise_level'] = self._resolve_noise_level(
                kwargs.get('noise_level'), paper_texture
            )
        
        if ink_bleeding is not None:
            current_contrast = kwargs.get('contrast_factor', self.parameters.get('contrast_factor', 0.85))
            mapped_kwargs['contrast_factor'] = self._resolve_contrast_factor(
                current_contrast, ink_bleeding
            )
        
        # Merge mapped parameters with remaining kwargs
        final_kwargs = {**kwargs, **mapped_kwargs}
        
        # Update parameters
        super().set_parameters(**final_kwargs)
        
        # Validate new parameter values
        self._validate_parameters()