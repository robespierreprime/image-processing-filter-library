"""
Noise filter for adding various types of noise to images.

This module contains the NoiseFilter class which adds gaussian, salt-pepper,
and uniform noise to images for artistic and simulation effects.
"""

import numpy as np
from ...core.base_filter import BaseFilter
from ...core.protocols import DataType, ColorFormat
from ...core.utils import FilterValidationError
from ..registry import register_filter


@register_filter(category="artistic")
class NoiseFilter(BaseFilter):
    """
    Filter that adds various types of noise to images.
    
    This filter supports three types of noise:
    - Gaussian: Normally distributed random values added to pixels
    - Salt-pepper: Random pixels set to minimum or maximum values
    - Uniform: Uniformly distributed random values added to pixels
    
    Parameters:
        noise_type (str): Type of noise ("gaussian", "salt_pepper", "uniform")
        intensity (float): Noise intensity (0.0-1.0)
                          - 0.0 = no noise (no change)
                          - 1.0 = maximum noise intensity
        salt_pepper_ratio (float): Ratio of salt to pepper for salt-pepper noise (0.0-1.0)
                                  - 0.0 = all pepper (black pixels)
                                  - 0.5 = equal salt and pepper
                                  - 1.0 = all salt (white pixels)
    """
    
    def __init__(self, noise_type: str = "gaussian", intensity: float = 0.1, 
                 salt_pepper_ratio: float = 0.5):
        """
        Initialize the NoiseFilter.
        
        Args:
            noise_type: Type of noise ("gaussian", "salt_pepper", "uniform")
            intensity: Noise intensity (0.0-1.0, default 0.1)
            salt_pepper_ratio: Salt to pepper ratio for salt-pepper noise (0.0-1.0, default 0.5)
        """
        super().__init__(
            name="noise",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,  # Will handle RGB, RGBA, and GRAYSCALE
            category="artistic",
            noise_type=noise_type,
            intensity=intensity,
            salt_pepper_ratio=salt_pepper_ratio
        )
    
    def _validate_parameters(self) -> None:
        """
        Validate filter parameters.
        
        Raises:
            FilterValidationError: If parameters are outside valid ranges or invalid types
        """
        noise_type = self.parameters.get('noise_type', 'gaussian')
        intensity = self.parameters.get('intensity', 0.1)
        salt_pepper_ratio = self.parameters.get('salt_pepper_ratio', 0.5)
        
        # Validate noise_type
        valid_noise_types = ['gaussian', 'salt_pepper', 'uniform']
        if not isinstance(noise_type, str):
            raise FilterValidationError(
                f"noise_type must be a string, got {type(noise_type).__name__}"
            )
        if noise_type not in valid_noise_types:
            raise FilterValidationError(
                f"noise_type must be one of {valid_noise_types}, got '{noise_type}'"
            )
        
        # Validate intensity
        if not isinstance(intensity, (int, float)):
            raise FilterValidationError(
                f"intensity must be a number, got {type(intensity).__name__}"
            )
        if intensity < 0.0 or intensity > 1.0:
            raise FilterValidationError(
                f"intensity must be in range [0.0, 1.0], got {intensity}"
            )
        
        # Validate salt_pepper_ratio
        if not isinstance(salt_pepper_ratio, (int, float)):
            raise FilterValidationError(
                f"salt_pepper_ratio must be a number, got {type(salt_pepper_ratio).__name__}"
            )
        if salt_pepper_ratio < 0.0 or salt_pepper_ratio > 1.0:
            raise FilterValidationError(
                f"salt_pepper_ratio must be in range [0.0, 1.0], got {salt_pepper_ratio}"
            )
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data for NoiseFilter.
        
        Extends BaseFilter validation to ensure the input is compatible
        with noise operations.
        
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
            # Grayscale is acceptable for noise
            pass
        
        return True
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply noise to the input image.
        
        Adds the specified type of noise to the image based on the noise_type parameter.
        
        Args:
            data: Input numpy array containing image data
            **kwargs: Additional parameters (can override noise_type, intensity, salt_pepper_ratio)
            
        Returns:
            Numpy array with noise applied
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Update parameters with any kwargs
        if kwargs:
            self.set_parameters(**kwargs)
        
        # Validate input data and parameters
        self.validate_input(data)
        self._validate_parameters()
        
        noise_type = self.parameters.get('noise_type', 'gaussian')
        intensity = self.parameters.get('intensity', 0.1)
        
        def noise_operation():
            # Handle identity case (no change needed)
            if intensity == 0.0:
                return data.copy() if not self._should_use_inplace(data) else data
            
            # Determine if we should use in-place processing
            use_inplace = self._should_use_inplace(data)
            
            if use_inplace:
                # Process in-place for memory efficiency
                result = data
            else:
                # Create a copy for processing
                result = data.copy()
            
            # Apply the appropriate noise algorithm
            if noise_type == 'gaussian':
                self._apply_gaussian_noise(result, intensity)
            elif noise_type == 'salt_pepper':
                salt_pepper_ratio = self.parameters.get('salt_pepper_ratio', 0.5)
                self._apply_salt_pepper_noise(result, intensity, salt_pepper_ratio)
            elif noise_type == 'uniform':
                self._apply_uniform_noise(result, intensity)
            
            return result
        
        # Execute with timing and progress tracking
        self._update_progress(0.0)
        result = self._measure_execution_time(noise_operation)
        self._update_progress(1.0)
        
        # Record metadata
        self._record_shapes(data, result)
        self._track_memory_usage(data, result, used_inplace=np.shares_memory(data, result))
        
        return result
    
    def _apply_gaussian_noise(self, data: np.ndarray, intensity: float) -> None:
        """
        Apply gaussian noise to the image data in-place.
        
        Adds normally distributed random values to pixel intensities.
        The noise is scaled by the intensity parameter.
        Preserves alpha channel for RGBA images.
        
        Args:
            data: Image data array to modify in-place
            intensity: Noise intensity (0.0-1.0)
        """
        # Determine if we have RGBA and need to preserve alpha
        has_alpha = data.ndim == 3 and data.shape[-1] == 4
        
        if has_alpha:
            # Apply noise only to RGB channels, preserve alpha
            rgb_data = data[..., :3]
            alpha_data = data[..., 3].copy()  # Preserve original alpha
            noise_shape = rgb_data.shape
            target_data = rgb_data
        else:
            # Apply noise to all data
            noise_shape = data.shape
            target_data = data
        
        # Generate gaussian noise with mean=0 and std proportional to intensity
        # Scale standard deviation based on data type and intensity
        if data.dtype == np.uint8:
            # For uint8, use intensity to scale noise relative to full range (255)
            noise_std = intensity * 255.0 * 0.1  # 0.1 factor to make intensity=1.0 reasonable
            noise = np.random.normal(0, noise_std, noise_shape).astype(np.float32)
            
            # Add noise and clamp to valid range
            noisy_data = target_data.astype(np.float32) + noise
            target_data[...] = np.clip(noisy_data, 0, 255).astype(np.uint8)
            
        elif data.dtype in [np.float32, np.float64]:
            # For float, assume [0, 1] range and scale noise accordingly
            noise_std = intensity * 0.1  # 0.1 factor to make intensity=1.0 reasonable
            noise = np.random.normal(0, noise_std, noise_shape).astype(data.dtype)
            
            # Add noise and clamp to valid range
            target_data[...] = np.clip(target_data + noise, 0.0, 1.0)
            
        else:
            # For other data types, try to determine appropriate scaling
            if np.issubdtype(data.dtype, np.integer):
                max_val = np.iinfo(data.dtype).max
                noise_std = intensity * max_val * 0.1
            else:
                # Assume [0, 1] range for other float types
                noise_std = intensity * 0.1
            
            noise = np.random.normal(0, noise_std, noise_shape).astype(data.dtype)
            
            # Add noise and clamp to appropriate range
            if np.issubdtype(data.dtype, np.integer):
                max_val = np.iinfo(data.dtype).max
                min_val = np.iinfo(data.dtype).min
                noisy_data = target_data.astype(np.float64) + noise
                target_data[...] = np.clip(noisy_data, min_val, max_val).astype(data.dtype)
            else:
                target_data[...] = np.clip(target_data + noise, 0.0, 1.0)
        
        # Restore alpha channel if it was preserved
        if has_alpha:
            data[..., 3] = alpha_data
    
    def _apply_salt_pepper_noise(self, data: np.ndarray, intensity: float, 
                                salt_pepper_ratio: float) -> None:
        """
        Apply salt-and-pepper noise to the image data in-place.
        
        Randomly sets pixels to minimum (pepper) or maximum (salt) values
        based on the intensity and salt_pepper_ratio parameters.
        Preserves alpha channel for RGBA images.
        
        Args:
            data: Image data array to modify in-place
            intensity: Noise intensity (0.0-1.0) - fraction of pixels to affect
            salt_pepper_ratio: Ratio of salt to pepper (0.0-1.0)
                              0.0 = all pepper, 0.5 = equal, 1.0 = all salt
        """
        # Determine if we have RGBA and need to preserve alpha
        has_alpha = data.ndim == 3 and data.shape[-1] == 4
        
        if has_alpha:
            # Apply noise only to RGB channels, preserve alpha
            target_data = data[..., :3]
            alpha_data = data[..., 3].copy()  # Preserve original alpha
            noise_shape = target_data.shape
        else:
            # Apply noise to all data
            target_data = data
            noise_shape = data.shape
        
        # Generate random mask for pixels to affect
        noise_mask = np.random.random(noise_shape) < intensity
        
        # Among the affected pixels, determine which are salt vs pepper
        salt_mask = np.random.random(noise_shape) < salt_pepper_ratio
        
        # Combine masks: pixels that are both affected by noise AND should be salt
        final_salt_mask = noise_mask & salt_mask
        # Pixels that are affected by noise but should be pepper
        final_pepper_mask = noise_mask & (~salt_mask)
        
        # Determine min/max values based on data type
        if data.dtype == np.uint8:
            min_val, max_val = 0, 255
        elif data.dtype in [np.float32, np.float64]:
            min_val, max_val = 0.0, 1.0
        elif np.issubdtype(data.dtype, np.integer):
            min_val = np.iinfo(data.dtype).min
            max_val = np.iinfo(data.dtype).max
        else:
            # Default for other float types
            min_val, max_val = 0.0, 1.0
        
        # Apply salt and pepper noise
        target_data[final_salt_mask] = max_val  # Salt (white)
        target_data[final_pepper_mask] = min_val  # Pepper (black)
        
        # Restore alpha channel if it was preserved
        if has_alpha:
            data[..., 3] = alpha_data
    
    def _apply_uniform_noise(self, data: np.ndarray, intensity: float) -> None:
        """
        Apply uniform noise to the image data in-place.
        
        Adds uniformly distributed random values to pixel intensities.
        The noise range is scaled by the intensity parameter.
        Preserves alpha channel for RGBA images.
        
        Args:
            data: Image data array to modify in-place
            intensity: Noise intensity (0.0-1.0)
        """
        # Determine if we have RGBA and need to preserve alpha
        has_alpha = data.ndim == 3 and data.shape[-1] == 4
        
        if has_alpha:
            # Apply noise only to RGB channels, preserve alpha
            rgb_data = data[..., :3]
            alpha_data = data[..., 3].copy()  # Preserve original alpha
            noise_shape = rgb_data.shape
            target_data = rgb_data
        else:
            # Apply noise to all data
            noise_shape = data.shape
            target_data = data
        
        # Generate uniform noise in range [-intensity*scale, +intensity*scale]
        if data.dtype == np.uint8:
            # For uint8, scale noise relative to full range (255)
            noise_range = intensity * 255.0 * 0.2  # 0.2 factor to make intensity=1.0 reasonable
            noise = np.random.uniform(-noise_range, noise_range, noise_shape).astype(np.float32)
            
            # Add noise and clamp to valid range
            noisy_data = target_data.astype(np.float32) + noise
            target_data[...] = np.clip(noisy_data, 0, 255).astype(np.uint8)
            
        elif data.dtype in [np.float32, np.float64]:
            # For float, assume [0, 1] range and scale noise accordingly
            noise_range = intensity * 0.2  # 0.2 factor to make intensity=1.0 reasonable
            noise = np.random.uniform(-noise_range, noise_range, noise_shape).astype(data.dtype)
            
            # Add noise and clamp to valid range
            target_data[...] = np.clip(target_data + noise, 0.0, 1.0)
            
        else:
            # For other data types, try to determine appropriate scaling
            if np.issubdtype(data.dtype, np.integer):
                max_val = np.iinfo(data.dtype).max
                noise_range = intensity * max_val * 0.2
            else:
                # Assume [0, 1] range for other float types
                noise_range = intensity * 0.2
            
            noise = np.random.uniform(-noise_range, noise_range, noise_shape).astype(data.dtype)
            
            # Add noise and clamp to appropriate range
            if np.issubdtype(data.dtype, np.integer):
                max_val = np.iinfo(data.dtype).max
                min_val = np.iinfo(data.dtype).min
                noisy_data = target_data.astype(np.float64) + noise
                target_data[...] = np.clip(noisy_data, min_val, max_val).astype(data.dtype)
            else:
                target_data[...] = np.clip(target_data + noise, 0.0, 1.0)
        
        # Restore alpha channel if it was preserved
        if has_alpha:
            data[..., 3] = alpha_data