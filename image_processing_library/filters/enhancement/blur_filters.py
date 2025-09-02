"""
Blur effect filters for image smoothing and motion effects.

This module contains filters for blur operations:
- GaussianBlurFilter: Applies gaussian blur using convolution
- MotionBlurFilter: Creates directional motion blur
"""

import numpy as np
from scipy import ndimage
import math
from ...core.base_filter import BaseFilter
from ...core.protocols import DataType, ColorFormat
from ...core.utils import FilterValidationError
from ..registry import register_filter


@register_filter(category="enhancement")
class GaussianBlurFilter(BaseFilter):
    """
    Filter that applies gaussian blur using convolution.
    
    Gaussian blur smooths images by convolving with a gaussian kernel.
    The blur strength is controlled by the sigma parameter:
    - sigma = 0: No blur (identity transformation)
    - sigma > 0: Increasing blur strength
    
    The filter supports RGB, RGBA, and GRAYSCALE color formats.
    """
    
    def __init__(self, sigma: float = 1.0, kernel_size: int = None):
        """
        Initialize the GaussianBlurFilter.
        
        Args:
            sigma: Standard deviation for gaussian kernel (0.0-10.0 range)
            kernel_size: Size of convolution kernel (auto-calculated if None)
        """
        super().__init__(
            name="gaussian_blur",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,  # Will handle RGB, RGBA, and GRAYSCALE
            category="enhancement",
            sigma=sigma,
            kernel_size=kernel_size
        )
    
    def _validate_parameters(self) -> None:
        """
        Validate filter parameters.
        
        Ensures sigma value is within acceptable range (0.0-10.0) and
        kernel_size is valid if provided.
        
        Raises:
            FilterValidationError: If parameters are invalid
        """
        sigma = self.parameters.get('sigma', 1.0)
        kernel_size = self.parameters.get('kernel_size', None)
        
        # Validate sigma
        if not isinstance(sigma, (int, float)):
            raise FilterValidationError("Sigma must be a numeric value")
        
        if sigma < 0:
            raise FilterValidationError("Sigma must be non-negative")
        
        if sigma < 0.0 or sigma > 10.0:
            raise FilterValidationError(
                f"Sigma must be in range [0.0, 10.0], got {sigma}"
            )
        
        # Validate kernel_size if provided
        if kernel_size is not None:
            if not isinstance(kernel_size, int):
                raise FilterValidationError("Kernel size must be an integer")
            
            if kernel_size <= 0:
                raise FilterValidationError("Kernel size must be positive")
            
            if kernel_size % 2 == 0:
                raise FilterValidationError("Kernel size must be odd")
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data for GaussianBlurFilter.
        
        Extends BaseFilter validation to ensure the input is compatible
        with gaussian blur operations.
        
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
            # Grayscale is acceptable for gaussian blur
            pass
        
        return True
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply gaussian blur to the input image.
        
        Applies gaussian blur using convolution with a gaussian kernel.
        For sigma = 0, returns the original image unchanged (identity case).
        
        Args:
            data: Input numpy array containing image data
            **kwargs: Additional parameters (can override sigma, kernel_size)
            
        Returns:
            Numpy array with gaussian blur applied
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Validate input data
        self.validate_input(data)
        
        # Update parameters with any kwargs
        if kwargs:
            self.set_parameters(**kwargs)
        
        self._validate_parameters()
        
        sigma = self.parameters.get('sigma', 1.0)
        kernel_size = self.parameters.get('kernel_size', None)
        
        def gaussian_blur_operation():
            # Handle identity case (sigma = 0)
            if abs(sigma) < 1e-6:
                # Return copy to maintain consistent behavior
                return data.copy()
            
            # Determine if we should use in-place processing
            use_inplace = self._should_use_inplace(data)
            
            # For gaussian blur, we need to create a copy since scipy.ndimage.gaussian_filter
            # doesn't support true in-place operation for all cases
            if use_inplace and data.dtype in [np.float32, np.float64]:
                # Can work in-place for float types
                result = data.copy()
            else:
                # Create a copy for processing
                result = data.astype(np.float64)
            
            # Apply gaussian blur based on image dimensions
            if data.ndim == 2:
                # Grayscale image
                if data.dtype == np.uint8:
                    # Convert to float for processing
                    float_data = data.astype(np.float64)
                    blurred = ndimage.gaussian_filter(float_data, sigma=sigma)
                    result = np.clip(blurred, 0, 255).astype(np.uint8)
                else:
                    # Float data - apply directly
                    result = ndimage.gaussian_filter(result, sigma=sigma)
                    if data.dtype in [np.float32, np.float64]:
                        result = np.clip(result, 0.0, 1.0).astype(data.dtype)
                    
            elif data.ndim == 3:
                # Color image (RGB or RGBA)
                if data.dtype == np.uint8:
                    # Convert to float for processing
                    float_data = data.astype(np.float64)
                    # Apply gaussian filter to each channel separately
                    for channel in range(data.shape[2]):
                        float_data[:, :, channel] = ndimage.gaussian_filter(
                            float_data[:, :, channel], sigma=sigma
                        )
                    result = np.clip(float_data, 0, 255).astype(np.uint8)
                else:
                    # Float data - apply to each channel
                    for channel in range(data.shape[2]):
                        result[:, :, channel] = ndimage.gaussian_filter(
                            result[:, :, channel], sigma=sigma
                        )
                    if data.dtype in [np.float32, np.float64]:
                        result = np.clip(result, 0.0, 1.0).astype(data.dtype)
            
            return result
        
        # Execute with timing and progress tracking
        self._update_progress(0.0)
        result = self._measure_execution_time(gaussian_blur_operation)
        self._update_progress(1.0)
        
        # Record metadata
        self._record_shapes(data, result)
        self._track_memory_usage(data, result, used_inplace=np.shares_memory(data, result))
        
        return result


@register_filter(category="enhancement")
class MotionBlurFilter(BaseFilter):
    """
    Filter that creates directional motion blur effects.
    
    Motion blur simulates the effect of camera or subject movement during
    image capture by applying a linear blur in a specified direction.
    The blur is controlled by distance and angle parameters:
    - distance = 0: No blur (identity transformation)
    - distance > 0: Increasing blur strength
    - angle: Direction of blur in degrees (0-360)
    
    The filter supports RGB, RGBA, and GRAYSCALE color formats.
    """
    
    def __init__(self, distance: int = 5, angle: float = 0.0):
        """
        Initialize the MotionBlurFilter.
        
        Args:
            distance: Blur distance in pixels (0-50 range)
            angle: Blur direction in degrees (0-360 range)
        """
        super().__init__(
            name="motion_blur",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,  # Will handle RGB, RGBA, and GRAYSCALE
            category="enhancement",
            distance=distance,
            angle=angle
        )
    
    def _validate_parameters(self) -> None:
        """
        Validate filter parameters.
        
        Ensures distance is within acceptable range (0-50) and
        angle is within valid range (0-360).
        
        Raises:
            FilterValidationError: If parameters are invalid
        """
        distance = self.parameters.get('distance', 5)
        angle = self.parameters.get('angle', 0.0)
        
        # Validate distance
        if not isinstance(distance, (int, float)):
            raise FilterValidationError("Distance must be a numeric value")
        
        if distance < 0:
            raise FilterValidationError("Distance must be non-negative")
        
        if distance < 0 or distance > 50:
            raise FilterValidationError(
                f"Distance must be in range [0, 50], got {distance}"
            )
        
        # Validate angle
        if not isinstance(angle, (int, float)):
            raise FilterValidationError("Angle must be a numeric value")
        
        if angle < 0 or angle >= 360:
            raise FilterValidationError(
                f"Angle must be in range [0, 360), got {angle}"
            )
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data for MotionBlurFilter.
        
        Extends BaseFilter validation to ensure the input is compatible
        with motion blur operations.
        
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
            # Grayscale is acceptable for motion blur
            pass
        
        return True
    
    def _create_motion_kernel(self, distance: int, angle: float) -> np.ndarray:
        """
        Create a linear motion blur kernel based on distance and angle.
        
        Args:
            distance: Blur distance in pixels
            angle: Blur direction in degrees
            
        Returns:
            2D numpy array representing the motion blur kernel
        """
        if distance == 0:
            # Return identity kernel for no blur
            return np.array([[1.0]])
        
        # Convert angle to radians
        angle_rad = math.radians(angle)
        
        # Calculate direction vector
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        
        # Calculate kernel size - needs to be large enough to contain the motion line
        # Use the maximum extent in x and y directions
        max_extent = max(abs(distance * dx), abs(distance * dy))
        kernel_size = int(2 * max_extent + 3)  # Add padding
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size
        
        # Create empty kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
        
        # Calculate center of kernel
        center = kernel_size // 2
        
        # Use Bresenham-like algorithm to draw a continuous line
        # Sample points along the line more densely
        num_samples = max(distance * 2, 10)  # Ensure enough samples
        
        for i in range(num_samples + 1):
            # Calculate position along the line from -distance/2 to +distance/2
            t = (i / num_samples - 0.5) * distance
            
            # Calculate pixel coordinates
            x = center + t * dx
            y = center + t * dy
            
            # Use bilinear interpolation to distribute weight to nearby pixels
            x_floor = int(math.floor(x))
            y_floor = int(math.floor(y))
            x_frac = x - x_floor
            y_frac = y - y_floor
            
            # Distribute weight to 4 neighboring pixels
            for dy_offset in [0, 1]:
                for dx_offset in [0, 1]:
                    px = x_floor + dx_offset
                    py = y_floor + dy_offset
                    
                    if 0 <= px < kernel_size and 0 <= py < kernel_size:
                        # Calculate bilinear weight
                        weight_x = (1 - x_frac) if dx_offset == 0 else x_frac
                        weight_y = (1 - y_frac) if dy_offset == 0 else y_frac
                        weight = weight_x * weight_y
                        
                        kernel[py, px] += weight
        
        # Normalize kernel so sum equals 1
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        else:
            # Fallback to identity if something went wrong
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[center, center] = 1.0
        
        return kernel
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply motion blur to the input image.
        
        Applies directional motion blur using convolution with a linear kernel.
        For distance = 0, returns the original image unchanged (identity case).
        
        Args:
            data: Input numpy array containing image data
            **kwargs: Additional parameters (can override distance, angle)
            
        Returns:
            Numpy array with motion blur applied
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Validate input data
        self.validate_input(data)
        
        # Update parameters with any kwargs
        if kwargs:
            self.set_parameters(**kwargs)
        
        self._validate_parameters()
        
        distance = self.parameters.get('distance', 5)
        angle = self.parameters.get('angle', 0.0)
        
        def motion_blur_operation():
            # Handle identity case (distance = 0)
            if distance == 0:
                # Return copy to maintain consistent behavior
                return data.copy()
            
            # Create motion blur kernel
            kernel = self._create_motion_kernel(distance, angle)
            
            # Determine if we should use in-place processing
            use_inplace = self._should_use_inplace(data)
            
            # For motion blur, we need to create a copy since convolution
            # doesn't support true in-place operation
            if data.dtype == np.uint8:
                # Convert to float for processing
                float_data = data.astype(np.float64)
            else:
                float_data = data.copy()
            
            # Apply motion blur based on image dimensions
            if data.ndim == 2:
                # Grayscale image
                if data.dtype == np.uint8:
                    blurred = ndimage.convolve(float_data, kernel, mode='reflect')
                    result = np.clip(blurred, 0, 255).astype(np.uint8)
                else:
                    # Float data - apply directly
                    result = ndimage.convolve(float_data, kernel, mode='reflect')
                    if data.dtype in [np.float32, np.float64]:
                        result = np.clip(result, 0.0, 1.0).astype(data.dtype)
                    
            elif data.ndim == 3:
                # Color image (RGB or RGBA)
                if data.dtype == np.uint8:
                    # Apply convolution to each channel separately
                    for channel in range(data.shape[2]):
                        float_data[:, :, channel] = ndimage.convolve(
                            float_data[:, :, channel], kernel, mode='reflect'
                        )
                    result = np.clip(float_data, 0, 255).astype(np.uint8)
                else:
                    # Float data - apply to each channel
                    for channel in range(data.shape[2]):
                        float_data[:, :, channel] = ndimage.convolve(
                            float_data[:, :, channel], kernel, mode='reflect'
                        )
                    if data.dtype in [np.float32, np.float64]:
                        result = np.clip(float_data, 0.0, 1.0).astype(data.dtype)
                    else:
                        result = float_data
            
            return result
        
        # Execute with timing and progress tracking
        self._update_progress(0.0)
        result = self._measure_execution_time(motion_blur_operation)
        self._update_progress(1.0)
        
        # Record metadata
        self._record_shapes(data, result)
        self._track_memory_usage(data, result, used_inplace=np.shares_memory(data, result))
        
        return result