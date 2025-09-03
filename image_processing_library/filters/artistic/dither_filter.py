"""
Dithering filter for creating stylized reduced-color representations.

This module contains the DitherFilter class which applies various dithering
algorithms including Floyd-Steinberg, Bayer, and random threshold dithering.
"""

import numpy as np
from ...core.base_filter import BaseFilter
from ...core.protocols import DataType, ColorFormat
from ...core.utils import FilterValidationError
from ..registry import register_filter


@register_filter(category="artistic")
class DitherFilter(BaseFilter):
    """
    Filter that applies dithering effects with multiple pattern types.
    
    This filter supports three types of dithering algorithms:
    - Floyd-Steinberg: Error diffusion dithering with proper error distribution
    - Bayer: Ordered dithering using Bayer matrices of configurable sizes
    - Random: Random threshold dithering with random threshold values
    
    All algorithms quantize colors to a specified number of levels before applying
    the dithering pattern to create stylized reduced-color representations.
    
    Parameters:
        pattern_type (str): Type of dithering pattern ("floyd_steinberg", "bayer", "random")
        levels (int): Number of quantization levels per channel (2-256)
                     - 2 = binary (black/white)
                     - Higher values = more color gradations
        bayer_size (int): Size of Bayer matrix for Bayer dithering (2, 4, 8, 16, 32, or 64)
                         - Only used when pattern_type is "bayer"
                         - Larger sizes create finer dithering patterns
        pixel_step (int): Size of pixel blocks for chunky/pixelated dithering (1-64)
                         - 1 = normal pixel-by-pixel dithering
                         - Higher values create larger "pixels" for retro/chunky effects
    """
    
    def __init__(self, pattern_type: str = "floyd_steinberg", levels: int = 8, 
                 bayer_size: int = 4, pixel_step: int = 1):
        """
        Initialize the DitherFilter.
        
        Args:
            pattern_type: Type of dithering pattern ("floyd_steinberg", "bayer", "random")
            levels: Number of quantization levels per channel (2-256, default 8)
            bayer_size: Size of Bayer matrix for Bayer dithering (2, 4, 8, 16, 32, or 64, default 4)
            pixel_step: Size of pixel blocks for chunky dithering (1-64, default 1)
        """
        super().__init__(
            name="dither",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,  # Will handle RGB, RGBA, and GRAYSCALE
            category="artistic",
            pattern_type=pattern_type,
            levels=levels,
            bayer_size=bayer_size,
            pixel_step=pixel_step
        )
    
    def _validate_parameters(self) -> None:
        """
        Validate filter parameters.
        
        Raises:
            FilterValidationError: If parameters are outside valid ranges or invalid types
        """
        pattern_type = self.parameters.get('pattern_type', 'floyd_steinberg')
        levels = self.parameters.get('levels', 8)
        bayer_size = self.parameters.get('bayer_size', 4)
        pixel_step = self.parameters.get('pixel_step', 1)
        
        # Validate pattern_type
        valid_pattern_types = ['floyd_steinberg', 'bayer', 'random']
        if not isinstance(pattern_type, str):
            raise FilterValidationError(
                f"pattern_type must be a string, got {type(pattern_type).__name__}"
            )
        if pattern_type not in valid_pattern_types:
            raise FilterValidationError(
                f"pattern_type must be one of {valid_pattern_types}, got '{pattern_type}'"
            )
        
        # Validate levels
        if not isinstance(levels, int):
            raise FilterValidationError(
                f"levels must be an integer, got {type(levels).__name__}"
            )
        if levels < 2 or levels > 256:
            raise FilterValidationError(
                f"levels must be in range [2, 256], got {levels}"
            )
        
        # Validate bayer_size
        if not isinstance(bayer_size, int):
            raise FilterValidationError(
                f"bayer_size must be an integer, got {type(bayer_size).__name__}"
            )
        valid_bayer_sizes = [2, 4, 8, 16, 32, 64]
        if bayer_size not in valid_bayer_sizes:
            raise FilterValidationError(
                f"bayer_size must be one of {valid_bayer_sizes}, got {bayer_size}"
            )
        
        # Validate pixel_step
        if not isinstance(pixel_step, int):
            raise FilterValidationError(
                f"pixel_step must be an integer, got {type(pixel_step).__name__}"
            )
        if pixel_step < 1 or pixel_step > 64:
            raise FilterValidationError(
                f"pixel_step must be in range [1, 64], got {pixel_step}"
            )
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data for DitherFilter.
        
        Extends BaseFilter validation to ensure the input is compatible
        with dithering operations.
        
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
            # Grayscale is acceptable for dithering
            pass
        
        return True
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply dithering to the input image.
        
        Applies the specified dithering algorithm to create stylized reduced-color
        representations of the input image.
        
        Args:
            data: Input numpy array containing image data
            **kwargs: Additional parameters (can override pattern_type, levels, bayer_size)
            
        Returns:
            Numpy array with dithering applied
            
        Raises:
            FilterValidationError: If input data is invalid
        """
        # Update parameters with any kwargs
        if kwargs:
            self.set_parameters(**kwargs)
        
        # Validate input data and parameters
        self.validate_input(data)
        self._validate_parameters()
        
        pattern_type = self.parameters.get('pattern_type', 'floyd_steinberg')
        levels = self.parameters.get('levels', 8)
        pixel_step = self.parameters.get('pixel_step', 1)
        
        def dither_operation():
            # Determine if we should use in-place processing
            # Note: Dithering algorithms typically need to create copies for processing
            # so we'll generally not use in-place processing
            use_inplace = False  # Dithering algorithms need working copies
            
            # Create a copy for processing
            result = data.copy()
            
            # Apply pixel step preprocessing if needed
            if pixel_step > 1:
                result = self._apply_pixel_step_preprocessing(result, pixel_step)
            
            # Apply the appropriate dithering algorithm
            if pattern_type == 'floyd_steinberg':
                self._apply_floyd_steinberg_dither(result, levels)
            elif pattern_type == 'bayer':
                bayer_size = self.parameters.get('bayer_size', 4)
                self._apply_bayer_dither(result, levels, bayer_size)
            elif pattern_type == 'random':
                self._apply_random_dither(result, levels)
            
            # Apply pixel step postprocessing if needed
            if pixel_step > 1:
                result = self._apply_pixel_step_postprocessing(result, pixel_step)
            
            return result
        
        # Execute with timing and progress tracking
        self._update_progress(0.0)
        result = self._measure_execution_time(dither_operation)
        self._update_progress(1.0)
        
        # Record metadata
        self._record_shapes(data, result)
        self._track_memory_usage(data, result, used_inplace=False)
        
        return result
    
    def _apply_floyd_steinberg_dither(self, data: np.ndarray, levels: int) -> None:
        """
        Apply Floyd-Steinberg error diffusion dithering to the image data in-place.
        
        Implements the Floyd-Steinberg error diffusion algorithm:
        - Process pixels left-to-right, top-to-bottom
        - Quantize each pixel to the nearest level
        - Distribute quantization error to neighboring pixels using the error matrix:
          [0, 0, 7/16]
          [3/16, 5/16, 1/16]
        
        Preserves alpha channel for RGBA images.
        
        Args:
            data: Image data array to modify in-place
            levels: Number of quantization levels per channel
        """
        # Determine if we have RGBA and need to preserve alpha
        has_alpha = data.ndim == 3 and data.shape[-1] == 4
        
        if has_alpha:
            # Process only RGB channels, preserve alpha
            alpha_data = data[..., 3].copy()
            work_data = data[..., :3].astype(np.float64)
            channels = 3
        elif data.ndim == 3:
            # RGB image
            work_data = data.astype(np.float64)
            channels = data.shape[-1]
        else:
            # Grayscale image
            work_data = data.astype(np.float64)
            channels = 1
        
        # Determine data range for quantization
        if data.dtype == np.uint8:
            data_min, data_max = 0.0, 255.0
        elif data.dtype in [np.float32, np.float64]:
            data_min, data_max = 0.0, 1.0
        else:
            # For other data types, use actual min/max
            data_min = float(np.min(data))
            data_max = float(np.max(data))
        
        # Calculate quantization step
        # For proper quantization to exactly 'levels' number of values,
        # we need to divide the range into 'levels' equal parts
        step = (data_max - data_min) / (levels - 1) if levels > 1 else 0
        
        # Floyd-Steinberg error diffusion matrix
        # Errors are distributed to: right, bottom-left, bottom, bottom-right
        # with weights: 7/16, 3/16, 5/16, 1/16
        error_weights = np.array([7/16, 3/16, 5/16, 1/16])
        
        # Get image dimensions
        if work_data.ndim == 2:
            height, width = work_data.shape
            work_data = work_data.reshape(height, width, 1)  # Add channel dimension
        else:
            height, width, _ = work_data.shape
        
        # Process each pixel left-to-right, top-to-bottom
        for y in range(height):
            # Update progress periodically
            if y % max(1, height // 10) == 0:
                progress = 0.1 + (y / height) * 0.8  # Progress from 10% to 90%
                self._update_progress(progress)
            
            for x in range(width):
                for c in range(channels):
                    # Get current pixel value
                    old_pixel = work_data[y, x, c]
                    
                    # Quantize to nearest level
                    level_index = round((old_pixel - data_min) / step)
                    level_index = max(0, min(levels - 1, level_index))
                    new_pixel = data_min + level_index * step
                    
                    # Set quantized value
                    work_data[y, x, c] = new_pixel
                    
                    # Calculate quantization error
                    error = old_pixel - new_pixel
                    
                    # Distribute error to neighboring pixels
                    # Right pixel (x+1, y)
                    if x + 1 < width:
                        work_data[y, x + 1, c] += error * error_weights[0]
                    
                    # Bottom-left pixel (x-1, y+1)
                    if y + 1 < height and x - 1 >= 0:
                        work_data[y + 1, x - 1, c] += error * error_weights[1]
                    
                    # Bottom pixel (x, y+1)
                    if y + 1 < height:
                        work_data[y + 1, x, c] += error * error_weights[2]
                    
                    # Bottom-right pixel (x+1, y+1)
                    if y + 1 < height and x + 1 < width:
                        work_data[y + 1, x + 1, c] += error * error_weights[3]
        
        # Convert back to original data type and shape
        if data.ndim == 2:
            # Grayscale - remove channel dimension
            work_data = work_data.reshape(height, width)
        
        # Clamp values to valid range
        work_data = np.clip(work_data, data_min, data_max)
        
        # Convert back to original data type
        if data.dtype == np.uint8:
            result_data = work_data.astype(np.uint8)
        elif data.dtype in [np.float32, np.float64]:
            result_data = work_data.astype(data.dtype)
        else:
            result_data = work_data.astype(data.dtype)
        
        # Update the original data array
        if has_alpha:
            data[..., :3] = result_data
            data[..., 3] = alpha_data  # Restore alpha channel
        else:
            data[...] = result_data
    
    def _apply_bayer_dither(self, data: np.ndarray, levels: int, bayer_size: int) -> None:
        """
        Apply Bayer ordered dithering to the image data in-place.
        
        Implements Bayer ordered dithering using threshold matrices:
        - Generate Bayer matrix of specified size (2x2, 4x4, or 8x8)
        - Tile the matrix across the image
        - Compare each pixel against the threshold matrix
        - Quantize based on threshold comparison
        
        Preserves alpha channel for RGBA images.
        
        Args:
            data: Image data array to modify in-place
            levels: Number of quantization levels per channel
            bayer_size: Size of Bayer matrix (2, 4, or 8)
        """
        # Generate Bayer matrix
        bayer_matrix = self._generate_bayer_matrix(bayer_size)
        
        # Determine if we have RGBA and need to preserve alpha
        has_alpha = data.ndim == 3 and data.shape[-1] == 4
        
        if has_alpha:
            # Process only RGB channels, preserve alpha
            alpha_data = data[..., 3].copy()
            work_data = data[..., :3].astype(np.float64)
            channels = 3
        elif data.ndim == 3:
            # RGB image
            work_data = data.astype(np.float64)
            channels = data.shape[-1]
        else:
            # Grayscale image
            work_data = data.astype(np.float64)
            channels = 1
        
        # Determine data range for quantization
        if data.dtype == np.uint8:
            data_min, data_max = 0.0, 255.0
        elif data.dtype in [np.float32, np.float64]:
            data_min, data_max = 0.0, 1.0
        else:
            # For other data types, use actual min/max
            data_min = float(np.min(data))
            data_max = float(np.max(data))
        
        # Calculate quantization step
        # For proper quantization to exactly 'levels' number of values,
        # we need to divide the range into 'levels' equal parts
        step = (data_max - data_min) / (levels - 1) if levels > 1 else 0
        
        # Get image dimensions
        if work_data.ndim == 2:
            height, width = work_data.shape
            work_data = work_data.reshape(height, width, 1)  # Add channel dimension
        else:
            height, width, _ = work_data.shape
        
        # Create tiled threshold matrix to match image size
        # Tile the Bayer matrix across the entire image
        tiles_y = (height + bayer_size - 1) // bayer_size
        tiles_x = (width + bayer_size - 1) // bayer_size
        
        # Create tiled matrix
        tiled_matrix = np.tile(bayer_matrix, (tiles_y, tiles_x))
        
        # Crop to exact image size
        threshold_matrix = tiled_matrix[:height, :width]
        
        # Normalize threshold matrix to [0, 1] range
        threshold_matrix = threshold_matrix / (bayer_size * bayer_size)
        
        # Apply Bayer dithering
        for c in range(channels):
            # Update progress
            progress = 0.1 + (c / channels) * 0.8
            self._update_progress(progress)
            
            # Get channel data
            channel_data = work_data[..., c]
            
            # Normalize pixel values to [0, 1] range for comparison
            normalized_pixels = (channel_data - data_min) / (data_max - data_min)
            
            # Calculate which quantization level each pixel should be at
            level_indices = normalized_pixels * (levels - 1)
            
            # Get the fractional part (determines if we round up or down)
            fractional_part = level_indices - np.floor(level_indices)
            
            # Compare fractional part against threshold matrix
            # If fractional part > threshold, round up; otherwise round down
            should_round_up = fractional_part > threshold_matrix
            
            # Calculate final quantized values
            base_levels = np.floor(level_indices).astype(int)
            final_levels = base_levels + should_round_up.astype(int)
            
            # Clamp to valid level range
            final_levels = np.clip(final_levels, 0, levels - 1)
            
            # Convert back to original data range
            quantized_values = data_min + final_levels * step
            
            # Update the channel data
            work_data[..., c] = quantized_values
        
        # Convert back to original data type and shape
        if data.ndim == 2:
            # Grayscale - remove channel dimension
            work_data = work_data.reshape(height, width)
        
        # Clamp values to valid range
        work_data = np.clip(work_data, data_min, data_max)
        
        # Convert back to original data type
        if data.dtype == np.uint8:
            result_data = work_data.astype(np.uint8)
        elif data.dtype in [np.float32, np.float64]:
            result_data = work_data.astype(data.dtype)
        else:
            result_data = work_data.astype(data.dtype)
        
        # Update the original data array
        if has_alpha:
            data[..., :3] = result_data
            data[..., 3] = alpha_data  # Restore alpha channel
        else:
            data[...] = result_data
    
    def _generate_bayer_matrix(self, size: int) -> np.ndarray:
        """
        Generate a Bayer matrix of the specified size.
        
        Bayer matrices are used for ordered dithering and have a recursive structure:
        - 2x2 base matrix
        - Larger matrices are built by recursively combining smaller ones
        
        Args:
            size: Size of the matrix (2, 4, 8, 16, 32, or 64)
            
        Returns:
            Numpy array containing the Bayer matrix
        """
        if size == 2:
            # Base 2x2 Bayer matrix
            return np.array([
                [0, 2],
                [3, 1]
            ], dtype=np.float64)
        
        elif size in [4, 8, 16, 32, 64]:
            # Recursively build larger matrices from smaller ones
            half_size = size // 2
            base = self._generate_bayer_matrix(half_size)
            
            # Build the larger matrix using the recursive Bayer pattern
            top_left = 4 * base
            top_right = 4 * base + 2
            bottom_left = 4 * base + 3
            bottom_right = 4 * base + 1
            
            return np.block([
                [top_left, top_right],
                [bottom_left, bottom_right]
            ]).astype(np.float64)
        
        else:
            # This should not happen due to parameter validation
            raise ValueError(f"Unsupported Bayer matrix size: {size}")
    
    def _apply_random_dither(self, data: np.ndarray, levels: int) -> None:
        """
        Apply random threshold dithering to the image data in-place.
        
        Implements random threshold dithering:
        - Generate random threshold matrix for each pixel
        - Compare pixel values against random thresholds
        - Quantize based on threshold comparison
        
        This creates a noisy but unstructured dithering pattern that can be
        useful for certain artistic effects.
        
        Preserves alpha channel for RGBA images.
        
        Args:
            data: Image data array to modify in-place
            levels: Number of quantization levels per channel
        """
        # Determine if we have RGBA and need to preserve alpha
        has_alpha = data.ndim == 3 and data.shape[-1] == 4
        
        if has_alpha:
            # Process only RGB channels, preserve alpha
            alpha_data = data[..., 3].copy()
            work_data = data[..., :3].astype(np.float64)
            channels = 3
        elif data.ndim == 3:
            # RGB image
            work_data = data.astype(np.float64)
            channels = data.shape[-1]
        else:
            # Grayscale image
            work_data = data.astype(np.float64)
            channels = 1
        
        # Determine data range for quantization
        if data.dtype == np.uint8:
            data_min, data_max = 0.0, 255.0
        elif data.dtype in [np.float32, np.float64]:
            data_min, data_max = 0.0, 1.0
        else:
            # For other data types, use actual min/max
            data_min = float(np.min(data))
            data_max = float(np.max(data))
        
        # Calculate quantization step
        # For proper quantization to exactly 'levels' number of values,
        # we need to divide the range into 'levels' equal parts
        step = (data_max - data_min) / (levels - 1) if levels > 1 else 0
        
        # Get image dimensions
        if work_data.ndim == 2:
            height, width = work_data.shape
            work_data = work_data.reshape(height, width, 1)  # Add channel dimension
        else:
            height, width, _ = work_data.shape
        
        # Apply random threshold dithering
        for c in range(channels):
            # Update progress
            progress = 0.1 + (c / channels) * 0.8
            self._update_progress(progress)
            
            # Get channel data
            channel_data = work_data[..., c]
            
            # Normalize pixel values to [0, 1] range for comparison
            normalized_pixels = (channel_data - data_min) / (data_max - data_min)
            
            # Calculate which quantization level each pixel should be at
            level_indices = normalized_pixels * (levels - 1)
            
            # Get the fractional part (determines if we round up or down)
            fractional_part = level_indices - np.floor(level_indices)
            
            # Generate random threshold matrix with same shape as the image
            # Each pixel gets a random threshold value between 0 and 1
            random_thresholds = np.random.random((height, width))
            
            # Compare fractional part against random thresholds
            # If fractional part > threshold, round up; otherwise round down
            should_round_up = fractional_part > random_thresholds
            
            # Calculate final quantized values
            base_levels = np.floor(level_indices).astype(int)
            final_levels = base_levels + should_round_up.astype(int)
            
            # Clamp to valid level range
            final_levels = np.clip(final_levels, 0, levels - 1)
            
            # Convert back to original data range
            quantized_values = data_min + final_levels * step
            
            # Update the channel data
            work_data[..., c] = quantized_values
        
        # Convert back to original data type and shape
        if data.ndim == 2:
            # Grayscale - remove channel dimension
            work_data = work_data.reshape(height, width)
        
        # Clamp values to valid range
        work_data = np.clip(work_data, data_min, data_max)
        
        # Convert back to original data type
        if data.dtype == np.uint8:
            result_data = work_data.astype(np.uint8)
        elif data.dtype in [np.float32, np.float64]:
            result_data = work_data.astype(data.dtype)
        else:
            result_data = work_data.astype(data.dtype)
        
        # Update the original data array
        if has_alpha:
            data[..., :3] = result_data
            data[..., 3] = alpha_data  # Restore alpha channel
        else:
            data[...] = result_data
    def _apply_pixel_step_preprocessing(self, data: np.ndarray, pixel_step: int) -> np.ndarray:
        """
        Apply pixel step preprocessing to create chunky/pixelated effect.
        
        This method downsamples the image by averaging pixel blocks, creating
        a pixelated effect that will be maintained through the dithering process.
        
        Args:
            data: Input image data
            pixel_step: Size of pixel blocks to average
            
        Returns:
            Preprocessed image data with chunky pixels
        """
        if pixel_step <= 1:
            return data
        
        # Get image dimensions
        if data.ndim == 2:
            height, width = data.shape
            channels = 1
            work_data = data.reshape(height, width, 1)
        else:
            height, width, channels = data.shape
            work_data = data
        
        # Calculate new dimensions (rounded down to fit complete blocks)
        new_height = (height // pixel_step) * pixel_step
        new_width = (width // pixel_step) * pixel_step
        
        # Crop to fit complete blocks
        cropped_data = work_data[:new_height, :new_width]
        
        # Reshape to group pixels into blocks
        # Shape: (blocks_y, pixel_step, blocks_x, pixel_step, channels)
        blocks_y = new_height // pixel_step
        blocks_x = new_width // pixel_step
        
        blocked_data = cropped_data.reshape(
            blocks_y, pixel_step, blocks_x, pixel_step, channels
        )
        
        # Average each block
        # Shape: (blocks_y, blocks_x, channels)
        averaged_blocks = np.mean(blocked_data, axis=(1, 3))
        
        # Expand blocks back to original size
        # Shape: (blocks_y, 1, blocks_x, 1, channels)
        expanded_blocks = averaged_blocks[:, np.newaxis, :, np.newaxis, :]
        
        # Tile to fill the pixel_step x pixel_step area
        # Shape: (blocks_y, pixel_step, blocks_x, pixel_step, channels)
        tiled_blocks = np.tile(expanded_blocks, (1, pixel_step, 1, pixel_step, 1))
        
        # Reshape back to image format
        result = tiled_blocks.reshape(new_height, new_width, channels)
        
        # Handle any remaining pixels at edges (pad with nearest values)
        if new_height < height or new_width < width:
            padded_result = np.zeros_like(work_data)
            padded_result[:new_height, :new_width] = result
            
            # Fill right edge
            if new_width < width:
                padded_result[:new_height, new_width:] = result[:, -1:, :]
            
            # Fill bottom edge
            if new_height < height:
                padded_result[new_height:, :new_width] = result[-1:, :, :]
            
            # Fill bottom-right corner
            if new_height < height and new_width < width:
                padded_result[new_height:, new_width:] = result[-1:, -1:, :]
            
            result = padded_result
        
        # Convert back to original shape if needed
        if data.ndim == 2:
            result = result.reshape(height, width)
        
        return result.astype(data.dtype)
    
    def _apply_pixel_step_postprocessing(self, data: np.ndarray, pixel_step: int) -> np.ndarray:
        """
        Apply pixel step postprocessing to ensure chunky pixels are maintained.
        
        This method ensures that the dithered result maintains the chunky pixel
        effect by re-averaging any blocks that may have been modified during dithering.
        
        Args:
            data: Dithered image data
            pixel_step: Size of pixel blocks
            
        Returns:
            Postprocessed image data with consistent chunky pixels
        """
        if pixel_step <= 1:
            return data
        
        # For most dithering algorithms, the preprocessing should be sufficient
        # However, Floyd-Steinberg error diffusion might spread errors across
        # block boundaries, so we need to re-enforce the block structure
        
        # Get image dimensions
        if data.ndim == 2:
            height, width = data.shape
            channels = 1
            work_data = data.reshape(height, width, 1)
        else:
            height, width, channels = data.shape
            work_data = data
        
        # Calculate dimensions for complete blocks
        new_height = (height // pixel_step) * pixel_step
        new_width = (width // pixel_step) * pixel_step
        
        # Process complete blocks
        if new_height > 0 and new_width > 0:
            cropped_data = work_data[:new_height, :new_width]
            
            # Reshape to group pixels into blocks
            blocks_y = new_height // pixel_step
            blocks_x = new_width // pixel_step
            
            blocked_data = cropped_data.reshape(
                blocks_y, pixel_step, blocks_x, pixel_step, channels
            )
            
            # For each block, use the most common value (mode) to maintain
            # the quantized appearance while preserving the chunky effect
            result_blocks = np.zeros_like(blocked_data)
            
            for by in range(blocks_y):
                for bx in range(blocks_x):
                    block = blocked_data[by, :, bx, :, :]
                    
                    # For each channel, find the most common value in the block
                    for c in range(channels):
                        channel_block = block[:, :, c]
                        unique_values, counts = np.unique(channel_block, return_counts=True)
                        most_common_value = unique_values[np.argmax(counts)]
                        result_blocks[by, :, bx, :, c] = most_common_value
            
            # Reshape back to image format
            result = result_blocks.reshape(new_height, new_width, channels)
            
            # Copy result back to work_data
            work_data[:new_height, :new_width] = result
        
        # Convert back to original shape if needed
        if data.ndim == 2:
            work_data = work_data.reshape(height, width)
        
        return work_data.astype(data.dtype)