"""
Background removal filter using AI models for automatic background detection and removal.

This module provides a filter that removes backgrounds from images using
pre-trained AI models through the rembg library.
"""

import numpy as np
from PIL import Image
import io
from typing import Dict, Any, Optional

from ...core.base_filter import BaseFilter
from ...core.protocols import DataType, ColorFormat
from ...core.utils import FilterExecutionError


class BackgroundRemoverFilter(BaseFilter):
    """
    Background removal filter using AI models.
    
    This filter automatically detects and removes backgrounds from images
    using pre-trained AI models. It supports various models optimized for
    different types of subjects (general objects, humans, etc.).
    
    The filter outputs images with transparent backgrounds (RGBA format).
    """
    
    # Available models and their descriptions
    AVAILABLE_MODELS = {
        'u2net': 'General purpose model (default)',
        'u2netp': 'Lightweight version of u2net',
        'u2net_human_seg': 'Optimized for human subjects',
        'silueta': 'Good for objects with clear silhouettes',
        'isnet-general-use': 'High quality general purpose model'
    }
    
    def __init__(self,
                 model: str = 'u2net',
                 alpha_matting: bool = False,
                 alpha_matting_foreground_threshold: int = 240,
                 alpha_matting_background_threshold: int = 10,
                 **kwargs):
        """
        Initialize the background remover filter.
        
        Args:
            model: AI model to use for background removal
            alpha_matting: Enable alpha matting for better edge quality
            alpha_matting_foreground_threshold: Foreground threshold for alpha matting
            alpha_matting_background_threshold: Background threshold for alpha matting
            **kwargs: Additional parameters passed to BaseFilter
        """
        super().__init__(
            name="Background Remover",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,  # Input RGB, output RGBA
            category="technical",
            model=model,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            **kwargs
        )
        
        # Validate parameters
        self._validate_parameters()
        
        # Initialize rembg session (lazy loading)
        self._session = None
    
    def _validate_parameters(self) -> None:
        """Validate filter parameters are within acceptable ranges."""
        params = self.parameters
        
        if params['model'] not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model '{params['model']}'. "
                           f"Available models: {list(self.AVAILABLE_MODELS.keys())}")
        
        if not isinstance(params['alpha_matting'], bool):
            raise ValueError("alpha_matting must be a boolean")
        
        if not (0 <= params['alpha_matting_foreground_threshold'] <= 255):
            raise ValueError("alpha_matting_foreground_threshold must be between 0 and 255")
        
        if not (0 <= params['alpha_matting_background_threshold'] <= 255):
            raise ValueError("alpha_matting_background_threshold must be between 0 and 255")
    
    def _get_session(self):
        """Get or create rembg session (lazy loading)."""
        if self._session is None:
            try:
                from rembg import new_session
                self._session = new_session(self.parameters['model'])
            except ImportError:
                raise FilterExecutionError(
                    "rembg library not found. Install it with: pip install rembg"
                )
            except Exception as e:
                raise FilterExecutionError(f"Failed to create rembg session: {e}")
        
        return self._session
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply background removal to the input image data.
        
        Args:
            data: Input image as numpy array (H, W, 3) with RGB values 0-255
            **kwargs: Additional parameters (merged with instance parameters)
            
        Returns:
            Processed image array with transparent background (H, W, 4) RGBA format
            
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
            
            # Convert numpy array to PIL Image
            try:
                if processed_data.ndim == 2:
                    # Grayscale to RGB
                    processed_data = np.stack([processed_data] * 3, axis=-1)
                
                pil_image = Image.fromarray(processed_data)
                
                # Convert to bytes for rembg
                img_bytes = io.BytesIO()
                pil_image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                input_data = img_bytes.read()
                
            except Exception as e:
                raise FilterExecutionError(f"Failed to convert image to PIL format: {e}")
            
            self._update_progress(0.2)
            
            # Remove background using rembg
            try:
                from rembg import remove
                
                session = self._get_session()
                
                # Apply background removal
                if self.parameters['alpha_matting']:
                    # Use alpha matting for better edge quality
                    output_data = remove(
                        input_data,
                        session=session,
                        alpha_matting=True,
                        alpha_matting_foreground_threshold=self.parameters['alpha_matting_foreground_threshold'],
                        alpha_matting_background_threshold=self.parameters['alpha_matting_background_threshold']
                    )
                else:
                    output_data = remove(input_data, session=session)
                
            except ImportError:
                raise FilterExecutionError(
                    "rembg library not found. Install it with: pip install rembg"
                )
            except Exception as e:
                raise FilterExecutionError(f"Background removal failed: {e}")
            
            self._update_progress(0.8)
            
            # Convert result back to numpy array
            try:
                output_image = Image.open(io.BytesIO(output_data))
                
                # Ensure output is RGBA
                if output_image.mode != 'RGBA':
                    output_image = output_image.convert('RGBA')
                
                result = np.array(output_image)
                
            except Exception as e:
                raise FilterExecutionError(f"Failed to convert result to numpy array: {e}")
            
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
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data with support for RGB and grayscale images.
        
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
        
        # If 3D, check channel count (RGB or RGBA)
        if data.ndim == 3 and data.shape[-1] not in [3, 4]:
            from ...core.utils import FilterValidationError
            raise FilterValidationError(
                f"3D image must have 3 or 4 channels, got {data.shape[-1]}"
            )
        
        # Validate data range
        self._validate_data_range(data)
        
        return True
    
    def set_parameters(self, **kwargs) -> None:
        """
        Update filter parameters with validation.
        
        Args:
            **kwargs: Parameter names and values to update
            
        Raises:
            ValueError: If invalid parameter names or values provided
        """
        # Validate parameter names
        valid_params = {
            'model', 'alpha_matting', 'alpha_matting_foreground_threshold',
            'alpha_matting_background_threshold'
        }
        
        invalid_params = set(kwargs.keys()) - valid_params
        if invalid_params:
            raise ValueError(f"Invalid parameters: {invalid_params}")
        
        # If model is being changed, reset session
        if 'model' in kwargs and kwargs['model'] != self.parameters.get('model'):
            self._session = None
        
        # Update parameters
        super().set_parameters(**kwargs)
        
        # Validate new parameter values
        self._validate_parameters()
    
    def get_available_models(self) -> Dict[str, str]:
        """
        Get available background removal models.
        
        Returns:
            Dictionary mapping model names to descriptions
        """
        return self.AVAILABLE_MODELS.copy()
    
    def get_model_info(self) -> str:
        """
        Get information about the currently selected model.
        
        Returns:
            Description of the current model
        """
        current_model = self.parameters['model']
        return f"{current_model}: {self.AVAILABLE_MODELS.get(current_model, 'Unknown model')}"