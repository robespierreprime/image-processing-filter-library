"""
Image input/output functionality.

This module provides functions for loading and saving images with support for
PIL and OpenCV formats, automatic format detection, and numpy array conversions.
"""

import numpy as np
from PIL import Image, ImageFile
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
import cv2
import os

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Supported image formats
SUPPORTED_FORMATS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'
}

# PIL to OpenCV format mapping
PIL_TO_CV2_FORMATS = {
    'RGB': cv2.COLOR_RGB2BGR,
    'RGBA': cv2.COLOR_RGBA2BGRA,
    'L': None  # Grayscale, no conversion needed
}

CV2_TO_PIL_FORMATS = {
    3: 'RGB',  # 3 channels -> RGB
    4: 'RGBA', # 4 channels -> RGBA
    1: 'L'     # 1 channel -> Grayscale
}


class ImageIOError(Exception):
    """Exception raised for image I/O operations."""
    pass


def load_image_pil(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image using PIL and convert to numpy array.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        numpy array with shape (height, width, channels) for RGB/RGBA
        or (height, width) for grayscale
        
    Raises:
        ImageIOError: If the image cannot be loaded
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ImageIOError(f"Image file not found: {file_path}")
        
        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ImageIOError(f"Unsupported image format: {file_path.suffix}")
        
        with Image.open(file_path) as img:
            # Convert to RGB if image has palette or other modes
            if img.mode in ('P', 'CMYK', 'YCbCr'):
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            return img_array
            
    except Exception as e:
        raise ImageIOError(f"Failed to load image {file_path}: {str(e)}")


def load_image_cv2(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image using OpenCV and convert to RGB format.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        numpy array with shape (height, width, channels) in RGB format
        
    Raises:
        ImageIOError: If the image cannot be loaded
    """
    try:
        file_path = str(file_path)
        
        if not os.path.exists(file_path):
            raise ImageIOError(f"Image file not found: {file_path}")
        
        # Load image with OpenCV
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ImageIOError(f"Failed to load image: {file_path}")
        
        # Convert BGR to RGB if image has 3 channels
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        
        return img
        
    except Exception as e:
        raise ImageIOError(f"Failed to load image {file_path}: {str(e)}")


def load_image(file_path: Union[str, Path], backend: str = 'pil') -> np.ndarray:
    """
    Load an image using specified backend.
    
    Args:
        file_path: Path to the image file
        backend: Backend to use ('pil' or 'cv2')
        
    Returns:
        numpy array containing the image data
        
    Raises:
        ImageIOError: If the image cannot be loaded
        ValueError: If backend is not supported
    """
    if backend == 'pil':
        return load_image_pil(file_path)
    elif backend == 'cv2':
        return load_image_cv2(file_path)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'pil' or 'cv2'")


def save_image_pil(image: np.ndarray, file_path: Union[str, Path], 
                   quality: int = 95, **kwargs) -> None:
    """
    Save an image using PIL with format detection and quality settings.
    
    Args:
        image: numpy array containing image data
        file_path: Path where to save the image
        quality: JPEG quality (1-100), ignored for other formats
        **kwargs: Additional PIL save parameters
        
    Raises:
        ImageIOError: If the image cannot be saved
    """
    try:
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Detect format from extension
        format_ext = file_path.suffix.lower()
        if format_ext not in SUPPORTED_FORMATS:
            raise ImageIOError(f"Unsupported image format: {format_ext}")
        
        # Convert numpy array to PIL Image
        pil_image = numpy_to_pil(image)
        
        # Set up save parameters
        save_kwargs = kwargs.copy()
        
        if format_ext in ['.jpg', '.jpeg']:
            save_kwargs['quality'] = quality
            save_kwargs['optimize'] = True
        elif format_ext == '.png':
            save_kwargs['optimize'] = True
        
        # Save the image
        pil_image.save(file_path, **save_kwargs)
        
    except Exception as e:
        raise ImageIOError(f"Failed to save image {file_path}: {str(e)}")


def save_image_cv2(image: np.ndarray, file_path: Union[str, Path], 
                   quality: int = 95) -> None:
    """
    Save an image using OpenCV.
    
    Args:
        image: numpy array containing image data in RGB format
        file_path: Path where to save the image
        quality: JPEG quality (1-100)
        
    Raises:
        ImageIOError: If the image cannot be saved
    """
    try:
        file_path = str(file_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.shape[2] == 4:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            else:
                image_bgr = image
        else:
            image_bgr = image
        
        # Set up compression parameters
        if file_path.lower().endswith(('.jpg', '.jpeg')):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif file_path.lower().endswith('.png'):
            # PNG compression level (0-9)
            compression = int((100 - quality) / 10)
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        else:
            params = []
        
        # Save the image
        success = cv2.imwrite(file_path, image_bgr, params)
        
        if not success:
            raise ImageIOError(f"OpenCV failed to save image: {file_path}")
        
    except Exception as e:
        raise ImageIOError(f"Failed to save image {file_path}: {str(e)}")


def save_image(image: np.ndarray, file_path: Union[str, Path], 
               backend: str = 'pil', quality: int = 95, **kwargs) -> None:
    """
    Save an image using specified backend with automatic format detection.
    
    Args:
        image: numpy array containing image data
        file_path: Path where to save the image
        backend: Backend to use ('pil' or 'cv2')
        quality: Image quality (1-100)
        **kwargs: Additional save parameters (PIL only)
        
    Raises:
        ImageIOError: If the image cannot be saved
        ValueError: If backend is not supported
    """
    if backend == 'pil':
        save_image_pil(image, file_path, quality, **kwargs)
    elif backend == 'cv2':
        save_image_cv2(image, file_path, quality)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'pil' or 'cv2'")


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image with proper format handling.
    
    Args:
        image: numpy array with shape (height, width) for grayscale
               or (height, width, channels) for RGB/RGBA
               
    Returns:
        PIL Image object
        
    Raises:
        ImageIOError: If the array format is not supported
    """
    try:
        # Ensure array is in correct data type
        if image.dtype != np.uint8:
            # Normalize to 0-255 range if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Handle different array shapes
        if len(image.shape) == 2:
            # Grayscale image
            return Image.fromarray(image, mode='L')
        elif len(image.shape) == 3:
            if image.shape[2] == 1:
                # Single channel, squeeze to 2D
                return Image.fromarray(image.squeeze(), mode='L')
            elif image.shape[2] == 3:
                # RGB image
                return Image.fromarray(image, mode='RGB')
            elif image.shape[2] == 4:
                # RGBA image
                return Image.fromarray(image, mode='RGBA')
            else:
                raise ImageIOError(f"Unsupported number of channels: {image.shape[2]}")
        else:
            raise ImageIOError(f"Unsupported array shape: {image.shape}")
            
    except Exception as e:
        raise ImageIOError(f"Failed to convert numpy array to PIL Image: {str(e)}")


def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        numpy array with appropriate shape and dtype
        
    Raises:
        ImageIOError: If conversion fails
    """
    try:
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Ensure uint8 dtype
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        return image_array
        
    except Exception as e:
        raise ImageIOError(f"Failed to convert PIL Image to numpy array: {str(e)}")


def get_image_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about an image file without loading the full image.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dictionary containing image information (width, height, mode, format, etc.)
        
    Raises:
        ImageIOError: If the image info cannot be retrieved
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ImageIOError(f"Image file not found: {file_path}")
        
        with Image.open(file_path) as img:
            info = {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format,
                'size': img.size,
                'filename': str(file_path),
                'file_size': file_path.stat().st_size
            }
            
            # Add additional info if available
            if hasattr(img, 'info'):
                info['metadata'] = img.info
            
            return info
            
    except Exception as e:
        raise ImageIOError(f"Failed to get image info for {file_path}: {str(e)}")


def validate_image_array(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that a numpy array represents a valid image.
    
    Args:
        image: numpy array to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(image, np.ndarray):
        return False, "Input is not a numpy array"
    
    if image.size == 0:
        return False, "Array is empty"
    
    if len(image.shape) not in [2, 3]:
        return False, f"Invalid array shape: {image.shape}. Expected 2D or 3D array"
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False, f"Invalid number of channels: {image.shape[2]}. Expected 1, 3, or 4"
    
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        return False, f"Unsupported dtype: {image.dtype}. Expected uint8, float32, or float64"
    
    return True, ""