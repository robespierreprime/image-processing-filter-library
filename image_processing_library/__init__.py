"""
Image Processing Filter Library

A unified, extensible framework for creating and managing image and video filters.

This library provides:
- A standardized filter interface using typing.Protocol
- Base filter class with common functionality (progress tracking, error handling)
- Filter chaining and execution queue system
- Preset management for saving/loading filter configurations
- Organized filter categories (artistic, enhancement, technical)
- Comprehensive I/O support for images and videos
- Performance optimization features (memory management, chunked processing)

Note: A significant portion of this codebase was generated with assistance from 
Claude.

Quick Start:
    >>> from image_processing_library import BaseFilter, DataType, ColorFormat
    >>> from image_processing_library.media_io import load_image, save_image
    >>> from image_processing_library.filters.artistic import GlitchFilter
    
    >>> # Load an image
    >>> image = load_image("input.jpg")
    
    >>> # Apply a filter
    >>> glitch = GlitchFilter(intensity=0.5)
    >>> result = glitch.apply(image)
    
    >>> # Save the result
    >>> save_image(result, "output.jpg")
"""

# Core components
from .core.protocols import FilterProtocol, DataType, ColorFormat
from .core.base_filter import BaseFilter, FilterMetadata
from .core.execution_queue import ExecutionQueue, FilterStep
from .core.preset_manager import PresetManager, PresetMetadata
from .core.utils import (
    FilterError,
    FilterValidationError,
    FilterExecutionError,
    PresetError,
    UnsupportedFormatError,
    MemoryManager,
    MemoryError,
    ChunkedProcessor,
)

# I/O components
from .media_io import (
    load_image,
    save_image,
    VideoReader,
    VideoWriter,
    ImageIOError,
    VideoIOError,
)

# Filter registry
from .filters import (
    FilterRegistry,
    get_registry,
    register_filter,
    list_filters,
    get_filter,
)

# Filter categories (import the modules to make them available)
from .filters import artistic, enhancement, technical

__version__ = "1.0.0"
__author__ = "Image Processing Library Team"
__license__ = "MIT"
__description__ = (
    "A unified, extensible framework for image and video processing filters"
)

# Main public API
__all__ = [
    # Core interfaces and classes
    "FilterProtocol",
    "DataType",
    "ColorFormat",
    "BaseFilter",
    "FilterMetadata",
    # Execution and management
    "ExecutionQueue",
    "FilterStep",
    "PresetManager",
    "PresetMetadata",
    # I/O operations
    "load_image",
    "save_image",
    "VideoReader",
    "VideoWriter",
    # Filter registry
    "FilterRegistry",
    "get_registry",
    "register_filter",
    "list_filters",
    "get_filter",
    # Filter categories
    "artistic",
    "enhancement",
    "technical",
    # Utilities and exceptions
    "MemoryManager",
    "ChunkedProcessor",
    "FilterError",
    "FilterValidationError",
    "FilterExecutionError",
    "PresetError",
    "UnsupportedFormatError",
    "ImageIOError",
    "VideoIOError",
]
