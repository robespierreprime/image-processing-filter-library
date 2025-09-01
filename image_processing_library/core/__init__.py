"""
Core module for the image processing library.

Contains the fundamental interfaces, base classes, and utilities.
"""

from .protocols import FilterProtocol, DataType, ColorFormat
from .base_filter import BaseFilter, FilterMetadata
from .utils import (
    FilterError,
    FilterValidationError,
    FilterExecutionError,
    PresetError,
    UnsupportedFormatError,
    MemoryManager,
    MemoryError,
    ChunkedProcessor,
)

from .execution_queue import ExecutionQueue, FilterStep
from .preset_manager import PresetManager, PresetMetadata

__all__ = [
    "FilterProtocol",
    "DataType",
    "ColorFormat", 
    "BaseFilter",
    "FilterMetadata",
    "ExecutionQueue",
    "FilterStep",
    "PresetManager",
    "PresetMetadata",
    "FilterError",
    "FilterValidationError",
    "FilterExecutionError",
    "PresetError",
    "UnsupportedFormatError",
    "MemoryManager",
    "MemoryError",
    "ChunkedProcessor",
]