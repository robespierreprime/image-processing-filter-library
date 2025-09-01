"""
Protocol definitions and enums for the image processing library.

Defines the FilterProtocol interface and supporting enumerations.
"""

from typing import Protocol, Any, Dict
from enum import Enum
import numpy as np


class DataType(Enum):
    """Enumeration for supported data types."""

    IMAGE = "image"
    VIDEO = "video"


class ColorFormat(Enum):
    """Enumeration for supported color formats."""

    RGB = "rgb"
    RGBA = "rgba"
    GRAYSCALE = "grayscale"


class FilterProtocol(Protocol):
    """
    Protocol defining the interface for all filters.

    This protocol establishes the contract that all filters must follow,
    ensuring consistency across different filter implementations while
    maintaining flexibility through duck typing.
    """

    name: str
    data_type: DataType
    color_format: ColorFormat
    category: str

    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply the filter to input data.

        Args:
            data: Input numpy array containing image or video data
            **kwargs: Additional filter-specific parameters

        Returns:
            Processed numpy array with filter applied

        Raises:
            FilterValidationError: If input data is invalid
            FilterExecutionError: If filter processing fails
        """
        ...

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current filter parameters.

        Returns:
            Dictionary containing current parameter values
        """
        ...

    def set_parameters(self, **kwargs) -> None:
        """
        Update filter parameters.

        Args:
            **kwargs: Parameter names and values to update

        Raises:
            ValueError: If invalid parameter names or values provided
        """
        ...

    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data format and dimensions.

        Args:
            data: Input numpy array to validate

        Returns:
            True if input is valid

        Raises:
            FilterValidationError: If input data is invalid
        """
        ...
