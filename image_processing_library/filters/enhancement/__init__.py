"""
Enhancement filters for improving image quality and appearance.

This package contains filters organized by functionality:
- color_filters: Basic color manipulation (invert, hue rotation, saturation)
- correction_filters: Image correction (gamma correction, contrast)
- blur_filters: Blur effects (gaussian blur, motion blur)
"""

# Import filter modules to ensure they are discoverable by the registry
from . import color_filters
from . import correction_filters
from . import blur_filters

__all__ = [
    'color_filters',
    'correction_filters', 
    'blur_filters'
]