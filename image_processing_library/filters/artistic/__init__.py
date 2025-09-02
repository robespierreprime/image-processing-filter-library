"""
Artistic filters for creative image processing effects.

This package contains filters for artistic and creative effects:
- glitch: Glitch effects and digital artifacts
- print_simulation: Print media simulation effects
- dither_filter: Dithering algorithms for stylized color reduction
- rgb_shift_filter: RGB channel shifting for chromatic aberration
- noise_filter: Various noise types for texture and degradation effects
"""

from .glitch import GlitchFilter
from .print_simulation import PrintSimulationFilter
from .dither_filter import DitherFilter
from .rgb_shift_filter import RGBShiftFilter
from .noise_filter import NoiseFilter

__all__ = [
    'GlitchFilter', 
    'PrintSimulationFilter',
    'DitherFilter',
    'RGBShiftFilter',
    'NoiseFilter'
]