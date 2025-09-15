# Migration Guide

This guide helps you migrate existing filter code to use the new Image Processing Filter Library framework.

## Overview

The library provides a unified interface for image and video processing filters. If you have existing filter functions or classes, this guide will help you adapt them to the new architecture.

## Migrating Function-Based Filters

### Before (Function-based)
```python
def my_old_filter(image, param1=0.5, param2=10):
    # Direct image processing
    result = image.copy()
    # ... processing logic ...
    return result
```

### After (Class-based)
```python
from image_processing_library import BaseFilter, DataType, ColorFormat

class MyNewFilter(BaseFilter):
    def __init__(self, param1=0.5, param2=10):
        super().__init__(
            name="my_new_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="enhancement",
            param1=param1,
            param2=param2
        )
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        self.validate_input(data)
        
        def _apply():
            self._update_progress(0.0)
            result = data.copy()
            
            # Your original processing logic here
            param1 = self.parameters['param1']
            param2 = self.parameters['param2']
            # ... processing ...
            
            self._update_progress(1.0)
            return result
        
        return self._measure_execution_time(_apply)
```

## Migrating Class-Based Filters

### Before (Custom class)
```python
class OldFilter:
    def __init__(self, param=0.5):
        self.param = param
    
    def process(self, image):
        # Processing logic
        return processed_image
```

### After (BaseFilter inheritance)
```python
class NewFilter(BaseFilter):
    def __init__(self, param=0.5):
        super().__init__(
            name="new_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="artistic",
            param=param
        )
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        self.validate_input(data)
        
        def _apply():
            # Use your original processing logic
            param = self.parameters['param']
            # ... original logic ...
            return result
        
        return self._measure_execution_time(_apply)
```

## Key Changes

1. **Method Name**: Change from custom method names to `apply()`
2. **Inheritance**: Inherit from `BaseFilter` instead of custom base classes
3. **Initialization**: Call `super().__init__()` with metadata
4. **Parameters**: Access via `self.parameters` dictionary
5. **Progress**: Add progress tracking with `self._update_progress()`
6. **Validation**: Use built-in `self.validate_input()`
7. **Timing**: Wrap logic in `self._measure_execution_time()`

## Benefits of Migration

- Standardized interface across all filters
- Built-in progress tracking and error handling
- Automatic input validation
- Integration with execution queues and presets
- Performance monitoring and optimization
- Consistent parameter management
#
# New Enhancement and Artistic Filters (v2.0+)

### New Filters Available

The library now includes a comprehensive set of enhancement and artistic filters:

#### Enhancement Filters
- `InvertFilter` - Invert RGB color values
- `GammaCorrectionFilter` - Brightness adjustment via gamma correction
- `ContrastFilter` - Contrast adjustment
- `SaturationFilter` - Color saturation adjustment in HSV space
- `HueRotationFilter` - Hue rotation in HSV space
- `GaussianBlurFilter` - Gaussian blur with convolution
- `MotionBlurFilter` - Directional motion blur

#### Artistic Filters
- `RGBShiftFilter` - Independent RGB channel shifting
- `NoiseFilter` - Various noise types (Gaussian, Salt-pepper, Uniform)

### Updated Existing Filters

#### GlitchFilter
The `GlitchFilter` has been enhanced to use the new `RGBShiftFilter` internally:

**Before:**
```python
# Internal color shifting was hardcoded
glitch = GlitchFilter(intensity=0.5, shift_amount=10)
```

**After:**
```python
# Same API, but now uses RGBShiftFilter internally for better quality
glitch = GlitchFilter(intensity=0.5, shift_amount=10, corruption_probability=0.2)
# New parameter: corruption_probability for JPEG-like artifacts
```

#### PrintSimulationFilter
The `PrintSimulationFilter` now uses the new `NoiseFilter` internally:

**Before:**
```python
# Internal noise generation was basic
print_sim = PrintSimulationFilter(dot_gain=0.2, paper_texture=0.3)
```

**After:**
```python
# Same API, but now uses NoiseFilter internally for better texture simulation
print_sim = PrintSimulationFilter(
    dot_gain=0.2, 
    paper_texture=0.3, 
    ink_bleeding=0.1  # New parameter
)
```

### Filter Import Changes

#### New Import Paths
```python
# Enhancement filters
from image_processing_library.filters.enhancement.color_filters import (
    InvertFilter, SaturationFilter, HueRotationFilter
)
from image_processing_library.filters.enhancement.correction_filters import (
    GammaCorrectionFilter, ContrastFilter
)
from image_processing_library.filters.enhancement.blur_filters import (
    GaussianBlurFilter, MotionBlurFilter
)

# Artistic filters
from image_processing_library.filters.artistic.rgb_shift_filter import RGBShiftFilter
from image_processing_library.filters.artistic.noise_filter import NoiseFilter
```

#### Existing Imports (Still Valid)
```python
# These imports continue to work
from image_processing_library.filters.artistic.glitch import GlitchFilter
from image_processing_library.filters.artistic.print_simulation import PrintSimulationFilter
from image_processing_library.filters.technical.background_remover import BackgroundRemoverFilter
```

### Filter Registry Updates

All new filters are automatically registered and discoverable:

```python
from image_processing_library.filters.registry import list_filters

# List all enhancement filters
enhancement_filters = list_filters(category="enhancement")
print(f"Enhancement filters: {len(enhancement_filters)}")

# List all artistic filters  
artistic_filters = list_filters(category="artistic")
print(f"Artistic filters: {len(artistic_filters)}")
```

### Parameter Validation Improvements

All new filters include comprehensive parameter validation:

```python
# This will raise FilterValidationError with clear message
try:
    gamma_filter = GammaCorrectionFilter(gamma=-1.0)  # Invalid: negative gamma
except FilterValidationError as e:
    print(f"Validation error: {e}")

# This will raise FilterValidationError with range information
try:
    contrast_filter = ContrastFilter(contrast_factor=5.0)  # Invalid: > 3.0
except FilterValidationError as e:
    print(f"Validation error: {e}")
```

### Memory Management Improvements

All new filters support improved memory management:

```python
# Automatic chunked processing for large images
large_image = load_image("very_large_image.jpg")  # e.g., 8000x6000 pixels

# This will automatically use chunked processing if needed
blur_filter = GaussianBlurFilter(sigma=2.0)
result = blur_filter.apply(large_image)  # Handles memory efficiently
```

### Performance Optimizations

#### Identity Parameter Optimization
Filters now skip processing when parameters indicate no change:

```python
# These operations return the original image unchanged (fast)
gamma_filter = GammaCorrectionFilter(gamma=1.0)  # No change
contrast_filter = ContrastFilter(contrast_factor=1.0)  # No change
blur_filter = GaussianBlurFilter(sigma=0.0)  # No blur
```

#### Vectorized Operations
All new filters use optimized numpy operations:

```python
# These operations are now significantly faster
saturation_filter = SaturationFilter(saturation_factor=1.5)
hue_filter = HueRotationFilter(rotation_degrees=120)
```

### Backward Compatibility

#### Existing Code Continues to Work
All existing filter usage patterns continue to work without changes:

```python
# This code from v1.x continues to work in v2.0+
from image_processing_library.filters.artistic.glitch import GlitchFilter

glitch = GlitchFilter(intensity=0.5, shift_amount=10)
result = glitch.apply(image)
```

#### Filter Registry Compatibility
Existing filter discovery code continues to work:

```python
# This v1.x code continues to work
from image_processing_library.filters.registry import get_filter

GlitchFilterClass = get_filter("glitch")
glitch_instance = GlitchFilterClass(intensity=0.3)
```

### Recommended Migration Steps

1. **Update imports** to use the new organized structure (optional but recommended)
2. **Review parameter ranges** for any custom filter usage
3. **Test with large images** to benefit from improved memory management
4. **Explore new filters** for enhanced functionality
5. **Update documentation** to reference new filter capabilities

### Example Migration

#### Before (v1.x)
```python
from image_processing_library.filters.artistic.glitch import GlitchFilter
from image_processing_library.core.execution_queue import ExecutionQueue

# Limited filter options
queue = ExecutionQueue()
queue.add_filter(GlitchFilter, {"intensity": 0.5})
result = queue.execute(image)
```

#### After (v2.0+)
```python
from image_processing_library.filters.enhancement.color_filters import SaturationFilter
from image_processing_library.filters.enhancement.correction_filters import ContrastFilter
from image_processing_library.filters.artistic.rgb_shift_filter import RGBShiftFilter
from image_processing_library.filters.artistic.noise_filter import NoiseFilter
from image_processing_library.core.execution_queue import ExecutionQueue

# Rich filter ecosystem with precise control
queue = ExecutionQueue()
queue.add_filter(ContrastFilter, {"contrast_factor": 1.2})
queue.add_filter(SaturationFilter, {"saturation_factor": 1.3})
queue.add_filter(RGBShiftFilter, {
    "red_shift": (2, 0), 
    "green_shift": (0, 0), 
    "blue_shift": (-2, 0)
})
queue.add_filter(NoiseFilter, {"noise_type": "gaussian", "intensity": 0.02})

result = queue.execute(image)
```