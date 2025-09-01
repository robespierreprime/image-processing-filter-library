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