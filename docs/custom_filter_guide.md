# Creating Custom Filters

This guide explains how to create custom filters using the Image Processing Filter Library framework.

## Filter Architecture

All filters in the library follow a standardized interface defined by the `FilterProtocol`. The easiest way to create a custom filter is to inherit from the `BaseFilter` class, which provides common functionality.

## Basic Filter Structure

```python
from image_processing_library import BaseFilter, DataType, ColorFormat
import numpy as np

class MyCustomFilter(BaseFilter):
    def __init__(self, parameter1=default_value, parameter2=default_value):
        super().__init__(
            name="my_custom_filter",
            data_type=DataType.IMAGE,  # or DataType.VIDEO
            color_format=ColorFormat.RGB,  # RGB, RGBA, or GRAYSCALE
            category="enhancement",  # artistic, enhancement, or technical
            parameter1=parameter1,
            parameter2=parameter2
        )
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the filter to input data."""
        self.validate_input(data)
        
        def _apply_filter():
            self._update_progress(0.0)
            
            # Your filter logic here
            result = data.copy()
            # ... processing ...
            
            self._update_progress(1.0)
            return result
        
        return self._measure_execution_time(_apply_filter)
```

## Key Components

### 1. Initialization
- Call `super().__init__()` with filter metadata
- Store parameters for later use
- Specify data type (IMAGE/VIDEO) and color format requirements

### 2. Apply Method
- Always call `self.validate_input(data)` first
- Use `self._measure_execution_time()` wrapper for timing
- Update progress with `self._update_progress()`
- Return processed numpy array

### 3. Progress Tracking
Use `self._update_progress(progress)` where progress is 0.0 to 1.0:

```python
self._update_progress(0.0)    # Start
self._update_progress(0.5)    # Halfway
self._update_progress(1.0)    # Complete
```

## Best Practices

1. **Input Validation**: Always validate input before processing
2. **Progress Updates**: Provide regular progress updates for long operations
3. **Error Handling**: Let the base class handle timing and errors
4. **Memory Efficiency**: Consider in-place operations when possible
5. **Parameter Access**: Use `self.parameters.get('param_name', default)`

## Registration

Register your filter for use with the filter registry:

```python
from image_processing_library.filters import get_registry
registry = get_registry()
registry.register_filter(MyCustomFilter)
```