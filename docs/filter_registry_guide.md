# Filter Registry Guide

This guide explains how filters are registered and discovered in the image processing library.

## Filter Registration System

The library uses an automatic filter registration system that discovers and registers filters when modules are imported. All filters are organized by categories and can be discovered programmatically.

## Available Filters

### Enhancement Filters

Enhancement filters improve image quality and appearance:

| Filter Name | Class | Description | Parameters |
|-------------|-------|-------------|------------|
| `invert` | `InvertFilter` | Inverts RGB color values | None |
| `gamma_correction` | `GammaCorrectionFilter` | Applies gamma correction | `gamma` (0.1-3.0) |
| `contrast` | `ContrastFilter` | Adjusts image contrast | `contrast_factor` (0.0-3.0) |
| `saturation` | `SaturationFilter` | Adjusts color saturation | `saturation_factor` (0.0-3.0) |
| `hue_rotation` | `HueRotationFilter` | Rotates hue values | `rotation_degrees` (0-360) |
| `gaussian_blur` | `GaussianBlurFilter` | Applies gaussian blur | `sigma` (0.0-10.0) |
| `motion_blur` | `MotionBlurFilter` | Creates directional blur | `distance` (0-50), `angle` (0-360) |

### Artistic Filters

Artistic filters create creative and stylized effects:

| Filter Name | Class | Description | Parameters |
|-------------|-------|-------------|------------|
| `rgb_shift` | `RGBShiftFilter` | Shifts RGB channels | `red_shift`, `green_shift`, `blue_shift` |
| `noise` | `NoiseFilter` | Adds various noise types | `noise_type`, `intensity`, `salt_pepper_ratio` |
| `Glitch Effect` | `GlitchFilter` | Creates glitch effects | Various glitch parameters |
| `Print Simulation` | `PrintSimulationFilter` | Simulates print media | Print simulation parameters |

### Technical Filters

Technical filters for specialized processing:

| Filter Name | Class | Description | Parameters |
|-------------|-------|-------------|------------|
| `Background Remover` | `BackgroundRemoverFilter` | Removes image backgrounds | Background removal parameters |

## Using the Filter Registry

### Basic Usage

```python
from image_processing_library.filters import get_registry, auto_discover_filters

# Auto-discover all filters
auto_discover_filters()

# Get the registry
registry = get_registry()

# List all filters
all_filters = registry.list_filters()
print("Available filters:", all_filters)

# List filters by category
enhancement_filters = registry.list_filters(category='enhancement')
artistic_filters = registry.list_filters(category='artistic')

# Get a specific filter
filter_class = registry.get_filter('invert')
filter_instance = registry.create_filter_instance('invert')
```

### Filter Discovery by Criteria

```python
from image_processing_library.core.protocols import DataType, ColorFormat

# Find filters that support specific data types
image_filters = registry.list_filters(data_type=DataType.IMAGE)

# Find filters that support specific color formats
rgb_filters = registry.list_filters(color_format=ColorFormat.RGB)

# Combine criteria
rgb_enhancement_filters = registry.list_filters(
    category='enhancement',
    color_format=ColorFormat.RGB
)
```

### Filter Metadata

```python
# Get detailed information about a filter
metadata = registry.get_filter_metadata('gamma_correction')
print(f"Category: {metadata['category']}")
print(f"Data Type: {metadata['data_type']}")
print(f"Color Format: {metadata['color_format']}")
print(f"Description: {metadata['description']}")
```

### Creating Filter Instances

```python
# Simple filter (no parameters)
invert_filter = registry.create_filter_instance('invert')

# Parameterized filter
gamma_filter = registry.create_filter_instance(
    'gamma_correction', 
    gamma=1.2
)

blur_filter = registry.create_filter_instance(
    'gaussian_blur',
    sigma=2.0
)
```

## Filter Categories

### Enhancement Category
Filters that improve image quality:
- Color corrections (gamma, contrast)
- Color manipulations (invert, saturation, hue rotation)
- Blur effects (gaussian, motion)

### Artistic Category
Filters that create creative effects:
- Stylization (dither, glitch)
- Color effects (RGB shift, noise)
- Print simulation

### Technical Category
Filters for specialized processing:
- Background removal
- Object detection/segmentation

## Automatic Registration

Filters are automatically registered when their modules are imported. The registration system:

1. Scans filter modules for classes that implement `FilterProtocol`
2. Extracts metadata (name, category, data type, color format)
3. Registers filters in the global registry
4. Organizes filters by category

### Registration Decorator

Filters use the `@register_filter` decorator for automatic registration:

```python
from image_processing_library.filters.registry import register_filter
from image_processing_library.core.base_filter import BaseFilter

@register_filter(category="enhancement")
class MyFilter(BaseFilter):
    name = "my_filter"
    category = "enhancement"
    data_type = DataType.IMAGE
    color_format = ColorFormat.RGB
    
    def apply(self, data):
        # Filter implementation
        pass
```

## Filter Validation

The registry validates filter implementations:

```python
# Validate a filter class
validation_result = registry.validate_filter_metadata(MyFilterClass)

if validation_result['valid']:
    print("Filter is valid")
else:
    print("Validation errors:", validation_result['errors'])
    print("Warnings:", validation_result['warnings'])
```

## Best Practices

1. **Use Auto-Discovery**: Call `auto_discover_filters()` at application startup
2. **Check Registration**: Verify filters are registered before using them
3. **Handle Errors**: Use try-catch when creating filter instances
4. **Validate Parameters**: Check parameter ranges before creating instances
5. **Use Metadata**: Query filter capabilities before applying them

## Troubleshooting

### Filter Not Found
If a filter is not found:
1. Ensure the module is imported
2. Check that `auto_discover_filters()` was called
3. Verify the filter name (use `list_filters()` to see available names)

### Duplicate Registration Warnings
Warnings about duplicate registration are normal when:
- Modules are imported multiple times
- Tests import filters directly
- The same filter is registered through different paths

These warnings don't affect functionality.

### Parameter Errors
When creating filter instances:
- Check parameter names and types
- Verify parameter ranges using filter documentation
- Use `get_filter_metadata()` to see expected parameters