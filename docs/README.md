# Documentation

Comprehensive documentation for the image processing library.

## Documentation Files

### Core Documentation
- [API Reference](api_reference.md) - Complete function and class documentation
- [Filter Guide](filter_guide.md) - Comprehensive guide to all available filters with examples
- [Filter Registry Guide](filter_registry_guide.md) - Working with the filter registry system
- [Custom Filter Guide](custom_filter_guide.md) - Creating your own filters
- [Migration Guide](migration_guide.md) - Upgrading from older versions

### Examples and Tutorials
- [Examples](../examples/README.md) - Usage examples and demo scripts
- [Enhancement Filters Demo](../examples/enhancement_filters_demo.py) - Complete enhancement filter examples
- [Artistic Filters Demo](../examples/artistic_filters_demo.py) - Complete artistic filter examples

## Quick Start

### Available Filter Categories

#### Enhancement Filters
Improve or adjust basic image properties:
- **InvertFilter** - Create negative effects
- **GammaCorrectionFilter** - Brightness adjustment
- **ContrastFilter** - Contrast adjustment
- **SaturationFilter** - Color saturation control
- **HueRotationFilter** - Color shifting
- **GaussianBlurFilter** - Smooth blur effects
- **MotionBlurFilter** - Directional blur effects

#### Artistic Filters
Create special effects and stylized appearances:
- **DitherFilter** - Retro dithering effects (Floyd-Steinberg, Bayer, Random)
- **RGBShiftFilter** - Chromatic aberration effects
- **NoiseFilter** - Various noise types (Gaussian, Salt-pepper, Uniform)
- **GlitchFilter** - Digital glitch effects
- **PrintSimulationFilter** - Realistic printing artifacts

#### Technical Filters
Specialized processing tasks:
- **BackgroundRemoverFilter** - AI-powered background removal

### Basic Usage

```python
from image_processing_library.filters.enhancement.color_filters import InvertFilter
from image_processing_library.media_io.image_io import load_image, save_image

# Load image
image = load_image("input.jpg")

# Apply filter
invert_filter = InvertFilter()
result = invert_filter.apply(image)

# Save result
save_image(result, "output.jpg")
```

### Filter Chaining

```python
from image_processing_library.core.execution_queue import ExecutionQueue
from image_processing_library.filters.enhancement.correction_filters import GammaCorrectionFilter, ContrastFilter

# Create execution queue
queue = ExecutionQueue()
queue.add_filter(GammaCorrectionFilter, {"gamma": 0.8})
queue.add_filter(ContrastFilter, {"contrast_factor": 1.2})

# Execute chain
result = queue.execute(image)
```

## How It Works

- **Unified Interface**: All filters implement a common `FilterProtocol` interface
- **Filter Chaining**: Combine multiple filters using `ExecutionQueue`
- **Multi-Format Support**: Works with images and videos
- **Automatic Registration**: Filters are automatically discovered and registered
- **Memory Efficient**: Built-in chunked processing for large images
- **Progress Tracking**: Real-time progress updates for long operations
- **Parameter Validation**: Comprehensive input validation with clear error messages

## Filter Categories

- **Enhancement**: Basic image improvements (color, brightness, blur)
- **Artistic**: Creative effects and stylization
- **Technical**: Specialized processing (background removal, etc.)

## Getting Help

1. Check the [Filter Guide](filter_guide.md) for detailed filter documentation
2. Run the demo scripts in [examples/](../examples/) to see filters in action
3. Review the [API Reference](api_reference.md) for technical details
4. See the [Migration Guide](migration_guide.md) if upgrading from older versions