# Examples

Example scripts showing how to use the library.

## Scripts

### Basic Usage
- `basic_filter_usage.py` - Basic filter usage
- `filter_chaining.py` - Chain multiple filters  
- `preset_management.py` - Save/load filter presets
- `custom_filter_creation.py` - Create your own filters
- `video_processing.py` - Process videos

### Filter Demonstrations
- `enhancement_filters_demo.py` - Complete demonstration of all enhancement filters
- `artistic_filters_demo.py` - Complete demonstration of all artistic filters

### Advanced Features
- `example_chunked_processing.py` - Memory-efficient processing of large images
- `example_execution_queue.py` - Advanced filter chaining and execution
- `example_filter_registry.py` - Working with the filter registry

## Run examples

```bash
# Basic usage
python examples/basic_filter_usage.py

# Enhancement filters demo (creates 15+ example images)
python examples/enhancement_filters_demo.py

# Artistic filters demo (creates 20+ example images)
python examples/artistic_filters_demo.py

# Other examples
python examples/filter_chaining.py
python examples/preset_management.py
```

Examples create their own test images/videos and save outputs in the examples folder.

## Enhancement Filters Demo

The `enhancement_filters_demo.py` script demonstrates:
- **Color Filters**: Invert, saturation adjustment, hue rotation
- **Correction Filters**: Gamma correction, contrast adjustment
- **Blur Filters**: Gaussian blur, motion blur
- **Combined Effects**: Photo enhancement and vintage workflows

## Artistic Filters Demo

The `artistic_filters_demo.py` script demonstrates:
- **Dithering**: Floyd-Steinberg, Bayer, and random dithering
- **RGB Shift**: Chromatic aberration and glitch effects
- **Noise**: Gaussian, salt-pepper, and uniform noise
- **Special Effects**: Glitch and print simulation filters
- **Artistic Workflows**: Retro computer graphics, glitch art, vintage print effects