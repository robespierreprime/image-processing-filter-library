# Image Processing Filter Library

A simple library for image and video processing filters.

## Features

- **Standardized Filter Interface**: Using `typing.Protocol` for consistent filter contracts
- **Base Filter Class**: Common functionality including progress tracking, error handling, and parameter management
- **Filter Chaining**: Execute multiple filters in sequence with execution queues
- **Preset Management**: Save and load filter configurations for reuse
- **Organized Categories**: Filters organized into artistic, enhancement, and technical categories
- **Comprehensive I/O**: Support for various image and video formats
- **Performance Optimization**: Memory management and chunked processing for large files

## Quick Start

### Installation

```bash
git clone https://github.com/robespierreprime/image-processing-filter-library.git
cd image-processing-filter-library
pip install -e .
```

### Basic Usage

```python
from image_processing_library import load_image, save_image
from image_processing_library.filters.artistic import GlitchFilter

# Load an image
image = load_image("input.jpg")

# Apply a filter
glitch = GlitchFilter(intensity=0.5)
result = glitch.apply(image)

# Save the result
save_image(result, "output.jpg")
```

### Filter Chaining

```python
from image_processing_library import ExecutionQueue
from image_processing_library.filters.artistic import GlitchFilter
from image_processing_library.filters.artistic import PrintSimulationFilter

# Create an execution queue
queue = ExecutionQueue()
queue.add_filter(GlitchFilter, {"intensity": 0.3})
queue.add_filter(PrintSimulationFilter, {"band_intensity": 20})

# Execute the filter chain
result = queue.execute(image)
```

## Requirements

- Python 3.8+
- NumPy, Pillow, OpenCV, psutil

## Attribution

A significant portion of this codebase was generated with assistance from Claude. The core architecture, filter implementations, and documentation were developed through AI-assisted programming.