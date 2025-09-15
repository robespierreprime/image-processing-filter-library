# Design Document

## Overview

This design outlines the implementation of a comprehensive set of image enhancement and manipulation filters for the existing image processing library. The new filters will be organized into logical categories and integrate seamlessly with the existing BaseFilter architecture and FilterRegistry system.

The implementation will add 9 new filter classes across two main categories:
- **Enhancement filters**: Basic color manipulations and corrections (invert, hue rotation, saturation, gamma correction, contrast)
- **Effect filters**: Advanced processing and artistic effects (dither, blur, RGB shift, noise)

Additionally, two existing artistic filters (glitch and print simulation) will be refactored to utilize the new RGB shift and noise filters as internal components.

## Architecture

### Filter Organization

The new filters will be organized within the existing directory structure:

```
image_processing_library/filters/
├── enhancement/
│   ├── __init__.py
│   ├── color_filters.py      # InvertFilter, HueRotationFilter, SaturationFilter
│   ├── correction_filters.py # GammaCorrectionFilter, ContrastFilter
│   └── blur_filters.py       # GaussianBlurFilter, MotionBlurFilter
├── artistic/
│   ├── __init__.py
│   ├── glitch.py            # Updated to use RGBShiftFilter
│   ├── print_simulation.py  # Updated to use NoiseFilter
│   ├── dither_filter.py     # DitherFilter
│   ├── rgb_shift_filter.py  # RGBShiftFilter
│   └── noise_filter.py      # NoiseFilter
```

### Class Hierarchy

All new filters will inherit from `BaseFilter` and implement the `FilterProtocol`:

```
BaseFilter (existing)
├── InvertFilter
├── HueRotationFilter
├── SaturationFilter
├── GammaCorrectionFilter
├── ContrastFilter
├── GaussianBlurFilter
├── MotionBlurFilter
├── DitherFilter
├── RGBShiftFilter
└── NoiseFilter
```

### Filter Categories

- **Enhancement**: `invert`, `hue_rotation`, `saturation`, `gamma_correction`, `contrast`, `gaussian_blur`, `motion_blur`
- **Artistic**: `dither`, `rgb_shift`, `noise`

## Components and Interfaces

### 1. Color Manipulation Filters

#### InvertFilter
- **Purpose**: Inverts RGB color values
- **Parameters**: None (simple inversion)
- **Algorithm**: `output = 255 - input` for uint8 images
- **Color Format**: RGB, RGBA (preserves alpha)

#### HueRotationFilter
- **Purpose**: Rotates hue values in HSV color space
- **Parameters**: 
  - `rotation_degrees` (float, 0-360): Degrees to rotate hue
- **Algorithm**: Convert RGB → HSV, rotate H channel, convert back to RGB
- **Color Format**: RGB, RGBA

#### SaturationFilter
- **Purpose**: Adjusts color saturation
- **Parameters**:
  - `saturation_factor` (float, 0.0-3.0): Saturation multiplier (1.0 = no change)
- **Algorithm**: Convert RGB → HSV, multiply S channel, convert back to RGB
- **Color Format**: RGB, RGBA

### 2. Correction Filters

#### GammaCorrectionFilter
- **Purpose**: Applies gamma correction for brightness adjustment
- **Parameters**:
  - `gamma` (float, 0.1-3.0): Gamma value (1.0 = no change, <1 = brighter, >1 = darker)
- **Algorithm**: `output = (input / 255.0) ^ (1/gamma) * 255`
- **Color Format**: RGB, RGBA, GRAYSCALE

#### ContrastFilter
- **Purpose**: Adjusts image contrast
- **Parameters**:
  - `contrast_factor` (float, 0.0-3.0): Contrast multiplier (1.0 = no change)
- **Algorithm**: `output = (input - 128) * contrast_factor + 128`
- **Color Format**: RGB, RGBA, GRAYSCALE

### 3. Blur Filters

#### GaussianBlurFilter
- **Purpose**: Applies gaussian blur using convolution
- **Parameters**:
  - `sigma` (float, 0.0-10.0): Standard deviation for gaussian kernel
  - `kernel_size` (int, auto-calculated): Size of convolution kernel
- **Algorithm**: Convolve with 2D gaussian kernel
- **Color Format**: RGB, RGBA, GRAYSCALE
- **Implementation**: Use scipy.ndimage.gaussian_filter or custom implementation

#### MotionBlurFilter
- **Purpose**: Creates directional motion blur
- **Parameters**:
  - `distance` (int, 0-50): Blur distance in pixels
  - `angle` (float, 0-360): Blur direction in degrees
- **Algorithm**: Create linear kernel based on angle and distance, apply convolution
- **Color Format**: RGB, RGBA, GRAYSCALE

### 4. Effect Filters

#### DitherFilter
- **Purpose**: Applies dithering with multiple algorithms
- **Parameters**:
  - `pattern_type` (str): "floyd_steinberg", "bayer", "random"
  - `levels` (int, 2-256): Number of quantization levels per channel
  - `bayer_size` (int, 2-8): Size of Bayer matrix (for Bayer dithering)
- **Algorithms**:
  - Floyd-Steinberg: Error diffusion dithering
  - Bayer: Ordered dithering using Bayer matrices
  - Random: Random threshold dithering
- **Color Format**: RGB, RGBA, GRAYSCALE

#### RGBShiftFilter
- **Purpose**: Shifts individual color channels for chromatic aberration effects
- **Parameters**:
  - `red_shift` (tuple[int, int]): (x, y) pixel shift for red channel
  - `green_shift` (tuple[int, int]): (x, y) pixel shift for green channel
  - `blue_shift` (tuple[int, int]): (x, y) pixel shift for blue channel
  - `edge_mode` (str): "clip", "wrap", "reflect" - how to handle edge pixels
- **Algorithm**: Apply independent 2D translations to each color channel
- **Color Format**: RGB, RGBA

#### NoiseFilter
- **Purpose**: Adds various types of noise to images
- **Parameters**:
  - `noise_type` (str): "gaussian", "salt_pepper", "uniform"
  - `intensity` (float, 0.0-1.0): Noise intensity
  - `salt_pepper_ratio` (float, 0.0-1.0): Ratio of salt to pepper for salt-pepper noise
- **Algorithms**:
  - Gaussian: Add normally distributed random values
  - Salt-pepper: Randomly set pixels to min/max values
  - Uniform: Add uniformly distributed random values
- **Color Format**: RGB, RGBA, GRAYSCALE

## Data Models

### Filter Parameter Validation

Each filter will implement parameter validation using a common pattern:

```python
def _validate_parameters(self) -> None:
    """Validate filter parameters are within acceptable ranges."""
    # Implementation specific to each filter
    pass
```

### Common Parameter Types

- **Range parameters**: Validated using min/max bounds
- **Enum parameters**: Validated against allowed string values
- **Tuple parameters**: Validated for correct length and element types

### Memory Management

All filters will support:
- Automatic in-place processing for large images
- Chunked processing for memory-constrained environments
- Memory usage tracking and reporting

## Error Handling

### Validation Errors
- **FilterValidationError**: Raised for invalid input data or parameters
- Clear error messages indicating the specific validation failure
- Parameter range information in error messages

### Execution Errors
- **FilterExecutionError**: Raised for processing failures
- Wrapped underlying exceptions (e.g., from scipy, PIL)
- Cleanup of partial results on failure

### Edge Cases
- **Zero parameters**: Filters return original image unchanged
- **Boundary conditions**: Proper handling of pixel shifts and convolutions at image edges
- **Data type conversion**: Automatic handling of float/int conversions with proper scaling

## Testing Strategy

### Unit Tests

Each filter will have comprehensive unit tests covering:

1. **Parameter Validation**
   - Valid parameter ranges
   - Invalid parameter rejection
   - Edge case parameters (0, maximum values)

2. **Functionality Tests**
   - Expected output for known inputs
   - Identity operations (no-change parameters)
   - Boundary condition handling

3. **Integration Tests**
   - Filter registration with FilterRegistry
   - Compatibility with BaseFilter features
   - Progress tracking and metadata collection

4. **Performance Tests**
   - Memory usage validation
   - Processing time benchmarks
   - Chunked processing verification

### Test Data

- **Synthetic images**: Generated test patterns for predictable results
- **Real images**: Sample images of various sizes and formats
- **Edge cases**: Single-pixel images, very large images, unusual aspect ratios

### Regression Tests

- **Existing filter compatibility**: Ensure updates to glitch and print simulation filters don't break existing functionality
- **Filter registry**: Verify all filters are properly discoverable
- **Parameter persistence**: Ensure parameter changes are properly handled

## Implementation Strategy

### Incremental Development Approach

The implementation will follow a filter-by-filter approach to enable frequent commits and incremental testing:

1. **Single Filter Implementation**: Each filter will be implemented as a complete, standalone unit
2. **Individual Testing**: Each filter will have its own test suite that can be run independently
3. **Incremental Registration**: Filters will be registered individually as they're completed
4. **Commit-Ready Units**: Each filter implementation will be a complete, commit-ready unit

### Development Order

The filters will be implemented in the following order to minimize dependencies:

**Phase 1: Basic Enhancement Filters (No Dependencies)**
1. InvertFilter - Simplest implementation, good starting point
2. GammaCorrectionFilter - Basic mathematical transformation
3. ContrastFilter - Similar to gamma correction

**Phase 2: Color Space Filters (HSV Dependencies)**
4. SaturationFilter - Introduces HSV color space handling
5. HueRotationFilter - Builds on HSV infrastructure

**Phase 3: Advanced Effect Filters (Independent)**
6. NoiseFilter - Standalone noise generation
7. RGBShiftFilter - Independent channel shifting

**Phase 4: Convolution-Based Filters**
8. GaussianBlurFilter - Introduces convolution operations
9. MotionBlurFilter - Builds on convolution infrastructure

**Phase 5: Complex Algorithms**
10. DitherFilter - Most complex algorithms, multiple patterns

**Phase 6: Filter Updates (Depends on Phase 3)**
11. Update GlitchFilter to use RGBShiftFilter
12. Update PrintSimulationFilter to use NoiseFilter

### Per-Filter Implementation Pattern

Each filter implementation will follow this complete pattern:

1. **Filter Class Implementation**
   - Complete filter class with all methods
   - Parameter validation
   - Core algorithm implementation
   - Error handling

2. **Unit Tests**
   - Parameter validation tests
   - Functionality tests
   - Edge case tests
   - Integration tests

3. **Registration**
   - Automatic registration with FilterRegistry
   - Category assignment
   - Metadata setup

4. **Documentation**
   - Docstrings and parameter documentation
   - Usage examples in tests

### Implementation Details

#### Color Space Conversions (For Saturation/Hue Filters)
- Use `colorsys` module for RGB ↔ HSV conversion
- Handle vectorized operations using numpy
- Preserve alpha channel for RGBA images

#### Convolution Operations (For Blur Filters)
- Use `scipy.ndimage` for efficient convolution
- Implement custom kernels for motion blur
- Handle edge modes appropriately (reflect, constant, etc.)

#### Dithering Algorithms (For Dither Filter)
```python
def floyd_steinberg_dither(image, levels):
    # Error diffusion matrix: [0, 0, 7/16], [3/16, 5/16, 1/16]
    # Process pixels left-to-right, top-to-bottom
    # Distribute quantization error to neighboring pixels
```

#### Memory Optimization (All Filters)
- **In-place operations**: Modify input arrays when safe to do so
- **Chunked processing**: Process large images in tiles
- **Memory monitoring**: Track peak memory usage during processing

#### Filter Updates (Phase 6)
- **GlitchFilter Enhancement**: Replace internal color channel shifting with RGBShiftFilter
- **PrintSimulationFilter Enhancement**: Replace internal noise generation with NoiseFilter
- Maintain backward compatibility with existing parameters

## Performance Considerations

### Optimization Strategies

1. **Vectorized Operations**: Use numpy broadcasting and vectorized functions
2. **Memory Layout**: Ensure cache-friendly memory access patterns
3. **Algorithm Selection**: Choose efficient algorithms for each operation
4. **Early Termination**: Skip processing when parameters indicate no change

### Benchmarking

- **Processing Time**: Measure filter execution time for various image sizes
- **Memory Usage**: Track peak memory consumption
- **Throughput**: Images processed per second for batch operations

### Scalability

- **Large Images**: Support for images larger than available RAM through chunking
- **Parallel Processing**: Design filters to be thread-safe for future parallelization
- **Progressive Processing**: Support for progressive/streaming image processing

## Integration Points

### Filter Registry Integration

All new filters will be automatically registered using the existing registry system:
- Automatic discovery through module scanning
- Proper categorization (enhancement vs artistic)
- Metadata extraction for filter capabilities

### CLI Integration

New filters will be accessible through the existing CLI interface:
- Parameter specification through command-line arguments
- Batch processing support
- Progress reporting for long-running operations

### API Consistency

All filters will maintain consistency with existing API patterns:
- Parameter naming conventions
- Return value formats
- Error handling patterns
- Documentation standards