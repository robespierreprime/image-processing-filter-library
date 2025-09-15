# Filter Guide

This guide provides comprehensive documentation for all available filters in the image processing library, including usage examples and parameter descriptions.

## Enhancement Filters

Enhancement filters are designed to improve or adjust basic image properties like color, brightness, contrast, and sharpness.

### InvertFilter

Inverts RGB color values to create a negative effect.

**Category:** enhancement  
**Parameters:** None  
**Color Formats:** RGB, RGBA  

**Usage:**
```python
from image_processing_library.filters.enhancement.color_filters import InvertFilter
from image_processing_library.media_io.image_io import load_image, save_image

# Load image
image = load_image("input.jpg")

# Create and apply filter
invert_filter = InvertFilter()
result = invert_filter.apply(image)

# Save result
save_image(result, "inverted.jpg")
```

**Effect:** Transforms each RGB pixel value using the formula: `output = 255 - input`. This creates a photographic negative effect where bright areas become dark and vice versa.

---

### GammaCorrectionFilter

Applies gamma correction for brightness adjustment using power law transformation.

**Category:** enhancement  
**Parameters:**
- `gamma` (float, 0.1-3.0): Gamma value (1.0 = no change, <1 = brighter, >1 = darker)

**Color Formats:** RGB, RGBA, GRAYSCALE  

**Usage:**
```python
from image_processing_library.filters.enhancement.correction_filters import GammaCorrectionFilter

# Create filter with gamma = 0.5 (brighter)
gamma_filter = GammaCorrectionFilter(gamma=0.5)
result = gamma_filter.apply(image)

# Adjust gamma dynamically
gamma_filter.set_parameters(gamma=2.0)  # Darker
result_dark = gamma_filter.apply(image)
```

**Effect:** Applies the transformation `output = (input / 255.0) ^ (1/gamma) * 255`. Values less than 1.0 brighten the image, while values greater than 1.0 darken it.

---

### ContrastFilter

Adjusts image contrast by scaling pixel values around a midpoint.

**Category:** enhancement  
**Parameters:**
- `contrast_factor` (float, 0.0-3.0): Contrast multiplier (1.0 = no change)

**Color Formats:** RGB, RGBA, GRAYSCALE  

**Usage:**
```python
from image_processing_library.filters.enhancement.correction_filters import ContrastFilter

# Increase contrast
contrast_filter = ContrastFilter(contrast_factor=1.5)
result = contrast_filter.apply(image)

# Decrease contrast
contrast_filter.set_parameters(contrast_factor=0.5)
result_low = contrast_filter.apply(image)
```

**Effect:** Scales pixel values around the midpoint (128 for uint8 images) using: `output = (input - 128) * contrast_factor + 128`. Values greater than 1.0 increase contrast, while values less than 1.0 decrease it.

---

### SaturationFilter

Adjusts color saturation by modifying the saturation component in HSV color space.

**Category:** enhancement  
**Parameters:**
- `saturation_factor` (float, 0.0-3.0): Saturation multiplier (1.0 = no change)

**Color Formats:** RGB, RGBA  

**Usage:**
```python
from image_processing_library.filters.enhancement.color_filters import SaturationFilter

# Increase saturation
saturation_filter = SaturationFilter(saturation_factor=1.5)
result = saturation_filter.apply(image)

# Create grayscale effect
saturation_filter.set_parameters(saturation_factor=0.0)
grayscale = saturation_filter.apply(image)
```

**Effect:** Converts RGB to HSV, multiplies the saturation channel by the factor, then converts back to RGB. A factor of 0.0 creates grayscale, 1.0 preserves original colors, and values > 1.0 increase vibrancy.

---

### HueRotationFilter

Rotates hue values in HSV color space to shift colors around the color wheel.

**Category:** enhancement  
**Parameters:**
- `rotation_degrees` (float, 0-360): Degrees to rotate hue

**Color Formats:** RGB, RGBA  

**Usage:**
```python
from image_processing_library.filters.enhancement.color_filters import HueRotationFilter

# Rotate hue by 120 degrees
hue_filter = HueRotationFilter(rotation_degrees=120)
result = hue_filter.apply(image)

# Create complementary colors
hue_filter.set_parameters(rotation_degrees=180)
complementary = hue_filter.apply(image)
```

**Effect:** Converts RGB to HSV, adds the rotation value to the hue channel (with wraparound at 360°), then converts back to RGB. This shifts all colors around the color wheel by the specified amount.

---

### GaussianBlurFilter

Applies gaussian blur using convolution with a gaussian kernel.

**Category:** enhancement  
**Parameters:**
- `sigma` (float, 0.0-10.0): Standard deviation for gaussian kernel
- `kernel_size` (int, optional): Size of convolution kernel (auto-calculated if not provided)

**Color Formats:** RGB, RGBA, GRAYSCALE  

**Usage:**
```python
from image_processing_library.filters.enhancement.blur_filters import GaussianBlurFilter

# Light blur
blur_filter = GaussianBlurFilter(sigma=1.0)
result = blur_filter.apply(image)

# Heavy blur
blur_filter.set_parameters(sigma=5.0)
heavy_blur = blur_filter.apply(image)
```

**Effect:** Convolves the image with a 2D gaussian kernel. Higher sigma values create stronger blur effects. When sigma is 0, the original image is returned unchanged.

---

### MotionBlurFilter

Creates directional motion blur effects using linear kernel convolution.

**Category:** enhancement  
**Parameters:**
- `distance` (int, 0-50): Blur distance in pixels
- `angle` (float, 0-360): Blur direction in degrees

**Color Formats:** RGB, RGBA, GRAYSCALE  

**Usage:**
```python
from image_processing_library.filters.enhancement.blur_filters import MotionBlurFilter

# Horizontal motion blur
motion_filter = MotionBlurFilter(distance=10, angle=0)
result = motion_filter.apply(image)

# Diagonal motion blur
motion_filter.set_parameters(distance=15, angle=45)
diagonal_blur = motion_filter.apply(image)
```

**Effect:** Creates a linear motion kernel based on the angle and distance, then applies convolution. This simulates the blur that occurs when objects move during camera exposure.

## Artistic Filters

Artistic filters create special effects and stylized appearances.



### RGBShiftFilter

Shifts individual RGB color channels independently to create chromatic aberration effects.

**Category:** artistic  
**Parameters:**
- `red_shift` (tuple[int, int]): (x, y) pixel shift for red channel
- `green_shift` (tuple[int, int]): (x, y) pixel shift for green channel  
- `blue_shift` (tuple[int, int]): (x, y) pixel shift for blue channel
- `edge_mode` (str): "clip", "wrap", or "reflect" - how to handle edge pixels

**Color Formats:** RGB, RGBA  

**Usage:**
```python
from image_processing_library.filters.artistic.rgb_shift_filter import RGBShiftFilter

# Create chromatic aberration effect
rgb_shift = RGBShiftFilter(
    red_shift=(2, 0),
    green_shift=(0, 0), 
    blue_shift=(-2, 0),
    edge_mode="clip"
)
result = rgb_shift.apply(image)

# Vertical shift effect
rgb_shift.set_parameters(
    red_shift=(0, 3),
    green_shift=(0, 0),
    blue_shift=(0, -3)
)
vertical_shift = rgb_shift.apply(image)
```

**Effect:** Applies independent 2D translations to each color channel, creating color fringing effects similar to chromatic aberration in lenses or glitch effects in digital media.

---

### NoiseFilter

Adds various types of noise to images for texture simulation and artistic effects.

**Category:** artistic  
**Parameters:**
- `noise_type` (str): "gaussian", "salt_pepper", or "uniform"
- `intensity` (float, 0.0-1.0): Noise intensity
- `salt_pepper_ratio` (float, 0.0-1.0): Ratio of salt to pepper (for salt_pepper noise only)

**Color Formats:** RGB, RGBA, GRAYSCALE  

**Usage:**
```python
from image_processing_library.filters.artistic.noise_filter import NoiseFilter

# Gaussian noise
noise_filter = NoiseFilter(noise_type="gaussian", intensity=0.1)
result = noise_filter.apply(image)

# Salt and pepper noise
noise_filter.set_parameters(
    noise_type="salt_pepper", 
    intensity=0.05, 
    salt_pepper_ratio=0.5
)
salt_pepper = noise_filter.apply(image)

# Uniform noise
noise_filter.set_parameters(noise_type="uniform", intensity=0.2)
uniform_noise = noise_filter.apply(image)
```

**Effect:**
- **Gaussian**: Adds normally distributed random values to pixel intensities
- **Salt-pepper**: Randomly sets pixels to minimum (pepper) or maximum (salt) values
- **Uniform**: Adds uniformly distributed random values within a specified range

## Filter Chaining Examples

Combine multiple filters for complex effects:

```python
from image_processing_library.core.execution_queue import ExecutionQueue

# Create execution queue
queue = ExecutionQueue()

# Add filters in sequence
queue.add_filter(GammaCorrectionFilter, {"gamma": 0.8})
queue.add_filter(ContrastFilter, {"contrast_factor": 1.2})
queue.add_filter(SaturationFilter, {"saturation_factor": 1.3})
queue.add_filter(GaussianBlurFilter, {"sigma": 0.5})

# Execute the chain
result = queue.execute(image)
```

## Performance Tips

1. **Parameter Optimization**: Use identity values (gamma=1.0, contrast=1.0, etc.) to skip processing
2. **Memory Management**: Process large images in chunks using the library's built-in chunking
3. **Filter Order**: Apply computationally expensive filters (blur) last in chains
4. **Batch Processing**: Use ExecutionQueue for consistent parameter application across multiple images

## Common Use Cases

### Photo Enhancement Workflow
```python
# Basic photo enhancement
queue = ExecutionQueue()
queue.add_filter(GammaCorrectionFilter, {"gamma": 0.9})  # Slightly brighter
queue.add_filter(ContrastFilter, {"contrast_factor": 1.1})  # More contrast
queue.add_filter(SaturationFilter, {"saturation_factor": 1.2})  # More vibrant
```

### Artistic Effect Workflow
```python
# Vintage/retro effect
queue = ExecutionQueue()
queue.add_filter(SaturationFilter, {"saturation_factor": 0.7})  # Desaturate
queue.add_filter(HueRotationFilter, {"rotation_degrees": 15})  # Warm tint
queue.add_filter(NoiseFilter, {"noise_type": "gaussian", "intensity": 0.05})  # Film grain
```

### Glitch Art Workflow
```python
# Digital glitch effect
queue = ExecutionQueue()
queue.add_filter(RGBShiftFilter, {
    "red_shift": (3, 1), 
    "green_shift": (0, 0), 
    "blue_shift": (-2, -1)
})
queue.add_filter(NoiseFilter, {"noise_type": "salt_pepper", "intensity": 0.02})
```