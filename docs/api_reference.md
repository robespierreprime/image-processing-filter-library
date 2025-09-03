# API Reference

## Core Classes

### FilterProtocol
Protocol defining the interface for all filters.

**Attributes:**
- `name: str` - Filter name
- `data_type: DataType` - IMAGE or VIDEO
- `color_format: ColorFormat` - RGB, RGBA, or GRAYSCALE
- `category: str` - Filter category

**Methods:**
- `apply(data: np.ndarray, **kwargs) -> np.ndarray` - Apply filter
- `get_parameters() -> Dict[str, Any]` - Get current parameters
- `set_parameters(**kwargs) -> None` - Update parameters
- `validate_input(data: np.ndarray) -> bool` - Validate input

### BaseFilter
Base implementation providing common filter functionality.

**Constructor:**
```python
BaseFilter(name, data_type, color_format, category, **parameters)
```

**Methods:**
- `set_progress_callback(callback: Callable[[float], None])` - Set progress callback
- `apply(data: np.ndarray) -> np.ndarray` - Apply filter (override in subclasses)
- `get_parameters() -> Dict[str, Any]` - Get parameters
- `set_parameters(**kwargs) -> None` - Update parameters
- `validate_input(data: np.ndarray) -> bool` - Validate input

### ExecutionQueue
Manages filter execution with chaining support.

**Methods:**
- `add_filter(filter_class, parameters, save_intermediate=False, save_path=None)`
- `execute(input_data: np.ndarray) -> np.ndarray` - Execute filter chain
- `set_progress_callback(callback: Callable[[float, str], None])` - Set progress callback

### PresetManager
Manages saving and loading of filter presets.

**Constructor:**
```python
PresetManager(presets_dir="presets")
```

**Methods:**
- `save_preset(name, execution_queue, description="", author=None) -> str`
- `load_preset(name: str) -> ExecutionQueue`

## I/O Functions

### Image I/O
- `load_image(path: str) -> np.ndarray` - Load image file
- `save_image(data: np.ndarray, path: str)` - Save image file
- `get_image_info(path: str) -> dict` - Get image metadata

### Video I/O
- `VideoReader(path: str)` - Video reading class
- `VideoWriter(path: str, fps: float, codec: str)` - Video writing class
- `load_video(path: str) -> np.ndarray` - Load entire video
- `save_video(frames: np.ndarray, path: str, fps: float)` - Save video

## Available Filters

### Enhancement Filters

#### InvertFilter
Inverts RGB color values to create a negative effect.
- **Parameters:** None
- **Color Formats:** RGB, RGBA

#### GammaCorrectionFilter
Applies gamma correction for brightness adjustment.
- **Parameters:**
  - `gamma` (float, 0.1-3.0): Gamma value (1.0 = no change)
- **Color Formats:** RGB, RGBA, GRAYSCALE

#### ContrastFilter
Adjusts image contrast by scaling pixel values around a midpoint.
- **Parameters:**
  - `contrast_factor` (float, 0.0-3.0): Contrast multiplier (1.0 = no change)
- **Color Formats:** RGB, RGBA, GRAYSCALE

#### SaturationFilter
Adjusts color saturation in HSV color space.
- **Parameters:**
  - `saturation_factor` (float, 0.0-3.0): Saturation multiplier (1.0 = no change)
- **Color Formats:** RGB, RGBA

#### HueRotationFilter
Rotates hue values in HSV color space.
- **Parameters:**
  - `rotation_degrees` (float, 0-360): Degrees to rotate hue
- **Color Formats:** RGB, RGBA

#### GaussianBlurFilter
Applies gaussian blur using convolution.
- **Parameters:**
  - `sigma` (float, 0.0-10.0): Standard deviation for gaussian kernel
  - `kernel_size` (int, optional): Size of convolution kernel
- **Color Formats:** RGB, RGBA, GRAYSCALE

#### MotionBlurFilter
Creates directional motion blur effects.
- **Parameters:**
  - `distance` (int, 0-50): Blur distance in pixels
  - `angle` (float, 0-360): Blur direction in degrees
- **Color Formats:** RGB, RGBA, GRAYSCALE

### Artistic Filters

#### DitherFilter
Applies dithering effects with multiple pattern types.
- **Parameters:**
  - `pattern_type` (str): "floyd_steinberg", "bayer", or "random"
  - `levels` (int, 2-256): Number of quantization levels per channel (fixed to work correctly)
  - `bayer_size` (int, 2-64): Size of Bayer matrix (supports larger sizes for high-res images)
  - `pixel_step` (int, 1-64): Size of pixel blocks for chunky/pixelated dithering (NEW!)
- **Color Formats:** RGB, RGBA, GRAYSCALE

#### RGBShiftFilter
Shifts individual RGB color channels independently.
- **Parameters:**
  - `red_shift` (tuple[int, int]): (x, y) pixel shift for red channel
  - `green_shift` (tuple[int, int]): (x, y) pixel shift for green channel
  - `blue_shift` (tuple[int, int]): (x, y) pixel shift for blue channel
  - `edge_mode` (str): "clip", "wrap", or "reflect"
- **Color Formats:** RGB, RGBA

#### NoiseFilter
Adds various types of noise to images.
- **Parameters:**
  - `noise_type` (str): "gaussian", "salt_pepper", or "uniform"
  - `intensity` (float, 0.0-1.0): Noise intensity
  - `salt_pepper_ratio` (float, 0.0-1.0): Ratio of salt to pepper (salt_pepper only)
- **Color Formats:** RGB, RGBA, GRAYSCALE

#### GlitchFilter
Creates digital glitch effects with pixel shifts and compression artifacts.
- **Parameters:**
  - `intensity` (float, 0.0-1.0): Overall glitch intensity
  - `shift_amount` (int, 0-20): Maximum pixel shift amount
  - `corruption_probability` (float, 0.0-1.0): Probability of corruption effects
- **Color Formats:** RGB, RGBA

#### PrintSimulationFilter
Simulates realistic printing artifacts and degradation.
- **Parameters:**
  - `dot_gain` (float, 0.0-0.5): Dot gain simulation intensity
  - `paper_texture` (float, 0.0-1.0): Paper texture simulation intensity
  - `ink_bleeding` (float, 0.0-1.0): Ink bleeding effect intensity
- **Color Formats:** RGB, RGBA

#### BackgroundRemoverFilter
Removes backgrounds using AI models.
- **Parameters:**
  - `model_name` (str): AI model to use for background removal
  - `confidence_threshold` (float, 0.0-1.0): Confidence threshold for removal
- **Color Formats:** RGB, RGBA

## Filter Registry
- `register_filter(name: str, filter_class: type)` - Register filter
- `get_filter(name: str) -> type` - Get filter class by name
- `list_filters(category: str = None) -> List[dict]` - List available filters

## Enums

### DataType
- `DataType.IMAGE` - Image data
- `DataType.VIDEO` - Video data

### ColorFormat  
- `ColorFormat.RGB` - 3-channel RGB
- `ColorFormat.RGBA` - 4-channel RGBA
- `ColorFormat.GRAYSCALE` - Single channel grayscale