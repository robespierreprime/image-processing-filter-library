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