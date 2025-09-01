"""
Example demonstrating the ExecutionQueue functionality.

This example shows how to use the ExecutionQueue to chain multiple filters
together with progress tracking and intermediate saving.
"""

import numpy as np
from image_processing_library.core import (
    ExecutionQueue, BaseFilter, DataType, ColorFormat
)


class BrightnessFilter(BaseFilter):
    """Example filter that adjusts image brightness."""
    
    def __init__(self, brightness=0.0, **kwargs):
        super().__init__(
            name="brightness_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="enhancement",
            brightness=brightness,
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply brightness adjustment."""
        self.validate_input(data)
        
        self._update_progress(0.0)
        
        # Apply brightness adjustment
        brightness = self.parameters.get('brightness', 0.0)
        result = np.clip(data + brightness, 0, 1)
        
        self._update_progress(1.0)
        return result


class ContrastFilter(BaseFilter):
    """Example filter that adjusts image contrast."""
    
    def __init__(self, contrast=1.0, **kwargs):
        super().__init__(
            name="contrast_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="enhancement",
            contrast=contrast,
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply contrast adjustment."""
        self.validate_input(data)
        
        self._update_progress(0.0)
        
        # Apply contrast adjustment (simple linear scaling)
        contrast = self.parameters.get('contrast', 1.0)
        result = np.clip(data * contrast, 0, 1)
        
        self._update_progress(1.0)
        return result


def progress_callback(progress: float, filter_name: str):
    """Callback function to display progress updates."""
    print(f"Progress: {progress:.1%} - {filter_name}")


def main():
    """Demonstrate ExecutionQueue functionality."""
    print("ExecutionQueue Example")
    print("=" * 50)
    
    # Create sample image data (normalized to [0, 1])
    print("Creating sample image data...")
    image_data = np.random.rand(100, 100, 3).astype(np.float32)
    print(f"Original image shape: {image_data.shape}")
    print(f"Original image range: [{image_data.min():.3f}, {image_data.max():.3f}]")
    
    # Create execution queue
    print("\nCreating execution queue...")
    queue = ExecutionQueue()
    queue.set_progress_callback(progress_callback)
    
    # Add filters to the queue
    print("Adding filters to queue...")
    
    # Step 1: Increase brightness
    queue.add_filter(
        BrightnessFilter,
        {'brightness': 0.1},
        save_intermediate=True,
        save_path="/tmp/after_brightness.npy"
    )
    
    # Step 2: Increase contrast
    queue.add_filter(
        ContrastFilter,
        {'contrast': 1.2},
        save_intermediate=True,
        save_path="/tmp/after_contrast.npy"
    )
    
    # Step 3: Slight brightness reduction
    queue.add_filter(
        BrightnessFilter,
        {'brightness': -0.05}
    )
    
    print(f"Queue has {queue.get_step_count()} steps")
    
    # Display queue information
    print("\nQueue steps:")
    for i in range(queue.get_step_count()):
        step_info = queue.get_step_info(i)
        print(f"  Step {i + 1}: {step_info['filter_name']} - {step_info['parameters']}")
        if step_info['save_intermediate']:
            print(f"    -> Save to: {step_info['save_path']}")
    
    # Execute the queue
    print("\nExecuting filter queue...")
    result = queue.execute(image_data)
    
    print(f"\nProcessing complete!")
    print(f"Final image shape: {result.shape}")
    print(f"Final image range: [{result.min():.3f}, {result.max():.3f}]")
    
    # Demonstrate queue manipulation
    print("\nDemonstrating queue manipulation...")
    
    # Remove the middle step
    print("Removing step 2 (contrast filter)...")
    queue.remove_step(1)
    print(f"Queue now has {queue.get_step_count()} steps")
    
    # Execute modified queue
    print("Executing modified queue...")
    result2 = queue.execute(image_data)
    print(f"Modified result range: [{result2.min():.3f}, {result2.max():.3f}]")
    
    # Clear queue and add new filters
    print("\nClearing queue and adding new filters...")
    queue.clear()
    
    # Add filter instances instead of classes
    brightness_filter = BrightnessFilter(brightness=0.2)
    contrast_filter = ContrastFilter(contrast=0.8)
    
    queue.add_filter(brightness_filter)
    queue.add_filter(contrast_filter)
    
    print("Executing queue with filter instances...")
    result3 = queue.execute(image_data)
    print(f"Instance result range: [{result3.min():.3f}, {result3.max():.3f}]")
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()