#!/usr/bin/env python3
"""
Preset Management Example

This example demonstrates how to save and load filter presets.
"""

import numpy as np
from image_processing_library import (
    ExecutionQueue, PresetManager, 
    load_image, save_image
)
from image_processing_library.filters.artistic import GlitchFilter
from image_processing_library.filters.artistic import PrintSimulationFilter

def main():
    """Demonstrate preset management functionality."""
    
    # Create sample image
    image = create_sample_image()
    print("Created sample image with shape:", image.shape)
    
    # Initialize preset manager
    preset_manager = PresetManager("examples/presets")
    
    # Example 1: Create and save a preset
    print("\n1. Creating and Saving Presets...")
    
    # Create a "vintage effect" preset
    vintage_queue = ExecutionQueue()
    vintage_queue.add_filter(
        PrintSimulationFilter,
        {"band_intensity": 20, "band_frequency": 25}
    )
    vintage_queue.add_filter(
        GlitchFilter,
        {"shift_intensity": 3, "line_width": 2}
    )
    
    preset_path = preset_manager.save_preset(
        name="vintage_effect",
        execution_queue=vintage_queue,
        description="A vintage look with subtle glitch effects",
        author="Example User"
    )
    print(f"   Saved vintage_effect preset to: {preset_path}")
    
    # Create a "heavy distortion" preset
    distortion_queue = ExecutionQueue()
    distortion_queue.add_filter(
        GlitchFilter,
        {"shift_intensity": 20, "line_width": 5, "glitch_probability": 0.7}
    )
    distortion_queue.add_filter(
        PrintSimulationFilter,
        {"band_intensity": 25, "band_frequency": 60}
    )
    
    preset_manager.save_preset(
        name="heavy_distortion",
        execution_queue=distortion_queue,
        description="Heavy glitch and print artifacts for dramatic effect"
    )
    print("   Saved heavy_distortion preset")
    
    # Example 2: Load and apply presets
    print("\n2. Loading and Applying Presets...")
    
    # Load the vintage effect preset
    loaded_vintage = preset_manager.load_preset("vintage_effect")
    
    def progress_callback(progress, filter_name):
        print(f"   {filter_name}: {progress:.1%}")
    
    loaded_vintage.set_progress_callback(progress_callback)
    vintage_result = loaded_vintage.execute(image)
    save_image(vintage_result, "examples/output_vintage_preset.jpg")
    print("   Applied vintage_effect preset")
    
    # Load the heavy distortion preset
    loaded_distortion = preset_manager.load_preset("heavy_distortion")
    loaded_distortion.set_progress_callback(progress_callback)
    distortion_result = loaded_distortion.execute(image)
    save_image(distortion_result, "examples/output_distortion_preset.jpg")
    print("   Applied heavy_distortion preset")
    
    # Example 3: List available presets
    print("\n3. Available Presets:")
    import os
    preset_dir = "examples/presets"
    if os.path.exists(preset_dir):
        presets = [f[:-5] for f in os.listdir(preset_dir) if f.endswith('.json')]
        for preset in presets:
            print(f"   - {preset}")
    
    print("\nPreset management examples completed!")

def create_sample_image():
    """Create a sample image with circular patterns."""
    height, width = 200, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    center_x, center_y = width // 2, height // 2
    
    for y in range(height):
        for x in range(width):
            # Calculate distance from center
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Create concentric circles with different colors
            if distance < 30:
                image[y, x] = [255, 255, 100]  # Yellow center
            elif distance < 60:
                image[y, x] = [255, 100, 100]  # Red ring
            elif distance < 90:
                image[y, x] = [100, 255, 100]  # Green ring
            else:
                image[y, x] = [100, 100, 255]  # Blue background
    
    return image

if __name__ == "__main__":
    main()