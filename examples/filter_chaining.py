#!/usr/bin/env python3
"""
Filter Chaining Example

This example demonstrates how to chain multiple filters using ExecutionQueue.
"""

import numpy as np
from image_processing_library import ExecutionQueue, load_image, save_image
from image_processing_library.filters.artistic import GlitchFilter
from image_processing_library.filters.artistic import PrintSimulationFilter

def main():
    """Demonstrate filter chaining with ExecutionQueue."""
    
    # Create or load an image
    image = create_sample_image()
    print("Original image shape:", image.shape)
    
    # Example 1: Basic filter chaining
    print("\n1. Basic Filter Chaining...")
    queue = ExecutionQueue()
    
    # Add filters to the queue
    queue.add_filter(
        GlitchFilter,
        {"shift_intensity": 5, "line_width": 2}
    )
    
    queue.add_filter(
        PrintSimulationFilter,
        {"band_intensity": 15, "band_frequency": 30}
    )
    
    # Set up progress tracking for the entire queue
    def queue_progress(progress, filter_name):
        print(f"   {filter_name}: {progress:.1%}")
    
    queue.set_progress_callback(queue_progress)
    
    # Execute the filter chain
    result = queue.execute(image)
    save_image(result, "examples/output_chained.jpg")
    
    # Example 2: Chaining with intermediate saves
    print("\n2. Chaining with Intermediate Saves...")
    queue_with_saves = ExecutionQueue()
    
    queue_with_saves.add_filter(
        GlitchFilter,
        {"shift_intensity": 8, "line_width": 3},
        save_intermediate=True,
        save_path="examples/intermediate_glitch.jpg"
    )
    
    queue_with_saves.add_filter(
        PrintSimulationFilter,
        {"band_intensity": 18, "band_frequency": 40},
        save_intermediate=True,
        save_path="examples/intermediate_print.jpg"
    )
    
    queue_with_saves.set_progress_callback(queue_progress)
    final_result = queue_with_saves.execute(image)
    save_image(final_result, "examples/output_chained_with_saves.jpg")
    
    # Example 3: Complex processing pipeline
    print("\n3. Complex Processing Pipeline...")
    complex_queue = ExecutionQueue()
    
    # Multiple passes with different parameters
    complex_queue.add_filter(GlitchFilter, {"shift_intensity": 3})
    complex_queue.add_filter(PrintSimulationFilter, {"band_intensity": 12})
    complex_queue.add_filter(GlitchFilter, {"shift_intensity": 15, "line_width": 4})
    
    complex_queue.set_progress_callback(queue_progress)
    complex_result = complex_queue.execute(image)
    save_image(complex_result, "examples/output_complex_pipeline.jpg")
    
    print("\nFilter chaining examples completed!")

def create_sample_image():
    """Create a sample image with geometric patterns."""
    height, width = 200, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a checkerboard pattern with colors
    square_size = 20
    for y in range(height):
        for x in range(width):
            square_x = x // square_size
            square_y = y // square_size
            
            if (square_x + square_y) % 2 == 0:
                image[y, x] = [255, 100, 100]  # Light red
            else:
                image[y, x] = [100, 100, 255]  # Light blue
    
    return image

if __name__ == "__main__":
    main()