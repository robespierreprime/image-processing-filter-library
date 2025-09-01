#!/usr/bin/env python3
"""
Basic Filter Usage Example

This example demonstrates how to use individual filters from the library.
"""

import numpy as np
from image_processing_library import load_image, save_image
from image_processing_library.filters.artistic import GlitchFilter
from image_processing_library.filters.artistic import PrintSimulationFilter


def main():
    """Demonstrate basic filter usage."""

    # Create a sample image (or load from file)
    # image = load_image("input.jpg")  # Uncomment to load from file
    image = create_sample_image()

    print("Original image shape:", image.shape)

    # Example 1: Apply Glitch Filter
    print("\n1. Applying Glitch Filter...")
    glitch_filter = GlitchFilter(
        shift_intensity=10, line_width=3, glitch_probability=0.8
    )

    # Set up progress tracking
    def progress_callback(progress):
        print(f"   Progress: {progress:.1%}")

    glitch_filter.set_progress_callback(progress_callback)
    glitched_image = glitch_filter.apply(image)

    print(f"   Execution time: {glitch_filter.metadata.execution_time:.3f}s")
    save_image(glitched_image, "examples/output_glitch.jpg")

    # Example 2: Apply Print Simulation Filter
    print("\n2. Applying Print Simulation Filter...")
    print_filter = PrintSimulationFilter(
        band_intensity=20, band_frequency=50, noise_level=15
    )

    print_filter.set_progress_callback(progress_callback)
    printed_image = print_filter.apply(image)

    print(f"   Execution time: {print_filter.metadata.execution_time:.3f}s")
    save_image(printed_image, "examples/output_print.jpg")

    # Example 3: Parameter modification
    print("\n3. Modifying filter parameters...")
    glitch_filter.set_parameters(shift_intensity=20, line_width=5)
    params = glitch_filter.get_parameters()
    print(f"   Updated parameters: {params}")

    intense_glitch = glitch_filter.apply(image)
    save_image(intense_glitch, "examples/output_intense_glitch.jpg")

    print("\nBasic filter usage examples completed!")


def create_sample_image():
    """Create a sample RGB image for demonstration."""
    # Create a colorful gradient image
    height, width = 200, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            image[y, x, 0] = int(255 * x / width)  # Red gradient
            image[y, x, 1] = int(255 * y / height)  # Green gradient
            image[y, x, 2] = int(255 * (x + y) / (width + height))  # Blue gradient

    return image


if __name__ == "__main__":
    main()
