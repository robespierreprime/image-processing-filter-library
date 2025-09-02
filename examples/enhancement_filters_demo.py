#!/usr/bin/env python3
"""
Enhancement Filters Demo

This example demonstrates all the enhancement filters available in the library,
including color manipulation, correction, and blur effects.
"""

import numpy as np
from PIL import Image
import os

from image_processing_library.filters.enhancement.color_filters import (
    InvertFilter, SaturationFilter, HueRotationFilter
)
from image_processing_library.filters.enhancement.correction_filters import (
    GammaCorrectionFilter, ContrastFilter
)
from image_processing_library.filters.enhancement.blur_filters import (
    GaussianBlurFilter, MotionBlurFilter
)
from image_processing_library.media_io.image_io import save_image


def create_test_image():
    """Create a colorful test image with various patterns."""
    # Create a 400x300 test image with gradients and patterns
    width, height = 400, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create color gradients
    for y in range(height):
        for x in range(width):
            # Red gradient
            if x < width // 3:
                image[y, x, 0] = int(255 * x / (width // 3))
            # Green gradient  
            elif x < 2 * width // 3:
                image[y, x, 1] = int(255 * (x - width // 3) / (width // 3))
            # Blue gradient
            else:
                image[y, x, 2] = int(255 * (x - 2 * width // 3) / (width // 3))
            
            # Add some patterns
            if (x // 20 + y // 20) % 2 == 0:
                image[y, x] = np.minimum(image[y, x] + 50, 255)
    
    return image


def demo_color_filters():
    """Demonstrate color manipulation filters."""
    print("Creating color filter examples...")
    
    # Create test image
    image = create_test_image()
    save_image(image, "examples/original_color.jpg")
    
    # Invert Filter
    invert_filter = InvertFilter()
    inverted = invert_filter.apply(image)
    save_image(inverted, "examples/inverted.jpg")
    print("✓ Created inverted.jpg")
    
    # Saturation Filter - Various levels
    saturation_filter = SaturationFilter(saturation_factor=0.0)  # Grayscale
    grayscale = saturation_filter.apply(image)
    save_image(grayscale, "examples/grayscale.jpg")
    print("✓ Created grayscale.jpg")
    
    saturation_filter.set_parameters(saturation_factor=2.0)  # High saturation
    vibrant = saturation_filter.apply(image)
    save_image(vibrant, "examples/vibrant.jpg")
    print("✓ Created vibrant.jpg")
    
    # Hue Rotation Filter - Different rotations
    hue_filter = HueRotationFilter(rotation_degrees=120)
    hue_shifted = hue_filter.apply(image)
    save_image(hue_shifted, "examples/hue_shifted_120.jpg")
    print("✓ Created hue_shifted_120.jpg")
    
    hue_filter.set_parameters(rotation_degrees=240)
    hue_shifted_240 = hue_filter.apply(image)
    save_image(hue_shifted_240, "examples/hue_shifted_240.jpg")
    print("✓ Created hue_shifted_240.jpg")


def demo_correction_filters():
    """Demonstrate correction filters."""
    print("\nCreating correction filter examples...")
    
    image = create_test_image()
    
    # Gamma Correction - Brighter
    gamma_filter = GammaCorrectionFilter(gamma=0.5)
    brighter = gamma_filter.apply(image)
    save_image(brighter, "examples/gamma_bright.jpg")
    print("✓ Created gamma_bright.jpg")
    
    # Gamma Correction - Darker
    gamma_filter.set_parameters(gamma=2.0)
    darker = gamma_filter.apply(image)
    save_image(darker, "examples/gamma_dark.jpg")
    print("✓ Created gamma_dark.jpg")
    
    # Contrast Filter - High contrast
    contrast_filter = ContrastFilter(contrast_factor=2.0)
    high_contrast = contrast_filter.apply(image)
    save_image(high_contrast, "examples/high_contrast.jpg")
    print("✓ Created high_contrast.jpg")
    
    # Contrast Filter - Low contrast
    contrast_filter.set_parameters(contrast_factor=0.3)
    low_contrast = contrast_filter.apply(image)
    save_image(low_contrast, "examples/low_contrast.jpg")
    print("✓ Created low_contrast.jpg")


def demo_blur_filters():
    """Demonstrate blur filters."""
    print("\nCreating blur filter examples...")
    
    # Create a more detailed test image for blur effects
    image = create_test_image()
    
    # Add some sharp details
    height, width = image.shape[:2]
    for i in range(0, width, 40):
        image[:, i:i+2] = [255, 255, 255]  # White vertical lines
    for i in range(0, height, 30):
        image[i:i+2, :] = [255, 255, 255]  # White horizontal lines
    
    save_image(image, "examples/original_detailed.jpg")
    
    # Gaussian Blur - Light
    gaussian_filter = GaussianBlurFilter(sigma=1.0)
    light_blur = gaussian_filter.apply(image)
    save_image(light_blur, "examples/gaussian_light.jpg")
    print("✓ Created gaussian_light.jpg")
    
    # Gaussian Blur - Heavy
    gaussian_filter.set_parameters(sigma=5.0)
    heavy_blur = gaussian_filter.apply(image)
    save_image(heavy_blur, "examples/gaussian_heavy.jpg")
    print("✓ Created gaussian_heavy.jpg")
    
    # Motion Blur - Horizontal
    motion_filter = MotionBlurFilter(distance=15, angle=0)
    horizontal_motion = motion_filter.apply(image)
    save_image(horizontal_motion, "examples/motion_horizontal.jpg")
    print("✓ Created motion_horizontal.jpg")
    
    # Motion Blur - Diagonal
    motion_filter.set_parameters(distance=20, angle=45)
    diagonal_motion = motion_filter.apply(image)
    save_image(diagonal_motion, "examples/motion_diagonal.jpg")
    print("✓ Created motion_diagonal.jpg")


def demo_combined_effects():
    """Demonstrate combining multiple enhancement filters."""
    print("\nCreating combined effect examples...")
    
    from image_processing_library.core.execution_queue import ExecutionQueue
    
    image = create_test_image()
    
    # Photo enhancement workflow
    queue = ExecutionQueue()
    queue.add_filter(GammaCorrectionFilter, {"gamma": 0.9})
    queue.add_filter(ContrastFilter, {"contrast_factor": 1.2})
    queue.add_filter(SaturationFilter, {"saturation_factor": 1.3})
    queue.add_filter(GaussianBlurFilter, {"sigma": 0.5})
    
    enhanced = queue.execute(image)
    save_image(enhanced, "examples/photo_enhanced.jpg")
    print("✓ Created photo_enhanced.jpg")
    
    # Vintage effect workflow
    queue = ExecutionQueue()
    queue.add_filter(SaturationFilter, {"saturation_factor": 0.7})
    queue.add_filter(HueRotationFilter, {"rotation_degrees": 15})
    queue.add_filter(GammaCorrectionFilter, {"gamma": 1.2})
    queue.add_filter(ContrastFilter, {"contrast_factor": 0.9})
    
    vintage = queue.execute(image)
    save_image(vintage, "examples/vintage_effect.jpg")
    print("✓ Created vintage_effect.jpg")


def main():
    """Run all enhancement filter demonstrations."""
    print("Enhancement Filters Demo")
    print("=" * 50)
    
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    
    # Run demonstrations
    demo_color_filters()
    demo_correction_filters()
    demo_blur_filters()
    demo_combined_effects()
    
    print("\n" + "=" * 50)
    print("All examples created successfully!")
    print("Check the examples/ directory for output images.")
    print("\nGenerated files:")
    print("- Color filters: inverted.jpg, grayscale.jpg, vibrant.jpg, hue_shifted_*.jpg")
    print("- Correction filters: gamma_*.jpg, *_contrast.jpg")
    print("- Blur filters: gaussian_*.jpg, motion_*.jpg")
    print("- Combined effects: photo_enhanced.jpg, vintage_effect.jpg")


if __name__ == "__main__":
    main()