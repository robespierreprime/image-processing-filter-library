#!/usr/bin/env python3
"""
Artistic Filters Demo

This example demonstrates all the artistic filters available in the library,
including dithering, noise, RGB shift, and other special effects.
"""

import numpy as np
from PIL import Image
import os

from image_processing_library.filters.artistic.dither_filter import DitherFilter
from image_processing_library.filters.artistic.rgb_shift_filter import RGBShiftFilter
from image_processing_library.filters.artistic.noise_filter import NoiseFilter
from image_processing_library.filters.artistic.glitch import GlitchFilter
from image_processing_library.filters.artistic.print_simulation import PrintSimulationFilter
from image_processing_library.media_io.image_io import save_image


def create_test_image():
    """Create a colorful test image with various patterns."""
    # Create a 400x300 test image with smooth gradients and geometric shapes
    width, height = 400, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create smooth color gradients
    for y in range(height):
        for x in range(width):
            # Radial gradient
            center_x, center_y = width // 2, height // 2
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            # Color based on position and distance
            image[y, x, 0] = int(255 * (1 - distance / max_distance))  # Red
            image[y, x, 1] = int(255 * x / width)  # Green
            image[y, x, 2] = int(255 * y / height)  # Blue
    
    # Add some geometric shapes for better dithering demonstration
    # Circle
    center_x, center_y = width // 4, height // 4
    for y in range(height):
        for x in range(width):
            if (x - center_x)**2 + (y - center_y)**2 < 50**2:
                image[y, x] = [255, 255, 0]  # Yellow circle
    
    # Rectangle
    image[height//2:height//2+60, 3*width//4-30:3*width//4+30] = [0, 255, 255]  # Cyan rectangle
    
    return image


def demo_dither_filters():
    """Demonstrate dithering filters."""
    print("Creating dithering filter examples...")
    
    image = create_test_image()
    save_image(image, "examples/original_artistic.jpg")
    
    # Floyd-Steinberg dithering with different levels
    dither_filter = DitherFilter(pattern_type="floyd_steinberg", levels=4)
    fs_4_levels = dither_filter.apply(image)
    save_image(fs_4_levels, "examples/dither_floyd_steinberg_4.jpg")
    print("✓ Created dither_floyd_steinberg_4.jpg")
    
    dither_filter.set_parameters(levels=8)
    fs_8_levels = dither_filter.apply(image)
    save_image(fs_8_levels, "examples/dither_floyd_steinberg_8.jpg")
    print("✓ Created dither_floyd_steinberg_8.jpg")
    
    # Bayer dithering with different matrix sizes
    dither_filter.set_parameters(pattern_type="bayer", levels=6, bayer_size=2)
    bayer_2x2 = dither_filter.apply(image)
    save_image(bayer_2x2, "examples/dither_bayer_2x2.jpg")
    print("✓ Created dither_bayer_2x2.jpg")
    
    dither_filter.set_parameters(bayer_size=4)
    bayer_4x4 = dither_filter.apply(image)
    save_image(bayer_4x4, "examples/dither_bayer_4x4.jpg")
    print("✓ Created dither_bayer_4x4.jpg")
    
    dither_filter.set_parameters(bayer_size=8)
    bayer_8x8 = dither_filter.apply(image)
    save_image(bayer_8x8, "examples/dither_bayer_8x8.jpg")
    print("✓ Created dither_bayer_8x8.jpg")
    
    # Random dithering
    dither_filter.set_parameters(pattern_type="random", levels=5)
    random_dither = dither_filter.apply(image)
    save_image(random_dither, "examples/dither_random.jpg")
    print("✓ Created dither_random.jpg")


def demo_rgb_shift_filters():
    """Demonstrate RGB shift filters."""
    print("\nCreating RGB shift filter examples...")
    
    image = create_test_image()
    
    # Horizontal chromatic aberration
    rgb_shift = RGBShiftFilter(
        red_shift=(3, 0),
        green_shift=(0, 0),
        blue_shift=(-3, 0),
        edge_mode="clip"
    )
    horizontal_shift = rgb_shift.apply(image)
    save_image(horizontal_shift, "examples/rgb_shift_horizontal.jpg")
    print("✓ Created rgb_shift_horizontal.jpg")
    
    # Vertical chromatic aberration
    rgb_shift.set_parameters(
        red_shift=(0, 2),
        green_shift=(0, 0),
        blue_shift=(0, -2)
    )
    vertical_shift = rgb_shift.apply(image)
    save_image(vertical_shift, "examples/rgb_shift_vertical.jpg")
    print("✓ Created rgb_shift_vertical.jpg")
    
    # Diagonal shift for glitch effect
    rgb_shift.set_parameters(
        red_shift=(4, 2),
        green_shift=(-2, 1),
        blue_shift=(1, -3)
    )
    diagonal_shift = rgb_shift.apply(image)
    save_image(diagonal_shift, "examples/rgb_shift_diagonal.jpg")
    print("✓ Created rgb_shift_diagonal.jpg")
    
    # Wrap edge mode for different effect
    rgb_shift.set_parameters(
        red_shift=(10, 0),
        green_shift=(0, 0),
        blue_shift=(-10, 0),
        edge_mode="wrap"
    )
    wrap_shift = rgb_shift.apply(image)
    save_image(wrap_shift, "examples/rgb_shift_wrap.jpg")
    print("✓ Created rgb_shift_wrap.jpg")


def demo_noise_filters():
    """Demonstrate noise filters."""
    print("\nCreating noise filter examples...")
    
    image = create_test_image()
    
    # Gaussian noise - light
    noise_filter = NoiseFilter(noise_type="gaussian", intensity=0.05)
    gaussian_light = noise_filter.apply(image)
    save_image(gaussian_light, "examples/noise_gaussian_light.jpg")
    print("✓ Created noise_gaussian_light.jpg")
    
    # Gaussian noise - heavy
    noise_filter.set_parameters(intensity=0.2)
    gaussian_heavy = noise_filter.apply(image)
    save_image(gaussian_heavy, "examples/noise_gaussian_heavy.jpg")
    print("✓ Created noise_gaussian_heavy.jpg")
    
    # Salt and pepper noise
    noise_filter.set_parameters(
        noise_type="salt_pepper", 
        intensity=0.02, 
        salt_pepper_ratio=0.5
    )
    salt_pepper = noise_filter.apply(image)
    save_image(salt_pepper, "examples/noise_salt_pepper.jpg")
    print("✓ Created noise_salt_pepper.jpg")
    
    # Salt and pepper with more salt
    noise_filter.set_parameters(salt_pepper_ratio=0.8)
    more_salt = noise_filter.apply(image)
    save_image(more_salt, "examples/noise_more_salt.jpg")
    print("✓ Created noise_more_salt.jpg")
    
    # Uniform noise
    noise_filter.set_parameters(noise_type="uniform", intensity=0.1)
    uniform_noise = noise_filter.apply(image)
    save_image(uniform_noise, "examples/noise_uniform.jpg")
    print("✓ Created noise_uniform.jpg")


def demo_glitch_effects():
    """Demonstrate glitch and print simulation filters."""
    print("\nCreating glitch and print simulation examples...")
    
    image = create_test_image()
    
    # Glitch filter - light
    glitch_filter = GlitchFilter(
        intensity=0.3,
        shift_amount=5,
        corruption_probability=0.1
    )
    light_glitch = glitch_filter.apply(image)
    save_image(light_glitch, "examples/glitch_light.jpg")
    print("✓ Created glitch_light.jpg")
    
    # Glitch filter - heavy
    glitch_filter.set_parameters(
        intensity=0.8,
        shift_amount=15,
        corruption_probability=0.3
    )
    heavy_glitch = glitch_filter.apply(image)
    save_image(heavy_glitch, "examples/glitch_heavy.jpg")
    print("✓ Created glitch_heavy.jpg")
    
    # Print simulation - light
    print_filter = PrintSimulationFilter(
        dot_gain=0.1,
        paper_texture=0.2,
        ink_bleeding=0.1
    )
    light_print = print_filter.apply(image)
    save_image(light_print, "examples/print_simulation_light.jpg")
    print("✓ Created print_simulation_light.jpg")
    
    # Print simulation - heavy
    print_filter.set_parameters(
        dot_gain=0.3,
        paper_texture=0.6,
        ink_bleeding=0.4
    )
    heavy_print = print_filter.apply(image)
    save_image(heavy_print, "examples/print_simulation_heavy.jpg")
    print("✓ Created print_simulation_heavy.jpg")


def demo_artistic_workflows():
    """Demonstrate combining artistic filters for complex effects."""
    print("\nCreating artistic workflow examples...")
    
    from image_processing_library.core.execution_queue import ExecutionQueue
    
    image = create_test_image()
    
    # Retro computer graphics effect
    queue = ExecutionQueue()
    queue.add_filter(DitherFilter, {
        "pattern_type": "bayer", 
        "levels": 8, 
        "bayer_size": 2
    })
    queue.add_filter(RGBShiftFilter, {
        "red_shift": (1, 0),
        "green_shift": (0, 0),
        "blue_shift": (-1, 0)
    })
    
    retro_effect = queue.execute(image)
    save_image(retro_effect, "examples/retro_computer.jpg")
    print("✓ Created retro_computer.jpg")
    
    # Glitch art workflow
    queue = ExecutionQueue()
    queue.add_filter(RGBShiftFilter, {
        "red_shift": (3, 1),
        "green_shift": (0, 0),
        "blue_shift": (-2, -1)
    })
    queue.add_filter(NoiseFilter, {
        "noise_type": "salt_pepper", 
        "intensity": 0.01
    })
    queue.add_filter(DitherFilter, {
        "pattern_type": "random", 
        "levels": 16
    })
    
    glitch_art = queue.execute(image)
    save_image(glitch_art, "examples/glitch_art.jpg")
    print("✓ Created glitch_art.jpg")
    
    # Vintage print effect
    queue = ExecutionQueue()
    queue.add_filter(NoiseFilter, {
        "noise_type": "gaussian", 
        "intensity": 0.03
    })
    queue.add_filter(DitherFilter, {
        "pattern_type": "floyd_steinberg", 
        "levels": 12
    })
    queue.add_filter(PrintSimulationFilter, {
        "dot_gain": 0.2,
        "paper_texture": 0.4,
        "ink_bleeding": 0.2
    })
    
    vintage_print = queue.execute(image)
    save_image(vintage_print, "examples/vintage_print.jpg")
    print("✓ Created vintage_print.jpg")


def main():
    """Run all artistic filter demonstrations."""
    print("Artistic Filters Demo")
    print("=" * 50)
    
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    
    # Run demonstrations
    demo_dither_filters()
    demo_rgb_shift_filters()
    demo_noise_filters()
    demo_glitch_effects()
    demo_artistic_workflows()
    
    print("\n" + "=" * 50)
    print("All examples created successfully!")
    print("Check the examples/ directory for output images.")
    print("\nGenerated files:")
    print("- Dithering: dither_*.jpg")
    print("- RGB Shift: rgb_shift_*.jpg")
    print("- Noise: noise_*.jpg")
    print("- Glitch: glitch_*.jpg")
    print("- Print Simulation: print_simulation_*.jpg")
    print("- Artistic Workflows: retro_computer.jpg, glitch_art.jpg, vintage_print.jpg")


if __name__ == "__main__":
    main()