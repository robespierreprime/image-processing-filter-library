#!/usr/bin/env python3
"""
Demonstration of the improved dithering filter with larger Bayer matrices and fixed levels.

This example shows:
1. Larger Bayer matrix sizes (16x16, 32x32, 64x64) for high-resolution images
2. Correct levels quantization (levels=2 now produces exactly 2 colors)
3. Comparison between different Bayer sizes and levels
4. Saves example images to demonstrate the visual improvements
"""

import numpy as np
import os
from image_processing_library.filters.artistic.dither_filter import DitherFilter
from image_processing_library.media_io.image_io import save_image


def create_test_image(size=(512, 512)):
    """Create a test image with gradients for demonstrating dithering effects."""
    height, width = size

    # Create horizontal gradient
    gradient_h = np.linspace(0, 255, width, dtype=np.uint8)
    gradient_h = np.tile(gradient_h.reshape(1, -1), (height // 2, 1))

    # Create vertical gradient
    gradient_v = np.linspace(0, 255, height // 2, dtype=np.uint8)
    gradient_v = np.tile(gradient_v.reshape(-1, 1), (1, width))

    # Combine gradients
    combined = np.vstack([gradient_h, gradient_v])

    # Convert to RGB
    return np.stack([combined, combined, combined], axis=-1)


def demonstrate_bayer_sizes():
    """Demonstrate different Bayer matrix sizes for high-resolution images."""
    print("=== Bayer Matrix Size Demonstration ===")
    
    # Create output directory
    output_dir = "dither_demo_output"
    os.makedirs(output_dir, exist_ok=True)

    # Create high-resolution test image
    test_image = create_test_image((512, 512))
    print(f"Test image shape: {test_image.shape}")
    
    # Save original for comparison
    save_image(test_image, os.path.join(output_dir, "00_original_gradient.png"))
    print(f"  ✓ Saved original image: {output_dir}/00_original_gradient.png")

    bayer_sizes = [4, 8, 16, 32, 64]
    levels = 4

    for bayer_size in bayer_sizes:
        print(f"\nTesting Bayer {bayer_size}x{bayer_size} matrix:")

        filter_instance = DitherFilter(
            pattern_type="bayer", levels=levels, bayer_size=bayer_size
        )

        result = filter_instance.apply(test_image.copy())
        unique_colors = len(np.unique(result))
        
        # Save result
        filename = f"01_bayer_{bayer_size:02d}x{bayer_size:02d}_levels{levels}.png"
        filepath = os.path.join(output_dir, filename)
        save_image(result, filepath)

        print(f"  ✓ Successfully processed {result.shape} image")
        print(f"  ✓ Result has {unique_colors} unique colors (expected ≤ {levels})")
        print(f"  ✓ Execution time: {filter_instance.metadata.execution_time:.3f}s")
        print(f"  ✓ Saved: {filepath}")


def demonstrate_levels_accuracy():
    """Demonstrate that levels parameter now works correctly."""
    print("\n=== Levels Parameter Accuracy Demonstration ===")
    
    # Create output directory
    output_dir = "dither_demo_output"
    os.makedirs(output_dir, exist_ok=True)

    # Create gradient test image
    gradient = np.linspace(0, 255, 256, dtype=np.uint8)
    test_image = np.tile(gradient.reshape(1, -1, 1), (100, 1, 3))
    
    # Save gradient for reference
    save_image(test_image, os.path.join(output_dir, "02_gradient_original.png"))
    print(f"  ✓ Saved gradient reference: {output_dir}/02_gradient_original.png")

    test_levels = [2, 4, 8, 16]

    for levels in test_levels:
        print(f"\nTesting levels={levels}:")

        # Test all dithering methods
        methods = [
            ("Floyd-Steinberg", "floyd_steinberg", {}),
            ("Bayer 8x8", "bayer", {"bayer_size": 8}),
            ("Random", "random", {}),
        ]

        for method_name, pattern_type, extra_params in methods:
            filter_instance = DitherFilter(
                pattern_type=pattern_type, levels=levels, **extra_params
            )

            result = filter_instance.apply(test_image.copy())
            unique_colors = len(np.unique(result))
            
            # Save result
            method_safe = method_name.lower().replace("-", "_").replace(" ", "_")
            filename = f"03_{method_safe}_levels{levels:02d}.png"
            filepath = os.path.join(output_dir, filename)
            save_image(result, filepath)

            print(
                f"  {method_name:15}: {unique_colors:2d} unique colors (expected ≤ {levels})"
            )

            # Verify quantization is correct
            if unique_colors <= levels:
                print(f"  {'':15}  ✓ Correct quantization")
            else:
                print(f"  {'':15}  ✗ Too many colors!")
            
            print(f"  {'':15}  ✓ Saved: {filename}")


def demonstrate_high_res_performance():
    """Demonstrate performance with different configurations on high-res images."""
    print("\n=== High-Resolution Performance Demonstration ===")

    # Test different image sizes
    sizes = [(256, 256), (512, 512), (1024, 1024)]

    for size in sizes:
        print(f"\nTesting {size[0]}x{size[1]} image:")

        test_image = create_test_image(size)

        # Test different Bayer sizes
        bayer_configs = [
            (8, "Small pattern"),
            (16, "Medium pattern"),
            (32, "Large pattern"),
            (64, "Very large pattern"),
        ]

        for bayer_size, description in bayer_configs:
            filter_instance = DitherFilter(
                pattern_type="bayer", levels=4, bayer_size=bayer_size
            )

            result = filter_instance.apply(test_image.copy())
            execution_time = filter_instance.metadata.execution_time

            print(
                f"  Bayer {bayer_size:2d}x{bayer_size:2d} ({description:17}): {execution_time:.3f}s"
            )


def demonstrate_visual_differences():
    """Demonstrate visual differences between different Bayer sizes."""
    print("\n=== Visual Pattern Differences ===")
    
    # Create output directory
    output_dir = "dither_demo_output"
    os.makedirs(output_dir, exist_ok=True)

    # Create a uniform gray image to show dithering patterns clearly
    gray_value = 128
    test_image = np.full((128, 128, 3), gray_value, dtype=np.uint8)  # Larger for better visibility
    
    # Save original gray image
    save_image(test_image, os.path.join(output_dir, "04_gray_original.png"))
    print(f"  ✓ Saved gray reference: {output_dir}/04_gray_original.png")

    bayer_sizes = [2, 4, 8, 16, 32]
    levels = 2  # Binary dithering for clear pattern visibility

    print(f"Testing binary dithering (levels={levels}) on uniform gray image:")

    for bayer_size in bayer_sizes:
        filter_instance = DitherFilter(
            pattern_type="bayer", levels=levels, bayer_size=bayer_size
        )

        result = filter_instance.apply(test_image.copy())
        unique_values = np.unique(result)
        
        # Save pattern
        filename = f"05_pattern_bayer_{bayer_size:02d}x{bayer_size:02d}_binary.png"
        filepath = os.path.join(output_dir, filename)
        save_image(result, filepath)

        # Count black and white pixels
        black_pixels = np.sum(result == 0)
        white_pixels = np.sum(result == 255)
        total_pixels = result.size

        print(
            f"  Bayer {bayer_size:2d}x{bayer_size:2d}: {len(unique_values)} colors, "
            f"{black_pixels/total_pixels:.1%} black, {white_pixels/total_pixels:.1%} white"
        )
        print(f"  {'':15}  ✓ Saved: {filename}")


def create_comparison_showcase():
    """Create a comprehensive comparison showcase image."""
    print("\n=== Creating Comparison Showcase ===")
    
    # Create output directory
    output_dir = "dither_demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a more interesting test image with various features
    size = (256, 256)
    height, width = size
    
    # Create a test image with different regions
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Top-left: horizontal gradient
    for x in range(width // 2):
        test_image[:height//2, x] = int(255 * x / (width // 2))
    
    # Top-right: vertical gradient  
    for y in range(height // 2):
        test_image[y, width//2:] = int(255 * y / (height // 2))
    
    # Bottom-left: diagonal pattern
    for y in range(height // 2, height):
        for x in range(width // 2):
            test_image[y, x] = int(255 * ((x + y) % 64) / 64)
    
    # Bottom-right: circular pattern
    center_x, center_y = width * 3 // 4, height * 3 // 4
    for y in range(height // 2, height):
        for x in range(width // 2, width):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            test_image[y, x] = int(255 * (dist % 32) / 32)
    
    # Save original
    save_image(test_image, os.path.join(output_dir, "06_showcase_original.png"))
    print(f"  ✓ Saved showcase original: {output_dir}/06_showcase_original.png")
    
    # Test different configurations
    configs = [
        ("Binary Floyd-Steinberg", "floyd_steinberg", {"levels": 2}),
        ("Binary Bayer 4x4", "bayer", {"levels": 2, "bayer_size": 4}),
        ("Binary Bayer 16x16", "bayer", {"levels": 2, "bayer_size": 16}),
        ("4-Level Floyd-Steinberg", "floyd_steinberg", {"levels": 4}),
        ("4-Level Bayer 8x8", "bayer", {"levels": 4, "bayer_size": 8}),
        ("4-Level Bayer 32x32", "bayer", {"levels": 4, "bayer_size": 32}),
    ]
    
    for i, (name, pattern_type, params) in enumerate(configs):
        filter_instance = DitherFilter(pattern_type=pattern_type, **params)
        result = filter_instance.apply(test_image.copy())
        
        # Save result
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        filename = f"07_{i+1:02d}_{safe_name}.png"
        filepath = os.path.join(output_dir, filename)
        save_image(result, filepath)
        
        unique_colors = len(np.unique(result))
        print(f"  ✓ {name}: {unique_colors} colors -> {filename}")


def demonstrate_pixel_step():
    """Demonstrate the new pixel_step parameter for chunky/pixelated dithering."""
    print("\n=== Pixel Step (Chunky Dithering) Demonstration ===")
    
    # Create output directory
    output_dir = "dither_demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a test image that shows the chunky effect well
    size = (128, 128)
    height, width = size
    
    # Create a more complex pattern to show chunky effect
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create diagonal stripes
    for y in range(height):
        for x in range(width):
            value = int(255 * ((x + y) % 32) / 32)
            test_image[y, x] = value
    
    # Save original
    save_image(test_image, os.path.join(output_dir, "08_chunky_original.png"))
    print(f"  ✓ Saved chunky test image: {output_dir}/08_chunky_original.png")
    
    # Test different pixel steps
    pixel_steps = [1, 2, 4, 8, 16]
    
    print(f"\nTesting pixel_step parameter (chunky/pixelated effect):")
    
    for pixel_step in pixel_steps:
        # Binary dithering with chunky pixels
        filter_instance = DitherFilter(
            pattern_type="bayer",
            levels=2,
            bayer_size=4,
            pixel_step=pixel_step
        )
        
        result = filter_instance.apply(test_image.copy())
        unique_colors = len(np.unique(result))
        
        # Save result
        filename = f"09_chunky_step{pixel_step:02d}_binary.png"
        filepath = os.path.join(output_dir, filename)
        save_image(result, filepath)
        
        print(f"  Step {pixel_step:2d}: {unique_colors} colors, chunky {pixel_step}x{pixel_step} blocks -> {filename}")
    
    # Test with 4-level dithering for comparison
    print(f"\nTesting 4-level dithering with chunky pixels:")
    
    for pixel_step in [1, 4, 8]:
        filter_instance = DitherFilter(
            pattern_type="floyd_steinberg",
            levels=4,
            pixel_step=pixel_step
        )
        
        result = filter_instance.apply(test_image.copy())
        unique_colors = len(np.unique(result))
        
        # Save result
        filename = f"10_chunky_floyd_step{pixel_step:02d}_4levels.png"
        filepath = os.path.join(output_dir, filename)
        save_image(result, filepath)
        
        print(f"  Floyd step {pixel_step:2d}: {unique_colors} colors -> {filename}")


if __name__ == "__main__":
    print("Dithering Filter Improvements Demo")
    print("=" * 50)

    try:
        demonstrate_bayer_sizes()
        demonstrate_levels_accuracy()
        demonstrate_high_res_performance()
        demonstrate_visual_differences()
        create_comparison_showcase()
        demonstrate_pixel_step()

        print("\n" + "=" * 50)
        print("✓ All demonstrations completed successfully!")
        print("\nKey improvements:")
        print(
            "1. Bayer matrices now support sizes up to 64x64 for high-resolution images"
        )
        print(
            "2. Levels parameter now works correctly (levels=2 produces exactly 2 colors)"
        )
        print("3. Better performance and pattern quality for large images")
        print("4. NEW: pixel_step parameter for chunky/pixelated dithering effects")
        
        print(f"\n📁 All output images saved to: dither_demo_output/")
        print("   Check the images to see the visual improvements!")
        print("   - 00_original_gradient.png: Original test image")
        print("   - 01_bayer_*.png: Different Bayer matrix sizes")
        print("   - 02-03_*.png: Levels accuracy demonstration")
        print("   - 04-05_*.png: Pattern differences on uniform gray")
        print("   - 06-07_*.png: Comprehensive showcase")
        print("   - 08-10_*.png: NEW pixel_step chunky dithering effects")

    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        raise
