#!/usr/bin/env python3
"""
Custom Filter Creation Example

This example demonstrates how to create custom filters using the library framework.
"""

import numpy as np
from image_processing_library import (
    BaseFilter,
    DataType,
    ColorFormat,
    load_image,
    save_image,
    get_registry,
)


class SepiaFilter(BaseFilter):
    """Custom sepia tone filter."""

    def __init__(self, intensity=1.0):
        super().__init__(
            name="sepia",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="artistic",
            intensity=intensity,
        )

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply sepia tone effect to the image."""
        self.validate_input(data)

        def _apply_sepia():
            self._update_progress(0.0)

            # Sepia transformation matrix
            sepia_matrix = np.array(
                [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
            )

            # Apply sepia transformation
            result = data.copy()
            original_shape = result.shape

            # Reshape for matrix multiplication
            pixels = result.reshape(-1, 3).astype(np.float32)

            self._update_progress(0.3)

            # Apply sepia matrix
            sepia_pixels = np.dot(pixels, sepia_matrix.T)

            self._update_progress(0.7)

            # Blend with original based on intensity
            intensity = self.parameters.get("intensity", 1.0)
            blended_pixels = pixels * (1 - intensity) + sepia_pixels * intensity

            # Clip values and convert back to uint8
            blended_pixels = np.clip(blended_pixels, 0, 255).astype(np.uint8)
            result = blended_pixels.reshape(original_shape)

            self._update_progress(1.0)
            return result

        return self._measure_execution_time(_apply_sepia)


class EdgeDetectionFilter(BaseFilter):
    """Custom edge detection filter using Sobel operator."""

    def __init__(self, threshold=50):
        super().__init__(
            name="edge_detection",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="enhancement",
            threshold=threshold,
        )

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply edge detection to the image."""
        self.validate_input(data)

        def _apply_edge_detection():
            self._update_progress(0.0)

            # Convert to grayscale for edge detection
            if len(data.shape) == 3:
                gray = np.dot(data[..., :3], [0.299, 0.587, 0.114])
            else:
                gray = data.copy()

            self._update_progress(0.2)

            # Sobel operators
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            # Apply convolution
            height, width = gray.shape
            edges = np.zeros_like(gray)

            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    region = gray[y - 1 : y + 2, x - 1 : x + 2]

                    gx = np.sum(region * sobel_x)
                    gy = np.sum(region * sobel_y)

                    magnitude = np.sqrt(gx * gx + gy * gy)
                    edges[y, x] = magnitude

                if y % 10 == 0:  # Update progress periodically
                    progress = 0.2 + 0.6 * y / (height - 2)
                    self._update_progress(progress)

            self._update_progress(0.8)

            # Apply threshold
            threshold = self.parameters.get("threshold", 50)
            edges = (edges > threshold) * 255

            # Convert back to RGB if needed
            if len(data.shape) == 3:
                result = np.stack([edges, edges, edges], axis=2).astype(np.uint8)
            else:
                result = edges.astype(np.uint8)

            self._update_progress(1.0)
            return result

        return self._measure_execution_time(_apply_edge_detection)


def main():
    """Demonstrate custom filter creation and usage."""

    # Create sample image
    image = create_sample_image()
    print("Created sample image with shape:", image.shape)

    # Example 1: Use custom Sepia filter
    print("\n1. Applying Custom Sepia Filter...")
    sepia_filter = SepiaFilter(intensity=0.8)

    def progress_callback(progress):
        print(f"   Progress: {progress:.1%}")

    sepia_filter.set_progress_callback(progress_callback)
    sepia_result = sepia_filter.apply(image)

    print(f"   Execution time: {sepia_filter.metadata.execution_time:.3f}s")
    save_image(sepia_result, "examples/output_custom_sepia.jpg")

    # Example 2: Use custom Edge Detection filter
    print("\n2. Applying Custom Edge Detection Filter...")
    edge_filter = EdgeDetectionFilter(threshold=30)
    edge_filter.set_progress_callback(progress_callback)
    edge_result = edge_filter.apply(image)

    print(f"   Execution time: {edge_filter.metadata.execution_time:.3f}s")
    save_image(edge_result, "examples/output_custom_edges.jpg")

    # Example 3: Register custom filters for use with registry
    print("\n3. Registering Custom Filters...")
    registry = get_registry()
    registry.register_filter(SepiaFilter)
    registry.register_filter(EdgeDetectionFilter)

    from image_processing_library.filters import list_filters

    filters = list_filters()
    print("   Available filters after registration:")
    for filter_name in filters:
        if filter_name in ["sepia", "edge_detection"]:
            print(f"     - {filter_name}")

    print("\nCustom filter creation examples completed!")


def create_sample_image():
    """Create a sample image with various shapes and colors."""
    height, width = 150, 200
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create some geometric shapes
    # Rectangle
    image[20:60, 30:80] = [255, 100, 100]  # Red rectangle

    # Circle
    center_x, center_y = 140, 75
    for y in range(height):
        for x in range(width):
            if (x - center_x) ** 2 + (y - center_y) ** 2 < 25**2:
                image[y, x] = [100, 255, 100]  # Green circle

    # Triangle (approximate)
    for y in range(90, 130):
        for x in range(50, 90):
            if x - 50 < (y - 90) * 0.5 and 90 - x + 50 < (y - 90) * 0.5:
                image[y, x] = [100, 100, 255]  # Blue triangle

    return image


if __name__ == "__main__":
    main()
