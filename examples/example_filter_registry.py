"""
Example demonstrating the filter registry and discovery functionality.

This example shows how to:
1. Register filters manually
2. Use the decorator for automatic registration
3. Discover and list filters by category
4. Create filter instances from the registry
"""

import numpy as np
from image_processing_library.core.base_filter import BaseFilter
from image_processing_library.core.protocols import DataType, ColorFormat
from image_processing_library.filters import (
    get_registry,
    register_filter,
    list_filters,
    get_filter,
    auto_discover_filters
)


# Example 1: Manual registration
class BlurFilter(BaseFilter):
    """A simple blur filter for demonstration."""
    
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(
            name="blur_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="enhancement",
            kernel_size=kernel_size,
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply blur effect (simplified implementation)."""
        # This is a simplified blur - in reality you'd use cv2.blur or similar
        return data * 0.9  # Just darken the image for demo


# Example 2: Automatic registration using decorator
@register_filter(category="artistic")
class VintageFilter(BaseFilter):
    """A vintage effect filter."""
    
    def __init__(self, sepia_strength=0.8, **kwargs):
        super().__init__(
            name="vintage_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="artistic",
            sepia_strength=sepia_strength,
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply vintage effect (simplified implementation)."""
        # Simplified vintage effect
        return np.clip(data * 1.1, 0, 255)


@register_filter()
class NoiseReductionFilter(BaseFilter):
    """A noise reduction filter."""
    
    def __init__(self, strength=0.5, **kwargs):
        super().__init__(
            name="noise_reduction",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="enhancement",
            strength=strength,
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply noise reduction (simplified implementation)."""
        return data  # No-op for demo


def main():
    """Demonstrate filter registry functionality."""
    
    print("=== Filter Registry Demo ===\n")
    
    # Get the global registry
    registry = get_registry()
    
    # Manual registration
    print("1. Manual Registration:")
    registry.register_filter(BlurFilter)
    print(f"   Registered BlurFilter")
    
    # The decorated filters are automatically registered when imported
    print("\n2. Automatic Registration (via decorators):")
    print(f"   VintageFilter and NoiseReductionFilter were auto-registered")
    
    # List all filters
    print(f"\n3. All Registered Filters:")
    all_filters = list_filters()
    for filter_name in all_filters:
        metadata = registry.get_filter_metadata(filter_name)
        print(f"   - {filter_name} (category: {metadata['category']})")
    
    # List filters by category
    print(f"\n4. Filters by Category:")
    categories = registry.list_categories()
    for category in categories:
        filters_in_category = list_filters(category=category)
        print(f"   {category}: {filters_in_category}")
    
    # List filters by data type
    print(f"\n5. Filters by Data Type:")
    image_filters = list_filters(data_type=DataType.IMAGE)
    print(f"   Image filters: {image_filters}")
    
    # Create filter instances
    print(f"\n6. Creating Filter Instances:")
    
    # Create blur filter with custom parameters
    blur_filter = registry.create_filter_instance("blur_filter", kernel_size=7)
    print(f"   Created blur filter with kernel_size: {blur_filter.parameters['kernel_size']}")
    
    # Create vintage filter
    vintage_filter = get_filter("vintage_filter")()
    print(f"   Created vintage filter: {vintage_filter.name}")
    
    # Demonstrate filter usage
    print(f"\n7. Using Filters:")
    
    # Create sample image data
    sample_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    print(f"   Original image shape: {sample_image.shape}")
    
    # Apply blur filter
    blurred = blur_filter.apply(sample_image)
    print(f"   Applied blur filter, result shape: {blurred.shape}")
    print(f"   Blur filter execution time: {blur_filter.metadata.execution_time:.4f}s")
    
    # Apply vintage filter
    vintage = vintage_filter.apply(sample_image)
    print(f"   Applied vintage filter, result shape: {vintage.shape}")
    print(f"   Vintage filter execution time: {vintage_filter.metadata.execution_time:.4f}s")
    
    # Show filter metadata
    print(f"\n8. Filter Metadata:")
    for filter_name in all_filters:
        metadata = registry.get_filter_metadata(filter_name)
        print(f"   {filter_name}:")
        print(f"     Category: {metadata['category']}")
        print(f"     Data Type: {metadata['data_type']}")
        print(f"     Color Format: {metadata['color_format']}")
        print(f"     Description: {metadata['description'][:50]}...")
    
    print(f"\n=== Demo Complete ===")


if __name__ == "__main__":
    main()