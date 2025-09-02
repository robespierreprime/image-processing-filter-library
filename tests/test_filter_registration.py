"""
Test filter registration and discovery functionality.

This test verifies that all new enhancement filters are properly registered
and discoverable through the FilterRegistry system.
"""

import pytest
from image_processing_library.filters import (
    get_registry, 
    auto_discover_filters,
    list_filters,
    get_filter
)
from image_processing_library.core.protocols import DataType, ColorFormat


class TestFilterRegistration:
    """Test filter registration and discovery."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear registry and auto-discover filters
        registry = get_registry()
        registry.clear()
        auto_discover_filters()
    
    def test_all_new_filters_registered(self):
        """Test that all new enhancement filters are registered."""
        expected_filters = {
            # Enhancement filters
            'invert': 'enhancement',
            'hue_rotation': 'enhancement', 
            'saturation': 'enhancement',
            'gamma_correction': 'enhancement',
            'contrast': 'enhancement',
            'gaussian_blur': 'enhancement',
            'motion_blur': 'enhancement',
            
            # Artistic filters
            'dither': 'artistic',
            'rgb_shift': 'artistic',
            'noise': 'artistic'
        }
        
        registry = get_registry()
        registered_filters = registry.list_filters()
        
        for filter_name, expected_category in expected_filters.items():
            assert filter_name in registered_filters, f"Filter {filter_name} not registered"
            
            # Check category
            metadata = registry.get_filter_metadata(filter_name)
            assert metadata['category'] == expected_category, \
                f"Filter {filter_name} has wrong category: {metadata['category']} != {expected_category}"
    
    def test_filter_categories(self):
        """Test that filters are properly categorized."""
        registry = get_registry()
        categories = registry.list_categories()
        
        # Should have at least enhancement and artistic categories
        assert 'enhancement' in categories
        assert 'artistic' in categories
        
        # Check enhancement filters
        enhancement_filters = registry.get_filters_by_category('enhancement')
        expected_enhancement = [
            'invert', 'hue_rotation', 'saturation',
            'gamma_correction', 'contrast', 
            'gaussian_blur', 'motion_blur'
        ]
        
        for filter_name in expected_enhancement:
            assert filter_name in enhancement_filters, \
                f"Enhancement filter {filter_name} not in enhancement category"
        
        # Check artistic filters
        artistic_filters = registry.get_filters_by_category('artistic')
        expected_artistic = ['dither', 'rgb_shift', 'noise']
        
        for filter_name in expected_artistic:
            assert filter_name in artistic_filters, \
                f"Artistic filter {filter_name} not in artistic category"
    
    def test_filter_metadata_extraction(self):
        """Test that filter metadata is properly extracted."""
        registry = get_registry()
        
        # Test a few key filters
        test_filters = {
            'invert': {
                'category': 'enhancement',
                'data_type': DataType.IMAGE,
                'color_format': ColorFormat.RGB
            },
            'gamma_correction': {
                'category': 'enhancement', 
                'data_type': DataType.IMAGE,
                'color_format': ColorFormat.RGB
            },
            'dither': {
                'category': 'artistic',
                'data_type': DataType.IMAGE,
                'color_format': ColorFormat.RGB
            }
        }
        
        for filter_name, expected_metadata in test_filters.items():
            metadata = registry.get_filter_metadata(filter_name)
            
            for key, expected_value in expected_metadata.items():
                assert metadata[key] == expected_value, \
                    f"Filter {filter_name} metadata {key}: {metadata[key]} != {expected_value}"
    
    def test_filter_instantiation(self):
        """Test that filters can be instantiated through the registry."""
        registry = get_registry()
        
        # Test filters that don't require parameters
        simple_filters = ['invert']
        
        for filter_name in simple_filters:
            filter_instance = registry.create_filter_instance(filter_name)
            assert filter_instance is not None
            assert hasattr(filter_instance, 'apply')
            assert hasattr(filter_instance, 'name')
            assert filter_instance.name == filter_name
        
        # Test filters that require parameters
        parameterized_filters = {
            'gamma_correction': {'gamma': 1.2},
            'contrast': {'contrast_factor': 1.5},
            'saturation': {'saturation_factor': 1.3},
            'hue_rotation': {'rotation_degrees': 45},
            'gaussian_blur': {'sigma': 2.0},
            'motion_blur': {'distance': 10, 'angle': 45},
            'dither': {'pattern_type': 'floyd_steinberg', 'levels': 4},
            'rgb_shift': {
                'red_shift': (2, 0), 
                'green_shift': (0, 2), 
                'blue_shift': (-2, 0)
            },
            'noise': {'noise_type': 'gaussian', 'intensity': 0.1}
        }
        
        for filter_name, params in parameterized_filters.items():
            filter_instance = registry.create_filter_instance(filter_name, **params)
            assert filter_instance is not None
            assert hasattr(filter_instance, 'apply')
            assert hasattr(filter_instance, 'name')
            assert filter_instance.name == filter_name
    
    def test_filter_discovery_by_criteria(self):
        """Test filtering by data type and color format."""
        registry = get_registry()
        
        # All new filters should support IMAGE data type
        image_filters = registry.list_filters(data_type=DataType.IMAGE)
        expected_filters = [
            'invert', 'hue_rotation', 'saturation',
            'gamma_correction', 'contrast',
            'gaussian_blur', 'motion_blur',
            'dither', 'rgb_shift', 'noise'
        ]
        
        for filter_name in expected_filters:
            assert filter_name in image_filters, \
                f"Filter {filter_name} not found in IMAGE data type filters"
        
        # All new filters should support RGB color format
        rgb_filters = registry.list_filters(color_format=ColorFormat.RGB)
        
        for filter_name in expected_filters:
            assert filter_name in rgb_filters, \
                f"Filter {filter_name} not found in RGB color format filters"
    
    def test_existing_filters_still_work(self):
        """Test that existing filters are still registered and working."""
        registry = get_registry()
        
        # Check that existing filters are still there
        existing_filters = ['Glitch Effect', 'Print Simulation']
        
        for filter_name in existing_filters:
            # Should be able to get the filter
            filter_class = registry.get_filter(filter_name)
            assert filter_class is not None
            
            # Should have proper metadata
            metadata = registry.get_filter_metadata(filter_name)
            assert metadata['category'] == 'artistic'
    
    def test_filter_validation(self):
        """Test filter metadata validation."""
        registry = get_registry()
        
        # Test validation for a known good filter
        filter_class = registry.get_filter('invert')
        validation_result = registry.validate_filter_metadata(filter_class)
        
        assert validation_result['valid'] is True
        assert len(validation_result['errors']) == 0
        assert 'name' in validation_result['metadata']
        assert 'category' in validation_result['metadata']
        assert 'data_type' in validation_result['metadata']
        assert 'color_format' in validation_result['metadata']
    
    def test_auto_discovery_completeness(self):
        """Test that auto-discovery finds all expected filters."""
        # Clear registry and run auto-discovery
        registry = get_registry()
        registry.clear()
        
        discovered_filters = auto_discover_filters()
        
        # Should discover all new filters plus existing ones
        expected_minimum_filters = [
            'invert', 'hue_rotation', 'saturation',
            'gamma_correction', 'contrast',
            'gaussian_blur', 'motion_blur',
            'dither', 'rgb_shift', 'noise',
            'Glitch Effect', 'Print Simulation'
        ]
        
        for filter_name in expected_minimum_filters:
            assert filter_name in discovered_filters, \
                f"Auto-discovery did not find filter {filter_name}"
        
        # Verify they're actually registered
        registered_filters = registry.list_filters()
        for filter_name in expected_minimum_filters:
            assert filter_name in registered_filters, \
                f"Filter {filter_name} discovered but not registered"


if __name__ == "__main__":
    # Run basic registration test
    test = TestFilterRegistration()
    test.setup_method()
    
    print("Testing filter registration...")
    test.test_all_new_filters_registered()
    print("✓ All new filters registered")
    
    test.test_filter_categories()
    print("✓ Filter categories correct")
    
    test.test_filter_metadata_extraction()
    print("✓ Filter metadata extraction working")
    
    test.test_filter_instantiation()
    print("✓ Filter instantiation working")
    
    test.test_auto_discovery_completeness()
    print("✓ Auto-discovery complete")
    
    print("\nAll filter registration tests passed!")