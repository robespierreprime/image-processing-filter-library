#!/usr/bin/env python3
"""
Documentation Validation Script

This script validates that all documented filters are properly implemented
and accessible through the filter registry.
"""

import sys
from typing import List, Dict, Any

def validate_filter_imports():
    """Validate that all documented filters can be imported."""
    print("Validating filter imports...")
    
    # Enhancement filters
    try:
        from image_processing_library.filters.enhancement.color_filters import (
            InvertFilter, SaturationFilter, HueRotationFilter
        )
        from image_processing_library.filters.enhancement.correction_filters import (
            GammaCorrectionFilter, ContrastFilter
        )
        from image_processing_library.filters.enhancement.blur_filters import (
            GaussianBlurFilter, MotionBlurFilter
        )
        print("✓ Enhancement filters imported successfully")
    except ImportError as e:
        print(f"✗ Enhancement filter import failed: {e}")
        return False
    
    # Artistic filters
    try:
        from image_processing_library.filters.artistic.dither_filter import DitherFilter
        from image_processing_library.filters.artistic.rgb_shift_filter import RGBShiftFilter
        from image_processing_library.filters.artistic.noise_filter import NoiseFilter
        from image_processing_library.filters.artistic.glitch import GlitchFilter
        from image_processing_library.filters.artistic.print_simulation import PrintSimulationFilter
        print("✓ Artistic filters imported successfully")
    except ImportError as e:
        print(f"✗ Artistic filter import failed: {e}")
        return False
    
    return True


def validate_filter_registry():
    """Validate that all filters are properly registered."""
    print("\nValidating filter registry...")
    
    try:
        from image_processing_library.filters.registry import list_filters
        
        # Get all filters
        all_filters = list_filters()
        enhancement_filters = list_filters(category="enhancement")
        artistic_filters = list_filters(category="artistic")
        
        print(f"✓ Total filters registered: {len(all_filters)}")
        print(f"✓ Enhancement filters: {len(enhancement_filters)}")
        print(f"✓ Artistic filters: {len(artistic_filters)}")
        
        # Expected enhancement filters
        expected_enhancement = {
            "invert", "gamma_correction", "contrast", "saturation", 
            "hue_rotation", "gaussian_blur", "motion_blur"
        }
        
        # Expected artistic filters
        expected_artistic = {
            "dither", "rgb_shift", "noise", "glitch_effect", "print_simulation"
        }
        
        # Check enhancement filters
        enhancement_names = set(enhancement_filters)
        missing_enhancement = expected_enhancement - enhancement_names
        if missing_enhancement:
            print(f"✗ Missing enhancement filters: {missing_enhancement}")
            return False
        else:
            print("✓ All expected enhancement filters are registered")
        
        # Check artistic filters
        artistic_names = set(artistic_filters)
        missing_artistic = expected_artistic - artistic_names
        if missing_artistic:
            print(f"✗ Missing artistic filters: {missing_artistic}")
            return False
        else:
            print("✓ All expected artistic filters are registered")
        
        return True
        
    except ImportError as e:
        print(f"✗ Filter registry import failed: {e}")
        return False


def validate_filter_instantiation():
    """Validate that all filters can be instantiated with default parameters."""
    print("\nValidating filter instantiation...")
    
    filters_to_test = [
        # Enhancement filters
        ("InvertFilter", "image_processing_library.filters.enhancement.color_filters", {}),
        ("GammaCorrectionFilter", "image_processing_library.filters.enhancement.correction_filters", {"gamma": 1.0}),
        ("ContrastFilter", "image_processing_library.filters.enhancement.correction_filters", {"contrast_factor": 1.0}),
        ("SaturationFilter", "image_processing_library.filters.enhancement.color_filters", {"saturation_factor": 1.0}),
        ("HueRotationFilter", "image_processing_library.filters.enhancement.color_filters", {"rotation_degrees": 0}),
        ("GaussianBlurFilter", "image_processing_library.filters.enhancement.blur_filters", {"sigma": 1.0}),
        ("MotionBlurFilter", "image_processing_library.filters.enhancement.blur_filters", {"distance": 5, "angle": 0}),
        
        # Artistic filters
        ("DitherFilter", "image_processing_library.filters.artistic.dither_filter", {"pattern_type": "floyd_steinberg", "levels": 4}),
        ("RGBShiftFilter", "image_processing_library.filters.artistic.rgb_shift_filter", {
            "red_shift": (0, 0), "green_shift": (0, 0), "blue_shift": (0, 0)
        }),
        ("NoiseFilter", "image_processing_library.filters.artistic.noise_filter", {"noise_type": "gaussian", "intensity": 0.1}),
    ]
    
    success_count = 0
    for filter_name, module_path, params in filters_to_test:
        try:
            module = __import__(module_path, fromlist=[filter_name])
            filter_class = getattr(module, filter_name)
            filter_instance = filter_class(**params)
            print(f"✓ {filter_name} instantiated successfully")
            success_count += 1
        except Exception as e:
            print(f"✗ {filter_name} instantiation failed: {e}")
    
    print(f"\nInstantiation results: {success_count}/{len(filters_to_test)} filters successful")
    return success_count == len(filters_to_test)


def validate_parameter_validation():
    """Validate that filters properly validate their parameters."""
    print("\nValidating parameter validation...")
    
    try:
        from image_processing_library.filters.enhancement.correction_filters import GammaCorrectionFilter
        from image_processing_library.core.base_filter import FilterValidationError
        
        # Test invalid gamma (should raise error)
        try:
            invalid_filter = GammaCorrectionFilter(gamma=-1.0)
            print("✗ Parameter validation failed - invalid gamma accepted")
            return False
        except (FilterValidationError, ValueError):
            print("✓ Parameter validation working - invalid gamma rejected")
        
        # Test valid gamma (should work)
        try:
            valid_filter = GammaCorrectionFilter(gamma=1.5)
            print("✓ Parameter validation working - valid gamma accepted")
        except Exception as e:
            print(f"✗ Parameter validation failed - valid gamma rejected: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Parameter validation test failed - import error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("Documentation Validation")
    print("=" * 50)
    
    tests = [
        ("Filter Imports", validate_filter_imports),
        ("Filter Registry", validate_filter_registry),
        ("Filter Instantiation", validate_filter_instantiation),
        ("Parameter Validation", validate_parameter_validation),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        if test_func():
            passed_tests += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"Validation Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All documentation validation tests passed!")
        return 0
    else:
        print("❌ Some validation tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())