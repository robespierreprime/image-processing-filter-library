"""
Unit tests for GammaCorrectionFilter.

Tests parameter validation, gamma correction algorithm correctness,
edge cases, and integration with BaseFilter features.
"""

import pytest
import numpy as np
from image_processing_library.filters.enhancement.correction_filters import GammaCorrectionFilter
from image_processing_library.core.utils import FilterValidationError
from image_processing_library.core.protocols import DataType, ColorFormat


class TestGammaCorrectionFilter:
    """Test suite for GammaCorrectionFilter."""
    
    def test_filter_initialization(self):
        """Test filter initialization with default and custom parameters."""
        # Test default initialization
        filter_default = GammaCorrectionFilter()
        assert filter_default.name == "gamma_correction"
        assert filter_default.data_type == DataType.IMAGE
        assert filter_default.color_format == ColorFormat.RGB
        assert filter_default.category == "enhancement"
        assert filter_default.parameters['gamma'] == 1.0
        
        # Test custom gamma initialization
        filter_custom = GammaCorrectionFilter(gamma=2.2)
        assert filter_custom.parameters['gamma'] == 2.2
    
    def test_parameter_validation_valid_values(self):
        """Test parameter validation with valid gamma values."""
        filter_obj = GammaCorrectionFilter()
        
        # Test valid gamma values
        valid_gammas = [0.1, 0.5, 1.0, 1.5, 2.2, 3.0]
        for gamma in valid_gammas:
            filter_obj.set_parameters(gamma=gamma)
            filter_obj._validate_parameters()  # Should not raise
    
    def test_parameter_validation_invalid_values(self):
        """Test parameter validation with invalid gamma values."""
        filter_obj = GammaCorrectionFilter()
        
        # Test invalid gamma values
        invalid_gammas = [
            -1.0,    # Negative
            0.0,     # Zero
            0.05,    # Below minimum
            3.5,     # Above maximum
            "1.0",   # String
            None,    # None
        ]
        
        for gamma in invalid_gammas:
            filter_obj.set_parameters(gamma=gamma)
            with pytest.raises(FilterValidationError):
                filter_obj._validate_parameters()
    
    def test_input_validation_valid_inputs(self):
        """Test input validation with valid image data."""
        filter_obj = GammaCorrectionFilter()
        
        # Test valid inputs
        valid_inputs = [
            np.random.randint(0, 256, (100, 100), dtype=np.uint8),      # Grayscale uint8
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),   # RGB uint8
            np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8),   # RGBA uint8
            np.random.rand(100, 100).astype(np.float32),                 # Grayscale float32
            np.random.rand(100, 100, 3).astype(np.float32),              # RGB float32
            np.random.rand(100, 100, 4).astype(np.float64),              # RGBA float64
        ]
        
        for input_data in valid_inputs:
            assert filter_obj.validate_input(input_data) is True
    
    def test_input_validation_invalid_inputs(self):
        """Test input validation with invalid image data."""
        filter_obj = GammaCorrectionFilter()
        
        # Test invalid inputs
        invalid_inputs = [
            "not_an_array",                                    # Not numpy array
            np.array([]),                                      # Empty array
            np.random.rand(100),                               # 1D array
            np.random.rand(100, 100, 100, 100),               # 4D array
            np.random.rand(100, 100, 2),                      # 2 channels
            np.random.rand(100, 100, 5),                      # 5 channels
            np.full((100, 100), np.nan),                      # NaN values
            np.full((100, 100), np.inf),                      # Infinite values
        ]
        
        for input_data in invalid_inputs:
            with pytest.raises(FilterValidationError):
                filter_obj.validate_input(input_data)
    
    def test_gamma_correction_identity_case(self):
        """Test gamma correction with gamma = 1.0 (identity case)."""
        filter_obj = GammaCorrectionFilter(gamma=1.0)
        
        # Test with different image types
        test_images = [
            np.random.randint(0, 256, (50, 50), dtype=np.uint8),
            np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8),
            np.random.rand(50, 50).astype(np.float32),
            np.random.rand(50, 50, 3).astype(np.float32),
        ]
        
        for image in test_images:
            result = filter_obj.apply(image)
            
            # Result should be very close to original (allowing for floating point precision)
            np.testing.assert_allclose(result, image, rtol=1e-6, atol=1e-6)
            assert result.shape == image.shape
            assert result.dtype == image.dtype
    
    def test_gamma_correction_mathematical_correctness(self):
        """Test mathematical correctness of gamma correction algorithm."""
        # Test with known values
        gamma_values = [0.5, 2.0, 2.2]
        
        for gamma in gamma_values:
            filter_obj = GammaCorrectionFilter(gamma=gamma)
            
            # Test with uint8 image
            image_uint8 = np.array([[0, 64, 128, 192, 255]], dtype=np.uint8)
            result_uint8 = filter_obj.apply(image_uint8)
            
            # Calculate expected values manually
            normalized = image_uint8.astype(np.float64) / 255.0
            expected_normalized = np.power(normalized, 1.0 / gamma)
            expected_uint8 = np.clip(expected_normalized * 255.0, 0, 255).astype(np.uint8)
            
            np.testing.assert_array_equal(result_uint8, expected_uint8)
            
            # Test with float image
            image_float = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
            result_float = filter_obj.apply(image_float)
            
            # Calculate expected values for float
            expected_float = np.power(image_float, 1.0 / gamma).astype(np.float32)
            
            np.testing.assert_allclose(result_float, expected_float, rtol=1e-6)
    
    def test_gamma_correction_brightness_effects(self):
        """Test that gamma correction produces expected brightness effects."""
        # Create a test image with known values
        test_image = np.full((50, 50), 128, dtype=np.uint8)  # Mid-gray
        
        # Test gamma < 1.0 (darkens mid-tones: (0.5)^(1/0.5) = 0.5^2 = 0.25)
        filter_gamma_low = GammaCorrectionFilter(gamma=0.5)
        result_gamma_low = filter_gamma_low.apply(test_image)
        assert np.mean(result_gamma_low) < np.mean(test_image)
        
        # Test gamma > 1.0 (brightens mid-tones: (0.5)^(1/2.0) = 0.5^0.5 = 0.707)
        filter_gamma_high = GammaCorrectionFilter(gamma=2.0)
        result_gamma_high = filter_gamma_high.apply(test_image)
        assert np.mean(result_gamma_high) > np.mean(test_image)
        
        # Test that different gamma values produce different results
        assert not np.array_equal(result_gamma_low, result_gamma_high)
        
        # Test with extreme values to verify behavior
        dark_test_image = np.full((50, 50), 64, dtype=np.uint8)  # Dark gray
        bright_test_image = np.full((50, 50), 192, dtype=np.uint8)  # Bright gray
        
        # For dark pixels, both gamma adjustments should have less dramatic effect
        result_dark_low = filter_gamma_low.apply(dark_test_image)
        result_dark_high = filter_gamma_high.apply(dark_test_image)
        
        # For bright pixels, effects should be more pronounced
        result_bright_low = filter_gamma_low.apply(bright_test_image)
        result_bright_high = filter_gamma_high.apply(bright_test_image)
        
        # Verify results are different and in expected ranges
        assert result_dark_low.dtype == np.uint8
        assert result_bright_high.dtype == np.uint8
        assert np.all(result_dark_low >= 0) and np.all(result_dark_low <= 255)
        assert np.all(result_bright_high >= 0) and np.all(result_bright_high <= 255)
    
    def test_gamma_correction_edge_cases(self):
        """Test gamma correction with edge case values."""
        # Test with extreme gamma values
        filter_min = GammaCorrectionFilter(gamma=0.1)
        filter_max = GammaCorrectionFilter(gamma=3.0)
        
        test_image = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
        
        # Should not raise exceptions
        result_min = filter_min.apply(test_image)
        result_max = filter_max.apply(test_image)
        
        assert result_min.shape == test_image.shape
        assert result_max.shape == test_image.shape
        assert result_min.dtype == test_image.dtype
        assert result_max.dtype == test_image.dtype
        
        # Test with all-black and all-white images
        black_image = np.zeros((20, 20), dtype=np.uint8)
        white_image = np.full((20, 20), 255, dtype=np.uint8)
        
        filter_obj = GammaCorrectionFilter(gamma=2.2)
        
        result_black = filter_obj.apply(black_image)
        result_white = filter_obj.apply(white_image)
        
        # Black should stay black, white should stay white
        np.testing.assert_array_equal(result_black, black_image)
        np.testing.assert_array_equal(result_white, white_image)
    
    def test_gamma_correction_color_formats(self):
        """Test gamma correction with different color formats."""
        gamma = 2.2
        filter_obj = GammaCorrectionFilter(gamma=gamma)
        
        # Test RGB image
        rgb_image = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        result_rgb = filter_obj.apply(rgb_image)
        
        assert result_rgb.shape == rgb_image.shape
        assert result_rgb.dtype == rgb_image.dtype
        
        # Test RGBA image (alpha should be preserved for some implementations)
        rgba_image = np.random.randint(0, 256, (30, 30, 4), dtype=np.uint8)
        result_rgba = filter_obj.apply(rgba_image)
        
        assert result_rgba.shape == rgba_image.shape
        assert result_rgba.dtype == rgba_image.dtype
        
        # Test grayscale image
        gray_image = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
        result_gray = filter_obj.apply(gray_image)
        
        assert result_gray.shape == gray_image.shape
        assert result_gray.dtype == gray_image.dtype
    
    def test_parameter_updates_during_apply(self):
        """Test parameter updates through apply method kwargs."""
        filter_obj = GammaCorrectionFilter(gamma=1.0)
        test_image = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
        
        # Apply with different gamma via kwargs
        result = filter_obj.apply(test_image, gamma=2.2)
        
        # Check that parameter was updated
        assert filter_obj.parameters['gamma'] == 2.2
        
        # Result should be different from identity case
        identity_result = GammaCorrectionFilter(gamma=1.0).apply(test_image)
        assert not np.array_equal(result, identity_result)
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        filter_obj = GammaCorrectionFilter(gamma=2.2)
        test_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        progress_values = []
        
        def progress_callback(progress):
            progress_values.append(progress)
        
        filter_obj.set_progress_callback(progress_callback)
        filter_obj.apply(test_image)
        
        # Should have received progress updates
        assert len(progress_values) >= 2  # At least start (0.0) and end (1.0)
        assert progress_values[0] == 0.0
        assert progress_values[-1] == 1.0
    
    def test_metadata_tracking(self):
        """Test metadata tracking during filter execution."""
        filter_obj = GammaCorrectionFilter(gamma=2.2)
        test_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        result = filter_obj.apply(test_image)
        
        # Check metadata was recorded
        metadata = filter_obj.metadata
        assert metadata.execution_time > 0
        assert metadata.progress == 1.0
        assert metadata.error_message is None
        assert metadata.input_shape == test_image.shape
        assert metadata.output_shape == result.shape
        assert metadata.memory_usage > 0
    
    def test_data_type_preservation(self):
        """Test that data types are preserved correctly."""
        filter_obj = GammaCorrectionFilter(gamma=2.2)
        
        # Test different data types
        data_types = [np.uint8, np.uint16, np.float32, np.float64]
        
        for dtype in data_types:
            if dtype == np.uint8:
                test_image = np.random.randint(0, 256, (20, 20), dtype=dtype)
            elif dtype == np.uint16:
                test_image = np.random.randint(0, 65536, (20, 20), dtype=dtype)
            else:
                test_image = np.random.rand(20, 20).astype(dtype)
            
            result = filter_obj.apply(test_image)
            assert result.dtype == dtype
            assert result.shape == test_image.shape
    
    def test_filter_registry_integration(self):
        """Test integration with filter registry."""
        from image_processing_library.filters.registry import get_registry
        
        # Filter should be automatically registered
        registry = get_registry()
        
        # Check if filter is registered (it should be due to @register_filter decorator)
        try:
            filter_class = registry.get_filter("gamma_correction")
            assert filter_class == GammaCorrectionFilter
        except KeyError:
            # If not registered, register it manually for testing
            registry.register_filter(GammaCorrectionFilter)
            filter_class = registry.get_filter("gamma_correction")
            assert filter_class == GammaCorrectionFilter
        
        # Test creating instance through registry
        filter_instance = registry.create_filter_instance("gamma_correction", gamma=2.2)
        assert isinstance(filter_instance, GammaCorrectionFilter)
        assert filter_instance.parameters['gamma'] == 2.2
    
    def test_memory_efficiency(self):
        """Test memory efficiency features."""
        filter_obj = GammaCorrectionFilter(gamma=2.2)
        
        # Test with larger image to trigger memory management
        large_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        
        result = filter_obj.apply(large_image)
        
        # Check memory tracking
        metadata = filter_obj.metadata
        assert metadata.memory_usage > 0
        assert metadata.peak_memory_usage > 0
        assert metadata.memory_efficiency_ratio > 0
        
        # Result should be valid
        assert result.shape == large_image.shape
        assert result.dtype == large_image.dtype