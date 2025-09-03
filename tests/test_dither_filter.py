"""
Unit tests for DitherFilter.

Tests parameter validation, dithering algorithms, and edge cases for the DitherFilter class.
"""

import pytest
import numpy as np
from image_processing_library.filters.artistic.dither_filter import DitherFilter
from image_processing_library.core.utils import FilterValidationError


class TestDitherFilterInitialization:
    """Test DitherFilter initialization and parameter validation."""
    
    def test_default_initialization(self):
        """Test filter initialization with default parameters."""
        filter_instance = DitherFilter()
        
        assert filter_instance.name == "dither"
        assert filter_instance.category == "artistic"
        assert filter_instance.parameters['pattern_type'] == "floyd_steinberg"
        assert filter_instance.parameters['levels'] == 8
        assert filter_instance.parameters['bayer_size'] == 4
    
    def test_custom_initialization(self):
        """Test filter initialization with custom parameters."""
        filter_instance = DitherFilter(
            pattern_type="bayer",
            levels=16,
            bayer_size=8,
            pixel_step=4
        )
        
        assert filter_instance.parameters['pattern_type'] == "bayer"
        assert filter_instance.parameters['levels'] == 16
        assert filter_instance.parameters['bayer_size'] == 8
        assert filter_instance.parameters['pixel_step'] == 4
    
    def test_parameter_updates(self):
        """Test parameter updates after initialization."""
        filter_instance = DitherFilter()
        
        filter_instance.set_parameters(
            pattern_type="random",
            levels=4,
            bayer_size=2,
            pixel_step=8
        )
        
        params = filter_instance.get_parameters()
        assert params['pattern_type'] == "random"
        assert params['levels'] == 4
        assert params['bayer_size'] == 2
        assert params['pixel_step'] == 8


class TestDitherFilterParameterValidation:
    """Test parameter validation for DitherFilter."""
    
    def test_valid_pattern_types(self):
        """Test validation accepts valid pattern types."""
        valid_types = ["floyd_steinberg", "bayer", "random"]
        
        for pattern_type in valid_types:
            filter_instance = DitherFilter(pattern_type=pattern_type)
            filter_instance._validate_parameters()  # Should not raise
    
    def test_invalid_pattern_type_string(self):
        """Test validation rejects invalid pattern type strings."""
        filter_instance = DitherFilter(pattern_type="invalid_type")
        
        with pytest.raises(FilterValidationError, match="pattern_type must be one of"):
            filter_instance._validate_parameters()
    
    def test_invalid_pattern_type_non_string(self):
        """Test validation rejects non-string pattern types."""
        filter_instance = DitherFilter()
        
        with pytest.raises(FilterValidationError, match="pattern_type must be a string"):
            filter_instance.set_parameters(pattern_type=123)
    
    def test_valid_levels_range(self):
        """Test validation accepts valid levels values."""
        valid_levels = [2, 8, 16, 32, 128, 256]
        
        for levels in valid_levels:
            filter_instance = DitherFilter(levels=levels)
            filter_instance._validate_parameters()  # Should not raise
    
    def test_invalid_levels_range(self):
        """Test validation rejects levels values outside [2, 256]."""
        invalid_levels = [1, 0, 257, 300]
        
        for levels in invalid_levels:
            filter_instance = DitherFilter(levels=levels)
            
            with pytest.raises(FilterValidationError, match="levels must be in range"):
                filter_instance._validate_parameters()
    
    def test_invalid_levels_type(self):
        """Test validation rejects non-integer levels values."""
        filter_instance = DitherFilter()
        
        with pytest.raises(FilterValidationError, match="levels must be an integer"):
            filter_instance.set_parameters(levels=8.5)
    
    def test_valid_bayer_sizes(self):
        """Test validation accepts valid bayer_size values."""
        valid_sizes = [2, 4, 8, 16, 32, 64]
        
        for size in valid_sizes:
            filter_instance = DitherFilter(bayer_size=size)
            filter_instance._validate_parameters()  # Should not raise
    
    def test_invalid_bayer_sizes(self):
        """Test validation rejects invalid bayer_size values."""
        invalid_sizes = [1, 3, 6, 12, 128]
        
        for size in invalid_sizes:
            filter_instance = DitherFilter(bayer_size=size)
            
            with pytest.raises(FilterValidationError, match="bayer_size must be one of"):
                filter_instance._validate_parameters()
    
    def test_invalid_bayer_size_type(self):
        """Test validation rejects non-integer bayer_size values."""
        filter_instance = DitherFilter()
        
        with pytest.raises(FilterValidationError, match="bayer_size must be an integer"):
            filter_instance.set_parameters(bayer_size=4.0)
    
    def test_valid_pixel_step_range(self):
        """Test validation accepts valid pixel_step values."""
        valid_steps = [1, 2, 4, 8, 16, 32, 64]
        
        for step in valid_steps:
            filter_instance = DitherFilter(pixel_step=step)
            filter_instance._validate_parameters()  # Should not raise
    
    def test_invalid_pixel_step_range(self):
        """Test validation rejects pixel_step values outside [1, 64]."""
        invalid_steps = [0, -1, 65, 100]
        
        for step in invalid_steps:
            filter_instance = DitherFilter(pixel_step=step)
            
            with pytest.raises(FilterValidationError, match="pixel_step must be in range"):
                filter_instance._validate_parameters()
    
    def test_invalid_pixel_step_type(self):
        """Test validation rejects non-integer pixel_step values."""
        filter_instance = DitherFilter()
        
        with pytest.raises(FilterValidationError, match="pixel_step must be an integer"):
            filter_instance.set_parameters(pixel_step=4.0)


class TestDitherFilterInputValidation:
    """Test input validation for DitherFilter."""
    
    def test_valid_rgb_image(self):
        """Test validation accepts valid RGB images."""
        filter_instance = DitherFilter()
        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        assert filter_instance.validate_input(rgb_image) is True
    
    def test_valid_rgba_image(self):
        """Test validation accepts valid RGBA images."""
        filter_instance = DitherFilter()
        rgba_image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        
        assert filter_instance.validate_input(rgba_image) is True
    
    def test_valid_grayscale_image(self):
        """Test validation accepts valid grayscale images."""
        filter_instance = DitherFilter()
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        assert filter_instance.validate_input(gray_image) is True
    
    def test_invalid_dimensions(self):
        """Test validation rejects images with invalid dimensions."""
        filter_instance = DitherFilter()
        
        # 1D array
        with pytest.raises(FilterValidationError, match="Image data must be 2D or 3D"):
            filter_instance.validate_input(np.array([1, 2, 3]))
        
        # 4D array
        with pytest.raises(FilterValidationError, match="Image data must be 2D or 3D"):
            filter_instance.validate_input(np.random.randint(0, 256, (10, 10, 10, 3)))
    
    def test_invalid_channels(self):
        """Test validation rejects images with invalid channel counts."""
        filter_instance = DitherFilter()
        
        # 2 channels
        with pytest.raises(FilterValidationError, match="Color images must have 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_instance.validate_input(np.random.randint(0, 256, (100, 100, 2)))
        
        # 5 channels
        with pytest.raises(FilterValidationError, match="Color images must have 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_instance.validate_input(np.random.randint(0, 256, (100, 100, 5)))
    
    def test_empty_array(self):
        """Test validation rejects empty arrays."""
        filter_instance = DitherFilter()
        
        with pytest.raises(FilterValidationError, match="Input array cannot be empty"):
            filter_instance.validate_input(np.array([]))
    
    def test_non_numpy_array(self):
        """Test validation rejects non-numpy arrays."""
        filter_instance = DitherFilter()
        
        with pytest.raises(FilterValidationError, match="Input must be a numpy array"):
            filter_instance.validate_input([[1, 2, 3], [4, 5, 6]])


class TestDitherFilterFloydSteinberg:
    """Test Floyd-Steinberg dithering functionality."""
    
    def test_floyd_steinberg_basic_uint8(self):
        """Test Floyd-Steinberg dithering on uint8 images."""
        filter_instance = DitherFilter(pattern_type="floyd_steinberg", levels=4)
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Values should be quantized to specific levels
        expected_levels = [0, 85, 170, 255]  # 4 levels for uint8
        unique_values = np.unique(result)
        for value in unique_values:
            assert any(abs(value - level) <= 1 for level in expected_levels)
    
    def test_floyd_steinberg_grayscale(self):
        """Test Floyd-Steinberg dithering on grayscale images."""
        filter_instance = DitherFilter(pattern_type="floyd_steinberg", levels=2)
        original = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Should be quantized to 2 levels (0 and 255)
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert all(v in [0, 255] for v in unique_values)
    
    def test_floyd_steinberg_levels_scaling(self):
        """Test that different levels produce different quantization."""
        original = np.full((50, 50, 3), 128, dtype=np.uint8)
        
        # Test with 2 levels
        filter_2 = DitherFilter(pattern_type="floyd_steinberg", levels=2)
        result_2 = filter_2.apply(original.copy())
        
        # Test with 8 levels
        filter_8 = DitherFilter(pattern_type="floyd_steinberg", levels=8)
        result_8 = filter_8.apply(original.copy())
        
        # 8 levels should have more unique values than 2 levels
        unique_2 = len(np.unique(result_2))
        unique_8 = len(np.unique(result_8))
        assert unique_8 >= unique_2
    
    def test_floyd_steinberg_float_image(self):
        """Test Floyd-Steinberg dithering on float images."""
        filter_instance = DitherFilter(pattern_type="floyd_steinberg", levels=4)
        original = np.random.random((50, 50, 3)).astype(np.float32)
        
        result = filter_instance.apply(original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Values should be within [0, 1] range
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        
        # Should be quantized to specific levels
        expected_levels = [0.0, 1/3, 2/3, 1.0]  # 4 levels for float
        unique_values = np.unique(result)
        for value in unique_values:
            assert any(abs(value - level) <= 0.01 for level in expected_levels)


class TestDitherFilterBayer:
    """Test Bayer dithering functionality."""
    
    def test_bayer_basic_uint8(self):
        """Test Bayer dithering on uint8 images."""
        filter_instance = DitherFilter(pattern_type="bayer", levels=4, bayer_size=4)
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Values should be quantized to specific levels
        expected_levels = [0, 85, 170, 255]  # 4 levels for uint8
        unique_values = np.unique(result)
        for value in unique_values:
            assert any(abs(value - level) <= 1 for level in expected_levels)
    
    def test_bayer_different_sizes(self):
        """Test Bayer dithering with different matrix sizes."""
        original = np.full((64, 64, 3), 128, dtype=np.uint8)
        
        for bayer_size in [2, 4, 8, 16, 32, 64]:
            filter_instance = DitherFilter(
                pattern_type="bayer", 
                levels=4, 
                bayer_size=bayer_size
            )
            result = filter_instance.apply(original.copy())
            
            # Should produce valid results for all sizes
            assert result.shape == original.shape
            assert result.dtype == original.dtype
            
            # Should be quantized
            unique_values = np.unique(result)
            assert len(unique_values) <= 4  # At most 4 levels
    
    def test_bayer_matrix_generation(self):
        """Test Bayer matrix generation for different sizes."""
        filter_instance = DitherFilter()
        
        # Test 2x2 matrix
        matrix_2 = filter_instance._generate_bayer_matrix(2)
        assert matrix_2.shape == (2, 2)
        assert np.all(matrix_2 >= 0)
        assert np.all(matrix_2 < 4)  # Values should be 0-3 for 2x2
        
        # Test 4x4 matrix
        matrix_4 = filter_instance._generate_bayer_matrix(4)
        assert matrix_4.shape == (4, 4)
        assert np.all(matrix_4 >= 0)
        assert np.all(matrix_4 < 16)  # Values should be 0-15 for 4x4
        
        # Test 8x8 matrix
        matrix_8 = filter_instance._generate_bayer_matrix(8)
        assert matrix_8.shape == (8, 8)
        assert np.all(matrix_8 >= 0)
        assert np.all(matrix_8 < 64)  # Values should be 0-63 for 8x8
        
        # Test 16x16 matrix
        matrix_16 = filter_instance._generate_bayer_matrix(16)
        assert matrix_16.shape == (16, 16)
        assert np.all(matrix_16 >= 0)
        assert np.all(matrix_16 < 256)  # Values should be 0-255 for 16x16
    
    def test_bayer_grayscale(self):
        """Test Bayer dithering on grayscale images."""
        filter_instance = DitherFilter(pattern_type="bayer", levels=2, bayer_size=4)
        original = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Should be quantized to 2 levels (0 and 255)
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert all(v in [0, 255] for v in unique_values)


class TestDitherFilterRandom:
    """Test random threshold dithering functionality."""
    
    def test_random_basic_uint8(self):
        """Test random dithering on uint8 images."""
        filter_instance = DitherFilter(pattern_type="random", levels=4)
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Values should be quantized to specific levels
        expected_levels = [0, 85, 170, 255]  # 4 levels for uint8
        unique_values = np.unique(result)
        for value in unique_values:
            assert any(abs(value - level) <= 1 for level in expected_levels)
    
    def test_random_different_results(self):
        """Test that random dithering produces different results on repeated calls."""
        filter_instance = DitherFilter(pattern_type="random", levels=4)
        original = np.full((50, 50, 3), 128, dtype=np.uint8)
        
        result1 = filter_instance.apply(original.copy())
        result2 = filter_instance.apply(original.copy())
        
        # Results should be different due to randomness
        # (with very high probability)
        assert not np.array_equal(result1, result2)
    
    def test_random_grayscale(self):
        """Test random dithering on grayscale images."""
        filter_instance = DitherFilter(pattern_type="random", levels=2)
        original = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Should be quantized to 2 levels (0 and 255)
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert all(v in [0, 255] for v in unique_values)
    
    def test_random_levels_scaling(self):
        """Test that different levels produce different quantization."""
        original = np.full((50, 50, 3), 128, dtype=np.uint8)
        
        # Test with 2 levels
        filter_2 = DitherFilter(pattern_type="random", levels=2)
        result_2 = filter_2.apply(original.copy())
        
        # Test with 8 levels
        filter_8 = DitherFilter(pattern_type="random", levels=8)
        result_8 = filter_8.apply(original.copy())
        
        # 8 levels should have more unique values than 2 levels
        unique_2 = len(np.unique(result_2))
        unique_8 = len(np.unique(result_8))
        assert unique_8 >= unique_2


class TestDitherFilterEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_rgba_image_preserves_alpha(self):
        """Test that RGBA images preserve alpha channel correctly."""
        filter_instance = DitherFilter(pattern_type="floyd_steinberg", levels=4)
        original = np.random.randint(0, 256, (50, 50, 4), dtype=np.uint8)
        original_alpha = original[..., 3].copy()
        
        result = filter_instance.apply(original)
        
        # Alpha channel should be preserved
        assert np.array_equal(result[..., 3], original_alpha)
        
        # RGB channels should be dithered
        assert not np.array_equal(result[..., :3], original[..., :3])
    
    def test_single_pixel_image(self):
        """Test dithering on single pixel images."""
        filter_instance = DitherFilter(pattern_type="floyd_steinberg", levels=4)
        original = np.array([[[128, 128, 128]]], dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Should handle single pixel without error
        assert result.shape == original.shape
        assert result.dtype == original.dtype
    
    def test_minimum_levels(self):
        """Test dithering with minimum levels (2)."""
        filter_instance = DitherFilter(pattern_type="floyd_steinberg", levels=2)
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Should be quantized to only 2 levels
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert all(v in [0, 255] for v in unique_values)
    
    def test_maximum_levels(self):
        """Test dithering with maximum levels (256)."""
        filter_instance = DitherFilter(pattern_type="floyd_steinberg", levels=256)
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # With 256 levels, result should be very close to original
        # (since uint8 already has 256 levels)
        assert result.shape == original.shape
        assert result.dtype == original.dtype
    
    def test_parameter_override_in_apply(self):
        """Test parameter override during apply call."""
        filter_instance = DitherFilter(pattern_type="floyd_steinberg", levels=8)
        original = np.full((50, 50, 3), 128, dtype=np.uint8)
        
        # Override parameters in apply call
        result = filter_instance.apply(
            original, 
            pattern_type="bayer", 
            levels=2,
            bayer_size=2
        )
        
        # Should use overridden parameters
        assert filter_instance.parameters['pattern_type'] == "bayer"
        assert filter_instance.parameters['levels'] == 2
        assert filter_instance.parameters['bayer_size'] == 2
        
        # Result should reflect Bayer dithering with 2 levels
        unique_values = np.unique(result)
        assert len(unique_values) <= 2


class TestDitherFilterIntegration:
    """Test integration with BaseFilter features."""
    
    def test_progress_tracking(self):
        """Test that progress tracking works correctly."""
        filter_instance = DitherFilter(pattern_type="floyd_steinberg", levels=4)
        progress_values = []
        
        def progress_callback(progress):
            progress_values.append(progress)
        
        filter_instance.set_progress_callback(progress_callback)
        
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        filter_instance.apply(original)
        
        # Should have recorded progress values
        assert len(progress_values) > 0
        assert 0.0 in progress_values  # Should start at 0
        assert 1.0 in progress_values  # Should end at 1
    
    def test_metadata_recording(self):
        """Test that metadata is properly recorded."""
        filter_instance = DitherFilter(pattern_type="bayer", levels=4)
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Check metadata was recorded
        assert filter_instance.metadata.execution_time > 0
        assert filter_instance.metadata.input_shape == original.shape
        assert filter_instance.metadata.output_shape == result.shape
        assert filter_instance.metadata.progress == 1.0
    
    def test_memory_efficiency(self):
        """Test memory efficiency features."""
        filter_instance = DitherFilter(pattern_type="random", levels=4)
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Check memory tracking
        assert filter_instance.metadata.memory_usage > 0
        assert filter_instance.metadata.peak_memory_usage > 0
        
        # Dithering should not use in-place processing
        assert not filter_instance.metadata.used_inplace_processing


class TestDitherFilterPixelStep:
    """Test pixel step functionality for chunky/pixelated dithering."""
    
    def test_pixel_step_basic_functionality(self):
        """Test basic pixel step functionality."""
        filter_instance = DitherFilter(pattern_type="bayer", levels=4, pixel_step=4)
        original = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Should be quantized
        unique_values = len(np.unique(result))
        assert unique_values <= 4
    
    def test_pixel_step_chunky_effect(self):
        """Test that pixel_step creates chunky/pixelated effect."""
        # Create a gradient image
        gradient = np.linspace(0, 255, 64).astype(np.uint8)
        test_image = np.tile(gradient.reshape(1, -1, 1), (64, 1, 3))
        
        # Test with different pixel steps
        for pixel_step in [1, 2, 4, 8]:
            filter_instance = DitherFilter(
                pattern_type="bayer", 
                levels=4, 
                pixel_step=pixel_step
            )
            result = filter_instance.apply(test_image.copy())
            
            # Check that blocks of pixels have the same value
            if pixel_step > 1:
                # Sample a few blocks to verify they're uniform
                for y in range(0, 64 - pixel_step, pixel_step):
                    for x in range(0, 64 - pixel_step, pixel_step):
                        block = result[y:y+pixel_step, x:x+pixel_step]
                        # All pixels in the block should be the same
                        assert np.all(block == block[0, 0])
    
    def test_pixel_step_with_different_algorithms(self):
        """Test pixel_step works with all dithering algorithms."""
        original = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        pixel_step = 4
        
        algorithms = ["floyd_steinberg", "bayer", "random"]
        
        for algorithm in algorithms:
            filter_instance = DitherFilter(
                pattern_type=algorithm,
                levels=4,
                pixel_step=pixel_step
            )
            
            result = filter_instance.apply(original.copy())
            
            # Should produce valid results
            assert result.shape == original.shape
            assert result.dtype == original.dtype
            
            # Should be quantized
            unique_values = len(np.unique(result))
            assert unique_values <= 4
    
    def test_pixel_step_grayscale(self):
        """Test pixel_step works with grayscale images."""
        filter_instance = DitherFilter(
            pattern_type="bayer", 
            levels=2, 
            pixel_step=8
        )
        original = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Should be binary
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
    
    def test_pixel_step_edge_cases(self):
        """Test pixel_step with edge cases."""
        # Test with pixel_step = 1 (should be normal dithering)
        filter_normal = DitherFilter(pattern_type="bayer", levels=4, pixel_step=1)
        filter_chunky = DitherFilter(pattern_type="bayer", levels=4, pixel_step=4)
        
        original = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        
        result_normal = filter_normal.apply(original.copy())
        result_chunky = filter_chunky.apply(original.copy())
        
        # Results should be different
        assert not np.array_equal(result_normal, result_chunky)
        
        # Both should be properly quantized
        assert len(np.unique(result_normal)) <= 4
        assert len(np.unique(result_chunky)) <= 4


class TestDitherFilterQuantizationAccuracy:
    """Test quantization accuracy for different algorithms."""
    
    def test_quantization_levels_accuracy(self):
        """Test that quantization produces correct number of levels."""
        test_levels = [2, 4, 8, 16]
        patterns = ["floyd_steinberg", "bayer", "random"]
        
        for levels in test_levels:
            for pattern in patterns:
                filter_instance = DitherFilter(pattern_type=pattern, levels=levels)
                
                # Create gradient image to test quantization
                gradient = np.linspace(0, 255, 100).astype(np.uint8)
                gradient_image = np.tile(gradient.reshape(1, -1, 1), (50, 1, 3))
                
                result = filter_instance.apply(gradient_image)
                
                # Count unique values (allowing for some tolerance)
                unique_values = np.unique(result)
                
                # Should have at most the specified number of levels
                # (may be fewer if some levels are not used)
                assert len(unique_values) <= levels
    
    def test_pattern_quality_differences(self):
        """Test that different patterns produce visually different results."""
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        # Apply different dithering patterns
        floyd_filter = DitherFilter(pattern_type="floyd_steinberg", levels=4)
        bayer_filter = DitherFilter(pattern_type="bayer", levels=4, bayer_size=4)
        random_filter = DitherFilter(pattern_type="random", levels=4)
        
        floyd_result = floyd_filter.apply(original.copy())
        bayer_result = bayer_filter.apply(original.copy())
        random_result = random_filter.apply(original.copy())
        
        # Results should be different from each other
        assert not np.array_equal(floyd_result, bayer_result)
        assert not np.array_equal(floyd_result, random_result)
        assert not np.array_equal(bayer_result, random_result)
        
        # But all should have similar quantization levels
        floyd_unique = len(np.unique(floyd_result))
        bayer_unique = len(np.unique(bayer_result))
        random_unique = len(np.unique(random_result))
        
        # All should have similar number of unique values (within reasonable range)
        assert abs(floyd_unique - bayer_unique) <= 2
        assert abs(floyd_unique - random_unique) <= 2
        assert abs(bayer_unique - random_unique) <= 2