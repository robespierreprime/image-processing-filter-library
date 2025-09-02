"""
Unit tests for NoiseFilter.

Tests parameter validation, noise algorithms, and edge cases for the NoiseFilter class.
"""

import pytest
import numpy as np
from image_processing_library.filters.artistic.noise_filter import NoiseFilter
from image_processing_library.core.utils import FilterValidationError


class TestNoiseFilterInitialization:
    """Test NoiseFilter initialization and parameter validation."""
    
    def test_default_initialization(self):
        """Test filter initialization with default parameters."""
        filter_instance = NoiseFilter()
        
        assert filter_instance.name == "noise"
        assert filter_instance.category == "artistic"
        assert filter_instance.parameters['noise_type'] == "gaussian"
        assert filter_instance.parameters['intensity'] == 0.1
        assert filter_instance.parameters['salt_pepper_ratio'] == 0.5
    
    def test_custom_initialization(self):
        """Test filter initialization with custom parameters."""
        filter_instance = NoiseFilter(
            noise_type="salt_pepper",
            intensity=0.3,
            salt_pepper_ratio=0.7
        )
        
        assert filter_instance.parameters['noise_type'] == "salt_pepper"
        assert filter_instance.parameters['intensity'] == 0.3
        assert filter_instance.parameters['salt_pepper_ratio'] == 0.7
    
    def test_parameter_updates(self):
        """Test parameter updates after initialization."""
        filter_instance = NoiseFilter()
        
        filter_instance.set_parameters(
            noise_type="uniform",
            intensity=0.5,
            salt_pepper_ratio=0.2
        )
        
        params = filter_instance.get_parameters()
        assert params['noise_type'] == "uniform"
        assert params['intensity'] == 0.5
        assert params['salt_pepper_ratio'] == 0.2


class TestNoiseFilterParameterValidation:
    """Test parameter validation for NoiseFilter."""
    
    def test_valid_noise_types(self):
        """Test validation accepts valid noise types."""
        valid_types = ["gaussian", "salt_pepper", "uniform"]
        
        for noise_type in valid_types:
            filter_instance = NoiseFilter(noise_type=noise_type)
            filter_instance._validate_parameters()  # Should not raise
    
    def test_invalid_noise_type_string(self):
        """Test validation rejects invalid noise type strings."""
        filter_instance = NoiseFilter(noise_type="invalid_type")
        
        with pytest.raises(FilterValidationError, match="noise_type must be one of"):
            filter_instance._validate_parameters()
    
    def test_invalid_noise_type_non_string(self):
        """Test validation rejects non-string noise types."""
        filter_instance = NoiseFilter()
        filter_instance.set_parameters(noise_type=123)
        
        with pytest.raises(FilterValidationError, match="noise_type must be a string"):
            filter_instance._validate_parameters()
    
    def test_valid_intensity_range(self):
        """Test validation accepts valid intensity values."""
        valid_intensities = [0.0, 0.1, 0.5, 1.0]
        
        for intensity in valid_intensities:
            filter_instance = NoiseFilter(intensity=intensity)
            filter_instance._validate_parameters()  # Should not raise
    
    def test_invalid_intensity_range(self):
        """Test validation rejects intensity values outside [0.0, 1.0]."""
        invalid_intensities = [-0.1, 1.1, 2.0]
        
        for intensity in invalid_intensities:
            filter_instance = NoiseFilter(intensity=intensity)
            
            with pytest.raises(FilterValidationError, match="intensity must be in range"):
                filter_instance._validate_parameters()
    
    def test_invalid_intensity_type(self):
        """Test validation rejects non-numeric intensity values."""
        filter_instance = NoiseFilter()
        filter_instance.set_parameters(intensity="invalid")
        
        with pytest.raises(FilterValidationError, match="intensity must be a number"):
            filter_instance._validate_parameters()
    
    def test_valid_salt_pepper_ratio_range(self):
        """Test validation accepts valid salt_pepper_ratio values."""
        valid_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for ratio in valid_ratios:
            filter_instance = NoiseFilter(salt_pepper_ratio=ratio)
            filter_instance._validate_parameters()  # Should not raise
    
    def test_invalid_salt_pepper_ratio_range(self):
        """Test validation rejects salt_pepper_ratio values outside [0.0, 1.0]."""
        invalid_ratios = [-0.1, 1.1, 2.0]
        
        for ratio in invalid_ratios:
            filter_instance = NoiseFilter(salt_pepper_ratio=ratio)
            
            with pytest.raises(FilterValidationError, match="salt_pepper_ratio must be in range"):
                filter_instance._validate_parameters()
    
    def test_invalid_salt_pepper_ratio_type(self):
        """Test validation rejects non-numeric salt_pepper_ratio values."""
        filter_instance = NoiseFilter()
        filter_instance.set_parameters(salt_pepper_ratio="invalid")
        
        with pytest.raises(FilterValidationError, match="salt_pepper_ratio must be a number"):
            filter_instance._validate_parameters()


class TestNoiseFilterInputValidation:
    """Test input validation for NoiseFilter."""
    
    def test_valid_rgb_image(self):
        """Test validation accepts valid RGB images."""
        filter_instance = NoiseFilter()
        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        assert filter_instance.validate_input(rgb_image) is True
    
    def test_valid_rgba_image(self):
        """Test validation accepts valid RGBA images."""
        filter_instance = NoiseFilter()
        rgba_image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        
        assert filter_instance.validate_input(rgba_image) is True
    
    def test_valid_grayscale_image(self):
        """Test validation accepts valid grayscale images."""
        filter_instance = NoiseFilter()
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        assert filter_instance.validate_input(gray_image) is True
    
    def test_invalid_dimensions(self):
        """Test validation rejects images with invalid dimensions."""
        filter_instance = NoiseFilter()
        
        # 1D array
        with pytest.raises(FilterValidationError, match="Image data must be 2D or 3D"):
            filter_instance.validate_input(np.array([1, 2, 3]))
        
        # 4D array
        with pytest.raises(FilterValidationError, match="Image data must be 2D or 3D"):
            filter_instance.validate_input(np.random.randint(0, 256, (10, 10, 10, 3)))
    
    def test_invalid_channels(self):
        """Test validation rejects images with invalid channel counts."""
        filter_instance = NoiseFilter()
        
        # 2 channels
        with pytest.raises(FilterValidationError, match="Color images must have 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_instance.validate_input(np.random.randint(0, 256, (100, 100, 2)))
        
        # 5 channels
        with pytest.raises(FilterValidationError, match="Color images must have 3 \\(RGB\\) or 4 \\(RGBA\\) channels"):
            filter_instance.validate_input(np.random.randint(0, 256, (100, 100, 5)))
    
    def test_empty_array(self):
        """Test validation rejects empty arrays."""
        filter_instance = NoiseFilter()
        
        with pytest.raises(FilterValidationError, match="Input array cannot be empty"):
            filter_instance.validate_input(np.array([]))
    
    def test_non_numpy_array(self):
        """Test validation rejects non-numpy arrays."""
        filter_instance = NoiseFilter()
        
        with pytest.raises(FilterValidationError, match="Input must be a numpy array"):
            filter_instance.validate_input([[1, 2, 3], [4, 5, 6]])


class TestNoiseFilterGaussianNoise:
    """Test gaussian noise functionality."""
    
    def test_gaussian_noise_uint8(self):
        """Test gaussian noise application to uint8 images."""
        filter_instance = NoiseFilter(noise_type="gaussian", intensity=0.1)
        original = np.full((50, 50, 3), 128, dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Result should be different from original (with high probability)
        assert not np.array_equal(result, original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Values should be within valid range
        assert np.all(result >= 0)
        assert np.all(result <= 255)
    
    def test_gaussian_noise_float(self):
        """Test gaussian noise application to float images."""
        filter_instance = NoiseFilter(noise_type="gaussian", intensity=0.1)
        original = np.full((50, 50, 3), 0.5, dtype=np.float32)
        
        result = filter_instance.apply(original)
        
        # Result should be different from original (with high probability)
        assert not np.array_equal(result, original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Values should be within valid range
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
    
    def test_gaussian_noise_intensity_scaling(self):
        """Test that higher intensity produces more noise."""
        original = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        # Apply low intensity noise
        low_filter = NoiseFilter(noise_type="gaussian", intensity=0.05)
        low_result = low_filter.apply(original.copy())
        
        # Apply high intensity noise
        high_filter = NoiseFilter(noise_type="gaussian", intensity=0.3)
        high_result = high_filter.apply(original.copy())
        
        # Calculate variance from original
        low_variance = np.var(low_result.astype(np.float32) - original.astype(np.float32))
        high_variance = np.var(high_result.astype(np.float32) - original.astype(np.float32))
        
        # Higher intensity should produce higher variance
        assert high_variance > low_variance
    
    def test_gaussian_noise_zero_intensity(self):
        """Test that zero intensity produces no change."""
        filter_instance = NoiseFilter(noise_type="gaussian", intensity=0.0)
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Should be identical to original
        assert np.array_equal(result, original)


class TestNoiseFilterSaltPepperNoise:
    """Test salt-and-pepper noise functionality."""
    
    def test_salt_pepper_noise_basic(self):
        """Test basic salt-and-pepper noise application."""
        filter_instance = NoiseFilter(
            noise_type="salt_pepper", 
            intensity=0.1, 
            salt_pepper_ratio=0.5
        )
        original = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Result should be different from original (with high probability)
        assert not np.array_equal(result, original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Should contain some salt (255) and pepper (0) pixels
        unique_values = np.unique(result)
        assert 0 in unique_values or 255 in unique_values
    
    def test_salt_pepper_ratio_all_salt(self):
        """Test salt-pepper noise with all salt (ratio=1.0)."""
        filter_instance = NoiseFilter(
            noise_type="salt_pepper", 
            intensity=0.2, 
            salt_pepper_ratio=1.0
        )
        original = np.full((100, 100), 128, dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Should contain salt pixels (255) but no pepper pixels (0)
        changed_pixels = result != original
        if np.any(changed_pixels):
            changed_values = result[changed_pixels]
            assert np.all(changed_values == 255)  # All changed pixels should be salt
    
    def test_salt_pepper_ratio_all_pepper(self):
        """Test salt-pepper noise with all pepper (ratio=0.0)."""
        filter_instance = NoiseFilter(
            noise_type="salt_pepper", 
            intensity=0.2, 
            salt_pepper_ratio=0.0
        )
        original = np.full((100, 100), 128, dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Should contain pepper pixels (0) but no salt pixels (255)
        changed_pixels = result != original
        if np.any(changed_pixels):
            changed_values = result[changed_pixels]
            assert np.all(changed_values == 0)  # All changed pixels should be pepper
    
    def test_salt_pepper_intensity_scaling(self):
        """Test that higher intensity affects more pixels."""
        original = np.full((100, 100), 128, dtype=np.uint8)
        
        # Apply low intensity noise
        low_filter = NoiseFilter(noise_type="salt_pepper", intensity=0.05)
        low_result = low_filter.apply(original.copy())
        
        # Apply high intensity noise
        high_filter = NoiseFilter(noise_type="salt_pepper", intensity=0.2)
        high_result = high_filter.apply(original.copy())
        
        # Count changed pixels
        low_changed = np.sum(low_result != original)
        high_changed = np.sum(high_result != original)
        
        # Higher intensity should affect more pixels
        assert high_changed >= low_changed
    
    def test_salt_pepper_zero_intensity(self):
        """Test that zero intensity produces no change."""
        filter_instance = NoiseFilter(noise_type="salt_pepper", intensity=0.0)
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Should be identical to original
        assert np.array_equal(result, original)


class TestNoiseFilterUniformNoise:
    """Test uniform noise functionality."""
    
    def test_uniform_noise_uint8(self):
        """Test uniform noise application to uint8 images."""
        filter_instance = NoiseFilter(noise_type="uniform", intensity=0.1)
        original = np.full((50, 50, 3), 128, dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Result should be different from original (with high probability)
        assert not np.array_equal(result, original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Values should be within valid range
        assert np.all(result >= 0)
        assert np.all(result <= 255)
    
    def test_uniform_noise_float(self):
        """Test uniform noise application to float images."""
        filter_instance = NoiseFilter(noise_type="uniform", intensity=0.1)
        original = np.full((50, 50, 3), 0.5, dtype=np.float32)
        
        result = filter_instance.apply(original)
        
        # Result should be different from original (with high probability)
        assert not np.array_equal(result, original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Values should be within valid range
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
    
    def test_uniform_noise_intensity_scaling(self):
        """Test that higher intensity produces more noise."""
        original = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        # Apply low intensity noise
        low_filter = NoiseFilter(noise_type="uniform", intensity=0.05)
        low_result = low_filter.apply(original.copy())
        
        # Apply high intensity noise
        high_filter = NoiseFilter(noise_type="uniform", intensity=0.3)
        high_result = high_filter.apply(original.copy())
        
        # Calculate variance from original
        low_variance = np.var(low_result.astype(np.float32) - original.astype(np.float32))
        high_variance = np.var(high_result.astype(np.float32) - original.astype(np.float32))
        
        # Higher intensity should produce higher variance
        assert high_variance > low_variance
    
    def test_uniform_noise_zero_intensity(self):
        """Test that zero intensity produces no change."""
        filter_instance = NoiseFilter(noise_type="uniform", intensity=0.0)
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Should be identical to original
        assert np.array_equal(result, original)


class TestNoiseFilterEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_grayscale_image(self):
        """Test noise application to grayscale images."""
        filter_instance = NoiseFilter(noise_type="gaussian", intensity=0.1)
        original = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Result should be same shape and type
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        
        # Should be different (with high probability)
        assert not np.array_equal(result, original)
    
    def test_rgba_image_preserves_alpha(self):
        """Test that RGBA images preserve alpha channel correctly."""
        filter_instance = NoiseFilter(noise_type="gaussian", intensity=0.1)
        original = np.random.randint(0, 256, (50, 50, 4), dtype=np.uint8)
        original_alpha = original[..., 3].copy()
        
        result = filter_instance.apply(original)
        
        # Alpha channel should be preserved
        assert np.array_equal(result[..., 3], original_alpha)
        
        # RGB channels should be different (with high probability)
        assert not np.array_equal(result[..., :3], original[..., :3])
    
    def test_single_pixel_image(self):
        """Test noise application to single pixel images."""
        filter_instance = NoiseFilter(noise_type="gaussian", intensity=0.1)
        original = np.array([[[128, 128, 128]]], dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Should handle single pixel without error
        assert result.shape == original.shape
        assert result.dtype == original.dtype
    
    def test_parameter_override_in_apply(self):
        """Test parameter override during apply call."""
        filter_instance = NoiseFilter(noise_type="gaussian", intensity=0.1)
        original = np.full((50, 50, 3), 128, dtype=np.uint8)
        
        # Override parameters in apply call
        result = filter_instance.apply(
            original, 
            noise_type="salt_pepper", 
            intensity=0.2,
            salt_pepper_ratio=1.0
        )
        
        # Should use overridden parameters
        assert filter_instance.parameters['noise_type'] == "salt_pepper"
        assert filter_instance.parameters['intensity'] == 0.2
        assert filter_instance.parameters['salt_pepper_ratio'] == 1.0
        
        # Result should reflect salt-pepper noise
        changed_pixels = result != original
        if np.any(changed_pixels):
            changed_values = result[changed_pixels]
            assert np.all(changed_values == 255)  # All salt with ratio=1.0


class TestNoiseFilterIntegration:
    """Test integration with BaseFilter features."""
    
    def test_progress_tracking(self):
        """Test that progress tracking works correctly."""
        filter_instance = NoiseFilter()
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
        filter_instance = NoiseFilter()
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Check metadata was recorded
        assert filter_instance.metadata.execution_time > 0
        assert filter_instance.metadata.input_shape == original.shape
        assert filter_instance.metadata.output_shape == result.shape
        assert filter_instance.metadata.progress == 1.0
    
    def test_memory_efficiency(self):
        """Test memory efficiency features."""
        filter_instance = NoiseFilter()
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = filter_instance.apply(original)
        
        # Check memory tracking
        assert filter_instance.metadata.memory_usage > 0
        assert filter_instance.metadata.peak_memory_usage > 0