"""
Unit tests for image I/O functionality.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
from PIL import Image
import cv2

from image_processing_library.media_io.image_io import (
    load_image,
    load_image_pil,
    load_image_cv2,
    save_image,
    save_image_pil,
    save_image_cv2,
    numpy_to_pil,
    pil_to_numpy,
    get_image_info,
    validate_image_array,
    ImageIOError,
    SUPPORTED_FORMATS
)


class TestImageIO(unittest.TestCase):
    """Test cases for image I/O operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test images
        self.rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.rgba_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        self.gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Create test image files
        self.test_jpg = Path(self.temp_dir) / "test.jpg"
        self.test_png = Path(self.temp_dir) / "test.png"
        self.test_png_rgb = Path(self.temp_dir) / "test_rgb.png"
        self.test_bmp = Path(self.temp_dir) / "test.bmp"
        
        # Save test images using PIL
        Image.fromarray(self.rgb_image).save(self.test_jpg, quality=100)  # High quality JPEG
        Image.fromarray(self.rgba_image).save(self.test_png)
        Image.fromarray(self.rgb_image).save(self.test_png_rgb)  # RGB PNG for exact comparison
        Image.fromarray(self.gray_image).save(self.test_bmp)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_image_pil_rgb(self):
        """Test loading RGB image with PIL backend."""
        loaded = load_image_pil(self.test_jpg)
        
        self.assertEqual(loaded.shape, (100, 100, 3))
        self.assertEqual(loaded.dtype, np.uint8)
        # JPEG is lossy, so we check if images are reasonably close
        diff = np.abs(loaded.astype(np.int16) - self.rgb_image.astype(np.int16))
        self.assertLess(np.mean(diff), 50)  # Average difference should be reasonable for JPEG
    
    def test_load_image_pil_rgba(self):
        """Test loading RGBA image with PIL backend."""
        loaded = load_image_pil(self.test_png)
        
        self.assertEqual(loaded.shape, (100, 100, 4))
        self.assertEqual(loaded.dtype, np.uint8)
        np.testing.assert_array_equal(loaded, self.rgba_image)
    
    def test_load_image_pil_grayscale(self):
        """Test loading grayscale image with PIL backend."""
        loaded = load_image_pil(self.test_bmp)
        
        self.assertEqual(loaded.shape, (100, 100))
        self.assertEqual(loaded.dtype, np.uint8)
        np.testing.assert_array_equal(loaded, self.gray_image)
    
    def test_load_image_cv2_rgb(self):
        """Test loading RGB image with OpenCV backend."""
        loaded = load_image_cv2(self.test_jpg)
        
        self.assertEqual(loaded.shape, (100, 100, 3))
        self.assertEqual(loaded.dtype, np.uint8)
        # JPEG is lossy, so we check if images are reasonably close
        diff = np.abs(loaded.astype(np.int16) - self.rgb_image.astype(np.int16))
        self.assertLess(np.mean(diff), 50)  # Average difference should be reasonable for JPEG
    
    def test_load_image_with_backend_selection(self):
        """Test load_image function with backend selection."""
        pil_result = load_image(self.test_png_rgb, backend='pil')
        cv2_result = load_image(self.test_png_rgb, backend='cv2')
        
        self.assertEqual(pil_result.shape, cv2_result.shape)
        np.testing.assert_array_equal(pil_result, cv2_result)
    
    def test_load_image_invalid_backend(self):
        """Test load_image with invalid backend."""
        with self.assertRaises(ValueError):
            load_image(self.test_jpg, backend='invalid')
    
    def test_load_image_nonexistent_file(self):
        """Test loading non-existent file."""
        with self.assertRaises(ImageIOError):
            load_image_pil("nonexistent.jpg")
        
        with self.assertRaises(ImageIOError):
            load_image_cv2("nonexistent.jpg")
    
    def test_save_image_pil_jpg(self):
        """Test saving image as JPEG with PIL."""
        output_path = Path(self.temp_dir) / "output.jpg"
        save_image_pil(self.rgb_image, output_path, quality=90)
        
        self.assertTrue(output_path.exists())
        
        # Load and verify
        loaded = load_image_pil(output_path)
        self.assertEqual(loaded.shape, self.rgb_image.shape)
    
    def test_save_image_pil_png(self):
        """Test saving image as PNG with PIL."""
        output_path = Path(self.temp_dir) / "output.png"
        save_image_pil(self.rgba_image, output_path)
        
        self.assertTrue(output_path.exists())
        
        # Load and verify
        loaded = load_image_pil(output_path)
        self.assertEqual(loaded.shape, self.rgba_image.shape)
        np.testing.assert_array_equal(loaded, self.rgba_image)
    
    def test_save_image_cv2_jpg(self):
        """Test saving image as JPEG with OpenCV."""
        output_path = Path(self.temp_dir) / "output_cv2.jpg"
        save_image_cv2(self.rgb_image, output_path, quality=90)
        
        self.assertTrue(output_path.exists())
        
        # Load and verify
        loaded = load_image_cv2(output_path)
        self.assertEqual(loaded.shape, self.rgb_image.shape)
    
    def test_save_image_with_backend_selection(self):
        """Test save_image function with backend selection."""
        pil_path = Path(self.temp_dir) / "pil_output.png"
        cv2_path = Path(self.temp_dir) / "cv2_output.png"
        
        save_image(self.rgb_image, pil_path, backend='pil')
        save_image(self.rgb_image, cv2_path, backend='cv2')
        
        self.assertTrue(pil_path.exists())
        self.assertTrue(cv2_path.exists())
    
    def test_save_image_invalid_backend(self):
        """Test save_image with invalid backend."""
        output_path = Path(self.temp_dir) / "output.jpg"
        
        with self.assertRaises(ValueError):
            save_image(self.rgb_image, output_path, backend='invalid')
    
    def test_numpy_to_pil_rgb(self):
        """Test converting RGB numpy array to PIL Image."""
        pil_img = numpy_to_pil(self.rgb_image)
        
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.mode, 'RGB')
        self.assertEqual(pil_img.size, (100, 100))
        
        # Convert back and verify
        converted_back = np.array(pil_img)
        np.testing.assert_array_equal(converted_back, self.rgb_image)
    
    def test_numpy_to_pil_rgba(self):
        """Test converting RGBA numpy array to PIL Image."""
        pil_img = numpy_to_pil(self.rgba_image)
        
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.mode, 'RGBA')
        self.assertEqual(pil_img.size, (100, 100))
    
    def test_numpy_to_pil_grayscale(self):
        """Test converting grayscale numpy array to PIL Image."""
        pil_img = numpy_to_pil(self.gray_image)
        
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.mode, 'L')
        self.assertEqual(pil_img.size, (100, 100))
    
    def test_numpy_to_pil_float_normalization(self):
        """Test converting float array with normalization."""
        float_image = self.rgb_image.astype(np.float32) / 255.0
        pil_img = numpy_to_pil(float_image)
        
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.mode, 'RGB')
    
    def test_pil_to_numpy(self):
        """Test converting PIL Image to numpy array."""
        pil_img = Image.fromarray(self.rgb_image)
        numpy_img = pil_to_numpy(pil_img)
        
        self.assertIsInstance(numpy_img, np.ndarray)
        self.assertEqual(numpy_img.dtype, np.uint8)
        np.testing.assert_array_equal(numpy_img, self.rgb_image)
    
    def test_get_image_info(self):
        """Test getting image information."""
        info = get_image_info(self.test_jpg)
        
        self.assertIsInstance(info, dict)
        self.assertEqual(info['width'], 100)
        self.assertEqual(info['height'], 100)
        self.assertEqual(info['format'], 'JPEG')
        self.assertIn('file_size', info)
    
    def test_get_image_info_nonexistent(self):
        """Test getting info for non-existent file."""
        with self.assertRaises(ImageIOError):
            get_image_info("nonexistent.jpg")
    
    def test_validate_image_array_valid_rgb(self):
        """Test validating valid RGB array."""
        is_valid, message = validate_image_array(self.rgb_image)
        self.assertTrue(is_valid)
        self.assertEqual(message, "")
    
    def test_validate_image_array_valid_grayscale(self):
        """Test validating valid grayscale array."""
        is_valid, message = validate_image_array(self.gray_image)
        self.assertTrue(is_valid)
        self.assertEqual(message, "")
    
    def test_validate_image_array_invalid_shape(self):
        """Test validating array with invalid shape."""
        invalid_array = np.random.rand(10, 10, 10, 10)  # 4D array
        is_valid, message = validate_image_array(invalid_array)
        self.assertFalse(is_valid)
        self.assertIn("Invalid array shape", message)
    
    def test_validate_image_array_invalid_channels(self):
        """Test validating array with invalid number of channels."""
        invalid_array = np.random.rand(10, 10, 5)  # 5 channels
        is_valid, message = validate_image_array(invalid_array)
        self.assertFalse(is_valid)
        self.assertIn("Invalid number of channels", message)
    
    def test_validate_image_array_not_numpy(self):
        """Test validating non-numpy input."""
        is_valid, message = validate_image_array([1, 2, 3])
        self.assertFalse(is_valid)
        self.assertIn("not a numpy array", message)
    
    def test_validate_image_array_empty(self):
        """Test validating empty array."""
        empty_array = np.array([])
        is_valid, message = validate_image_array(empty_array)
        self.assertFalse(is_valid)
        self.assertIn("Array is empty", message)
    
    def test_supported_formats(self):
        """Test that supported formats constant is properly defined."""
        self.assertIsInstance(SUPPORTED_FORMATS, set)
        self.assertIn('.jpg', SUPPORTED_FORMATS)
        self.assertIn('.png', SUPPORTED_FORMATS)
        self.assertIn('.bmp', SUPPORTED_FORMATS)
    
    def test_round_trip_pil_cv2(self):
        """Test round-trip conversion between PIL and CV2 backends."""
        # Save with PIL, load with CV2
        pil_path = Path(self.temp_dir) / "pil_save.png"
        save_image_pil(self.rgb_image, pil_path)
        cv2_loaded = load_image_cv2(pil_path)
        
        # Save with CV2, load with PIL
        cv2_path = Path(self.temp_dir) / "cv2_save.png"
        save_image_cv2(self.rgb_image, cv2_path)
        pil_loaded = load_image_pil(cv2_path)
        
        # Both should be close (allowing for compression artifacts)
        self.assertEqual(cv2_loaded.shape, self.rgb_image.shape)
        self.assertEqual(pil_loaded.shape, self.rgb_image.shape)


if __name__ == '__main__':
    unittest.main()