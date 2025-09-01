"""
Unit tests for core protocols and enums.

Tests the FilterProtocol interface definition and supporting enumerations
to ensure proper type checking and protocol compliance.
"""

import unittest
import numpy as np
from typing import get_type_hints, get_origin, get_args
from unittest.mock import Mock

from image_processing_library.core.protocols import (
    FilterProtocol,
    DataType,
    ColorFormat
)


class TestDataType(unittest.TestCase):
    """Test cases for DataType enum."""
    
    def test_enum_values(self):
        """Test DataType enum has correct values."""
        self.assertEqual(DataType.IMAGE.value, "image")
        self.assertEqual(DataType.VIDEO.value, "video")
    
    def test_enum_members(self):
        """Test DataType enum has expected members."""
        expected_members = {"IMAGE", "VIDEO"}
        actual_members = set(DataType.__members__.keys())
        self.assertEqual(actual_members, expected_members)
    
    def test_enum_comparison(self):
        """Test DataType enum comparison operations."""
        self.assertEqual(DataType.IMAGE, DataType.IMAGE)
        self.assertNotEqual(DataType.IMAGE, DataType.VIDEO)
        
        # Test string comparison
        self.assertEqual(DataType.IMAGE.value, "image")
        self.assertEqual(DataType.VIDEO.value, "video")


class TestColorFormat(unittest.TestCase):
    """Test cases for ColorFormat enum."""
    
    def test_enum_values(self):
        """Test ColorFormat enum has correct values."""
        self.assertEqual(ColorFormat.RGB.value, "rgb")
        self.assertEqual(ColorFormat.RGBA.value, "rgba")
        self.assertEqual(ColorFormat.GRAYSCALE.value, "grayscale")
    
    def test_enum_members(self):
        """Test ColorFormat enum has expected members."""
        expected_members = {"RGB", "RGBA", "GRAYSCALE"}
        actual_members = set(ColorFormat.__members__.keys())
        self.assertEqual(actual_members, expected_members)
    
    def test_enum_comparison(self):
        """Test ColorFormat enum comparison operations."""
        self.assertEqual(ColorFormat.RGB, ColorFormat.RGB)
        self.assertNotEqual(ColorFormat.RGB, ColorFormat.RGBA)
        
        # Test string comparison
        self.assertEqual(ColorFormat.RGB.value, "rgb")
        self.assertEqual(ColorFormat.RGBA.value, "rgba")
        self.assertEqual(ColorFormat.GRAYSCALE.value, "grayscale")


class TestFilterProtocol(unittest.TestCase):
    """Test cases for FilterProtocol interface."""
    
    def test_protocol_attributes(self):
        """Test FilterProtocol has required attributes."""
        # Get type hints from the protocol
        hints = get_type_hints(FilterProtocol)
        
        # Check required attributes exist
        self.assertIn('name', hints)
        self.assertIn('data_type', hints)
        self.assertIn('color_format', hints)
        self.assertIn('category', hints)
        
        # Check attribute types
        self.assertEqual(hints['name'], str)
        self.assertEqual(hints['data_type'], DataType)
        self.assertEqual(hints['color_format'], ColorFormat)
        self.assertEqual(hints['category'], str)
    
    def test_protocol_methods(self):
        """Test FilterProtocol has required methods."""
        # Check that protocol defines required methods
        protocol_methods = {
            'apply', 'get_parameters', 'set_parameters', 'validate_input'
        }
        
        # Get all methods defined in the protocol
        defined_methods = set()
        for attr_name in dir(FilterProtocol):
            if not attr_name.startswith('_'):
                attr = getattr(FilterProtocol, attr_name)
                if callable(attr):
                    defined_methods.add(attr_name)
        
        # Check all required methods are present
        for method in protocol_methods:
            self.assertIn(method, defined_methods, f"Method {method} not found in FilterProtocol")
    
    def test_protocol_compliance_valid_class(self):
        """Test that a valid class implements FilterProtocol."""
        
        class ValidFilter:
            def __init__(self):
                self.name = "valid_filter"
                self.data_type = DataType.IMAGE
                self.color_format = ColorFormat.RGB
                self.category = "test"
            
            def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
                return data
            
            def get_parameters(self):
                return {}
            
            def set_parameters(self, **kwargs):
                pass
            
            def validate_input(self, data: np.ndarray) -> bool:
                return True
        
        # Create instance and check it has all required attributes/methods
        filter_instance = ValidFilter()
        
        # Check attributes
        self.assertTrue(hasattr(filter_instance, 'name'))
        self.assertTrue(hasattr(filter_instance, 'data_type'))
        self.assertTrue(hasattr(filter_instance, 'color_format'))
        self.assertTrue(hasattr(filter_instance, 'category'))
        
        # Check methods
        self.assertTrue(hasattr(filter_instance, 'apply'))
        self.assertTrue(hasattr(filter_instance, 'get_parameters'))
        self.assertTrue(hasattr(filter_instance, 'set_parameters'))
        self.assertTrue(hasattr(filter_instance, 'validate_input'))
        
        # Check method signatures work as expected
        test_data = np.ones((10, 10, 3))
        result = filter_instance.apply(test_data)
        self.assertIsInstance(result, np.ndarray)
        
        params = filter_instance.get_parameters()
        self.assertIsInstance(params, dict)
        
        # set_parameters should not raise
        filter_instance.set_parameters(test_param="value")
        
        # validate_input should return bool
        is_valid = filter_instance.validate_input(test_data)
        self.assertIsInstance(is_valid, bool)
    
    def test_protocol_compliance_missing_attributes(self):
        """Test detection of missing required attributes."""
        
        class IncompleteFilter:
            def __init__(self):
                self.name = "incomplete_filter"
                # Missing data_type, color_format, category
            
            def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
                return data
            
            def get_parameters(self):
                return {}
            
            def set_parameters(self, **kwargs):
                pass
            
            def validate_input(self, data: np.ndarray) -> bool:
                return True
        
        filter_instance = IncompleteFilter()
        
        # Check missing attributes
        self.assertTrue(hasattr(filter_instance, 'name'))
        self.assertFalse(hasattr(filter_instance, 'data_type'))
        self.assertFalse(hasattr(filter_instance, 'color_format'))
        self.assertFalse(hasattr(filter_instance, 'category'))
    
    def test_protocol_compliance_missing_methods(self):
        """Test detection of missing required methods."""
        
        class IncompleteFilter:
            def __init__(self):
                self.name = "incomplete_filter"
                self.data_type = DataType.IMAGE
                self.color_format = ColorFormat.RGB
                self.category = "test"
            
            def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
                return data
            
            # Missing get_parameters, set_parameters, validate_input
        
        filter_instance = IncompleteFilter()
        
        # Check missing methods
        self.assertTrue(hasattr(filter_instance, 'apply'))
        self.assertFalse(hasattr(filter_instance, 'get_parameters'))
        self.assertFalse(hasattr(filter_instance, 'set_parameters'))
        self.assertFalse(hasattr(filter_instance, 'validate_input'))
    
    def test_protocol_method_signatures(self):
        """Test that protocol method signatures are correctly defined."""
        
        # Create a mock that implements the protocol
        mock_filter = Mock()
        mock_filter.name = "mock_filter"
        mock_filter.data_type = DataType.IMAGE
        mock_filter.color_format = ColorFormat.RGB
        mock_filter.category = "test"
        
        # Test apply method signature
        test_data = np.ones((10, 10, 3))
        mock_filter.apply.return_value = test_data
        result = mock_filter.apply(test_data, param1="value")
        mock_filter.apply.assert_called_once_with(test_data, param1="value")
        self.assertIsInstance(result, np.ndarray)
        
        # Test get_parameters method signature
        mock_filter.get_parameters.return_value = {"param1": "value"}
        params = mock_filter.get_parameters()
        mock_filter.get_parameters.assert_called_once()
        self.assertIsInstance(params, dict)
        
        # Test set_parameters method signature
        mock_filter.set_parameters(param1="new_value", param2=42)
        mock_filter.set_parameters.assert_called_once_with(param1="new_value", param2=42)
        
        # Test validate_input method signature
        mock_filter.validate_input.return_value = True
        is_valid = mock_filter.validate_input(test_data)
        mock_filter.validate_input.assert_called_once_with(test_data)
        self.assertIsInstance(is_valid, bool)


class TestProtocolIntegration(unittest.TestCase):
    """Integration tests for protocol usage."""
    
    def test_protocol_with_isinstance_check(self):
        """Test protocol usage with isinstance checks (runtime)."""
        
        class ConformingFilter:
            def __init__(self):
                self.name = "conforming_filter"
                self.data_type = DataType.IMAGE
                self.color_format = ColorFormat.RGB
                self.category = "test"
            
            def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
                return data * 2
            
            def get_parameters(self):
                return {"multiplier": 2}
            
            def set_parameters(self, **kwargs):
                pass
            
            def validate_input(self, data: np.ndarray) -> bool:
                return isinstance(data, np.ndarray)
        
        filter_instance = ConformingFilter()
        
        # Test that the instance works as expected with protocol methods
        test_data = np.ones((5, 5, 3))
        
        # Test apply
        result = filter_instance.apply(test_data)
        np.testing.assert_array_equal(result, test_data * 2)
        
        # Test get_parameters
        params = filter_instance.get_parameters()
        self.assertEqual(params, {"multiplier": 2})
        
        # Test validate_input
        self.assertTrue(filter_instance.validate_input(test_data))
        self.assertFalse(filter_instance.validate_input("not an array"))
    
    def test_protocol_attribute_types(self):
        """Test that protocol attributes have correct types."""
        
        class TypedFilter:
            def __init__(self):
                self.name = "typed_filter"
                self.data_type = DataType.VIDEO
                self.color_format = ColorFormat.RGBA
                self.category = "artistic"
            
            def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
                return data
            
            def get_parameters(self):
                return {}
            
            def set_parameters(self, **kwargs):
                pass
            
            def validate_input(self, data: np.ndarray) -> bool:
                return True
        
        filter_instance = TypedFilter()
        
        # Check attribute types
        self.assertIsInstance(filter_instance.name, str)
        self.assertIsInstance(filter_instance.data_type, DataType)
        self.assertIsInstance(filter_instance.color_format, ColorFormat)
        self.assertIsInstance(filter_instance.category, str)
        
        # Check specific enum values
        self.assertEqual(filter_instance.data_type, DataType.VIDEO)
        self.assertEqual(filter_instance.color_format, ColorFormat.RGBA)


if __name__ == '__main__':
    unittest.main()