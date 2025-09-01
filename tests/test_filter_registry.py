"""
Tests for the filter registry and discovery functionality.

Tests filter registration, category management, automatic discovery,
and metadata validation.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from image_processing_library.filters.registry import (
    FilterRegistry,
    get_registry,
    register_filter,
    list_filters,
    get_filter,
    scan_and_register_filters,
    auto_discover_filters
)
from image_processing_library.core.protocols import DataType, ColorFormat, FilterProtocol
from image_processing_library.core.base_filter import BaseFilter


class MockFilter(BaseFilter):
    """Mock filter for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="mock_filter",
            data_type=DataType.IMAGE,
            color_format=ColorFormat.RGB,
            category="test",
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Mock apply method."""
        return data.copy()


class AnotherMockFilter(BaseFilter):
    """Another mock filter for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="another_mock_filter",
            data_type=DataType.VIDEO,
            color_format=ColorFormat.RGBA,
            category="artistic",
            **kwargs
        )
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Mock apply method."""
        return data * 0.5


class InvalidFilter:
    """Invalid filter class for testing validation."""
    
    name = "invalid_filter"
    # Missing required attributes and methods


class TestFilterRegistry(unittest.TestCase):
    """Test cases for FilterRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = FilterRegistry()
    
    def test_register_filter_success(self):
        """Test successful filter registration."""
        self.registry.register_filter(MockFilter)
        
        # Check filter is registered
        self.assertIn("mock_filter", self.registry._filters)
        self.assertEqual(self.registry._filters["mock_filter"], MockFilter)
        
        # Check category is registered
        self.assertIn("test", self.registry._categories)
        self.assertIn("mock_filter", self.registry._categories["test"])
        
        # Check metadata is stored
        metadata = self.registry.get_filter_metadata("mock_filter")
        self.assertEqual(metadata["category"], "test")
        self.assertEqual(metadata["data_type"], DataType.IMAGE)
        self.assertEqual(metadata["color_format"], ColorFormat.RGB)
    
    def test_register_filter_duplicate_name(self):
        """Test registering filter with duplicate name raises error."""
        self.registry.register_filter(MockFilter)
        
        with self.assertRaises(ValueError) as context:
            self.registry.register_filter(MockFilter)
        
        self.assertIn("already registered", str(context.exception))
    
    def test_register_filter_invalid_protocol(self):
        """Test registering invalid filter raises TypeError."""
        with self.assertRaises(TypeError) as context:
            self.registry.register_filter(InvalidFilter)
        
        self.assertIn("does not implement FilterProtocol", str(context.exception))
    
    def test_unregister_filter(self):
        """Test filter unregistration."""
        self.registry.register_filter(MockFilter)
        self.registry.unregister_filter("mock_filter")
        
        # Check filter is removed
        self.assertNotIn("mock_filter", self.registry._filters)
        self.assertNotIn("mock_filter", self.registry._categories["test"])
        self.assertNotIn("mock_filter", self.registry._filter_metadata)
    
    def test_unregister_nonexistent_filter(self):
        """Test unregistering nonexistent filter raises KeyError."""
        with self.assertRaises(KeyError):
            self.registry.unregister_filter("nonexistent")
    
    def test_get_filter(self):
        """Test getting registered filter."""
        self.registry.register_filter(MockFilter)
        filter_class = self.registry.get_filter("mock_filter")
        self.assertEqual(filter_class, MockFilter)
    
    def test_get_nonexistent_filter(self):
        """Test getting nonexistent filter raises KeyError."""
        with self.assertRaises(KeyError):
            self.registry.get_filter("nonexistent")
    
    def test_list_filters_all(self):
        """Test listing all filters."""
        self.registry.register_filter(MockFilter)
        self.registry.register_filter(AnotherMockFilter)
        
        filters = self.registry.list_filters()
        self.assertEqual(set(filters), {"mock_filter", "another_mock_filter"})
    
    def test_list_filters_by_category(self):
        """Test listing filters by category."""
        self.registry.register_filter(MockFilter)
        self.registry.register_filter(AnotherMockFilter)
        
        test_filters = self.registry.list_filters(category="test")
        self.assertEqual(test_filters, ["mock_filter"])
        
        artistic_filters = self.registry.list_filters(category="artistic")
        self.assertEqual(artistic_filters, ["another_mock_filter"])
    
    def test_list_filters_by_data_type(self):
        """Test listing filters by data type."""
        self.registry.register_filter(MockFilter)
        self.registry.register_filter(AnotherMockFilter)
        
        image_filters = self.registry.list_filters(data_type=DataType.IMAGE)
        self.assertEqual(image_filters, ["mock_filter"])
        
        video_filters = self.registry.list_filters(data_type=DataType.VIDEO)
        self.assertEqual(video_filters, ["another_mock_filter"])
    
    def test_list_filters_by_color_format(self):
        """Test listing filters by color format."""
        self.registry.register_filter(MockFilter)
        self.registry.register_filter(AnotherMockFilter)
        
        rgb_filters = self.registry.list_filters(color_format=ColorFormat.RGB)
        self.assertEqual(rgb_filters, ["mock_filter"])
        
        rgba_filters = self.registry.list_filters(color_format=ColorFormat.RGBA)
        self.assertEqual(rgba_filters, ["another_mock_filter"])
    
    def test_list_categories(self):
        """Test listing categories."""
        self.registry.register_filter(MockFilter)
        self.registry.register_filter(AnotherMockFilter)
        
        categories = self.registry.list_categories()
        self.assertEqual(set(categories), {"test", "artistic"})
    
    def test_get_filters_by_category(self):
        """Test getting filters by category."""
        self.registry.register_filter(MockFilter)
        self.registry.register_filter(AnotherMockFilter)
        
        test_filters = self.registry.get_filters_by_category("test")
        self.assertEqual(test_filters, ["mock_filter"])
    
    def test_get_filters_by_nonexistent_category(self):
        """Test getting filters by nonexistent category raises KeyError."""
        with self.assertRaises(KeyError):
            self.registry.get_filters_by_category("nonexistent")
    
    def test_create_filter_instance(self):
        """Test creating filter instance."""
        self.registry.register_filter(MockFilter)
        
        instance = self.registry.create_filter_instance("mock_filter")
        self.assertIsInstance(instance, MockFilter)
        self.assertEqual(instance.name, "mock_filter")
    
    def test_create_filter_instance_with_params(self):
        """Test creating filter instance with parameters."""
        self.registry.register_filter(MockFilter)
        
        instance = self.registry.create_filter_instance("mock_filter", test_param="value")
        self.assertIsInstance(instance, MockFilter)
        self.assertEqual(instance.parameters.get("test_param"), "value")
    
    def test_validate_filter_protocol_valid(self):
        """Test protocol validation for valid filter."""
        result = self.registry._validate_filter_protocol(MockFilter)
        self.assertTrue(result)
    
    def test_validate_filter_protocol_invalid(self):
        """Test protocol validation for invalid filter."""
        result = self.registry._validate_filter_protocol(InvalidFilter)
        self.assertFalse(result)
    
    def test_validate_filter_metadata_valid(self):
        """Test metadata validation for valid filter."""
        result = self.registry.validate_filter_metadata(MockFilter)
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
        self.assertEqual(result['metadata']['name'], 'mock_filter')
        self.assertEqual(result['metadata']['category'], 'test')
    
    def test_validate_filter_metadata_invalid(self):
        """Test metadata validation for invalid filter."""
        result = self.registry.validate_filter_metadata(InvalidFilter)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_clear_registry(self):
        """Test clearing the registry."""
        self.registry.register_filter(MockFilter)
        self.registry.register_filter(AnotherMockFilter)
        
        self.registry.clear()
        
        self.assertEqual(len(self.registry._filters), 0)
        self.assertEqual(len(self.registry._categories), 0)
        self.assertEqual(len(self.registry._filter_metadata), 0)


class TestFilterRegistryDecorator(unittest.TestCase):
    """Test cases for the register_filter decorator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear global registry
        get_registry().clear()
    
    def tearDown(self):
        """Clean up after tests."""
        get_registry().clear()
    
    def test_register_filter_decorator(self):
        """Test the register_filter decorator."""
        
        @register_filter(category="test")
        class DecoratedFilter(BaseFilter):
            def __init__(self, **kwargs):
                super().__init__(
                    name="decorated_filter",
                    data_type=DataType.IMAGE,
                    color_format=ColorFormat.RGB,
                    category="test",
                    **kwargs
                )
            
            def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
                return data
        
        # Check filter was automatically registered
        registry = get_registry()
        self.assertIn("decorated_filter", registry._filters)
        self.assertEqual(registry._filters["decorated_filter"], DecoratedFilter)
    
    def test_register_filter_decorator_invalid_filter(self):
        """Test decorator with invalid filter logs warning but doesn't fail."""
        
        with patch('warnings.warn') as mock_warn:
            @register_filter()
            class InvalidDecoratedFilter:
                pass
            
            # Should have warned about registration failure
            mock_warn.assert_called_once()
            
            # Filter should not be registered
            registry = get_registry()
            self.assertNotIn("InvalidDecoratedFilter", registry._filters)


class TestFilterDiscovery(unittest.TestCase):
    """Test cases for filter discovery functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = FilterRegistry()
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_scan_module_for_filters(self):
        """Test scanning a module for filters."""
        # Create a mock module with filter classes
        mock_module = MagicMock()
        mock_module.__name__ = "test_module"
        
        # Mock inspect.getmembers to return our test classes
        with patch('inspect.getmembers') as mock_getmembers:
            mock_getmembers.return_value = [
                ("MockFilter", MockFilter),
                ("AnotherMockFilter", AnotherMockFilter),
                ("InvalidFilter", InvalidFilter),
                ("_PrivateFilter", MockFilter)  # Should be skipped
            ]
            
            # Mock the __module__ attribute for each class
            MockFilter.__module__ = "test_module"
            AnotherMockFilter.__module__ = "test_module"
            InvalidFilter.__module__ = "test_module"
            
            registered = self.registry._scan_module_for_filters(mock_module)
            
            # Should register valid filters only
            self.assertEqual(len(registered), 2)
            self.assertIn("mock_filter", registered)
            self.assertIn("another_mock_filter", registered)
    
    @patch('importlib.import_module')
    @patch('pkgutil.walk_packages')
    def test_scan_and_register_filters_recursive(self, mock_walk, mock_import):
        """Test recursive package scanning."""
        # Mock package structure
        mock_package = MagicMock()
        mock_package.__file__ = str(Path(self.temp_dir) / "__init__.py")
        mock_import.return_value = mock_package
        
        # Mock pkgutil to return submodules
        mock_walk.return_value = [
            (None, "test_package.submodule1", False),
            (None, "test_package.submodule2", False)
        ]
        
        # Mock submodules
        mock_submodule1 = MagicMock()
        mock_submodule1.__name__ = "test_package.submodule1"
        mock_submodule2 = MagicMock()
        mock_submodule2.__name__ = "test_package.submodule2"
        
        def import_side_effect(module_name):
            if module_name == "test_package":
                return mock_package
            elif module_name == "test_package.submodule1":
                return mock_submodule1
            elif module_name == "test_package.submodule2":
                return mock_submodule2
            raise ImportError(f"No module named {module_name}")
        
        mock_import.side_effect = import_side_effect
        
        # Mock _scan_module_for_filters to return filter names
        with patch.object(self.registry, '_scan_module_for_filters') as mock_scan:
            mock_scan.side_effect = [["filter1"], ["filter2"]]
            
            registered = self.registry.scan_and_register_filters("test_package", recursive=True)
            
            self.assertEqual(registered, ["filter1", "filter2"])
            self.assertEqual(mock_scan.call_count, 2)
    
    @patch('importlib.import_module')
    def test_scan_and_register_filters_import_error(self, mock_import):
        """Test handling of import errors during scanning."""
        mock_import.side_effect = ImportError("Module not found")
        
        with self.assertRaises(ImportError):
            self.registry.scan_and_register_filters("nonexistent_package")


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        get_registry().clear()
    
    def tearDown(self):
        """Clean up after tests."""
        get_registry().clear()
    
    def test_list_filters_convenience(self):
        """Test list_filters convenience function."""
        get_registry().register_filter(MockFilter)
        
        filters = list_filters()
        self.assertIn("mock_filter", filters)
    
    def test_get_filter_convenience(self):
        """Test get_filter convenience function."""
        get_registry().register_filter(MockFilter)
        
        filter_class = get_filter("mock_filter")
        self.assertEqual(filter_class, MockFilter)
    
    @patch('image_processing_library.filters.registry.scan_and_register_filters')
    def test_auto_discover_filters(self, mock_scan):
        """Test auto_discover_filters function."""
        mock_scan.side_effect = [["filter1"], ["filter2"], ImportError("Package not found")]
        
        registered = auto_discover_filters()
        
        # Should have called scan for each package
        self.assertEqual(mock_scan.call_count, 3)
        self.assertEqual(registered, ["filter1", "filter2"])


if __name__ == '__main__':
    unittest.main()