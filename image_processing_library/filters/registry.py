"""
Filter registry for dynamic filter discovery and management.

This module provides the FilterRegistry class for organizing filters by category
and automatic filter registration using decorators.
"""

from typing import Dict, List, Type, Optional, Set, Any
from collections import defaultdict
import inspect
import importlib
import pkgutil
from pathlib import Path
from functools import wraps

from ..core.protocols import FilterProtocol, DataType, ColorFormat


class FilterRegistry:
    """
    Registry for dynamic filter discovery and category-based organization.
    
    This class manages filter registration, discovery, and organization by categories.
    It supports both manual registration and automatic registration using decorators.
    """
    
    def __init__(self):
        """Initialize the filter registry."""
        self._filters: Dict[str, Type[FilterProtocol]] = {}
        self._categories: Dict[str, Set[str]] = defaultdict(set)
        self._filter_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register_filter(self, filter_class: Type[FilterProtocol], 
                       category: Optional[str] = None) -> None:
        """
        Register a filter class with the registry.
        
        Args:
            filter_class: The filter class to register
            category: Optional category override (uses filter's category if not provided)
            
        Raises:
            ValueError: If filter class is invalid or already registered
            TypeError: If filter class doesn't implement FilterProtocol
        """
        # Validate filter class implements the protocol
        if not self._validate_filter_protocol(filter_class):
            raise TypeError(f"Filter class {filter_class.__name__} does not implement FilterProtocol")
        
        # Get filter name and category
        # Try to get name from class attribute first, then try instance
        filter_name = None
        filter_category = None
        
        if hasattr(filter_class, 'name'):
            filter_name = getattr(filter_class, 'name')
        if hasattr(filter_class, 'category'):
            filter_category = getattr(filter_class, 'category')
        
        # If not found as class attributes, try creating an instance
        if filter_name is None or filter_category is None:
            try:
                # Check if it's a BaseFilter subclass that needs parameters
                from ..core.base_filter import BaseFilter
                if BaseFilter in filter_class.__mro__:
                    # For BaseFilter subclasses, try with empty kwargs first
                    # If that fails, we'll use the class name
                    try:
                        temp_instance = filter_class()
                        if filter_name is None:
                            filter_name = getattr(temp_instance, 'name', filter_class.__name__)
                        if filter_category is None:
                            filter_category = getattr(temp_instance, 'category', 'uncategorized')
                    except:
                        # If empty kwargs don't work, use class name as fallback
                        if filter_name is None:
                            filter_name = filter_class.__name__
                        if filter_category is None:
                            filter_category = 'uncategorized'
                else:
                    temp_instance = filter_class()
                    if filter_name is None:
                        filter_name = getattr(temp_instance, 'name', filter_class.__name__)
                    if filter_category is None:
                        filter_category = getattr(temp_instance, 'category', 'uncategorized')
            except:
                # Fallback to class name
                if filter_name is None:
                    filter_name = filter_class.__name__
                if filter_category is None:
                    filter_category = 'uncategorized'
        
        # Apply category override if provided
        if category is not None:
            filter_category = category
        
        # Check if already registered
        if filter_name in self._filters:
            raise ValueError(f"Filter '{filter_name}' is already registered")
        
        # Register the filter
        self._filters[filter_name] = filter_class
        self._categories[filter_category].add(filter_name)
        
        # Get data_type and color_format from class or instance
        data_type = getattr(filter_class, 'data_type', None)
        color_format = getattr(filter_class, 'color_format', None)
        
        if data_type is None or color_format is None:
            try:
                from ..core.base_filter import BaseFilter
                if BaseFilter in filter_class.__mro__:
                    try:
                        temp_instance = filter_class()
                        if data_type is None:
                            data_type = getattr(temp_instance, 'data_type', None)
                        if color_format is None:
                            color_format = getattr(temp_instance, 'color_format', None)
                    except:
                        pass
                else:
                    temp_instance = filter_class()
                    if data_type is None:
                        data_type = getattr(temp_instance, 'data_type', None)
                    if color_format is None:
                        color_format = getattr(temp_instance, 'color_format', None)
            except:
                pass
        
        # Store metadata
        self._filter_metadata[filter_name] = {
            'category': filter_category,
            'data_type': data_type,
            'color_format': color_format,
            'class': filter_class,
            'module': filter_class.__module__,
            'description': (getattr(filter_class, '__doc__', '') or '').strip()
        }
    
    def unregister_filter(self, filter_name: str) -> None:
        """
        Unregister a filter from the registry.
        
        Args:
            filter_name: Name of the filter to unregister
            
        Raises:
            KeyError: If filter is not registered
        """
        if filter_name not in self._filters:
            raise KeyError(f"Filter '{filter_name}' is not registered")
        
        # Get category and remove from category set
        metadata = self._filter_metadata[filter_name]
        category = metadata['category']
        self._categories[category].discard(filter_name)
        
        # Remove empty categories
        if not self._categories[category]:
            del self._categories[category]
        
        # Remove from registry
        del self._filters[filter_name]
        del self._filter_metadata[filter_name]
    
    def get_filter(self, filter_name: str) -> Type[FilterProtocol]:
        """
        Get a filter class by name.
        
        Args:
            filter_name: Name of the filter to retrieve
            
        Returns:
            The filter class
            
        Raises:
            KeyError: If filter is not registered
        """
        if filter_name not in self._filters:
            raise KeyError(f"Filter '{filter_name}' is not registered")
        
        return self._filters[filter_name]
    
    def list_filters(self, category: Optional[str] = None, 
                    data_type: Optional[DataType] = None,
                    color_format: Optional[ColorFormat] = None) -> List[str]:
        """
        List registered filters with optional filtering.
        
        Args:
            category: Optional category to filter by
            data_type: Optional data type to filter by
            color_format: Optional color format to filter by
            
        Returns:
            List of filter names matching the criteria
        """
        filters = list(self._filters.keys())
        
        # Filter by category
        if category is not None:
            if category not in self._categories:
                return []
            filters = [f for f in filters if f in self._categories[category]]
        
        # Filter by data type
        if data_type is not None:
            filters = [f for f in filters 
                      if self._filter_metadata[f]['data_type'] == data_type]
        
        # Filter by color format
        if color_format is not None:
            filters = [f for f in filters 
                      if self._filter_metadata[f]['color_format'] == color_format]
        
        return sorted(filters)
    
    def list_categories(self) -> List[str]:
        """
        List all registered categories.
        
        Returns:
            List of category names
        """
        return sorted(self._categories.keys())
    
    def get_filters_by_category(self, category: str) -> List[str]:
        """
        Get all filters in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of filter names in the category
            
        Raises:
            KeyError: If category doesn't exist
        """
        if category not in self._categories:
            raise KeyError(f"Category '{category}' does not exist")
        
        return sorted(list(self._categories[category]))
    
    def get_filter_metadata(self, filter_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific filter.
        
        Args:
            filter_name: Name of the filter
            
        Returns:
            Dictionary containing filter metadata
            
        Raises:
            KeyError: If filter is not registered
        """
        if filter_name not in self._filter_metadata:
            raise KeyError(f"Filter '{filter_name}' is not registered")
        
        return self._filter_metadata[filter_name].copy()
    
    def create_filter_instance(self, filter_name: str, **kwargs) -> FilterProtocol:
        """
        Create an instance of a registered filter.
        
        Args:
            filter_name: Name of the filter to instantiate
            **kwargs: Parameters to pass to the filter constructor
            
        Returns:
            Filter instance
            
        Raises:
            KeyError: If filter is not registered
            TypeError: If filter constructor fails
        """
        filter_class = self.get_filter(filter_name)
        
        try:
            return filter_class(**kwargs)
        except Exception as e:
            raise TypeError(f"Failed to create instance of filter '{filter_name}': {e}")
    
    def _validate_filter_protocol(self, filter_class: Type) -> bool:
        """
        Validate that a class implements the FilterProtocol.
        
        Args:
            filter_class: Class to validate
            
        Returns:
            True if class implements the protocol
        """
        required_methods = ['apply', 'get_parameters', 'set_parameters', 'validate_input']
        
        # Check methods
        for method_name in required_methods:
            if not hasattr(filter_class, method_name):
                return False
            method = getattr(filter_class, method_name)
            if not callable(method):
                return False
        
        # For attributes, we need to check if they can be determined
        # either as class attributes or through instantiation
        required_attributes = ['name', 'data_type', 'color_format', 'category']
        
        # Try to create a temporary instance to check attributes
        try:
            # First check if attributes exist as class attributes
            for attr_name in required_attributes:
                if hasattr(filter_class, attr_name):
                    continue
                
                # If not a class attribute, check if it's set in __init__
                # by trying to create an instance with minimal parameters
                try:
                    # Try to create instance with empty kwargs first
                    temp_instance = filter_class()
                    if not hasattr(temp_instance, attr_name):
                        return False
                except:
                    # If that fails, the class likely requires parameters
                    # Check if BaseFilter is in the MRO (method resolution order)
                    from ..core.base_filter import BaseFilter
                    if BaseFilter in filter_class.__mro__:
                        # For BaseFilter subclasses, we know the pattern
                        return True
                    else:
                        # For other classes, we can't validate without knowing parameters
                        return False
            
            return True
            
        except Exception:
            # If we can't instantiate, check if it's a BaseFilter subclass
            try:
                from ..core.base_filter import BaseFilter
                return BaseFilter in filter_class.__mro__
            except:
                return False
    
    def scan_and_register_filters(self, package_path: str, 
                                 recursive: bool = True) -> List[str]:
        """
        Scan a package for filter classes and register them automatically.
        
        Args:
            package_path: Python package path to scan (e.g., 'image_processing_library.filters')
            recursive: Whether to scan subpackages recursively
            
        Returns:
            List of registered filter names
            
        Raises:
            ImportError: If package cannot be imported
        """
        registered_filters = []
        
        try:
            # Import the package
            package = importlib.import_module(package_path)
            package_dir = Path(package.__file__).parent
            
            # Get all modules in the package
            modules_to_scan = []
            
            if recursive:
                # Walk through all submodules recursively
                for importer, modname, ispkg in pkgutil.walk_packages(
                    [str(package_dir)], prefix=f"{package_path}."
                ):
                    modules_to_scan.append(modname)
            else:
                # Only scan direct submodules
                for importer, modname, ispkg in pkgutil.iter_modules([str(package_dir)]):
                    modules_to_scan.append(f"{package_path}.{modname}")
            
            # Scan each module for filter classes
            for module_name in modules_to_scan:
                try:
                    module = importlib.import_module(module_name)
                    filters_found = self._scan_module_for_filters(module)
                    registered_filters.extend(filters_found)
                except Exception as e:
                    # Log warning but continue scanning
                    import warnings
                    warnings.warn(f"Failed to scan module {module_name}: {e}")
                    
        except ImportError as e:
            raise ImportError(f"Cannot import package {package_path}: {e}")
        
        return registered_filters
    
    def _scan_module_for_filters(self, module) -> List[str]:
        """
        Scan a module for filter classes and register them.
        
        Args:
            module: The module to scan
            
        Returns:
            List of registered filter names from this module
        """
        registered_filters = []
        
        # Get all classes in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip if class is not defined in this module
            if obj.__module__ != module.__name__:
                continue
            
            # Skip if class name starts with underscore (private)
            if name.startswith('_'):
                continue
            
            # Check if class implements FilterProtocol
            if self._validate_filter_protocol(obj):
                try:
                    # Skip if already registered (check by class name first)
                    if name in self._filters:
                        continue
                    
                    # Register the filter and get the actual registered name
                    self.register_filter(obj)
                    
                    # Find the registered name (it might be different from class name)
                    for registered_name, registered_class in self._filters.items():
                        if registered_class == obj and registered_name not in registered_filters:
                            registered_filters.append(registered_name)
                            break
                    
                except Exception as e:
                    # Log warning but continue
                    import warnings
                    warnings.warn(f"Failed to register filter {name} from {module.__name__}: {e}")
        
        return registered_filters
    
    def validate_filter_metadata(self, filter_class: Type[FilterProtocol]) -> Dict[str, Any]:
        """
        Validate and extract metadata from a filter class.
        
        Args:
            filter_class: The filter class to validate
            
        Returns:
            Dictionary containing validation results and metadata
            
        Raises:
            ValueError: If filter metadata is invalid
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        # Check required attributes
        required_attrs = {
            'name': str,
            'data_type': DataType,
            'color_format': ColorFormat,
            'category': str
        }
        
        # Try to get attributes from class or instance
        for attr_name, expected_type in required_attrs.items():
            attr_value = None
            
            if hasattr(filter_class, attr_name):
                attr_value = getattr(filter_class, attr_name)
            else:
                # Try to create an instance to get the attribute
                try:
                    # Check if it's a BaseFilter subclass
                    from ..core.base_filter import BaseFilter
                    if BaseFilter in filter_class.__mro__:
                        # Try to create instance with empty kwargs
                        try:
                            temp_instance = filter_class()
                            if hasattr(temp_instance, attr_name):
                                attr_value = getattr(temp_instance, attr_name)
                        except:
                            # Skip validation for now as it requires constructor parameters
                            validation_result['warnings'].append(
                                f"Cannot validate attribute '{attr_name}' without instantiation"
                            )
                            continue
                    else:
                        temp_instance = filter_class()
                        if hasattr(temp_instance, attr_name):
                            attr_value = getattr(temp_instance, attr_name)
                except:
                    validation_result['errors'].append(f"Missing required attribute '{attr_name}'")
                    validation_result['valid'] = False
                    continue
            
            if attr_value is not None:
                # Type validation
                if expected_type in (DataType, ColorFormat):
                    if not isinstance(attr_value, expected_type):
                        validation_result['errors'].append(
                            f"Attribute '{attr_name}' must be of type {expected_type.__name__}"
                        )
                        validation_result['valid'] = False
                elif not isinstance(attr_value, expected_type):
                    validation_result['errors'].append(
                        f"Attribute '{attr_name}' must be of type {expected_type.__name__}"
                    )
                    validation_result['valid'] = False
                
                validation_result['metadata'][attr_name] = attr_value
        
        # Check for valid category names
        if 'category' in validation_result['metadata']:
            category = validation_result['metadata']['category']
            if not category or not isinstance(category, str) or not category.strip():
                validation_result['errors'].append("Category must be a non-empty string")
                validation_result['valid'] = False
            elif not category.replace('_', '').replace('-', '').isalnum():
                validation_result['warnings'].append(
                    "Category should contain only alphanumeric characters, hyphens, and underscores"
                )
        
        # Check method signatures
        method_checks = {
            'apply': ['data'],
            'get_parameters': [],
            'set_parameters': [],
            'validate_input': ['data']
        }
        
        for method_name, required_params in method_checks.items():
            if hasattr(filter_class, method_name):
                method = getattr(filter_class, method_name)
                if callable(method):
                    try:
                        sig = inspect.signature(method)
                        param_names = list(sig.parameters.keys())
                        
                        # Remove 'self' parameter for instance methods
                        if param_names and param_names[0] == 'self':
                            param_names = param_names[1:]
                        
                        # Check required parameters
                        for required_param in required_params:
                            if required_param not in param_names:
                                validation_result['warnings'].append(
                                    f"Method '{method_name}' should have parameter '{required_param}'"
                                )
                    except Exception:
                        validation_result['warnings'].append(
                            f"Could not inspect signature of method '{method_name}'"
                        )
                else:
                    validation_result['errors'].append(f"'{method_name}' is not callable")
                    validation_result['valid'] = False
            else:
                validation_result['errors'].append(f"Missing required method '{method_name}'")
                validation_result['valid'] = False
        
        return validation_result
    
    def clear(self) -> None:
        """Clear all registered filters and categories."""
        self._filters.clear()
        self._categories.clear()
        self._filter_metadata.clear()


# Global registry instance
_global_registry = FilterRegistry()


def get_registry() -> FilterRegistry:
    """
    Get the global filter registry instance.
    
    Returns:
        The global FilterRegistry instance
    """
    return _global_registry


def register_filter(category: Optional[str] = None):
    """
    Decorator for automatic filter registration.
    
    Args:
        category: Optional category override
        
    Returns:
        Decorator function
        
    Example:
        @register_filter(category="artistic")
        class MyFilter(BaseFilter):
            name = "my_filter"
            category = "artistic"
            # ... filter implementation
    """
    def decorator(filter_class: Type[FilterProtocol]) -> Type[FilterProtocol]:
        """
        Register the decorated filter class.
        
        Args:
            filter_class: The filter class to register
            
        Returns:
            The original filter class (unchanged)
        """
        try:
            _global_registry.register_filter(filter_class, category)
        except (ValueError, TypeError) as e:
            # Log warning but don't fail the import
            import warnings
            warnings.warn(f"Failed to register filter {filter_class.__name__}: {e}")
        
        return filter_class
    
    return decorator


def list_filters(category: Optional[str] = None, 
                data_type: Optional[DataType] = None,
                color_format: Optional[ColorFormat] = None) -> List[str]:
    """
    Convenience function to list filters from the global registry.
    
    Args:
        category: Optional category to filter by
        data_type: Optional data type to filter by
        color_format: Optional color format to filter by
        
    Returns:
        List of filter names matching the criteria
    """
    return _global_registry.list_filters(category, data_type, color_format)


def get_filter(filter_name: str) -> Type[FilterProtocol]:
    """
    Convenience function to get a filter from the global registry.
    
    Args:
        filter_name: Name of the filter to retrieve
        
    Returns:
        The filter class
        
    Raises:
        KeyError: If filter is not registered
    """
    return _global_registry.get_filter(filter_name)


def scan_and_register_filters(package_path: str, recursive: bool = True) -> List[str]:
    """
    Convenience function to scan and register filters from the global registry.
    
    Args:
        package_path: Python package path to scan
        recursive: Whether to scan subpackages recursively
        
    Returns:
        List of registered filter names
    """
    return _global_registry.scan_and_register_filters(package_path, recursive)


def auto_discover_filters() -> List[str]:
    """
    Automatically discover and register all filters in the library.
    
    Returns:
        List of registered filter names
    """
    registered_filters = []
    
    # Scan the main filter categories
    filter_packages = [
        'image_processing_library.filters.artistic',
        'image_processing_library.filters.enhancement', 
        'image_processing_library.filters.technical'
    ]
    
    for package in filter_packages:
        try:
            filters = scan_and_register_filters(package, recursive=True)
            registered_filters.extend(filters)
        except ImportError:
            # Package doesn't exist yet, skip
            pass
    
    return registered_filters