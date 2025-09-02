"""
Filters module containing all image and video processing filters.

Organized by categories: artistic, enhancement, technical.
"""

from .registry import (
    FilterRegistry,
    get_registry,
    register_filter,
    list_filters,
    get_filter,
    scan_and_register_filters,
    auto_discover_filters
)

# Import all filter modules to ensure registration decorators are executed
from . import artistic
from . import enhancement
from . import technical

__all__ = [
    "FilterRegistry",
    "get_registry", 
    "register_filter",
    "list_filters",
    "get_filter",
    "scan_and_register_filters",
    "auto_discover_filters"
]