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

__all__ = [
    "FilterRegistry",
    "get_registry", 
    "register_filter",
    "list_filters",
    "get_filter",
    "scan_and_register_filters",
    "auto_discover_filters"
]