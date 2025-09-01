#!/usr/bin/env python3
"""
Command-line interface for the Image Processing Filter Library.
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Image Processing Filter Library CLI",
        prog="image-filter"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="%(prog)s 1.0.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List filters command
    list_parser = subparsers.add_parser("list", help="List available filters")
    list_parser.add_argument(
        "--category", 
        help="Filter by category (artistic, enhancement, technical)"
    )
    
    # Apply filter command
    apply_parser = subparsers.add_parser("apply", help="Apply a filter to an image")
    apply_parser.add_argument("filter_name", help="Name of the filter to apply")
    apply_parser.add_argument("input_file", help="Input image file")
    apply_parser.add_argument("output_file", help="Output image file")
    apply_parser.add_argument("--params", help="Filter parameters as JSON string")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_filters_cmd(args.category)
    elif args.command == "apply":
        apply_filter_cmd(args.filter_name, args.input_file, args.output_file, args.params)
    else:
        parser.print_help()

def list_filters_cmd(category=None):
    """List available filters."""
    try:
        # Try package import first (when installed)
        from image_processing_library.filters import list_filters, get_registry, auto_discover_filters
    except ImportError:
        try:
            # Try relative import (when running from source)
            from .filters import list_filters, get_registry, auto_discover_filters
        except ImportError:
            # Handle direct script execution
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from filters import list_filters, get_registry, auto_discover_filters
    
    # Ensure filters are discovered and registered
    try:
        auto_discover_filters()
    except Exception as e:
        print(f"Warning: Could not auto-discover filters: {e}")
    
    filters = list_filters(category=category)
    
    if not filters:
        print("No filters found.")
        return
    
    print(f"Available filters{f' in category {category}' if category else ''}:")
    
    # Get registry to access filter metadata
    registry = get_registry()
    
    for filter_name in filters:
        # Get the filter's category from metadata
        try:
            filter_metadata = registry._filter_metadata.get(filter_name, {})
            filter_category = filter_metadata.get('category', 'unknown')
            print(f"  - {filter_name} ({filter_category})")
        except:
            print(f"  - {filter_name}")

def apply_filter_cmd(filter_name, input_file, output_file, params=None):
    """Apply a filter to an image."""
    import json
    try:
        # Try package import first (when installed)
        from image_processing_library.filters import get_filter, auto_discover_filters
        from image_processing_library.media_io import load_image, save_image
    except ImportError:
        try:
            # Try relative import (when running from source)
            from .filters import get_filter, auto_discover_filters
            from .media_io import load_image, save_image
        except ImportError:
            # Handle direct script execution
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from filters import get_filter, auto_discover_filters
            from media_io import load_image, save_image
    
    # Ensure filters are discovered and registered
    try:
        auto_discover_filters()
    except Exception as e:
        print(f"Warning: Could not auto-discover filters: {e}")
    
    try:
        # Load the image
        image = load_image(input_file)
        
        # Get the filter
        filter_class = get_filter(filter_name)
        if not filter_class:
            print(f"Filter '{filter_name}' not found.")
            sys.exit(1)
        
        # Parse parameters
        filter_params = {}
        if params:
            filter_params = json.loads(params)
        
        # Create and apply filter
        filter_instance = filter_class(**filter_params)
        result = filter_instance.apply(image)
        
        # Save the result
        save_image(result, output_file)
        print(f"Filter applied successfully. Output saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()