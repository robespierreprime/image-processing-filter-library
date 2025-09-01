# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-08-29

### Added
- Initial release of the Image Processing Filter Library
- Core filter protocol and base filter implementation
- Filter execution queue and chaining system
- Preset management for saving/loading filter configurations
- Filter registry with category-based organization
- Comprehensive I/O support for images and videos
- Performance optimization features (memory management, chunked processing)
- Built-in filters:
  - Artistic: Glitch filter
  - Technical: Print simulation, Background remover
- Command-line interface for basic operations
- Complete test suite with unit and integration tests
- Documentation and examples

### Features
- Protocol-based filter interface using `typing.Protocol`
- Progress tracking and error handling for all filters
- Memory-efficient processing with automatic optimization
- Support for RGB, RGBA, and grayscale image formats
- Video frame-by-frame processing capabilities
- JSON-based preset serialization
- Automatic filter discovery and registration
- Extensible architecture for custom filter development