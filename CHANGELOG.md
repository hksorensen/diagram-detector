# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-24

### Added
- Initial release
- YOLO11-based diagram detection
- PDF support with automatic page conversion
- Batch processing with auto-optimization
- Multiple output formats (JSON, CSV)
- Optional visualizations and cropped extractions
- CPU, CUDA, and MPS (Apple Silicon) support
- Command-line interface
- Python API
- Docker support
- Comprehensive documentation

### Models
- yolo11n (6 MB) - Fast, mobile-friendly
- yolo11s (22 MB) - Edge devices
- yolo11m (49 MB) - Production (recommended)
- yolo11l (63 MB) - High accuracy
- yolo11x (137 MB) - Research-grade

### Performance
- CPU: 50-80 images/second
- GPU: 500-800 images/second
- F1 Score: 0.74-0.82 (depending on model)
