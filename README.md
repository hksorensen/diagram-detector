# diagram-detector

Production-ready diagram detection for academic papers using YOLO11.

[![PyPI version](https://badge.fury.io/py/diagram-detector.svg)](https://badge.fury.io/py/diagram-detector)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🚀 **Fast**: 50-500 images/second depending on hardware
- 🌐 **Remote GPU**: Process on remote server via SSH
- 📄 **PDF Support**: Automatically processes PDF pages
- 🎯 **Accurate**: 98.49% Binary F1 score (optimized for page-level detection)
- 🔧 **Flexible**: CPU, CUDA, and MPS (Apple Silicon) support
- 📊 **Multiple Formats**: JSON, CSV, cropped images, visualizations
- 🔄 **Batch Optimized**: Auto-detects optimal batch size
- 💾 **Smart Caching**: Downloads models once, uses forever

## Installation

```bash
pip install diagram-detector
```

### From source:
```bash
git clone https://github.com/hksorensen/diagram-detector.git
cd diagram-detector
pip install -e .
```

## Quick Start

### Python API

```python
from diagram_detector import DiagramDetector

# Initialize detector
detector = DiagramDetector()

# Detect diagrams in images
results = detector.detect('path/to/images/')

# Or from PDFs
results = detector.detect_pdf('paper.pdf')

# Access results
for result in results:
    print(f"{result.filename}: {result.has_diagram} ({result.confidence:.2f})")
    
# Save results
detector.save_results(results, 'output/', format='json')  # or 'csv'

# Extract diagram crops
detector.save_crops(results, 'diagrams/')
```

### Command Line

```bash
# Detect in images
diagram-detect --input images/ --output results/

# Process PDF
diagram-detect --input paper.pdf --output results/ --save-crops

# With visualization
diagram-detect --input paper.pdf --visualize --confidence 0.35

# Batch processing
diagram-detect --input papers/*.pdf --output results/ --batch-size 16

# Remote GPU inference (process 300K images on remote server!)
diagram-detect --input images/ --remote user@gpu-server:22 --output results/
```

## Model

Uses YOLO11-medium (49 MB) optimized for production use. Model is automatically downloaded on first use from [HuggingFace Hub](https://huggingface.co/hksorensen/diagram-detector-model).

## Advanced Usage

### Custom Configuration

```python
detector = DiagramDetector(
    confidence=0.20,  # optimized default
    iou=0.30,  # optimized default
    device='cuda',  # or 'cpu', 'mps', 'auto'
    batch_size=32,  # or 'auto'
)

# Detect with options
results = detector.detect(
    'images/',
    save_crops=True,
    save_visualizations=True,
    crop_padding=10,  # pixels
)
```

### Batch Processing

```python
from pathlib import Path

# Process multiple PDFs
pdf_files = Path('papers/').glob('*.pdf')

for pdf in pdf_files:
    results = detector.detect_pdf(pdf)
    detector.save_results(results, f'results/{pdf.stem}/')
```

### Integration with Pipelines

```python
# Use in document processing pipeline
def process_paper(pdf_path):
    detector = DiagramDetector()
    
    # Detect diagrams
    results = detector.detect_pdf(pdf_path)
    
    # Filter high-confidence detections
    diagrams = [r for r in results if r.confidence > 0.5]
    
    # Extract for further analysis
    for diagram in diagrams:
        crop = diagram.get_crop()
        # ... process crop
```

## Performance

### Speed Benchmarks

| Hardware | Model | Batch Size | Speed |
|----------|-------|------------|-------|
| CPU (Intel i7) | yolo11n | 8 | ~50 img/s |
| CPU (Apple M1) | yolo11n | 8 | ~80 img/s |
| GPU (RTX 3090) | yolo11m | 32 | ~500 img/s |
| GPU (RTX 4090) | yolo11m | 64 | ~800 img/s |

### Memory Usage

| Model | CPU | GPU (batch=16) | GPU (batch=32) |
|-------|-----|----------------|----------------|
| yolo11n | 200 MB | 1 GB | 1.5 GB |
| yolo11m | 400 MB | 2 GB | 3 GB |
| yolo11l | 500 MB | 3 GB | 4.5 GB |

## Output Formats

### JSON

```json
{
  "filename": "figure1.jpg",
  "has_diagram": true,
  "count": 2,
  "confidence": 0.89,
  "detections": [
    {
      "bbox": [100, 150, 400, 500],
      "confidence": 0.89,
      "class": "diagram"
    }
  ]
}
```

### CSV

```csv
filename,has_diagram,count,max_confidence
figure1.jpg,true,2,0.89
figure2.jpg,false,0,0.00
```

## Development

```bash
# Clone repository
git clone https://github.com/hksorensen/diagram-detector.git
cd diagram-detector

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black .
flake8 .
mypy .
```

## Citation

If you use this detector in your research, please cite:

```bibtex
@software{diagram_detector,
  title = {diagram-detector: Production-ready diagram detection for academic papers},
  author = {Sørensen, Henrik Kragh},
  year = {2025},
  url = {https://github.com/hksorensen/diagram-detector}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built on [YOLO11](https://github.com/ultralytics/ultralytics) by Ultralytics
- Trained on curated dataset of academic diagrams
- Part of the dh4pmp Digital Humanities project

## Support

- 🐛 [Issue Tracker](https://github.com/hksorensen/diagram-detector/issues)
- 💬 [Discussions](https://github.com/hksorensen/diagram-detector/discussions)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
