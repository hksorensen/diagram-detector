# Diagram Detector Python Package - Implementation Guide

**Date:** December 24, 2024
**Version:** 1.0.0
**Status:** Package structure created, core modules need implementation

---

## 📦 PACKAGE STRUCTURE CREATED

```
diagram-detector-package/
├── pyproject.toml           ✅ Complete
├── README.md                ✅ Complete
├── CHANGELOG.md             ✅ Complete
├── LICENSE                  ✅ Complete
├── .gitignore               ✅ Complete
├── MANIFEST.in              ✅ Complete
├── requirements.txt         ✅ Complete
├── Dockerfile               ✅ Complete
├── docker-compose.yml       ✅ Complete
├── .github/
│   └── workflows/
│       └── ci.yml           ✅ Complete
├── diagram_detector/
│   ├── __init__.py          ✅ Created
│   ├── detector.py          ⏳ Need to implement
│   ├── models.py            ⏳ Need to implement
│   ├── utils.py             ⏳ Need to implement
│   ├── cli.py               ⏳ Need to implement
│   └── py.typed             ⏳ Create
├── tests/
│   ├── __init__.py          ⏳ Create
│   ├── test_detector.py     ⏳ Create
│   └── test_utils.py        ⏳ Create
└── docs/
    ├── index.md             ⏳ Create
    └── api.md               ⏳ Create
```

---

## 🎯 CORE MODULES TO IMPLEMENT

### 1. `diagram_detector/detector.py`

**Main class:** `DiagramDetector`

**Key methods:**
```python
class DiagramDetector:
    def __init__(
        self,
        model: str = 'yolo11m',
        confidence: float = 0.35,
        device: str = 'auto',
        batch_size: Union[int, str] = 'auto'
    ):
        """Initialize detector with model and configuration."""
        pass
    
    def detect(
        self,
        input_path: Union[str, Path],
        save_crops: bool = False,
        save_visualizations: bool = False,
        crop_padding: int = 10
    ) -> List[DetectionResult]:
        """Detect diagrams in images."""
        pass
    
    def detect_pdf(
        self,
        pdf_path: Union[str, Path],
        dpi: int = 200,
        **kwargs
    ) -> List[DetectionResult]:
        """Detect diagrams in PDF (converts to images first)."""
        pass
    
    def save_results(
        self,
        results: List[DetectionResult],
        output_dir: Union[str, Path],
        format: str = 'json'
    ):
        """Save results to JSON or CSV."""
        pass
    
    def save_crops(
        self,
        results: List[DetectionResult],
        output_dir: Union[str, Path],
        padding: int = 10
    ):
        """Extract and save cropped diagram regions."""
        pass
```

**Implementation notes:**
- Use existing `inference.py` from diagram-detection-pipeline as base
- Add PDF conversion using pdf2image
- Implement batch optimization
- Add progress bars with tqdm
- Support multiple devices (cpu, cuda, mps)

---

### 2. `diagram_detector/models.py`

**Data classes:**

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

@dataclass
class DiagramDetection:
    """Single diagram detection."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_name: str = "diagram"

@dataclass
class DetectionResult:
    """Detection results for one image/page."""
    filename: str
    page_number: Optional[int] = None  # For PDFs
    has_diagram: bool = False
    count: int = 0
    confidence: float = 0.0  # Max confidence
    detections: List[DiagramDetection] = None
    image: Optional[np.ndarray] = None  # Original image
    
    def get_crop(self, index: int = 0, padding: int = 10) -> np.ndarray:
        """Get cropped diagram region."""
        pass
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        pass
```

---

### 3. `diagram_detector/utils.py`

**Utility functions:**

```python
def list_models() -> List[str]:
    """List available models."""
    return ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']

def download_model(model_name: str, force: bool = False) -> Path:
    """Download model weights if not present."""
    # Check cache directory
    # Download from release or Hugging Face
    # Verify checksum
    pass

def detect_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def optimize_batch_size(
    model_size: str,
    device: str,
    image_size: int = 640
) -> int:
    """Calculate optimal batch size for device."""
    # Based on model size and available memory
    pass

def convert_pdf_to_images(
    pdf_path: Path,
    dpi: int = 200
) -> List[np.ndarray]:
    """Convert PDF pages to images."""
    from pdf2image import convert_from_path
    images = convert_from_path(pdf_path, dpi=dpi)
    return [np.array(img) for img in images]
```

---

### 4. `diagram_detector/cli.py`

**Command-line interface:**

```python
import argparse
from pathlib import Path
from .detector import DiagramDetector

def main():
    parser = argparse.ArgumentParser(
        description='Detect diagrams in images and PDFs'
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input file or directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='results',
        help='Output directory'
    )
    
    parser.add_argument(
        '--model', '-m',
        default='yolo11m',
        choices=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
        help='Model to use'
    )
    
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.35,
        help='Confidence threshold'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        default='auto',
        help='Batch size (or "auto")'
    )
    
    parser.add_argument(
        '--device', '-d',
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use'
    )
    
    parser.add_argument(
        '--save-crops',
        action='store_true',
        help='Save cropped diagram regions'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save visualizations with bounding boxes'
    )
    
    parser.add_argument(
        '--format', '-f',
        default='json',
        choices=['json', 'csv', 'both'],
        help='Output format'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DiagramDetector(
        model=args.model,
        confidence=args.confidence,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Run detection
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix == '.pdf':
        results = detector.detect_pdf(
            input_path,
            save_crops=args.save_crops,
            save_visualizations=args.visualize
        )
    else:
        results = detector.detect(
            input_path,
            save_crops=args.save_crops,
            save_visualizations=args.visualize
        )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format in ['json', 'both']:
        detector.save_results(results, output_dir, format='json')
    
    if args.format in ['csv', 'both']:
        detector.save_results(results, output_dir, format='csv')
    
    if args.save_crops:
        detector.save_crops(results, output_dir / 'crops')
    
    # Print summary
    total = len(results)
    with_diagrams = sum(1 for r in results if r.has_diagram)
    total_diagrams = sum(r.count for r in results)
    
    print(f"\n{'='*60}")
    print(f"DETECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total images/pages: {total}")
    print(f"  With diagrams: {with_diagrams} ({with_diagrams/total*100:.1f}%)")
    print(f"  Total diagrams detected: {total_diagrams}")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
```

---

## 🔧 IMPLEMENTATION PRIORITIES

### Phase 1: Core Detection (Essential)
1. ✅ Package structure
2. ⏳ `detector.py` - Main detection class
3. ⏳ `models.py` - Data models
4. ⏳ `utils.py` - Utility functions
5. ⏳ `cli.py` - Command-line interface

### Phase 2: Testing & Documentation
1. ⏳ Unit tests for all modules
2. ⏳ Integration tests
3. ⏳ API documentation
4. ⏳ Usage examples

### Phase 3: Distribution
1. ⏳ Test PyPI upload (TestPyPI first)
2. ⏳ Docker image build & test
3. ⏳ CI/CD pipeline verification
4. ⏳ Release v1.0.0

---

## 📝 IMPLEMENTATION NOTES

### Model Distribution

**Option 1: Bundle with package**
- Pro: Immediate use, no download
- Con: Large package size (50+ MB)

**Option 2: Download on first use** ← RECOMMENDED
- Pro: Small package, user choice of model
- Con: Requires internet on first use

**Implementation:**
```python
# In utils.py
MODEL_URLS = {
    'yolo11n': 'https://github.com/user/repo/releases/download/v1.0.0/yolo11n.pt',
    'yolo11m': 'https://github.com/user/repo/releases/download/v1.0.0/yolo11m.pt',
    # ...
}

CACHE_DIR = Path.home() / '.cache' / 'diagram-detector' / 'models'

def download_model(model_name: str) -> Path:
    cache_path = CACHE_DIR / f"{model_name}.pt"
    if cache_path.exists():
        return cache_path
    
    # Download from GitHub release
    # ...
```

### PDF Processing

Use `pdf2image` with automatic poppler detection:
```python
from pdf2image import convert_from_path

# Convert PDF to images
images = convert_from_path(
    pdf_path,
    dpi=200,  # Good balance of quality/speed
    fmt='jpeg'  # Faster than PNG
)
```

### Batch Optimization

```python
# Calculate optimal batch size based on:
# - Model size (nano vs x-large)
# - Device memory
# - Image size

BATCH_SIZES = {
    'cpu': {
        'yolo11n': 8,
        'yolo11m': 4,
        'yolo11l': 2,
    },
    'cuda': {
        'yolo11n': 64,
        'yolo11m': 32,
        'yolo11l': 16,
    }
}
```

---

## 🚀 USAGE EXAMPLES

### Simple Detection

```python
from diagram_detector import DiagramDetector

detector = DiagramDetector()
results = detector.detect('images/')

for result in results:
    if result.has_diagram:
        print(f"{result.filename}: {result.count} diagrams")
```

### PDF Processing

```python
detector = DiagramDetector(model='yolo11m')
results = detector.detect_pdf('paper.pdf')

# Save crops
detector.save_crops(results, 'diagrams/')

# Save results
detector.save_results(results, 'output/', format='json')
```

### Batch Processing

```python
from pathlib import Path

pdfs = Path('papers/').glob('*.pdf')

for pdf in pdfs:
    results = detector.detect_pdf(pdf)
    detector.save_results(results, f'results/{pdf.stem}/')
```

---

## 📦 DISTRIBUTION CHECKLIST

### Before Publishing to PyPI

- [ ] All core modules implemented
- [ ] Tests passing (>80% coverage)
- [ ] Documentation complete
- [ ] Examples verified
- [ ] CHANGELOG updated
- [ ] Version number correct
- [ ] License file included
- [ ] Requirements accurate
- [ ] Test on clean environment
- [ ] Test on multiple OS (Linux, macOS, Windows)
- [ ] Build package: `python -m build`
- [ ] Upload to TestPyPI first
- [ ] Test install from TestPyPI
- [ ] Upload to PyPI
- [ ] Create GitHub release
- [ ] Update documentation site

---

## 🔗 RESOURCES

### Reference Implementation
- Original inference code: `diagram-detection-pipeline/inference.py`
- Training pipeline: `diagram-detection-pipeline/pipeline.py`
- Models: `~/Downloads/diagram-detector-models/`

### Documentation to Write
1. API Reference
2. Quick Start Guide
3. Advanced Usage
4. Model Comparison
5. Performance Optimization
6. Troubleshooting Guide
7. Contributing Guidelines

### GitHub Repository Structure
```
diagram-detector/
├── .github/workflows/ci.yml
├── diagram_detector/
├── tests/
├── docs/
├── examples/
├── README.md
├── LICENSE
├── CHANGELOG.md
├── pyproject.toml
└── requirements.txt
```

---

## ✅ NEXT STEPS

1. **Implement core modules** (detector.py, models.py, utils.py, cli.py)
2. **Write tests** (unit + integration)
3. **Test locally** with pip install -e .
4. **Build package** with python -m build
5. **Upload to TestPyPI** for testing
6. **Create GitHub repository** and push
7. **Set up CI/CD** (GitHub Actions)
8. **Release v1.0.0** to PyPI
9. **Announce** and gather feedback

---

**Status:** Package structure complete, core implementation needed
**Timeline:** 1-2 days for full implementation + testing
**Priority:** Core detection functionality first, then polish

---

**End of implementation guide - December 24, 2024**
