"""
Pytest configuration and shared fixtures for diagram-detector tests.
"""

import pytest
import tempfile
import gzip
import json
from pathlib import Path
from diagram_detector import DetectionCache


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache(temp_cache_dir):
    """Create fresh DetectionCache instance."""
    return DetectionCache(cache_dir=temp_cache_dir)


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Sample PDF path for testing."""
    pdf_path = tmp_path / "test_sample.pdf"
    # Create a minimal fake PDF file
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
    return pdf_path


@pytest.fixture
def valid_detection_results():
    """Valid detection results (1 page, 2 detections)."""
    return [
        {
            "page_number": 0,
            "boxes": [[100, 100, 200, 200], [300, 300, 400, 400]],
            "scores": [0.95, 0.88],
            "class_ids": [0, 0],
            "labels": ["diagram", "diagram"],
        }
    ]


@pytest.fixture
def multi_page_results():
    """Multi-page detection results (10 pages)."""
    return [
        {
            "page_number": i,
            "boxes": [[i*10, i*10, i*20, i*20]],
            "scores": [0.9],
            "class_ids": [0],
            "labels": ["diagram"],
        }
        for i in range(10)
    ]


@pytest.fixture
def empty_detection_results():
    """Valid results but no detections found (1 page, 0 boxes)."""
    return [
        {
            "page_number": 0,
            "boxes": [],
            "scores": [],
            "class_ids": [],
            "labels": [],
        }
    ]


@pytest.fixture
def cache_params():
    """Standard cache parameters."""
    return {
        "model": "diagram-detector-v5",
        "confidence": 0.1,
        "iou": 0.3,
        "dpi": 200,
        "imgsz": 640,
    }
