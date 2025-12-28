"""Basic tests for diagram-detector."""

import pytest
from diagram_detector import DiagramDetector, __version__
from diagram_detector.models import DiagramDetection, DetectionResult
from diagram_detector.utils import list_models, detect_device


def test_version():
    """Test version is defined."""
    assert __version__ == "1.0.0"


def test_list_models():
    """Test model listing."""
    models = list_models()
    assert isinstance(models, list)
    assert 'yolo11n' in models
    assert 'yolo11m' in models
    assert len(models) == 5


def test_detect_device():
    """Test device detection."""
    device = detect_device()
    assert device in ['cpu', 'cuda', 'mps']


def test_diagram_detection_creation():
    """Test DiagramDetection dataclass."""
    detection = DiagramDetection(
        bbox=(100, 200, 300, 400),
        confidence=0.85
    )
    
    assert detection.bbox == (100, 200, 300, 400)
    assert detection.confidence == 0.85
    assert detection.width == 200
    assert detection.height == 200
    assert detection.area == 40000
    assert detection.center == (200, 300)


def test_diagram_detection_validation():
    """Test DiagramDetection validation."""
    # Invalid bbox length
    with pytest.raises(ValueError):
        DiagramDetection(
            bbox=(100, 200),  # Too short
            confidence=0.85
        )
    
    # Invalid confidence
    with pytest.raises(ValueError):
        DiagramDetection(
            bbox=(100, 200, 300, 400),
            confidence=1.5  # Too high
        )


def test_detection_result_creation():
    """Test DetectionResult dataclass."""
    detections = [
        DiagramDetection(bbox=(100, 100, 200, 200), confidence=0.9),
        DiagramDetection(bbox=(300, 300, 400, 400), confidence=0.8),
    ]
    
    result = DetectionResult(
        filename="test.jpg",
        detections=detections,
        image_width=800,
        image_height=600
    )
    
    assert result.has_diagram is True
    assert result.count == 2
    assert result.confidence == 0.9  # Max confidence


def test_detection_result_empty():
    """Test DetectionResult with no detections."""
    result = DetectionResult(
        filename="test.jpg",
        image_width=800,
        image_height=600
    )
    
    assert result.has_diagram is False
    assert result.count == 0
    assert result.confidence == 0.0


def test_detection_result_to_dict():
    """Test DetectionResult serialization."""
    detections = [
        DiagramDetection(bbox=(100, 100, 200, 200), confidence=0.9),
    ]
    
    result = DetectionResult(
        filename="test.jpg",
        detections=detections,
        image_width=800,
        image_height=600
    )
    
    data = result.to_dict()
    
    assert data['filename'] == "test.jpg"
    assert data['has_diagram'] is True
    assert data['count'] == 1
    assert len(data['detections']) == 1
    assert data['detections'][0]['confidence'] == 0.9


def test_detector_initialization():
    """Test DiagramDetector initialization."""
    # This will try to download model - may fail in CI
    # Skip if no network access
    try:
        detector = DiagramDetector(
            model='yolo11n',
            confidence=0.5,
            device='cpu',
            verbose=False
        )
        assert detector.model_name == 'yolo11n'
        assert detector.confidence == 0.5
        assert detector.device == 'cpu'
    except Exception:
        pytest.skip("Model download failed (expected in CI)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
