"""
diagram-detector: Production-ready diagram detection for academic papers.

This package provides tools for detecting diagrams in academic papers,
with support for both images and PDFs.
"""

__version__ = "1.0.0"
__author__ = "Henrik Kragh Sørensen"
__email__ = "hks@ku.dk"
__license__ = "MIT"

from .detector import DiagramDetector
from .models import DetectionResult, DiagramDetection
from .utils import list_models, download_model
from .remote_ssh import SSHRemoteDetector, RemoteConfig, parse_remote_string
from .remote_pdf import PDFRemoteDetector
from .cache import DetectionCache

__all__ = [
    "DiagramDetector",
    "DetectionResult",
    "DiagramDetection",
    "list_models",
    "download_model",
    "SSHRemoteDetector",
    "RemoteConfig",
    "parse_remote_string",
    "PDFRemoteDetector",
    "DetectionCache",
    "__version__",
]
