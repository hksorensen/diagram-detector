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
from .utils import list_models, download_model, compute_model_hash, verify_model_hash
from .remote_ssh import SSHRemoteDetector, RemoteConfig, parse_remote_string, is_remote_available, get_remote_endpoint
from .remote_pdf import PDFRemoteDetector
from .cache import DetectionCache
from .deployment import deploy_to_remote, check_remote_deployment

__all__ = [
    "DiagramDetector",
    "DetectionResult",
    "DiagramDetection",
    "list_models",
    "download_model",
    "compute_model_hash",
    "verify_model_hash",
    "SSHRemoteDetector",
    "RemoteConfig",
    "parse_remote_string",
    "is_remote_available",
    "get_remote_endpoint",
    "PDFRemoteDetector",
    "DetectionCache",
    "deploy_to_remote",
    "check_remote_deployment",
    "__version__",
]
