"""Utility functions for diagram detection."""

from pathlib import Path
from typing import List, Union, Optional
import torch
import numpy as np
from tqdm.auto import tqdm


# Model information
MODEL_INFO = {
    "v5": {
        "size_mb": 5.2,
        "params": "2.6M",
        "default_batch_cpu": 8,
        "default_batch_gpu": 64,
        "recommended_confidence": 0.20,
        "recommended_iou": 0.30,
        "description": "v5 clean dataset model - optimized for diagram counting",
    },
    "yolo11n": {
        "size_mb": 6,
        "params": "2.6M",
        "default_batch_cpu": 8,
        "default_batch_gpu": 64,
    },
    "yolo11s": {
        "size_mb": 22,
        "params": "9.4M",
        "default_batch_cpu": 4,
        "default_batch_gpu": 48,
    },
    "yolo11m": {
        "size_mb": 49,
        "params": "20.1M",
        "default_batch_cpu": 2,
        "default_batch_gpu": 32,
    },
    "yolo11l": {
        "size_mb": 63,
        "params": "25.3M",
        "default_batch_cpu": 1,
        "default_batch_gpu": 16,
    },
    "yolo11x": {
        "size_mb": 137,
        "params": "56.9M",
        "default_batch_cpu": 1,
        "default_batch_gpu": 8,
    },
}

# Hugging Face Hub repository
HF_REPO = "hksorensen/diagram-detector-models"

# GitHub release URLs (fallback)
GITHUB_RELEASE_BASE = "https://github.com/hksorensen/diagram-detector/releases/download/v1.0.0"

# Model URLs - multiple sources for robustness
MODEL_SOURCES = {
    "yolo11n": {
        "huggingface": f"{HF_REPO}/yolo11n.pt",
        "github": f"{GITHUB_RELEASE_BASE}/yolo11n.pt",
    },
    "yolo11s": {
        "huggingface": f"{HF_REPO}/yolo11s.pt",
        "github": f"{GITHUB_RELEASE_BASE}/yolo11s.pt",
    },
    "yolo11m": {
        "huggingface": f"{HF_REPO}/yolo11m.pt",
        "github": f"{GITHUB_RELEASE_BASE}/yolo11m.pt",
    },
    "yolo11l": {
        "huggingface": f"{HF_REPO}/yolo11l.pt",
        "github": f"{GITHUB_RELEASE_BASE}/yolo11l.pt",
    },
    "yolo11x": {
        "huggingface": f"{HF_REPO}/yolo11x.pt",
        "github": f"{GITHUB_RELEASE_BASE}/yolo11x.pt",
    },
}


def list_models() -> List[str]:
    """List available model names."""
    return list(MODEL_INFO.keys())


def get_cache_dir() -> Path:
    """Get cache directory for models."""
    cache_dir = Path.home() / ".cache" / "diagram-detector" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_path(model_name: str) -> Path:
    """Get cached model path."""
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}. Available: {list_models()}")

    cache_dir = get_cache_dir()
    return cache_dir / f"{model_name}.pt"


def get_metadata_path(model_name: str) -> Path:
    """Get cached metadata path."""
    cache_dir = get_cache_dir()
    return cache_dir / f"{model_name}_metadata.json"


def download_from_huggingface(
    model_name: str, model_path: Path, show_progress: bool = True
) -> bool:
    """
    Download model from Hugging Face Hub.

    Args:
        model_name: Model name
        model_path: Where to save model
        show_progress: Show download progress

    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import hf_hub_download

        # Extract repo and filename from MODEL_SOURCES
        repo_id = HF_REPO
        filename = f"{model_name}.pt"

        print(f"Downloading from Hugging Face Hub: {repo_id}/{filename}")

        # Download to temporary location first
        temp_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=get_cache_dir() / "hf_cache",
            resume_download=True,
        )

        # Copy to final location
        import shutil

        shutil.copy2(temp_path, model_path)

        # Also download metadata if available
        try:
            metadata_filename = f"{model_name}_metadata.json"
            temp_metadata = hf_hub_download(
                repo_id=repo_id,
                filename=metadata_filename,
                cache_dir=get_cache_dir() / "hf_cache",
            )
            metadata_path = get_metadata_path(model_name)
            shutil.copy2(temp_metadata, metadata_path)
            print("✓ Metadata downloaded")
        except Exception:
            # Metadata is optional
            pass

        return True

    except ImportError:
        print("⚠ huggingface_hub not installed, trying GitHub...")
        return False
    except Exception as e:
        print(f"⚠ Hugging Face download failed: {e}")
        return False


def download_from_github(model_name: str, model_path: Path, show_progress: bool = True) -> bool:
    """
    Download model from GitHub releases.

    Args:
        model_name: Model name
        model_path: Where to save model
        show_progress: Show download progress

    Returns:
        True if successful, False otherwise
    """
    try:
        import urllib.request

        url = MODEL_SOURCES[model_name]["github"]
        print(f"Downloading from GitHub: {url}")

        def progress_hook(block_num, block_size, total_size):
            if not show_progress or total_size <= 0:
                return
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = "=" * filled + "-" * (bar_length - filled)
            print(f"\r[{bar}] {percent:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, model_path, progress_hook)
        if show_progress:
            print()  # New line after progress

        # Try to download metadata
        try:
            metadata_url = url.replace(".pt", "_metadata.json")
            metadata_path = get_metadata_path(model_name)
            urllib.request.urlretrieve(metadata_url, metadata_path)
            print("✓ Metadata downloaded")
        except Exception:
            # Metadata is optional
            pass

        return True

    except Exception as e:
        print(f"⚠ GitHub download failed: {e}")
        return False


def download_model(model_name: str, force: bool = False, source: str = "auto") -> Path:
    """
    Download model weights if not present.

    Uses multiple sources with fallback:
    1. Hugging Face Hub (primary, requires huggingface_hub)
    2. GitHub releases (fallback)

    Args:
        model_name: Model name (yolo11n, yolo11m, etc.)
        force: Force re-download even if cached
        source: Download source ('auto', 'huggingface', 'github')

    Returns:
        Path to downloaded model

    Raises:
        ValueError: If model name unknown
        RuntimeError: If all download sources fail
    """
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}. Available: {list_models()}")

    model_path = get_model_path(model_name)

    # Check if already cached
    if model_path.exists() and not force:
        if model_path.stat().st_size > 0:  # Verify file is not empty
            return model_path
        else:
            # Corrupted file, remove it
            model_path.unlink()

    print(f"Downloading {model_name} model ({MODEL_INFO[model_name]['size_mb']} MB)...")

    # Try sources in order
    sources_to_try = []
    if source == "auto":
        sources_to_try = ["huggingface", "github"]
    elif source in ["huggingface", "github"]:
        sources_to_try = [source]
    else:
        raise ValueError(f"Unknown source: {source}. Use 'auto', 'huggingface', or 'github'")

    success = False
    last_error = None

    for src in sources_to_try:
        try:
            if src == "huggingface":
                success = download_from_huggingface(model_name, model_path)
            elif src == "github":
                success = download_from_github(model_name, model_path)

            if success:
                break
        except Exception as e:
            last_error = e
            continue

    if not success or not model_path.exists():
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()

        error_msg = f"Failed to download {model_name} from all sources"
        if last_error:
            error_msg += f": {last_error}"
        raise RuntimeError(error_msg)

    # Verify downloaded file
    if model_path.stat().st_size == 0:
        model_path.unlink()
        raise RuntimeError("Downloaded file is empty")

    print(f"✓ Model downloaded to {model_path}")
    return model_path


def load_model_metadata(model_name: str) -> Optional[dict]:
    """
    Load model metadata if available.

    Args:
        model_name: Model name

    Returns:
        Metadata dict or None if not available
    """
    metadata_path = get_metadata_path(model_name)

    if not metadata_path.exists():
        return None

    try:
        import json

        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def detect_device() -> str:
    """
    Auto-detect best available device.

    Returns:
        'cuda' or 'cpu'

    Note:
        MPS (Apple Silicon GPU) is intentionally disabled due to a critical bug in
        Ultralytics YOLO's MPS backend that corrupts bounding box coordinates.
        Using CPU ensures accurate results on Apple Silicon at the cost of speed.
    """
    if torch.cuda.is_available():
        return "cuda"

    # MPS is disabled due to YOLO coordinate corruption bug
    # See: Ultralytics YOLO MPS backend returns corrupted x-coordinates
    # Symptoms: bbox returns [y2, y1, garbage, y2] instead of [x1, y1, x2, y2]
    # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return "mps"

    return "cpu"


def get_device_info(device: str) -> dict:
    """Get information about device."""
    info = {"device": device}

    if device == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif device == "mps":
        info["name"] = "Apple Silicon (MPS)"
    else:
        info["name"] = "CPU"

    return info


def optimize_batch_size(
    model_name: str, device: str, available_memory_gb: Optional[float] = None
) -> int:
    """
    Calculate optimal batch size for device.

    Args:
        model_name: Model name
        device: Device type ('cpu', 'cuda', 'mps')
        available_memory_gb: Available memory in GB (auto-detected if None)

    Returns:
        Optimal batch size
    """
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}")

    model_info = MODEL_INFO[model_name]

    # Get default batch sizes
    if device == "cpu":
        base_batch = model_info["default_batch_cpu"]
    else:  # cuda or mps
        base_batch = model_info["default_batch_gpu"]

        # Adjust based on available memory
        if available_memory_gb is None and device == "cuda":
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        if available_memory_gb:
            # Scale batch size based on memory
            # Assume 2GB baseline for batch=16 on yolo11m
            memory_factor = available_memory_gb / 2.0
            base_batch = int(base_batch * memory_factor)

            # Clamp to reasonable range
            base_batch = max(1, min(128, base_batch))

    return base_batch


def convert_pdf_to_images(
    pdf_path: Union[str, Path],
    dpi: int = 200,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
    verbose: bool = True,
    grayscale: bool = True,
) -> List[np.ndarray]:
    """
    Convert PDF pages to images.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (higher = better quality, slower)
        first_page: First page to convert (1-indexed)
        last_page: Last page to convert (1-indexed)
        verbose: Show progress messages
        grayscale: Convert to grayscale (recommended for diagram detection)

    Returns:
        List of images as numpy arrays (RGB format, grayscale if enabled)
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image is required for PDF support. " "Install with: pip install pdf2image"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if verbose:
        mode_str = "grayscale" if grayscale else "RGB"
        print(f"Converting PDF to images (DPI={dpi}, {mode_str})...")

    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=first_page,
        last_page=last_page,
        fmt="jpeg",  # Faster than PNG
    )

    # Convert PIL Images to numpy arrays
    np_images = []
    for img in tqdm(images, desc="Converting pages", unit="page"):
        if grayscale:
            # Convert to grayscale, then back to RGB (YOLO expects 3 channels)
            img = img.convert("L").convert("RGB")
        np_images.append(np.array(img))

    return np_images


def load_image(image_path: Union[str, Path], grayscale: bool = True) -> np.ndarray:
    """
    Load image from file.

    Args:
        image_path: Path to image file
        grayscale: Convert to grayscale (recommended for diagram detection)

    Returns:
        Image as numpy array (RGB format, grayscale if enabled)
    """
    from PIL import Image

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    if grayscale:
        # Convert to grayscale, then back to RGB (YOLO expects 3 channels)
        img = img.convert("L").convert("RGB")
    else:
        img = img.convert("RGB")
    return np.array(img)


def save_json(data: dict, output_path: Union[str, Path]) -> None:
    """Save data as JSON."""
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(data: List[dict], output_path: Union[str, Path]) -> None:
    """Save data as CSV."""
    import csv

    if not data:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def validate_model_file(model_path: Path) -> bool:
    """
    Validate that model file exists and is loadable.

    Args:
        model_path: Path to model file

    Returns:
        True if valid
    """
    if not model_path.exists():
        return False

    try:
        # Try to load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        return isinstance(checkpoint, dict) or hasattr(checkpoint, "forward")
    except Exception:
        return False


def get_image_files(directory: Path, recursive: bool = False) -> List[Path]:
    """
    Get all image files in directory.

    Args:
        directory: Directory to search
        recursive: Search recursively

    Returns:
        List of image file paths
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if recursive:
        files = []
        for ext in image_extensions:
            files.extend(directory.rglob(f"*{ext}"))
            files.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        files = []
        for ext in image_extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(files)
