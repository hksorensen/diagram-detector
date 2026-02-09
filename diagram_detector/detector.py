"""Main diagram detection class."""

import gc
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from tqdm.auto import tqdm

from .cache import DetectionCache
from .models import DetectionResult, DiagramDetection
from .utils import (
    convert_pdf_to_images,
    detect_device,
    download_model,
    get_device_info,
    get_image_files,
    get_model_path,
    load_image,
    optimize_batch_size,
    save_csv,
    save_json,
)

# Filter out torch warnings that pollute logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logger = logging.getLogger(__name__)


class DiagramDetector:
    """
    Production-ready diagram detector for academic papers.

    Supports both images and PDFs with automatic batch optimization.
    """

    def __init__(
        self,
        model: str = "yolo11m",
        confidence: float = 0.20,
        iou: float = 0.30,
        device: str = "auto",
        batch_size: Union[int, str] = "auto",
        verbose: bool = True,
        cache: Union[bool, "DetectionCache", None] = True,
        imgsz: int = 640,
        tensorrt: bool = False,
    ):
        """
        Initialize detector.

        Args:
            model: Model name ('yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x', 'v5', etc.)
                   OR path to a local .pt file (will be automatically installed to cache)
            confidence: Confidence threshold (0.0-1.0, default: 0.20 - optimized via grid search)
            iou: IOU threshold for NMS (0.0-1.0, default: 0.30 - optimized via grid search)
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            batch_size: Batch size for inference (int or 'auto')
            verbose: Print progress information
            cache: Enable caching (True=default cache, False=disabled, DetectionCache=custom)
            imgsz: Image size for preprocessing (default: 640, must match training)
            tensorrt: Use TensorRT optimization with FP16 (default: False)
                     Requires NVIDIA GPU. First run exports optimized engine (slow),
                     subsequent runs use cached engine (2-3x faster inference).

        Examples:
            # Use a named model
            detector = DiagramDetector(model='yolo11m')

            # Use a local .pt file (automatically installs to cache)
            detector = DiagramDetector(model='/path/to/my_model.pt')
            detector = DiagramDetector(model='~/models/best.pt')
        """
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.verbose = verbose

        # Process model name early (needed for batch size optimization)
        from pathlib import Path as PathLib

        model_as_path = PathLib(model).expanduser()
        if model_as_path.exists() and model_as_path.suffix == ".pt":
            # Local .pt file - use filename as model name
            self.model_name = model_as_path.stem
        else:
            # Named model
            self.model_name = model

        # Initialize cache
        if cache is True:
            from .cache import DetectionCache

            self.cache = DetectionCache(compression=True, auto_cleanup=True)
        elif cache is False or cache is None:
            self.cache = None
        else:
            # Assume it's a DetectionCache instance
            self.cache = cache

        # Detect device
        if device == "auto":
            self.device = detect_device()
            if self.verbose:
                print(f"Auto-detected device: {self.device}")
        else:
            self.device = device

        # Get device info
        if self.verbose:
            device_info = get_device_info(self.device)
            print(f"Using: {device_info['name']}")
            if "memory_gb" in device_info:
                print(f"Memory: {device_info['memory_gb']:.1f} GB")

        # Optimize batch size (use model_name, not original model parameter)
        if batch_size == "auto":
            # For custom/unknown models, use a conservative default
            from .utils import MODEL_INFO

            if self.model_name in MODEL_INFO:
                self.batch_size = optimize_batch_size(self.model_name, self.device)
                if self.verbose:
                    print(f"Auto batch size: {self.batch_size}")
            else:
                # Unknown model - use conservative default
                self.batch_size = 4 if self.device == "cpu" else 8
                if self.verbose:
                    print(f"Auto batch size (default for custom model): {self.batch_size}")
        else:
            self.batch_size = batch_size

        self._tensorrt = tensorrt  # tracked for padding logic in _detect_batch

        # Handle model loading - support both named models and local .pt files
        if model_as_path.exists() and model_as_path.suffix == ".pt":
            # Local .pt file provided - install it to cache
            if self.verbose:
                print(f"Installing local model from: {model_as_path}")

            # Copy to cache
            from .utils import get_cache_dir as get_cache

            cache_dir = get_cache()
            model_path = cache_dir / f"{self.model_name}.pt"

            # Only copy if not already there or if source is newer
            if not model_path.exists() or model_as_path.stat().st_mtime > model_path.stat().st_mtime:
                import shutil

                shutil.copy2(model_as_path, model_path)
                if self.verbose:
                    print(f"✓ Model installed to cache: {model_path}")
            elif self.verbose:
                print(f"✓ Model already in cache: {model_path}")
        else:
            # Named model - use existing download logic
            model_path = get_model_path(self.model_name)
            if not model_path.exists():
                if self.verbose:
                    print("Model not found in cache, downloading...")
                download_model(self.model_name)

        # Load model
        if self.verbose:
            print(f"Loading {self.model_name} model...")

        from ultralytics import YOLO

        # Handle TensorRT optimization
        if tensorrt:
            if self.device not in ("cuda", "0", "1", "2", "3"):
                # Auto-detect might have chosen cuda
                if self.device != "auto" and not self.device.startswith("cuda"):
                    raise ValueError(
                        f"TensorRT requires NVIDIA GPU, but device is '{self.device}'. "
                        f"Use device='cuda' or device='auto' with an NVIDIA GPU."
                    )

            # Get GPU name for engine filename (engines are GPU-specific)
            try:
                import torch

                gpu_name = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
            except Exception:
                gpu_name = "unknown_gpu"

            # Engine path includes model name, imgsz, batch, and GPU for uniqueness
            # (engines are batch-size-specific and GPU-specific)
            engine_path = model_path.parent / f"{self.model_name}_imgsz{imgsz}_batch{self.batch_size}_{gpu_name}.engine"

            if engine_path.exists():
                if self.verbose:
                    print(f"✓ Loading cached TensorRT engine: {engine_path.name}")
                self.model = YOLO(str(engine_path))
            else:
                if self.verbose:
                    print("Exporting TensorRT engine (one-time, may take a few minutes)...")
                    print(f"  - Model: {model_path.name}")
                    print(f"  - GPU: {gpu_name}")
                    print("  - FP16: enabled")

                # Load PyTorch model first
                pt_model = YOLO(str(model_path))

                # Export to TensorRT with FP16
                engine_result = pt_model.export(
                    format="engine",
                    half=True,  # FP16 for speed
                    imgsz=imgsz,
                    batch=self.batch_size,
                    device=0,  # Use first GPU
                    verbose=self.verbose,
                )

                # Ultralytics returns the path to the exported engine
                exported_path = Path(engine_result)

                # Move to our cache location with descriptive name
                if exported_path.exists() and exported_path != engine_path:
                    import shutil

                    shutil.move(str(exported_path), str(engine_path))

                if self.verbose:
                    print(f"✓ TensorRT engine saved: {engine_path.name}")

                self.model = YOLO(str(engine_path))
        else:
            self.model = YOLO(str(model_path))

        # Reclaim GPU memory that may be held from a previous process (each image sub-batch
        # runs in a new process; the driver can retain memory across process exits).
        self._cuda_empty_cache_if_available()

        if self.verbose:
            print("✓ Model loaded")

    def _cuda_empty_cache_if_available(self) -> None:
        """Call torch.cuda.empty_cache() and gc.collect() if CUDA is available. No-op otherwise."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except Exception:
            pass

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "DiagramDetector":
        """
        Create DiagramDetector from YAML configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            DiagramDetector instance configured from file

        Example config.yaml:
            detector:
              model: v5
              confidence: 0.20
              iou: 0.30
              device: auto
              batch_size: auto
              verbose: true
        """
        import yaml

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if "detector" not in config:
            raise ValueError(
                "Config file missing 'detector' section. "
                "Expected format:\n"
                "detector:\n"
                "  model: v5\n"
                "  confidence: 0.20\n"
                "  ..."
            )

        detector_config = config["detector"]

        # Create instance with config values
        return cls(
            model=detector_config.get("model", "yolo11m"),
            confidence=detector_config.get("confidence", 0.20),
            iou=detector_config.get("iou", 0.30),
            device=detector_config.get("device", "auto"),
            batch_size=detector_config.get("batch_size", "auto"),
            verbose=detector_config.get("verbose", True),
            tensorrt=detector_config.get("tensorrt", False),
        )

    @classmethod
    def run_from_config(cls, config_path: Union[str, Path]) -> List[DetectionResult]:
        """
        Run complete detection workflow from YAML configuration.

        This method handles the full pipeline: load config, detect, save results/crops/viz.

        Args:
            config_path: Path to YAML config file

        Returns:
            List of DetectionResult objects

        Example config.yaml:
            detector:
              model: v5
              confidence: 0.20
              iou: 0.30
              device: auto

            paths:
              input: /path/to/pdfs
              output: /path/to/results
              detections: detections.json
              crops: crops/
              visualizations: visualizations/

            options:
              dpi: 300
              save_crops: true
              save_visualizations: false
              crop_padding: 10
              format: json
        """
        import yaml

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Create detector
        detector = cls.from_config(config_path)

        # Get paths config
        if "paths" not in config:
            raise ValueError("Config file missing 'paths' section")

        paths = config["paths"]
        input_path = Path(paths["input"])
        output_dir = Path(paths.get("output", "results"))

        # Get options
        options = config.get("options", {})
        dpi = options.get("dpi", 200)
        save_crops = options.get("save_crops", False)
        save_viz = options.get("save_visualizations", False)
        crop_padding = options.get("crop_padding", 10)
        output_format = options.get("format", "json")

        # Run detection
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            results = detector.detect_pdf(
                input_path,
                dpi=dpi,
                store_images=save_crops or save_viz,
            )
        else:
            results = detector.detect(
                input_path,
                store_images=save_crops or save_viz,
            )

        # Save results
        if output_format in ["json", "both"]:
            detector.save_results(results, output_dir, format="json")

        if output_format in ["csv", "both"]:
            detector.save_results(results, output_dir, format="csv")

        # Save crops if requested
        if save_crops:
            crops_path = paths.get("crops", "crops")
            crops_dir = output_dir / crops_path if not Path(crops_path).is_absolute() else Path(crops_path)
            detector.save_crops(results, crops_dir, padding=crop_padding)

        # Save visualizations if requested
        if save_viz:
            viz_path = paths.get("visualizations", "visualizations")
            viz_dir = output_dir / viz_path if not Path(viz_path).is_absolute() else Path(viz_path)
            detector.save_visualizations(results, viz_dir)

        return results

    def detect(
        self,
        input_path: Union[str, Path, List[str], List[Path]],
        save_crops: bool = False,
        save_visualizations: bool = False,
        crop_padding: int = 10,
        store_images: bool = False,
    ) -> List[DetectionResult]:
        """
        Detect diagrams in images.

        Args:
            input_path: Path to image, directory, or list of paths
            save_crops: Whether to extract cropped diagram regions
            save_visualizations: Whether to save images with bboxes drawn
            crop_padding: Pixels to add around crops
            store_images: Whether to store images in results (uses more memory)

        Returns:
            List of DetectionResult objects
        """
        # Parse input
        input_path = Path(input_path) if isinstance(input_path, (str, Path)) else input_path

        if isinstance(input_path, list):
            image_paths = [Path(p) for p in input_path]
        elif input_path.is_dir():
            # Check for manifest file in directory (explicit ordering)
            manifest_path = input_path / "manifest.txt"
            if manifest_path.exists():
                print(
                    "[DEBUG] detect(): Found manifest file - using EXPLICIT ordering (priority_list preserved!)",
                    flush=True,
                )
                image_paths = []
                with open(manifest_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # Manifest contains filenames only, resolve relative to directory
                            img_path = input_path / line
                            if img_path.exists():
                                image_paths.append(img_path)
                            else:
                                print(f"[WARNING] Manifest references missing file: {line}", flush=True)
                print(f"[DEBUG] detect(): Loaded {len(image_paths)} images from manifest", flush=True)
                if len(image_paths) >= 3:
                    print(f"[DEBUG]   Manifest order first 3: {[p.name for p in image_paths[:3]]}", flush=True)
                    print(f"[DEBUG]   Manifest order last 3: {[p.name for p in image_paths[-3:]]}", flush=True)
            else:
                # No manifest - fall back to directory scan (recursive for subdirs)
                print("[DEBUG] detect(): No manifest found, scanning directory recursively", flush=True)
                image_paths = get_image_files(input_path, recursive=True)

            if not image_paths:
                raise ValueError(f"No images found in {input_path}")
        else:
            image_paths = [input_path]

        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"[DIAGNOSTIC] detect(): Processing {len(image_paths)} image paths")
        batch_sz = int(self.batch_size) if isinstance(self.batch_size, int) else min(32, len(image_paths)) or 1
        # Debug progress to stdout so it appears in inference.log when run remotely
        num_batches = (len(image_paths) + batch_sz - 1) // batch_sz
        print(
            f"[DEBUG] detect(): starting, {len(image_paths)} images, batch_size={self.batch_size}, num_batches={num_batches}",
            flush=True,
        )
        if self.verbose:
            print(f"\nProcessing {len(image_paths)} image(s)...")

        # Run batch inference
        results = []
        batch_index = 0

        # Process in batches
        for i in tqdm(
            range(0, len(image_paths), batch_sz),
            desc="Detecting",
            disable=not self.verbose,
            unit="batch",
        ):
            batch_paths = image_paths[i : i + batch_sz]
            batch_index += 1
            print(
                f"[DEBUG] detect(): batch {batch_index}/{num_batches}, running _detect_batch on {len(batch_paths)} images (indices {i}-{i + len(batch_paths) - 1})",
                flush=True,
            )
            batch_results = self._detect_batch(
                batch_paths, store_images=store_images or save_crops or save_visualizations
            )
            print(
                f"[DEBUG] detect(): batch {batch_index}/{num_batches} returned {len(batch_results)} results (expected {len(batch_paths)})",
                flush=True,
            )
            logger.debug(
                f"[DIAGNOSTIC] detect(): Batch {batch_index} returned {len(batch_results)} results (expected {len(batch_paths)})"
            )
            results.extend(batch_results)

        print(f"[DEBUG] detect(): complete, total {len(results)} results", flush=True)
        logger.debug(f"[DIAGNOSTIC] detect(): Returning {len(results)} total results")
        return results

    def detect_pdf(
        self,
        pdf_path: Union[str, Path],
        dpi: int = 200,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> List[DetectionResult]:
        """
        Detect diagrams in PDF.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for PDF conversion
            first_page: First page to process (1-indexed)
            last_page: Last page to process (1-indexed)
            use_cache: Whether to use cache (default: True)
            **kwargs: Additional arguments passed to detect()

        Returns:
            List of DetectionResult objects (one per page)
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Check cache first (only if full PDF, not page ranges)
        if use_cache and self.cache and first_page is None and last_page is None:
            cached_results = self.cache.get(
                pdf_path,
                model=self.model_name,
                confidence=self.confidence,
                iou=self.iou,
                dpi=dpi,
                imgsz=self.imgsz,
            )

            if cached_results is not None:
                if self.verbose:
                    print(f"✓ Loaded {len(cached_results)} pages from cache")

                # Convert dicts back to DetectionResult objects
                from .models import DetectionResult, DiagramDetection

                results = []
                for result_dict in cached_results:
                    detections = [
                        DiagramDetection(
                            bbox=tuple(d["bbox"]),
                            confidence=d["confidence"],
                            class_name=d.get("class", "diagram"),
                            class_id=d.get("class_id", 0),
                        )
                        for d in result_dict.get("detections", [])
                    ]

                    result = DetectionResult(
                        filename=result_dict["filename"],
                        page_number=result_dict.get("page_number"),
                        detections=detections,
                        image=None,  # Images not stored in cache
                        image_width=result_dict["image_width"],
                        image_height=result_dict["image_height"],
                    )
                    results.append(result)

                return results

        if self.verbose:
            print(f"Processing PDF: {pdf_path.name}")

        # Convert PDF to images
        images = convert_pdf_to_images(pdf_path, dpi, first_page, last_page, verbose=self.verbose)

        if self.verbose:
            print(f"✓ Converted {len(images)} pages")

        # Run detection on all pages
        results = []

        # logger.info(f"Detecting {len(images)} pages")
        for page_num, image in enumerate(
            tqdm(
                images, desc="Detecting", unit="page", disable=False, total=len(images), leave=False
            ),  # WAS: disable=not self.verbose
            start=first_page or 1,
        ):
            # Create temporary result with image
            temp_result = self._detect_image(
                image,
                filename=f"{pdf_path.stem}_page{page_num}.jpg",
                store_image=(
                    kwargs.get("store_images", False)
                    or kwargs.get("save_crops", False)
                    or kwargs.get("save_visualizations", False)
                ),
            )
            temp_result.page_number = page_num
            results.append(temp_result)

        # Cache results (only if full PDF, not page ranges)
        if use_cache and self.cache and first_page is None and last_page is None:
            # Convert results to dicts for caching
            results_dicts = [r.to_dict(include_image=False) for r in results]

            self.cache.set(
                pdf_path,
                model=self.model_name,
                confidence=self.confidence,
                iou=self.iou,
                dpi=dpi,
                imgsz=self.imgsz,
                results=results_dicts,
            )

        return results

    def _detect_batch(self, image_paths: List[Path], store_images: bool = False) -> List[DetectionResult]:
        """Run inference on batch of images."""
        import sys

        # Debug to stdout so it appears in inference.log when run remotely
        print(f"[DEBUG] _detect_batch(): entry, {len(image_paths)} images", flush=True)

        # Load images if needed
        if store_images:
            images = [load_image(p) for p in image_paths]
        else:
            images = None

        # Run YOLO inference with error handling
        import io
        import subprocess
        import warnings

        # Get current ulimit for diagnostics
        try:
            ulimit_result = subprocess.run(["sh", "-c", "ulimit -n"], capture_output=True, text=True, timeout=1)
            current_ulimit = ulimit_result.stdout.strip() if ulimit_result.returncode == 0 else "unknown"
        except Exception:
            current_ulimit = "unknown"

        # Capture warnings and stderr to diagnose silent failures
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            # Capture stderr (YOLO might write errors there)
            old_stderr = sys.stderr
            stderr_capture = io.StringIO()
            sys.stderr = stderr_capture

            try:
                # TensorRT engines have a static batch size — the last sub-batch
                # may be smaller.  Pad by repeating the first image; results for
                # the padding slots are discarded after predict().
                predict_paths = image_paths
                pad_count = 0
                if self._tensorrt and len(image_paths) < self.batch_size:
                    pad_count = self.batch_size - len(image_paths)
                    predict_paths = list(image_paths) + [image_paths[0]] * pad_count
                    print(
                        f"[DEBUG] _detect_batch(): TensorRT pad {len(image_paths)} → {len(predict_paths)} (padding {pad_count})",
                        flush=True,
                    )

                print(f"[DEBUG] _detect_batch(): about to model.predict() on {len(predict_paths)} images", flush=True)
                yolo_results = self.model.predict(
                    source=[str(p) for p in predict_paths],
                    conf=self.confidence,
                    iou=self.iou,
                    device=self.device,
                    verbose=False,
                    stream=False,
                    imgsz=self.imgsz,  # Matches training config (config.yaml:image_size)
                )

                # Trim padding slots
                if pad_count > 0:
                    yolo_results = yolo_results[: len(image_paths)]

                print(f"[DEBUG] _detect_batch(): predict() returned {len(yolo_results)} results", flush=True)
                try:
                    import torch

                    if torch.cuda.is_available():
                        used_mb = torch.cuda.memory_allocated() / (1024**2)
                        reserved_mb = torch.cuda.memory_reserved() / (1024**2)
                        print(
                            f"[DEBUG] _detect_batch(): GPU memory allocated={used_mb:.1f} MB reserved={reserved_mb:.1f} MB",
                            flush=True,
                        )
                except Exception:
                    pass
            except Exception as e:
                sys.stderr = old_stderr
                logger.error(f"YOLO predict() raised exception: {e}")
                raise
            finally:
                sys.stderr = old_stderr
                stderr_output = stderr_capture.getvalue()

        # Check for warnings or stderr output
        if warning_list:
            for w in warning_list:
                logger.warning(f"YOLO warning during predict(): {w.message}")

        if stderr_output.strip():
            logger.warning(f"YOLO stderr output: {stderr_output.strip()}")

        # CRITICAL: Verify all images were processed
        # YOLO can silently return fewer results if it hits resource limits
        if len(yolo_results) != len(image_paths):
            error_msg = (
                f"YOLO INFERENCE FAILURE: Processed only {len(yolo_results)}/{len(image_paths)} images\n"
                f"Missing: {len(image_paths) - len(yolo_results)} images were not processed\n"
                f"\n"
                f"System info:\n"
                f"  File descriptor limit (ulimit -n): {current_ulimit}\n"
                f"  Batch size: {len(image_paths)} images\n"
                f"\n"
                f"Likely causes:\n"
                f"  1. File descriptor exhaustion (current ulimit: {current_ulimit}, need ~{len(image_paths) * 2})\n"
                f"  2. Out of memory (GPU or system RAM)\n"
                f"  3. Corrupted image files\n"
                f"  4. YOLO internal error (check stderr above)\n"
                f"\n"
                f"Diagnostics:\n"
                f"  Stderr output: {stderr_output if stderr_output.strip() else 'None'}\n"
                f"  Warnings: {[str(w.message) for w in warning_list] if warning_list else 'None'}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Parse results
        results = []
        for i, (yolo_result, image_path) in enumerate(zip(yolo_results, image_paths)):
            result = self._parse_yolo_result(yolo_result, filename=image_path.name, image=images[i] if images else None)
            results.append(result)

        # Force garbage collection to close file descriptors leaked by YOLO
        # predict().  Interval kept low (64) because multiple inference workers
        # share the same process FD table — with N workers the effective
        # interval is N×64 images between cleanups.
        if not hasattr(self, "_images_processed"):
            self._images_processed = 0
        self._images_processed += len(image_paths)

        if self._images_processed % 64 < len(image_paths):
            gc.collect()

        # Free GPU memory after each inference batch so the next sub-batch (or next process)
        # starts with a clean slate; helps when many image sub-batches run in sequence.
        self._cuda_empty_cache_if_available()
        print(f"[DEBUG] _detect_batch(): gc and empty_cache done, returning {len(results)} results", flush=True)

        return results

    def _detect_image(self, image: np.ndarray, filename: str, store_image: bool = False) -> DetectionResult:
        """Run inference on single image array."""
        # Run YOLO inference
        yolo_results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,  # Matches training config (config.yaml:image_size)
            device=self.device,
            verbose=False,
        )

        result = self._parse_yolo_result(yolo_results[0], filename=filename, image=image if store_image else None)

        # Note: gc.collect() removed from here as it's too expensive per-image
        # File descriptor cleanup is handled in _detect_batch() every 256 images
        # PDF processing uses _detect_batch() for remote inference anyway

        return result

    def _parse_yolo_result(self, yolo_result, filename: str, image: Optional[np.ndarray] = None) -> DetectionResult:
        """Parse YOLO result into DetectionResult."""
        # Get image dimensions first (needed for clamping)
        if image is not None:
            height, width = image.shape[:2]
        elif yolo_result.orig_shape is not None:
            height, width = yolo_result.orig_shape
        else:
            height, width = 0, 0

        detections = []

        if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
            for box in yolo_result.boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()

                # Clamp coordinates to valid range [0, width/height]
                # This prevents negative coordinates or coordinates beyond image bounds
                if width > 0 and height > 0:
                    bbox[0] = max(0.0, min(bbox[0], float(width)))  # x1
                    bbox[1] = max(0.0, min(bbox[1], float(height)))  # y1
                    bbox[2] = max(0.0, min(bbox[2], float(width)))  # x2
                    bbox[3] = max(0.0, min(bbox[3], float(height)))  # y2

                    # Normalize bbox coordinates: ensure x1 < x2 and y1 < y2
                    # (should be rare after MPS fix above, but kept as safety check)
                    if bbox[0] > bbox[2]:
                        bbox[0], bbox[2] = bbox[2], bbox[0]  # Swap x1 and x2
                    if bbox[1] > bbox[3]:
                        bbox[1], bbox[3] = bbox[3], bbox[1]  # Swap y1 and y2

                    # Skip only if zero-area after normalization (degenerate box)
                    if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                        continue  # Skip zero-area bbox

                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = yolo_result.names[cls_id]

                detection = DiagramDetection(bbox=tuple(bbox), confidence=conf, class_name=cls_name, class_id=cls_id)
                detections.append(detection)

        return DetectionResult(
            filename=filename,
            detections=detections,
            image=image,
            image_width=width,
            image_height=height,
        )

    def save_results(self, results: List[DetectionResult], output_dir: Union[str, Path], format: str = "json") -> None:
        """
        Save detection results.

        Args:
            results: List of DetectionResult objects
            output_dir: Output directory
            format: Output format ('json' or 'csv')
        """
        import os

        output_dir = Path(output_dir)

        # Warn if using relative path
        if not output_dir.is_absolute() and self.verbose:
            print(f"⚠️  Using relative output path: {output_dir} (CWD: {os.getcwd()})")

        # Validate output directory is writable
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise PermissionError(
                f"Cannot create or write to output directory: {output_dir}\n"
                f"Error: {e}\n"
                f"Fix: Check directory permissions or choose a different output path."
            ) from e

        if format == "json":
            # Save as JSON
            data = [r.to_dict() for r in results]
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"[DIAGNOSTIC] save_results(): Saving {len(data)} results to JSON")
            output_path = output_dir / "detections.json"
            save_json(data, output_path)

            if self.verbose:
                print(f"✓ Results saved to {output_path}")

        elif format == "csv":
            # Save as CSV
            data = [r.to_csv_row() for r in results]
            output_path = output_dir / "detections.csv"
            save_csv(data, output_path)

            if self.verbose:
                print(f"✓ Results saved to {output_path}")

        else:
            raise ValueError(f"Unknown format: {format}. Use 'json' or 'csv'")

    def save_crops(self, results: List[DetectionResult], output_dir: Union[str, Path], padding: int = 10) -> None:
        """
        Extract and save cropped diagram regions.

        Args:
            results: List of DetectionResult objects
            output_dir: Output directory
            padding: Pixels to add around bbox

        Raises:
            ValueError: If results contain diagrams but images were not stored during detection
        """
        import os

        output_dir = Path(output_dir)

        # Warn if using relative path
        if not output_dir.is_absolute() and self.verbose:
            print(f"⚠️  Using relative output path: {output_dir} (CWD: {os.getcwd()})")

        # Validate output directory is writable
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise PermissionError(
                f"Cannot create or write to output directory: {output_dir}\n"
                f"Error: {e}\n"
                f"Fix: Check directory permissions or choose a different output path."
            ) from e

        # Check if any results have diagrams without stored images
        diagrams_without_images = [r.filename for r in results if r.has_diagram and r.image is None]

        if diagrams_without_images:
            raise ValueError(
                f"Cannot save crops: {len(diagrams_without_images)} result(s) have diagrams but images were not stored.\n"
                f"Fix: Pass store_images=True when calling detect() or detect_pdf().\n"
                f"Examples: detector.detect(path, store_images=True)\n"
                f"          detector.detect_pdf(path, store_images=True)\n"
                f"First few files affected: {diagrams_without_images[:3]}"
            )

        total_crops = 0

        for result in tqdm(results, desc="Extracting crops", disable=not self.verbose, unit="image"):
            if not result.has_diagram:
                continue

            # Extract each diagram
            for i, detection in enumerate(result.detections):
                crop = result.get_crop(i, padding)
                if crop is not None:
                    # Create filename
                    base_name = Path(result.filename).stem
                    crop_name = f"{base_name}_diagram{i+1}.jpg"
                    crop_path = output_dir / crop_name

                    # Save crop
                    from PIL import Image

                    Image.fromarray(crop).save(crop_path, "JPEG", quality=95)
                    total_crops += 1

        if self.verbose:
            print(f"✓ Saved {total_crops} diagram crops to {output_dir}")

    def save_visualizations(
        self, results: List[DetectionResult], output_dir: Union[str, Path], line_width: int = 3
    ) -> None:
        """
        Save images with bounding boxes drawn.

        Args:
            results: List of DetectionResult objects
            output_dir: Output directory
            line_width: Thickness of bbox lines

        Raises:
            ValueError: If results contain diagrams but images were not stored during detection
        """
        import os

        output_dir = Path(output_dir)

        # Warn if using relative path
        if not output_dir.is_absolute() and self.verbose:
            print(f"⚠️  Using relative output path: {output_dir} (CWD: {os.getcwd()})")

        # Validate output directory is writable
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise PermissionError(
                f"Cannot create or write to output directory: {output_dir}\n"
                f"Error: {e}\n"
                f"Fix: Check directory permissions or choose a different output path."
            ) from e

        # Check if any results have diagrams without stored images
        diagrams_without_images = [r.filename for r in results if r.has_diagram and r.image is None]

        if diagrams_without_images:
            raise ValueError(
                f"Cannot save visualizations: {len(diagrams_without_images)} result(s) have diagrams but images were not stored.\n"
                f"Fix: Pass store_images=True when calling detect() or detect_pdf().\n"
                f"Examples: detector.detect(path, store_images=True)\n"
                f"          detector.detect_pdf(path, store_images=True)\n"
                f"First few files affected: {diagrams_without_images[:3]}"
            )

        for result in tqdm(results, desc="Creating visualizations", disable=not self.verbose, unit="image"):
            if not result.has_diagram:
                continue

            vis_path = output_dir / result.filename
            result.save_visualization(vis_path, line_width=line_width)

        if self.verbose:
            print(f"✓ Saved {len(results)} visualizations to {output_dir}")
