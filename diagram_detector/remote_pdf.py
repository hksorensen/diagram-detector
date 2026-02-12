"""
Enhanced SSH Remote Inference - PDF-Optimized with Local Caching

Optimized for:
- PDF files as processing unit (not individual images)
- Local network (gigabit speeds)
- SQLite-based caching with gzip compression (thread-safe)
- Parallel local PDF extraction (CPU bottleneck)
- Local PDF → image extraction (less network traffic)
"""

import csv
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from .models import DetectionResult
from .remote_ssh import RemoteConfig, SSHRemoteDetector
from .utils import convert_pdf_to_images, save_json


def _append_timing_to_csv(
    log_path: Path,
    timestamp: datetime,
    num_pdfs: int,
    num_pages: int,
    extraction_time: float,
    inference_time: float,
    total_time: float,
    model: str,
    batch_size: int,
    image_batch_size: int,
    remote_host: str,
    num_cached: int = 0,
    num_detections: int = 0,
    max_workers: int = 0,
    tensorrt: bool = False,
    gpu_mem_used_mb: int = 0,
    gpu_mem_free_mb: int = 0,
) -> None:
    """
    Append timing data to CSV log file.

    Creates file with headers if it doesn't exist.

    Args:
        log_path: Path to CSV log file
        timestamp: Run timestamp
        num_pdfs: Number of PDFs processed
        num_pages: Total pages processed
        extraction_time: PDF extraction time (seconds)
        inference_time: Remote inference time (seconds)
        total_time: Total time (seconds)
        model: Model name
        batch_size: PDFs per batch
        image_batch_size: Images per inference batch
        remote_host: Remote server hostname
        num_cached: Number of PDFs served from cache
        num_detections: Total number of diagrams detected
        max_workers: Number of parallel PDF extraction workers
        tensorrt: Whether TensorRT optimization is enabled
    """
    log_path = Path(log_path)

    # Check if file exists to determine if we need headers
    file_exists = log_path.exists()

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate derived metrics
    pages_per_sec = num_pages / total_time if total_time > 0 else 0
    extraction_pct = (extraction_time / total_time * 100) if total_time > 0 else 0
    inference_pct = (inference_time / total_time * 100) if total_time > 0 else 0
    detections_per_page = num_detections / num_pages if num_pages > 0 else 0

    # Prepare row
    row = {
        "timestamp": timestamp.isoformat(),
        "num_pdfs": num_pdfs,
        "num_pages": num_pages,
        "num_detections": num_detections,
        "detections_per_page": round(detections_per_page, 3),
        "num_cached": num_cached,
        "extraction_time": round(extraction_time, 2),
        "inference_time": round(inference_time, 2),
        "total_time": round(total_time, 2),
        "pages_per_sec": round(pages_per_sec, 2),
        "extraction_pct": round(extraction_pct, 1),
        "inference_pct": round(inference_pct, 1),
        "model": model,
        "batch_size": batch_size,
        "image_batch_size": image_batch_size,
        "max_workers": max_workers,
        "tensorrt": tensorrt,
        "remote_host": remote_host,
        "gpu_mem_used_mb": gpu_mem_used_mb,
        "gpu_mem_free_mb": gpu_mem_free_mb,
    }

    # Write to CSV
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header if new file
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def _copy_inference_logs(
    batch_dir: Path,
    run_id: str,
    preserve_logs_dir: Path,
    pdf_batch_id: str,
) -> None:
    """
    Copy inference.log and inference.err from downloaded batch results into a persistent dir.

    Source: batch_dir / "results" / run_id / <sub_batch_id> / inference.log|.err
    (fallback: batch_dir / "results" / <sub_batch_id> if run_id path missing)
    Target: preserve_logs_dir / pdf_batch_id / <sub_batch_id> / inference.log|.err
    """
    import logging

    _log = logging.getLogger(__name__)

    # Check if results directory exists at all (may not if all PDFs failed extraction)
    results_dir = batch_dir / "results"
    if not results_dir.exists():
        _log.debug(
            "Preserve logs: batch_dir/results does not exist (no inference was run for %s). "
            "This is normal if all PDFs in the batch failed extraction.",
            pdf_batch_id,
        )
        return

    results_base = batch_dir / "results" / run_id
    if not results_base.exists():
        # Fallback: some installs or older code may write under results/ without run_id
        results_base = batch_dir / "results"
        if not results_base.exists():
            _log.warning(
                "Preserve logs: results_base does not exist (nothing to copy): %s",
                batch_dir / "results" / run_id,
            )
            return
    dest_base = preserve_logs_dir / pdf_batch_id
    dest_base.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for sub_dir in results_base.iterdir():
        if not sub_dir.is_dir():
            continue
        dest_sub = dest_base / sub_dir.name
        dest_sub.mkdir(parents=True, exist_ok=True)
        for name in ("inference.log", "inference.err"):
            src = sub_dir / name
            if src.exists():
                shutil.copy2(src, dest_sub / name)
        n_copied += 1
    _log.info(
        "Preserve logs: copied %d sub-batches to %s",
        n_copied,
        dest_base.resolve(),
    )


def _log_pdf_status(
    manifest_path: Path,
    pdf_name: str,
    status: str,
    pages_extracted: int,
    pages_detected: int,
    num_diagrams: int,
    error_type: str = "",
    error_message: str = "",
) -> None:
    """
    Log PDF processing status to manifest CSV.

    Status codes:
    - success: Processed with diagrams found (num_diagrams > 0)
    - success_no_diagrams: Processed successfully, 0 diagrams (legitimate empty)
    - extraction_failed: PDF extraction failed (0 pages extracted)
    - inference_failed: Inference stopped early (inference mismatch, partial results)
    - failed_empty: Result is [] - no pages at all (SHOULD NOT OCCUR)

    Args:
        manifest_path: Path to manifest CSV file
        pdf_name: Name of the PDF file
        status: Processing status code
        pages_extracted: Number of pages extracted from PDF
        pages_detected: Number of pages with detection results
        num_diagrams: Total number of diagrams detected
        error_type: Type of error (empty string if no error)
        error_message: Detailed error message (empty string if no error)
    """
    manifest_path = Path(manifest_path)

    # Check if file exists to determine if we need headers
    file_exists = manifest_path.exists()

    # Ensure parent directory exists
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare row
    row = {
        "pdf_name": pdf_name,
        "status": status,
        "pages_extracted": pages_extracted,
        "pages_detected": pages_detected,
        "num_diagrams": num_diagrams,
        "error_type": error_type,
        "error_message": error_message,
        "timestamp": datetime.now().isoformat(),
    }

    # Write to CSV with file locking
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header if new file
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def extract_pdf_to_jpegs(pdf_path: Path, output_dir: Path, dpi: int, show_progress: bool = False) -> List[Path]:
    """Extract all pages of a PDF to JPEG files in *output_dir*.

    Shared by the remote upload path and the local chunked-inference path so
    that extraction behaviour (naming, quality, progress) is identical.

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Directory in which to write ``page_0001.jpg`` … files.
            Created if it does not exist.
        dpi: Rendering resolution.
        show_progress: Show a tqdm progress bar during rendering.

    Returns:
        List of JPEG paths, one per page, in page order.

    Raises:
        Any exception from PyMuPDF / PIL propagates to the caller.
    """
    import shutil

    from PIL import Image

    images = convert_pdf_to_images(pdf_path, dpi=dpi, verbose=False, show_progress=show_progress)

    # CRITICAL FIX: Clear any existing JPEGs to prevent contamination
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    image_paths: List[Path] = []
    # Include PDF stem in JPEG filename for traceability
    pdf_stem = pdf_path.stem
    for page_num, img_array in enumerate(images, start=1):
        img_path = output_dir / f"{pdf_stem}_page_{page_num:04d}.jpg"
        Image.fromarray(img_array).save(img_path, "JPEG", quality=95)
        image_paths.append(img_path)

    return image_paths


class PDFRemoteDetector:
    """
    Remote detector optimized for PDF processing.

    Features:
    - Local PDF → image extraction (reduces network traffic)
    - Parallel extraction (utilize Mac CPU cores)
    - SQLite caching with gzip (thread-safe, compressed)
    - Batch processing at PDF level
    - Gigabit LAN optimized

    Performance:
    - Extraction: Parallel (4-8x speedup on multi-core)
    - Upload: Gigabit LAN (fast)
    - Inference: GPU (bottleneck, sequential)
    - Download: Gigabit LAN (fast)
    """

    def __init__(
        self,
        config: Optional[RemoteConfig] = None,
        batch_size: int = 10,  # PDFs per batch
        image_batch_size: int = 500,  # Images per upload batch
        gpu_batch_size: int = 32,  # GPU inference batch
        model: str = "yolo11m",
        confidence: float = 0.35,
        iou: float = 0.30,
        dpi: int = 200,
        imgsz: int = 640,
        cache_dir: Optional[Path] = None,
        parallel_extract: bool = True,
        max_workers: int = 4,
        verbose: bool = True,
        run_id: Optional[str] = None,
        config_dir: Optional[Path] = None,
        tensorrt: bool = False,
        upload_workers: Optional[int] = None,
        max_pdf_size_mb: Optional[float] = None,
        max_pdf_size_pages: Optional[int] = None,
        use_persistent_server: bool = False,
        remote_cleanup: bool = False,
        pdfs_on_remote: bool = False,  # PDFs already on remote (skip extraction)
    ):
        """
        Initialize PDF remote detector.

        Args:
            config: Remote configuration (None = use defaults for henrikkragh.dk)
            batch_size: PDFs per batch (10 = ~100-200 pages, good for gigabit LAN)
            image_batch_size: Images per upload batch (500 = good balance for gigabit LAN)
                             Increase for faster networks, decrease for slow connections
            gpu_batch_size: Images per GPU inference batch (32 = typical for 8GB VRAM)
                           Reduce if OOM errors occur
            model: Model to use
            confidence: Confidence threshold
            iou: IoU threshold for NMS (default: 0.30, optimal from grid search)
            dpi: DPI for PDF conversion
            imgsz: Image size for preprocessing (default: 640, must match training)
            cache_dir: Cache directory (None = use default)
            parallel_extract: Use parallel PDF extraction
            max_workers: Number of parallel extraction workers
            verbose: Print progress
            run_id: Unique run identifier (auto-generated if None)
            config_dir: Directory to store run configs (for git tracking)
            tensorrt: Use TensorRT optimization on remote (NVIDIA GPU only, 2-3x faster)
            upload_workers: Number of parallel rsync workers (None = auto: min(5, max(1, files//50)))
            max_pdf_size_mb: Skip PDFs larger than this (MB). None = no limit. Useful for slow networks.
            max_pdf_size_pages: Skip PDFs with more pages than this. None = no limit. Page count is read before extraction (fast).
            use_persistent_server: Keep model resident in GPU memory across sub-batches (~3-5s reload saved per batch)
            remote_cleanup: Delete remote JPGs after processing (default: False for debugging)
        """
        # Auto-load config if not provided
        if config is None:
            config = RemoteConfig.auto_load()

        self.config = config
        self.batch_size = batch_size
        self.image_batch_size = image_batch_size
        self.gpu_batch_size = gpu_batch_size
        self.model = model
        self.confidence = confidence
        self.iou = iou
        self.dpi = dpi
        self.imgsz = imgsz
        self.verbose = verbose
        self.parallel_extract = parallel_extract
        self.max_workers = max_workers
        self.max_pdf_size_mb = max_pdf_size_mb
        self.max_pdf_size_pages = max_pdf_size_pages
        self.remote_cleanup = remote_cleanup
        self.pdfs_on_remote = pdfs_on_remote  # Skip extraction if PDFs already on remote

        self.tensorrt = tensorrt

        # Initialize SSH detector for actual remote execution
        self.remote_detector = SSHRemoteDetector(
            config=config,
            batch_size=image_batch_size,  # Images per upload batch
            gpu_batch_size=gpu_batch_size,  # GPU inference batch
            model=model,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            verbose=self.verbose,  # Pass through verbose flag for detailed timing
            run_id=run_id,
            config_dir=config_dir,
            tensorrt=tensorrt,
            upload_workers=upload_workers,
            use_persistent_server=use_persistent_server,
        )

        # REMOTE CACHE DISABLED - Causing contamination issues
        # Local cache (on client) is still active and sufficient
        # Remote cache can be re-enabled later if needed for multi-machine setups
        self.cache = None

        # Work directory for remote PDF processing results (when PDFs already on remote)
        import tempfile

        self.work_dir = Path(tempfile.mkdtemp(prefix="remote_pdf_"))

        # if self.verbose:
        #     cache_stats = self.cache.stats()
        #     print(
        #         f"Cache: {cache_stats['num_pdfs']} PDFs, "
        #         f"{cache_stats['total_pages']:,} pages "
        #         f"({cache_stats['size_mb']:.1f} MB compressed)"
        #     )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup SSH resources.

        Stops persistent server if exiting due to exception,
        otherwise keeps it alive for reuse across batches.
        """
        # Stop server if there was an exception
        stop_server = exc_type is not None
        self.cleanup(stop_server=stop_server)
        return False

    def __del__(self):
        """Destructor - cleanup SSH resources as safety net."""
        try:
            # Always stop server when object is destroyed
            self.cleanup(stop_server=True)
        except Exception:
            pass  # Ignore errors during destruction

    def cleanup(self, stop_server: bool = False):
        """Clean up SSH control socket and connections.

        Args:
            stop_server: If True, also stop the persistent server.
        """
        if hasattr(self, "remote_detector") and self.remote_detector:
            self.remote_detector.cleanup(stop_server=stop_server)

    def _extract_pdf_pages(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        """Extract PDF pages to images locally (delegates to shared helper)."""
        return extract_pdf_to_jpegs(pdf_path, output_dir, self.dpi, show_progress=self.verbose)

    def _extract_pdfs_parallel(self, pdf_batch: List[Path], batch_dir: Path) -> tuple[Dict[str, List[Path]], float]:
        """
        Extract multiple PDFs in parallel.

        Args:
            pdf_batch: List of PDF paths
            batch_dir: Working directory

        Returns:
            Tuple of (Dict mapping PDF name to image paths, extraction time in seconds)
        """
        start_time = time.time()

        if self.verbose:
            print(f"  Extracting {len(pdf_batch)} PDFs in parallel ({self.max_workers} workers)...")

        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"[EXTRACTION] Starting extraction of {len(pdf_batch)} PDFs")

        pdf_images = {}

        if not self.parallel_extract or len(pdf_batch) == 1:
            # Sequential extraction
            for pdf_path in pdf_batch:
                pdf_size_mb = pdf_path.stat().st_size / 1024 / 1024
                if self.max_pdf_size_mb is not None and pdf_size_mb > self.max_pdf_size_mb:
                    logger.warning(
                        f"SKIPPING large PDF: {pdf_path.name} ({pdf_size_mb:.1f} MB) exceeds limit of {self.max_pdf_size_mb} MB"
                    )
                    pdf_images[pdf_path.name] = None
                    continue
                if self.max_pdf_size_pages is not None:
                    import fitz

                    with fitz.open(pdf_path) as doc:
                        num_pages = doc.page_count
                    if num_pages > self.max_pdf_size_pages:
                        logger.warning(
                            f"SKIPPING large PDF: {pdf_path.name} ({num_pages} pages) exceeds limit of {self.max_pdf_size_pages} pages"
                        )
                        pdf_images[pdf_path.name] = None
                        continue
                pdf_dir = batch_dir / pdf_path.stem
                image_paths = self._extract_pdf_pages(pdf_path, pdf_dir)
                pdf_images[pdf_path.name] = image_paths
        else:
            # Parallel extraction
            logger.debug(f"[EXTRACTION] Creating ThreadPoolExecutor with {self.max_workers} workers")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit extraction tasks in pdf_batch order, preserving order
                # CRITICAL: Use list of (future, pdf_path) tuples to maintain order
                futures_in_order = []
                skipped_pdfs = []

                for pdf_path in pdf_batch:
                    pdf_size_mb = pdf_path.stat().st_size / 1024 / 1024

                    # Check if PDF exceeds size limit
                    if self.max_pdf_size_mb is not None and pdf_size_mb > self.max_pdf_size_mb:
                        logger.warning(
                            f"SKIPPING large PDF: {pdf_path.name} ({pdf_size_mb:.1f} MB) "
                            f"exceeds limit of {self.max_pdf_size_mb} MB"
                        )
                        pdf_images[pdf_path.name] = None  # Mark as skipped
                        skipped_pdfs.append(pdf_path)
                        continue

                    # Check if PDF exceeds page limit (page_count reads only trailer — ~0.1 ms)
                    if self.max_pdf_size_pages is not None:
                        import fitz

                        with fitz.open(pdf_path) as doc:
                            num_pages = doc.page_count
                        if num_pages > self.max_pdf_size_pages:
                            logger.warning(
                                f"SKIPPING large PDF: {pdf_path.name} ({num_pages} pages) "
                                f"exceeds limit of {self.max_pdf_size_pages} pages"
                            )
                            pdf_images[pdf_path.name] = None  # Mark as skipped
                            skipped_pdfs.append(pdf_path)
                            continue

                    pdf_dir = batch_dir / pdf_path.stem
                    future = executor.submit(self._extract_pdf_pages, pdf_path, pdf_dir)
                    futures_in_order.append((future, pdf_path))

                logger.debug(
                    f"[EXTRACTION] Submitted {len(futures_in_order)} extraction tasks, skipped {len(skipped_pdfs)} large PDFs"
                )

                # CRITICAL FIX: Wait for results in submission order (pdf_batch order)
                # Do NOT use as_completed() which returns in completion order!
                from tqdm import tqdm

                completed_count = 0
                total_pages = 0
                extraction_start_time = time.time()

                if self.verbose:
                    progress = tqdm(
                        total=len(futures_in_order),
                        desc="  Extracting",
                        unit="PDF",
                        leave=False,
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]",
                    )

                for future, pdf_path in futures_in_order:
                    try:
                        # Timeout after 10 minutes per PDF (handles very large PDFs up to ~3000 pages)
                        # At 0.2 sec/page, 3000 pages = 600 seconds = 10 minutes
                        image_paths = future.result(timeout=600)
                        pdf_images[pdf_path.name] = image_paths
                        num_pages = len(image_paths) if image_paths else 0
                        total_pages += num_pages
                    except TimeoutError:
                        logger.error(f"PDF extraction TIMEOUT for {pdf_path.name} (exceeded 10 minutes)")
                        logger.error(f"  PDF path: {pdf_path}")
                        logger.error(f"  PDF size: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")
                        if self.verbose:
                            print(f"  ✗ {pdf_path.name}: Extraction timeout (>10 min)")
                        pdf_images[pdf_path.name] = None
                    except Exception as e:
                        # CRITICAL: Log full exception for debugging
                        import traceback

                        logger.error(f"PDF extraction FAILED for {pdf_path.name}")
                        logger.error(f"  Error: {type(e).__name__}: {e}")
                        logger.error(f"  PDF path: {pdf_path}")
                        logger.error(f"  PDF size: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")
                        logger.error(f"  Traceback:\n{''.join(traceback.format_exc())}")

                        if self.verbose:
                            print(f"  ✗ {pdf_path.name}: {type(e).__name__}: {e}")

                        # Mark as None instead of empty list
                        # This will cause caching to skip this PDF
                        pdf_images[pdf_path.name] = None

                    completed_count += 1
                    if self.verbose:
                        # Update progress bar with current page count and speed
                        elapsed = time.time() - extraction_start_time
                        pages_per_sec = total_pages / elapsed if elapsed > 0 else 0
                        progress.set_description(f"  Extracting ({total_pages} pages, {pages_per_sec:.1f} p/s)")
                        progress.update(1)

                if self.verbose:
                    progress.close()

        extraction_time = time.time() - start_time

        # Log extraction summary
        total_pages_extracted = sum(len(imgs) if imgs else 0 for imgs in pdf_images.values() if imgs is not None)
        failed_pdfs = sum(1 for imgs in pdf_images.values() if imgs is None)
        logger.debug(
            f"[EXTRACTION] Complete: {len(pdf_images)} PDFs, {total_pages_extracted} pages in {extraction_time:.1f}s ({total_pages_extracted/extraction_time:.1f} pages/s)"
        )
        if failed_pdfs > 0:
            logger.warning(f"[EXTRACTION] {failed_pdfs}/{len(pdf_images)} PDFs failed extraction")

        if self.verbose:
            print(
                f"  ✓ Extraction complete: {total_pages_extracted} pages in {extraction_time:.1f}s ({total_pages_extracted/extraction_time:.1f} pages/s)"
            )

        return pdf_images, extraction_time

    def _process_remote_pdfs(
        self,
        pdfs_on_remote: List[Path],
        remote_pdf_base: str,
        batch_id: str,
        gpu_batch_size: int = 32,
        manifest_path: Optional[Path] = None,
    ) -> tuple[Dict[str, List[DetectionResult]], float]:
        """
        Process PDFs that are already on remote server.

        Uses the server's infer_pdfs command to extract and detect PDFs
        directly on remote, avoiding local extraction and upload.

        Args:
            pdfs_on_remote: List of PDF paths (local paths, filenames used for remote lookup)
            remote_pdf_base: Base directory for PDFs on remote (e.g., ~/diagrams_in_arxiv/pdfs)
            batch_id: Batch identifier for logging
            gpu_batch_size: GPU inference batch size

        Returns:
            Tuple of (results_dict, processing_time)
            results_dict: {pdf_filename: [DetectionResult, ...]}
        """
        import json
        import logging
        import subprocess
        import time

        logger = logging.getLogger(__name__)

        start_time = time.time()

        # Build full remote PDF paths - need to check which subdirectory each PDF is in
        # PDFs can be in either published/ or arxiv/ subdirectories
        remote_pdf_paths = []

        # Check subdirectory for all PDFs in a single SSH command (much faster than per-PDF)

        # Build a single command that checks all PDFs and outputs their subdirectory
        check_commands = []
        for pdf in pdfs_on_remote:
            # For each PDF, output "arxiv:filename" or "published:filename" if found
            check_commands.append(
                f"if [ -f {remote_pdf_base}/published/{pdf.name} ]; then "
                f"echo 'published:{pdf.name}'; "
                f"elif [ -f {remote_pdf_base}/arxiv/{pdf.name} ]; then "
                f"echo 'arxiv:{pdf.name}'; "
                f"fi"
            )

        combined_check = "; ".join(check_commands)
        ssh_cmd = (
            ["ssh"]
            + self.remote_detector.config.ssh_port_args
            + [self.remote_detector.config.ssh_target, combined_check]
        )

        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
            # Parse output: each line is "subdir:filename"
            for line in result.stdout.strip().split("\n"):
                if line and ":" in line:
                    subdir, filename = line.split(":", 1)
                    remote_pdf_paths.append(f"{remote_pdf_base}/{subdir}/{filename}")

            # Warn about PDFs that weren't found
            found_names = {line.split(":", 1)[1] for line in result.stdout.strip().split("\n") if line and ":" in line}
            for pdf in pdfs_on_remote:
                if pdf.name not in found_names:
                    logger.warning(f"PDF not found in published/ or arxiv/: {pdf.name}")

        except Exception as e:
            logger.error(f"Failed to check PDF locations: {e}")
            raise

        if not remote_pdf_paths:
            raise RuntimeError(f"No PDFs found on remote in {remote_pdf_base}/{{published,arxiv}}/")

        # DETAILED LOGGING: Show what we found
        logger.info(f"[REMOTE PDF CHECK] Base directory: {remote_pdf_base}")
        logger.info(f"[REMOTE PDF CHECK] Found {len(remote_pdf_paths)} PDFs on remote")
        logger.info(f"[REMOTE PDF CHECK] First 3 paths: {remote_pdf_paths[:3]}")
        logger.info(f"[REMOTE PDF CHECK] Last 3 paths: {remote_pdf_paths[-3:]}")
        logger.info(f"Processing {len(remote_pdf_paths)} PDFs directly on remote (no upload needed)")

        # Start persistent server if not running
        if not self.remote_detector._server_proc:
            self.remote_detector._start_persistent_server()

        # DETAILED LOGGING: Before calling persistent server
        logger.info(f"[PERSISTENT SERVER] Sending {len(remote_pdf_paths)} PDF paths to server")
        logger.info(f"[PERSISTENT SERVER] Batch ID: {batch_id}")
        logger.info("[PERSISTENT SERVER] Sample paths being sent:")
        for i, path in enumerate(remote_pdf_paths[:5]):
            logger.info(f"[PERSISTENT SERVER]   [{i}] {path}")

        # Run inference using SSHRemoteDetector's infrastructure
        self.remote_detector._run_inference_pdfs(remote_pdf_paths, batch_id)

        # Download results from remote
        remote_output_dir = f"{self.remote_detector.config.remote_work_dir}/output/{batch_id}"
        local_output_dir = self.work_dir / f"remote_results_{batch_id}"
        local_output_dir.mkdir(parents=True, exist_ok=True)

        # DETAILED LOGGING: Download info
        logger.info(f"[DOWNLOAD RESULTS] Remote output: {remote_output_dir}")
        logger.info(f"[DOWNLOAD RESULTS] Local output: {local_output_dir}")

        # Download results directory
        rsync_cmd = (
            ["rsync", "-avz", "--timeout=300"]
            + self.remote_detector.config.rsync_ssh_args
            + [f"{self.remote_detector.config.ssh_target}:{remote_output_dir}/", str(local_output_dir) + "/"]
        )

        logger.debug(f"Downloading results from {remote_output_dir}")
        result = subprocess.run(rsync_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download remote results: {result.stderr}")

        # DETAILED LOGGING: What did we download?
        downloaded_files = list(local_output_dir.glob("*"))
        logger.info(f"[DOWNLOAD RESULTS] Downloaded {len(downloaded_files)} files")
        logger.info(f"[DOWNLOAD RESULTS] Files: {[f.name for f in downloaded_files]}")

        # Parse results from single detections.json file
        detections_file = local_output_dir / "detections.json"
        results_dict = {}

        if detections_file.exists():
            detections_size = detections_file.stat().st_size
            logger.info(f"[PARSE RESULTS] detections.json exists, size: {detections_size} bytes")

            with open(detections_file) as f:
                all_results = json.load(f)

            # DETAILED LOGGING: What's in the JSON?
            logger.info(f"[PARSE RESULTS] Total image results in JSON: {len(all_results)}")
            if all_results:
                unique_pdfs = set()
                for img_result in all_results:
                    filename = img_result.get("filename", "")
                    if "_page_" in filename:
                        pdf_stem = filename.split("_page_")[0]
                        unique_pdfs.add(pdf_stem)
                logger.info(f"[PARSE RESULTS] Unique PDF stems in JSON: {len(unique_pdfs)}")
                logger.info(f"[PARSE RESULTS] PDF stems: {sorted(unique_pdfs)[:10]}")
                logger.info(f"[PARSE RESULTS] First 3 filenames: {[r.get('filename') for r in all_results[:3]]}")
                logger.info(f"[PARSE RESULTS] Last 3 filenames: {[r.get('filename') for r in all_results[-3:]]}")

            # Convert to DetectionResult objects and group by PDF
            import re

            from .models import DetectionResult, DiagramDetection

            # DETAILED LOGGING: Matching setup
            logger.info(f"[MATCHING] Trying to match {len(pdfs_on_remote)} PDFs to {len(all_results)} image results")
            logger.info(f"[MATCHING] First 5 PDF stems to match: {[p.stem for p in pdfs_on_remote[:5]]}")

            # Group results by PDF stem (extract from image filename)
            match_count = 0
            for pdf in pdfs_on_remote:
                pdf_stem = pdf.stem
                # Find all results for this PDF (images like "pdfname_page_0001.jpg")
                pdf_results = []

                for img_result in all_results:
                    img_name = img_result["filename"]
                    # Extract PDF stem and page number from image filename
                    # Format: "pdfname_page_0001.jpg"
                    match = re.match(r"(.+)_page_(\d+)\.jpg$", img_name)
                    if match and match.group(1) == pdf_stem:
                        page_num = int(match.group(2))

                        # Convert detection format to match DiagramDetection model
                        detections = []
                        for det in img_result.get("detections", []):
                            detections.append(
                                DiagramDetection(
                                    bbox=tuple(det["bbox"]),
                                    confidence=det["confidence"],
                                    class_id=int(det.get("class", 0)),
                                    class_name="diagram",
                                )
                            )

                        pdf_results.append(
                            DetectionResult(
                                filename=img_name,
                                page_number=page_num,
                                detections=detections,
                            )
                        )

                if not pdf_results:
                    logger.warning(f"No results found for {pdf.name}")
                    # Log as failed to manifest
                    if manifest_path:
                        _log_pdf_status(
                            manifest_path=manifest_path,
                            pdf_name=pdf.name,
                            status="inference_failed",
                            pages_extracted=0,
                            pages_detected=0,
                            num_diagrams=0,
                            error_type="NoResults",
                            error_message="No detection results returned from remote processing",
                        )
                else:
                    match_count += 1
                    # Log success to manifest
                    if manifest_path:
                        total_diagrams = sum(r.count for r in pdf_results)
                        status = "success" if total_diagrams > 0 else "success_no_diagrams"
                        _log_pdf_status(
                            manifest_path=manifest_path,
                            pdf_name=pdf.name,
                            status=status,
                            pages_extracted=len(pdf_results),
                            pages_detected=len(pdf_results),
                            num_diagrams=total_diagrams,
                        )

                results_dict[pdf.name] = pdf_results

            # DETAILED LOGGING: Matching results
            logger.info(f"[MATCHING] Successfully matched {match_count}/{len(pdfs_on_remote)} PDFs")
            logger.info(f"[MATCHING] PDFs with no results: {len(pdfs_on_remote) - match_count}")

            # DETAILED LOGGING: Check if we can write to detections.json (verify permissions)
            try:
                test_write = detections_file.parent / "test_write.tmp"
                test_write.write_text("test")
                test_write.unlink()
                logger.info(f"[FILE ACCESS] Can write to {detections_file.parent}")
            except Exception as e:
                logger.warning(f"[FILE ACCESS] Cannot write to {detections_file.parent}: {e}")
        else:
            logger.error(f"detections.json not found in {local_output_dir}")
            for pdf in pdfs_on_remote:
                results_dict[pdf.name] = []

        processing_time = time.time() - start_time
        logger.info(f"✓ Remote PDF processing complete: {processing_time:.1f}s for {len(pdfs_on_remote)} PDFs")
        return results_dict, processing_time

    def _process_pdf_batch(
        self,
        pdf_batch: List[Path],
        batch_id: str,
        work_dir: Path,
        auto_git_commit: bool = False,
        gpu_batch_size: int = 32,
        manifest_path: Optional[Path] = None,
    ) -> tuple[Dict[str, List[DetectionResult]], float, float, int, int, Dict[str, str]]:
        """
        Process batch of PDFs.

        Args:
            pdf_batch: List of PDF paths
            batch_id: Batch identifier
            work_dir: Working directory
            auto_git_commit: Automatically git commit the run config

        Returns:
            Tuple of (results dict, extraction_time, inference_time, gpu_mem_used_mb, gpu_mem_free_mb,
                      batch_failed dict mapping pdf_name to failure status)
        """
        import logging

        logger = logging.getLogger(__name__)

        batch_dir = work_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Track failures: {pdf_name: status} for PDFs that did not produce results
        batch_failed: Dict[str, str] = {}

        # Check if PDFs are already on remote (optimization to skip extraction+upload)
        pdfs_on_remote_list = []
        pdfs_need_extraction = []

        if self.pdfs_on_remote:
            # PDFs stored on remote via storage backend - check which ones exist
            # Get expanded absolute path from remote (can't use ~ or $HOME with persistent server)
            import subprocess

            if self.verbose:
                print("  Checking which PDFs are already on remote...")

            # Expand path on remote to get absolute path
            ssh_cmd = (
                ["ssh"]
                + self.remote_detector.config.ssh_port_args
                + [self.remote_detector.config.ssh_target, "echo $HOME/diagrams_in_arxiv/pdfs"]
            )
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=5)
            remote_pdf_base = result.stdout.strip()
            if not remote_pdf_base or result.returncode != 0:
                raise RuntimeError(f"Failed to get remote PDF base path: {result.stderr}")

            # Check all PDFs in batch efficiently with one SSH call

            check_commands = []
            for pdf in pdf_batch:
                check_commands.append(
                    f"([ -f {remote_pdf_base}/published/{pdf.name} ] || [ -f {remote_pdf_base}/arxiv/{pdf.name} ]) && echo '{pdf.name}' || true"
                )

            combined_check = " && ".join(check_commands)
            ssh_cmd = ["ssh", "-p", str(self.config.port), f"{self.config.user}@{self.config.host}", combined_check]

            try:
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
                found_names = set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()

                for pdf in pdf_batch:
                    if pdf.name in found_names:
                        pdfs_on_remote_list.append(pdf)
                    else:
                        pdfs_need_extraction.append(pdf)

                if self.verbose and pdfs_on_remote_list:
                    saved_pct = 100 * len(pdfs_on_remote_list) / len(pdf_batch)
                    print(
                        f"  ✓ {len(pdfs_on_remote_list)}/{len(pdf_batch)} PDFs already on remote ({saved_pct:.0f}% skip extraction+upload)"
                    )

            except (subprocess.TimeoutExpired, Exception) as e:
                logger.warning(f"Failed to check remote PDFs: {e}, will extract all locally")
                pdfs_need_extraction = pdf_batch
        else:
            # No remote PDF optimization, extract all
            pdfs_need_extraction = pdf_batch

        # Extract only PDFs that need extraction (not on remote)
        if pdfs_need_extraction:
            pdf_images, extraction_time = self._extract_pdfs_parallel(pdfs_need_extraction, batch_dir)
        else:
            pdf_images = {}
            extraction_time = 0.0

        # Process PDFs that are already on remote (extract and detect remotely)
        remote_results = {}
        remote_processing_time = 0.0
        if pdfs_on_remote_list:
            logger.info(f"Processing {len(pdfs_on_remote_list)} PDFs directly on remote (skip extraction+upload)")

            # Retry logic: attempt remote processing with automatic retry on server death
            max_retries = 2  # Try up to 2 times (initial + 1 retry)
            retry_delay = 2  # seconds between retries

            for attempt in range(max_retries):
                try:
                    remote_results, remote_time = self._process_remote_pdfs(
                        pdfs_on_remote_list, remote_pdf_base, batch_id, gpu_batch_size, manifest_path
                    )
                    remote_processing_time = remote_time
                    logger.info(f"✓ Remote PDF processing completed ({remote_time:.1f}s)")
                    break  # Success - exit retry loop

                except RuntimeError as e:
                    # Check if it's a server death (not other RuntimeErrors)
                    if "Persistent server died" in str(e):
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Remote server died (attempt {attempt+1}/{max_retries}). "
                                f"Reconnecting in {retry_delay}s..."
                            )
                            time.sleep(retry_delay)

                            # Restart persistent server for retry
                            try:
                                if self.remote_detector._server_proc:
                                    logger.info("Cleaning up dead server process...")
                                    self.remote_detector._server_proc = None
                                logger.info("Restarting persistent server...")
                                self.remote_detector._start_persistent_server()
                                logger.info("✓ Server restarted, retrying inference...")
                            except Exception as restart_error:
                                logger.error(f"Failed to restart server: {restart_error}")
                                # Don't retry if we can't restart
                                break
                        else:
                            # Final attempt failed
                            logger.error(f"Remote PDF processing failed after {max_retries} attempts: {e}")
                            logger.warning(f"Falling back to local extraction for {len(pdfs_on_remote_list)} PDFs")
                            # Fall back to local extraction
                            fallback_images, fallback_time = self._extract_pdfs_parallel(pdfs_on_remote_list, batch_dir)
                            pdf_images.update(fallback_images)
                            extraction_time += fallback_time
                    else:
                        # Not a server death error - re-raise
                        raise

                except Exception as e:
                    # Other exceptions - log and fall back immediately
                    logger.error(f"Remote PDF processing failed: {e}")
                    logger.warning(f"Falling back to local extraction for {len(pdfs_on_remote_list)} PDFs")
                    # Fall back to local extraction
                    fallback_images, fallback_time = self._extract_pdfs_parallel(pdfs_on_remote_list, batch_dir)
                    pdf_images.update(fallback_images)
                    extraction_time += fallback_time
                    break  # Don't retry on non-server-death errors

        # Flatten to all images
        all_images = []
        pdf_page_counts = {}

        # ═══════════════════════════════════════════════════════════════════
        # CHECKPOINT 2: Verify extraction preserved pdf_batch order
        # ═══════════════════════════════════════════════════════════════════
        pdf_images_order = [name for name in pdf_images.keys()]
        pdf_batch_order = [p.name for p in pdf_batch if p.name in pdf_images]
        order_matches = pdf_images_order == pdf_batch_order

        # Log to contamination logger (silent, for monitoring)
        try:
            from diagrams_in_arxiv.contamination_logger import log_checkpoint

            log_checkpoint(
                checkpoint=f"CHECKPOINT 2 (REMOTE batch {batch_id})",
                status="PASS" if order_matches else "FAIL",
                expected=pdf_batch_order,
                actual=pdf_images_order,
                details=f"After extraction (remote path), batch {batch_id}",
            )
        except ImportError:
            pass

        if order_matches:
            # PASS: quiet (only log to file at DEBUG level)
            logger.debug(f"CHECKPOINT 2 PASS: Extraction order correct for batch {batch_id}")
        else:
            # FAIL: verbose (show details for debugging)
            logger.error("=" * 70)
            logger.error(f"CHECKPOINT 2 FAIL: ORDER MISMATCH in batch {batch_id}")
            logger.error("=" * 70)
            logger.error(f"  Expected (pdf_batch): {pdf_batch_order[:5]}")
            logger.error(f"  Actual (pdf_images):  {pdf_images_order[:5]}")
            logger.error("  This will cause contamination - results assigned to wrong PDFs!")
            logger.error("=" * 70)

        # CRITICAL FIX: Iterate in pdf_batch order, not pdf_images.items() order!
        # This ensures all_images matches the order used for result slicing
        logger.debug("Building all_images in pdf_batch order...")
        for pdf_path in pdf_batch:
            pdf_name = pdf_path.name
            if pdf_name not in pdf_images:
                # PDF extraction failed - already logged, skip
                continue

            image_paths = pdf_images[pdf_name]
            if image_paths is None:
                # Extraction failed or skipped - determine reason
                # (pdf_path already available from loop iteration)
                pdf_size_mb = pdf_path.stat().st_size / 1024 / 1024
                if self.max_pdf_size_mb is not None and pdf_size_mb > self.max_pdf_size_mb:
                    # Skipped due to size limit
                    logger.warning(
                        f"Skipping {pdf_name} due to size limit ({pdf_size_mb:.1f} MB > {self.max_pdf_size_mb} MB)"
                    )
                    batch_failed[pdf_name] = "size_limit_exceeded"
                    if manifest_path:
                        _log_pdf_status(
                            manifest_path=manifest_path,
                            pdf_name=pdf_name,
                            status="size_limit_exceeded",
                            pages_extracted=0,
                            pages_detected=0,
                            num_diagrams=0,
                            error_type="SizeLimitExceeded",
                            error_message=f"PDF size ({pdf_size_mb:.1f} MB) exceeds limit of {self.max_pdf_size_mb} MB",
                        )
                elif self.max_pdf_size_pages is not None:
                    # Check if skipped due to page limit (page count already read pre-extraction)
                    import fitz

                    with fitz.open(pdf_path) as doc:
                        num_pages = doc.page_count
                    if num_pages > self.max_pdf_size_pages:
                        logger.warning(
                            f"Skipping {pdf_name} due to page limit ({num_pages} pages > {self.max_pdf_size_pages} pages)"
                        )
                        batch_failed[pdf_name] = "page_limit_exceeded"
                        if manifest_path:
                            _log_pdf_status(
                                manifest_path=manifest_path,
                                pdf_name=pdf_name,
                                status="page_limit_exceeded",
                                pages_extracted=0,
                                pages_detected=0,
                                num_diagrams=0,
                                error_type="PageLimitExceeded",
                                error_message=f"PDF has {num_pages} pages, exceeds limit of {self.max_pdf_size_pages} pages",
                            )
                    else:
                        # Page limit set but this PDF is within limit — genuine extraction failure
                        logger.warning(f"Skipping {pdf_name} due to extraction failure (None)")
                        batch_failed[pdf_name] = "extraction_failed"
                        if manifest_path:
                            _log_pdf_status(
                                manifest_path=manifest_path,
                                pdf_name=pdf_name,
                                status="extraction_failed",
                                pages_extracted=0,
                                pages_detected=0,
                                num_diagrams=0,
                                error_type="ExtractionFailure",
                                error_message="PDF extraction returned None (exception occurred)",
                            )
                else:
                    # Extraction failed for other reason
                    logger.warning(f"Skipping {pdf_name} due to extraction failure (None)")
                    batch_failed[pdf_name] = "extraction_failed"
                    if manifest_path:
                        _log_pdf_status(
                            manifest_path=manifest_path,
                            pdf_name=pdf_name,
                            status="extraction_failed",
                            pages_extracted=0,
                            pages_detected=0,
                            num_diagrams=0,
                            error_type="ExtractionFailure",
                            error_message="PDF extraction returned None (exception occurred)",
                        )

                continue

            if len(image_paths) == 0:
                # Extraction returned empty list - this should not happen!
                # If convert_pdf_to_images returns [], it means the PDF had pages
                # but all failed to render, which is very unusual
                logger.error(f"UNEXPECTED: Extraction returned 0 pages for {pdf_name}")
                logger.error("  This suggests all pages failed to render (very unusual)")
                logger.error("  The PDF should have been caught earlier if it had 0 pages")

                batch_failed[pdf_name] = "extraction_failed"
                if manifest_path:
                    _log_pdf_status(
                        manifest_path=manifest_path,
                        pdf_name=pdf_name,
                        status="extraction_failed",
                        pages_extracted=0,
                        pages_detected=0,
                        num_diagrams=0,
                        error_type="ZeroPages",
                        error_message="PDF extraction returned 0 pages (all pages failed to render)",
                    )
                continue

            # FORENSIC LOGGING: Track image additions for contamination debugging
            if len(all_images) < 100:  # Log first few to avoid spam
                logger.debug(f"Adding {len(image_paths)} images for {pdf_name}")
                if len(image_paths) > 0:
                    logger.debug(f"  Sample: {image_paths[0].name}")
                    # Verify image filename contains PDF stem
                    expected_stem = pdf_name.rsplit(".", 1)[0]  # Remove .pdf extension
                    if expected_stem not in image_paths[0].name:
                        logger.warning("  WARNING: Image name doesn't contain PDF stem!")
                        logger.warning(f"    PDF name: {pdf_name}")
                        logger.warning(f"    Expected stem: {expected_stem}")
                        logger.warning(f"    Image name: {image_paths[0].name}")

            all_images.extend(image_paths)
            pdf_page_counts[pdf_name] = len(image_paths)

        # ═══════════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════════
        # CHECKPOINT 3: Verify all_images list built in correct order
        # ═══════════════════════════════════════════════════════════════════
        images_in_correct_order = True
        if len(all_images) > 0 and len(pdf_batch) > 0:
            # Build expected image sequence from first few PDFs
            expected_sequence = []
            for pdf_path in pdf_batch[: min(3, len(pdf_batch))]:
                pdf_stem = pdf_path.stem
                if pdf_path.name in pdf_page_counts:
                    page_count = pdf_page_counts[pdf_path.name]
                    for page_num in range(1, page_count + 1):
                        expected_sequence.append(f"{pdf_stem}_page_{page_num:04d}.jpg")

            # Compare expected vs actual
            actual_sequence = [img.name for img in all_images[: len(expected_sequence)]]
            images_in_correct_order = expected_sequence == actual_sequence

            # Log to contamination logger (silent, for monitoring)
            try:
                from diagrams_in_arxiv.contamination_logger import log_checkpoint

                expected_summary = [f"{p.stem}_page_*" for p in pdf_batch[:3]]
                actual_summary = [p.name for p in all_images[: min(20, len(all_images))]]
                log_checkpoint(
                    checkpoint=f"CHECKPOINT 3 (REMOTE batch {batch_id})",
                    status="PASS" if images_in_correct_order else "FAIL",
                    expected=expected_summary,
                    actual=actual_summary,
                    details=f"After building all_images (remote path), batch {batch_id}, checked {len(expected_sequence)} images",
                )
            except ImportError:
                pass

            if images_in_correct_order:
                # PASS: quiet (only log to file at DEBUG level)
                logger.debug(
                    f"CHECKPOINT 3 PASS: Image order correct for batch {batch_id} ({len(expected_sequence)} checked)"
                )
            else:
                # FAIL: verbose (show details for debugging)
                logger.error("=" * 70)
                logger.error(f"CHECKPOINT 3 FAIL: IMAGE ORDER MISMATCH in batch {batch_id}")
                logger.error("=" * 70)
                logger.error(f"  Expected: {expected_sequence[:10]}")
                logger.error(f"  Actual:   {actual_sequence[:10]}")
                logger.error("  This indicates manifest.txt was not used or is incorrect!")
                logger.error("=" * 70)
                # FATAL ERROR: Stop processing to prevent contamination
                raise RuntimeError(
                    f"CHECKPOINT 3 FATAL: Image order contamination detected in batch {batch_id}. "
                    f"Expected sequence doesn't match actual. This will cause results to be assigned to wrong PDFs. "
                    f"Stopping to prevent data corruption."
                )

        # ═══════════════════════════════════════════════════════════════════
        # OPTIMIZATION: Use remote results directly if available
        # ═══════════════════════════════════════════════════════════════════
        # If all PDFs were processed remotely via infer_pdfs, we already have
        # complete results from the remote server. Skip the redundant upload+inference!
        if remote_results and len(all_images) == 0:
            # All PDFs processed remotely - use results directly, NO UPLOAD!
            logger.info("=" * 70)
            logger.info("FULLY REMOTE PATH: Using results from remote infer_pdfs")
            logger.info("=" * 70)
            logger.info(f"  PDFs processed remotely:  {len(remote_results)}")
            logger.info(f"  Local extraction skipped: {len(pdfs_on_remote_list)}")
            logger.info("  Upload eliminated:        100%")
            logger.info("=" * 70)

            pdf_results = remote_results  # Already in correct format!
            results = []  # No image-level results from upload path
            inference_time = remote_processing_time
        else:
            # Traditional path: upload and infer local images
            # Only run traditional detection if we have images to process
            if len(all_images) > 0:
                if self.verbose:
                    print(f"  Running remote inference on {len(all_images)} images...")

                # Run remote inference on all images
                inference_start = time.time()
                results = self.remote_detector.detect(
                    all_images,
                    output_dir=batch_dir / "results",
                    cleanup=self.remote_cleanup,  # Configurable: default False for debugging
                    auto_git_commit=auto_git_commit,
                    gpu_batch_size=gpu_batch_size,
                )
                inference_time = time.time() - inference_start

                if self.verbose:
                    print(
                        f"  ✓ Inference complete: {inference_time:.1f}s ({len(all_images)/inference_time:.1f} images/s)"
                    )
            else:
                # No images to process
                logger.info("No local images - all PDFs processed via remote persistent server")
                results = []
                inference_time = remote_processing_time  # Use remote processing time

            # Validate inference returned correct number of results
            if len(results) != len(all_images):
                missing_count = len(all_images) - len(results)
                logger.error("=" * 70)
                logger.error(f"INFERENCE MISMATCH IN BATCH {batch_id}")
                logger.error("=" * 70)
                logger.error(f"Expected: {len(all_images)} results")
                logger.error(f"Received: {len(results)} results")
                logger.error(f"Missing:  {missing_count} results ({missing_count/len(all_images)*100:.1f}%)")
                logger.error("")
                logger.error(f"This indicates inference failed for {missing_count} images.")
                logger.error("Check inference.err logs for details (likely OOM or model errors).")
                logger.error("Affected PDFs in this batch will have incomplete or empty results.")
                logger.error("=" * 70)

            # Group results by PDF (traditional path only)
            pdf_results = {}
            result_idx = 0

        # Only slice results if using traditional upload path
        # (remote path already has pdf_results populated)
        if not (remote_results and len(all_images) == 0):
            for pdf_path in pdf_batch:
                # Check if PDF extraction succeeded
                if pdf_path.name not in pdf_page_counts:
                    # PDF extraction failed - skip this PDF, don't assign empty results
                    # The PDF will not be in batch_results, preventing caching attempts
                    logger.debug(f"Skipping {pdf_path.name} - not in pdf_page_counts (extraction failed)")
                    continue

                num_pages = pdf_page_counts[pdf_path.name]

                # Additional validation: Ensure we don't create empty results
                # This should NEVER happen now - kept as defense-in-depth
                if num_pages == 0:
                    logger.error(f"CRITICAL BUG: {pdf_path.name} has pdf_page_counts = 0")
                    logger.error("  This should have been caught earlier in the pipeline!")
                    logger.error("  Please investigate why this PDF reached this point")
                    continue

                pdf_result_list = results[result_idx : result_idx + num_pages]

                # ═══════════════════════════════════════════════════════════════════
                # VALIDATION: Verify filenames match PDF stem (catch contamination bug)
                # ═══════════════════════════════════════════════════════════════════
                expected_pdf_stem = pdf_path.stem
                filename_mismatches = []

                for page_idx, result in enumerate(pdf_result_list):
                    # Each result should have filename containing the PDF stem
                    if hasattr(result, "filename") and result.filename:
                        if expected_pdf_stem not in result.filename:
                            filename_mismatches.append(
                                {
                                    "page_idx": page_idx,
                                    "expected_stem": expected_pdf_stem,
                                    "actual_filename": result.filename,
                                    "result_global_idx": result_idx + page_idx,
                                }
                            )

                if filename_mismatches:
                    logger.error("")
                    logger.error(f"{'═' * 70}")
                    logger.error(f"CONTAMINATION DETECTED: Filename mismatch in batch {batch_id}")
                    logger.error(f"{'═' * 70}")
                    logger.error(f"PDF: {pdf_path.name}")
                    logger.error(f"Expected stem: {expected_pdf_stem}")
                    logger.error(f"Result slice: [{result_idx}:{result_idx + num_pages}]")
                    logger.error(f"Mismatches found: {len(filename_mismatches)}/{num_pages} pages")
                    logger.error("")

                    # Log first few mismatches for diagnosis
                    for mm in filename_mismatches[:5]:
                        logger.error(f"  Page {mm['page_idx'] + 1}:")
                        logger.error(f"    Expected stem:    {mm['expected_stem']}")
                        logger.error(f"    Actual filename:  {mm['actual_filename']}")
                        logger.error(f"    Result index:     {mm['result_global_idx']}")

                    if len(filename_mismatches) > 5:
                        logger.error(f"  ... and {len(filename_mismatches) - 5} more")

                    logger.error("")
                    logger.error("This indicates the result slicing is incorrect!")
                    logger.error("Possible causes:")
                    logger.error("  - pdf_page_counts is wrong for earlier PDFs")
                    logger.error("  - pdf_batch iteration order changed")
                    logger.error("  - all_images order doesn't match pdf_batch order")
                    logger.error(f"{'═' * 70}")

                    # Log forensic info for debugging
                    logger.debug("")
                    logger.debug("FORENSIC INFO:")
                    logger.debug(f"  Current PDF: {pdf_path.name}")
                    logger.debug(f"  Current result_idx: {result_idx}")
                    logger.debug(f"  Current num_pages: {num_pages}")
                    logger.debug(f"  Total results: {len(results)}")
                    logger.debug(f"  pdf_page_counts keys: {list(pdf_page_counts.keys())[:10]}")

                    # CRITICAL: Don't cache contaminated results!
                    batch_failed[pdf_path.name] = "contamination_detected"
                    if manifest_path:
                        _log_pdf_status(
                            manifest_path=manifest_path,
                            pdf_name=pdf_path.name,
                            status="contamination_detected",
                            pages_extracted=num_pages,
                            pages_detected=len(pdf_result_list),
                            num_diagrams=sum(r.count for r in pdf_result_list),
                            error_type="FilenameMismatch",
                            error_message=f"{len(filename_mismatches)}/{num_pages} pages have wrong filenames",
                        )

                    # Skip this PDF - don't add to pdf_results (prevents caching)
                    result_idx += len(pdf_result_list)
                    continue

                # CHECKPOINT 6: Result Count Validation
                # Verify we got expected number of results (= num_pages)
                if len(pdf_result_list) != num_pages:
                    logger.warning("")
                    logger.warning(f"⚠️  RESULT COUNT MISMATCH: {pdf_path.name}")
                    logger.warning(f"  Expected: {num_pages} results (= page count)")
                    logger.warning(f"  Actual:   {len(pdf_result_list)} results")
                    logger.warning(f"  Difference: {abs(len(pdf_result_list) - num_pages)}")
                    logger.warning("")
                    # This is a warning, not fatal - inference might skip some pages

                # CHECKPOINT 7: Page Number Sequencing
                # Verify results are ordered by page number within each PDF
                if len(pdf_result_list) > 1:
                    page_numbers = [r.page_number for r in pdf_result_list]
                    # Filter out None values before sorting (some results may lack page numbers)
                    valid_page_numbers = [p for p in page_numbers if p is not None]
                    if valid_page_numbers:
                        expected_order = sorted(valid_page_numbers)
                        if valid_page_numbers != expected_order:
                            logger.warning("")
                            logger.warning(f"⚠️  PAGE ORDER ISSUE: {pdf_path.name}")
                            logger.warning(f"  Page numbers: {page_numbers[:10]}")
                            logger.warning(f"  Expected:     {expected_order[:10]}")
                            logger.warning("  Results may be out of order!")
                            logger.warning("")

                # CHECKPOINT 8: Result Consistency
                # Basic sanity checks on detection results
                if len(pdf_result_list) > 0:
                    # Check for all-zero detections (suspicious)
                    all_zero = all(r.count == 0 for r in pdf_result_list)
                    if all_zero and len(pdf_result_list) > 5:
                        logger.debug(f"  Note: {pdf_path.name} has 0 diagrams on all {len(pdf_result_list)} pages")

                    # Check for unreasonably high detections (possible duplicate)
                    max_count = max(r.count for r in pdf_result_list)
                    if max_count > 20:
                        logger.warning(f"  ⚠️  High diagram count: {pdf_path.name} has {max_count} diagrams on one page")

                # Log success for first few PDFs (helps verify correct mapping)
                if len(pdf_results) < 3:
                    logger.debug(f"✓ All validation checkpoints passed for {pdf_path.name}")
                    if len(pdf_result_list) > 0:
                        logger.debug(f"  Sample filenames: {[r.filename for r in pdf_result_list[:2]]}")
                        logger.debug(f"  Page numbers: {[r.page_number for r in pdf_result_list[:3]]}")

                # CRITICAL: Check if slice returned empty results
                # This happens when inference returned fewer results than expected
                if len(pdf_result_list) == 0:
                    logger.error("")
                    logger.error(f"INFERENCE FAILURE: No results for {pdf_path.name}")
                    logger.error(f"  Pages extracted: {num_pages}")
                    logger.error("  Pages received:  0")
                    logger.error(f"  Result index:    {result_idx} (expected results at this index)")
                    logger.error(f"  Total results:   {len(results)} (inference stopped before this PDF)")
                    logger.error("")
                    logger.error("  Likely causes:")
                    logger.error("    - Out of memory (OOM) during inference")
                    logger.error("    - Model crash on corrupt/unusual images")
                    logger.error("    - Remote process killed")
                    logger.error("")
                    logger.error("  This PDF will be reprocessed on next run")

                    batch_failed[pdf_path.name] = "inference_failed"
                    if manifest_path:
                        _log_pdf_status(
                            manifest_path=manifest_path,
                            pdf_name=pdf_path.name,
                            status="inference_failed",
                            pages_extracted=num_pages,
                            pages_detected=0,
                            num_diagrams=0,
                            error_type="InferenceMismatch",
                            error_message=f"Inference stopped before this PDF (expected at index {result_idx}, but only {len(results)} results total)",
                        )

                    # Don't add to pdf_results - let it be reprocessed
                    continue

                # Check if inference returned fewer pages than expected
                if len(pdf_result_list) < num_pages:
                    logger.warning(f"PARTIAL RESULTS: {pdf_path.name} got {len(pdf_result_list)}/{num_pages} pages")
                    logger.warning("  Some pages failed inference")

                    # Log partial result as inference_failed
                    batch_failed[pdf_path.name] = "inference_failed"
                    total_diagrams = sum(r.count for r in pdf_result_list)
                    if manifest_path:
                        _log_pdf_status(
                            manifest_path=manifest_path,
                            pdf_name=pdf_path.name,
                            status="inference_failed",
                            pages_extracted=num_pages,
                            pages_detected=len(pdf_result_list),
                            num_diagrams=total_diagrams,
                            error_type="PartialResults",
                            error_message=f"Received {len(pdf_result_list)}/{num_pages} pages (some pages failed inference)",
                        )

                # Add page numbers
                for page_num, result in enumerate(pdf_result_list, start=1):
                    result.page_number = page_num

                pdf_results[pdf_path.name] = pdf_result_list
                result_idx += len(pdf_result_list)  # Use actual results received, not expected

                # Log successful processing (only if not already logged as partial)
                if manifest_path and len(pdf_result_list) == num_pages:
                    total_diagrams = sum(r.count for r in pdf_result_list)
                    status = "success" if total_diagrams > 0 else "success_no_diagrams"

                    _log_pdf_status(
                        manifest_path=manifest_path,
                        pdf_name=pdf_path.name,
                        status=status,
                        pages_extracted=num_pages,
                        pages_detected=len(pdf_result_list),
                        num_diagrams=total_diagrams,
                        error_type="",
                        error_message="",
                    )

        # Merge remote_results into pdf_results (for batches with PDFs processed remotely)
        if remote_results:
            logger.info(f"Merging {len(remote_results)} remote PDF results into batch results")
            pdf_results.update(remote_results)

        # Query GPU memory after inference
        gpu_mem_used, gpu_mem_free = self.remote_detector.get_gpu_memory()

        # Log batch timing breakdown
        total_time = extraction_time + inference_time
        total_pages = sum(len(page_results) for page_results in pdf_results.values())
        total_diagrams = sum(r.count for page_results in pdf_results.values() for r in page_results)

        logger.info(f"[BATCH {batch_id}] Timing breakdown:")
        if total_time > 0:
            logger.info(f"  Extraction:  {extraction_time:6.1f}s ({100*extraction_time/total_time:5.1f}%)")
            logger.info(f"  Inference:   {inference_time:6.1f}s ({100*inference_time/total_time:5.1f}%)")
            logger.info(f"  Total:       {total_time:6.1f}s")
            logger.info(f"  Throughput:  {total_pages/total_time:.1f} pages/s")
        else:
            logger.warning("  Total time is zero - batch processing may have failed")
        logger.info(f"  Results:     {len(pdf_results)} PDFs, {total_pages} pages, {total_diagrams} diagrams")

        # Cleanup batch directory
        shutil.rmtree(batch_dir, ignore_errors=True)

        return pdf_results, extraction_time, inference_time, gpu_mem_used, gpu_mem_free, batch_failed

    def detect_pdfs(
        self,
        pdf_paths: Union[Path, List[Path]],
        output_dir: Optional[Path] = None,
        use_cache: bool = True,
        force_reprocess: bool = False,
        auto_git_commit: bool = False,
        timing_log: Optional[Path] = None,
        debug_max_batches: Optional[int] = None,
        gpu_batch_size: int = 32,
        manifest_path: Optional[Path] = None,
        preserve_logs_dir: Optional[Path] = None,
    ) -> Dict[str, List[DetectionResult]]:
        """
        Process PDFs with remote inference and local caching.

        Args:
            pdf_paths: Single PDF, directory of PDFs, or list of PDF paths
            output_dir: Where to save results (None = don't save)
            use_cache: Use cached results if available
            force_reprocess: Force reprocessing even if cached
            auto_git_commit: Automatically git commit the run config
            timing_log: Path to CSV file for logging timing data (None = no logging)
            preserve_logs_dir: If set, copy inference.log and inference.err from each
                image sub-batch into this dir (pdf_batch_id / sub_batch_id / inference.*)
                for debugging. Created if missing.

        Returns:
            Dict mapping PDF filename to list of DetectionResult (one per page)
        """
        import logging

        logger = logging.getLogger(__name__)

        # Parse input
        if isinstance(pdf_paths, Path):
            if pdf_paths.is_dir():
                pdf_list = sorted(pdf_paths.glob("*.pdf"))
            else:
                pdf_list = [pdf_paths]
        elif isinstance(pdf_paths, list):
            pdf_list = [Path(p) for p in pdf_paths]
        else:
            pdf_list = [Path(pdf_paths)]

        if not pdf_list:
            raise ValueError("No PDF files found")

        # Track timing for logging
        run_start_time = time.time()
        total_extraction_time = 0.0
        total_inference_time = 0.0

        if self.verbose:
            print(f"\n{'='*60}")
            print("PDF REMOTE INFERENCE (detect_pdfs)")
            print(f"{'='*60}")
            print(f"PDFs: {len(pdf_list)}")
            print(f"Batch size: {self.batch_size} PDFs/batch")
            print(f"Model: {self.model}")
            print(f"Remote: {self.config.ssh_target}")
            print(f"Local cache: {'enabled' if use_cache else 'disabled'}")
            print(f"{'='*60}\n")

        # Check cache and filter
        to_process = []
        cached_results = {}

        if use_cache and not force_reprocess:
            if self.verbose:
                print("Checking cache...")

            for pdf_path in pdf_list:
                cached = None
                if self.cache:
                    cached = self.cache.get(
                        pdf_path,
                        model=self.model,
                        confidence=self.confidence,
                        iou=self.iou,
                        dpi=self.dpi,
                        imgsz=self.imgsz,
                    )
                if cached is not None:
                    # Convert cached dicts back to DetectionResult objects
                    cached_results[pdf_path.name] = [DetectionResult.from_dict(result_dict) for result_dict in cached]
                    # if self.verbose:
                    #     print(f"  ✓ {pdf_path.name} (cached)")
                else:
                    to_process.append(pdf_path)
                    # if self.verbose:
                    #     print(f"  • {pdf_path.name} (needs processing)")
        else:
            to_process = pdf_list

        if self.verbose:
            print(f"\nProcessing {len(to_process)} PDFs ({len(cached_results)} cached)...\n")

        # Process in batches (skip if all cached)
        all_results = cached_results.copy()

        # Accumulators for failure summary across all batches
        all_failed_pdfs: Dict[str, str] = {}  # pdf_name -> status

        if not to_process:
            if self.verbose:
                print("✓ All PDFs cached, no processing needed!")
        else:
            num_batches = (len(to_process) + self.batch_size - 1) // self.batch_size

            # Limit batches for debugging if requested
            if debug_max_batches is not None:
                num_batches = min(num_batches, debug_max_batches)
                if self.verbose:
                    print(f"⚠ DEBUG MODE: Processing only {num_batches} batch(es)")

            with tempfile.TemporaryDirectory() as temp_dir:
                work_dir = Path(temp_dir)

                for batch_idx in range(num_batches):
                    batch_start = batch_idx * self.batch_size
                    batch_end = min((batch_idx + 1) * self.batch_size, len(to_process))
                    batch_pdfs = to_process[batch_start:batch_end]
                    batch_id = f"batch_{batch_idx:04d}"

                    if self.verbose:
                        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ({len(batch_pdfs)} PDFs) ---")

                    batch_dir = work_dir / batch_id
                    try:
                        # Process batch (detect() may raise on sub-batch mismatch; we still copy logs in finally)
                        (
                            batch_results,
                            batch_extraction_time,
                            batch_inference_time,
                            gpu_mem_used,
                            gpu_mem_free,
                            batch_failed,
                        ) = self._process_pdf_batch(
                            batch_pdfs, batch_id, work_dir, auto_git_commit, gpu_batch_size, manifest_path
                        )
                        all_failed_pdfs.update(batch_failed)
                    finally:
                        # Copy inference logs even when _process_pdf_batch raises (e.g. sub-batch mismatch),
                        # so we have inference.err/log for the failing sub-batch for debugging
                        if preserve_logs_dir:
                            _copy_inference_logs(
                                batch_dir=batch_dir,
                                run_id=self.remote_detector.run_id,
                                preserve_logs_dir=Path(preserve_logs_dir),
                                pdf_batch_id=batch_id,
                            )

                    # Check for inference error logs (before temp dir cleanup)
                    err_files = list(batch_dir.glob("**/inference.err"))
                    for err_file in err_files:
                        if err_file.stat().st_size > 0:
                            logger.error("=" * 70)
                            logger.error(f"REMOTE INFERENCE ERRORS - {err_file.parent.name}")
                            logger.error("=" * 70)
                            with open(err_file, "r") as f:
                                for line in f:
                                    logger.error(f"  {line.rstrip()}")
                            logger.error("=" * 70)

                    # Accumulate timing
                    total_extraction_time += batch_extraction_time
                    total_inference_time += batch_inference_time

                    # Cache results
                    if use_cache:
                        for pdf_path in batch_pdfs:
                            # Skip failed PDFs (not in batch_results)
                            if pdf_path.name not in batch_results:
                                import logging

                                logger = logging.getLogger(__name__)
                                pdf_size_mb = pdf_path.stat().st_size / 1024 / 1024
                                if self.max_pdf_size_mb is not None and pdf_size_mb > self.max_pdf_size_mb:
                                    logger.info(
                                        f"Not caching {pdf_path.name} - skipped (size limit: {pdf_size_mb:.1f} MB > {self.max_pdf_size_mb} MB)"
                                    )
                                elif self.max_pdf_size_pages is not None:
                                    import fitz

                                    with fitz.open(pdf_path) as doc:
                                        num_pages = doc.page_count
                                    if num_pages > self.max_pdf_size_pages:
                                        logger.info(
                                            f"Not caching {pdf_path.name} - skipped (page limit: {num_pages} pages > {self.max_pdf_size_pages} pages)"
                                        )
                                    else:
                                        logger.warning(f"⚠ Not caching {pdf_path.name} - extraction/inference failed")
                                        logger.warning("  This PDF will be reprocessed on next run")
                                else:
                                    logger.warning(f"⚠ Not caching {pdf_path.name} - extraction/inference failed")
                                    logger.warning("  This PDF will be reprocessed on next run")
                                continue

                            # Convert DetectionResult objects to dicts for JSON serialization
                            results_list = batch_results[pdf_path.name]
                            results_dicts = [asdict(r) for r in results_list]

                            # Cache results (if cache enabled)
                            if self.cache:
                                self.cache.set(
                                    pdf_path,
                                    model=self.model,
                                    confidence=self.confidence,
                                    iou=self.iou,
                                    dpi=self.dpi,
                                    imgsz=self.imgsz,
                                    results=results_dicts,
                                )

                    # Add to results
                    all_results.update(batch_results)

                    # Count detections in this batch
                    batch_detections = sum(sum(r.count for r in results) for results in batch_results.values())

                    if self.verbose:
                        # Calculate detailed statistics
                        batch_pages = sum(len(results) for results in batch_results.values())
                        pages_with_diagrams = sum(
                            sum(1 for r in results if r.has_diagram) for results in batch_results.values()
                        )
                        pages_pct = (pages_with_diagrams / batch_pages * 100) if batch_pages > 0 else 0
                        avg_diagrams_per_page = batch_detections / batch_pages if batch_pages > 0 else 0

                        print(f"✓ Batch complete: {batch_detections} diagrams found")
                        print(f"  Pages: {batch_pages} total, {pages_with_diagrams} with diagrams ({pages_pct:.1f}%)")
                        print(f"  Average: {avg_diagrams_per_page:.2f} diagrams/page")

                        # Show failure counts if any PDFs failed in this batch
                        if batch_failed:
                            n_extraction = sum(1 for s in batch_failed.values() if s == "extraction_failed")
                            n_inference = sum(1 for s in batch_failed.values() if s == "inference_failed")
                            if n_extraction:
                                print(f"  ⚠ Extraction failed: {n_extraction} PDF(s)")
                            if n_inference:
                                print(f"  ⚠ Inference failed:  {n_inference} PDF(s)")
                            failed_names = list(batch_failed.keys())
                            print(
                                f"  Failed: {', '.join(failed_names[:10])}"
                                + (f" … +{len(failed_names)-10} more" if len(failed_names) > 10 else "")
                            )

                        # Estimate remaining time
                        pdfs_processed = batch_end
                        pdfs_remaining = len(to_process) - pdfs_processed
                        elapsed = time.time() - run_start_time

                        if pdfs_processed > 0 and pdfs_remaining > 0:
                            avg_time_per_pdf = elapsed / pdfs_processed
                            est_remaining_secs = avg_time_per_pdf * pdfs_remaining

                            # Format as hours:minutes:seconds
                            est_hours = int(est_remaining_secs // 3600)
                            est_mins = int((est_remaining_secs % 3600) // 60)
                            est_secs = int(est_remaining_secs % 60)

                            if est_hours > 0:
                                eta_str = f"{est_hours}h {est_mins}m {est_secs}s"
                            elif est_mins > 0:
                                eta_str = f"{est_mins}m {est_secs}s"
                            else:
                                eta_str = f"{est_secs}s"

                            print(f"  ETA: {eta_str} ({pdfs_remaining} PDFs remaining)")

                    # Log timing for this batch incrementally
                    if timing_log:
                        batch_pages = sum(len(results) for results in batch_results.values())
                        batch_total_time = batch_extraction_time + batch_inference_time

                        _append_timing_to_csv(
                            log_path=timing_log,
                            timestamp=datetime.now(),
                            num_pdfs=len(batch_pdfs),
                            num_pages=batch_pages,
                            extraction_time=batch_extraction_time,
                            inference_time=batch_inference_time,
                            total_time=batch_total_time,
                            model=self.model,
                            batch_size=self.batch_size,
                            image_batch_size=self.remote_detector.batch_size,
                            remote_host=self.config.host,
                            num_cached=0,  # This batch had no cached results
                            num_detections=batch_detections,
                            max_workers=self.max_workers,
                            tensorrt=self.tensorrt,
                            gpu_mem_used_mb=gpu_mem_used,
                            gpu_mem_free_mb=gpu_mem_free,
                        )

        # Save results if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for pdf_name, results in all_results.items():
                pdf_output = output_dir / f"{Path(pdf_name).stem}_results.json"
                data = [r.to_dict() for r in results]
                save_json(data, pdf_output)

            if self.verbose:
                print(f"\n✓ Results saved to {output_dir}")

        # Calculate totals for summary and logging
        total_pages = sum(len(results) for results in all_results.values())
        total_diagrams = sum(sum(r.count for r in results) for results in all_results.values())

        # Print summary
        if self.verbose:
            total_with_diagrams = sum(sum(1 for r in results if r.has_diagram) for results in all_results.values())

            print(f"\n{'='*60}")
            print("PDF REMOTE INFERENCE COMPLETE")
            print(f"{'='*60}")
            print(f"Total PDFs: {len(all_results)}")
            print(f"Total pages: {total_pages:,}")
            pct = total_with_diagrams / total_pages * 100 if total_pages > 0 else 0
            print(f"Pages with diagrams: {total_with_diagrams:,} ({pct:.1f}%)")
            print(f"Total diagrams: {total_diagrams:,}")

            # Calculate and show distribution of diagrams per page
            from collections import Counter

            diagram_counts = []
            confidences = []
            for results in all_results.values():
                for r in results:
                    diagram_counts.append(r.count)
                    # Collect confidence scores from detections
                    if r.detections:
                        confidences.extend([d.confidence for d in r.detections])

            count_dist = Counter(diagram_counts)
            print("\nDiagram distribution per page:")
            for count in sorted(count_dist.keys()):
                num_pages = count_dist[count]
                pct = num_pages / total_pages * 100 if total_pages > 0 else 0
                print(f"  {count} diagrams: {num_pages:,} pages ({pct:.1f}%)")

            # Show confidence statistics
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                min_conf = min(confidences)
                max_conf = max(confidences)
                print("\nConfidence scores:")
                print(f"  Average: {avg_conf:.3f}")
                print(f"  Range: {min_conf:.3f} - {max_conf:.3f}")
                print(f"  Threshold: {self.confidence}")

            # Flag unusual patterns that might indicate issues
            max_diagrams = max(diagram_counts) if diagram_counts else 0
            if max_diagrams > 20:
                print(f"\n⚠ WARNING: Found page(s) with {max_diagrams} diagrams")
                print("  This may indicate false positives or threshold issues")

            high_diagram_pages = [c for c in diagram_counts if c > 10]
            if high_diagram_pages:
                pct_high = len(high_diagram_pages) / len(diagram_counts) * 100
                print(f"  Pages with >10 diagrams: {len(high_diagram_pages)} ({pct_high:.1f}%)")

            if use_cache and self.cache:
                cache_stats = self.cache.stats()
                num_pdfs = cache_stats["num_pdfs"]
                size_mb = cache_stats["size_mb"]
                print(f"Cache: {num_pdfs} PDFs ({size_mb:.1f} MB)")

            # Failure breakdown (only shown when there were failures)
            if all_failed_pdfs:
                total_processed = len(all_results) + len(all_failed_pdfs)
                n_extraction = sum(1 for s in all_failed_pdfs.values() if s == "extraction_failed")
                n_inference = sum(1 for s in all_failed_pdfs.values() if s == "inference_failed")
                print("\nFailure summary:")
                print(f"  Successful:        {len(all_results):,} ({len(all_results)/total_processed*100:.1f}%)")
                if n_extraction:
                    print(f"  Extraction failed: {n_extraction:,} ({n_extraction/total_processed*100:.1f}%)")
                if n_inference:
                    print(f"  Inference failed:  {n_inference:,} ({n_inference/total_processed*100:.1f}%)")

            print(f"{'='*60}\n")

        # Write detection_failures.txt when there were failures and manifest tracking is active
        if all_failed_pdfs and manifest_path:
            failures_path = Path(manifest_path).parent / "detection_failures.txt"
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            with open(failures_path, "w") as f:
                f.write(f"# Detection failures — {datetime.now().isoformat()}\n")
                f.write(f"# {len(all_failed_pdfs)} failed PDF(s)\n")
                f.write("# status: extraction_failed | inference_failed\n\n")
                for pdf_name, status in sorted(all_failed_pdfs.items()):
                    f.write(f"{status}\t{pdf_name}\n")
            if self.verbose:
                print(f"✓ Failed PDFs saved to: {failures_path}")

        # Log timing data to CSV if requested (always log, even for fully cached runs)
        if timing_log:
            run_total_time = time.time() - run_start_time

            # Query final GPU memory state
            final_gpu_mem_used, final_gpu_mem_free = self.remote_detector.get_gpu_memory()

            _append_timing_to_csv(
                log_path=timing_log,
                timestamp=datetime.now(),
                num_pdfs=len(to_process),
                num_pages=total_pages,
                extraction_time=total_extraction_time,
                inference_time=total_inference_time,
                total_time=run_total_time,
                model=self.model,
                batch_size=self.batch_size,
                image_batch_size=self.remote_detector.batch_size,
                remote_host=self.config.host,
                num_cached=len(cached_results),
                num_detections=total_diagrams,
                max_workers=self.max_workers,
                tensorrt=self.tensorrt,
                gpu_mem_used_mb=final_gpu_mem_used,
                gpu_mem_free_mb=final_gpu_mem_free,
            )

            if self.verbose:
                print(f"✓ Timing logged to: {timing_log}")

        return all_results

    def get_cached_results(self, pdf_paths: Union[Path, List[Path]]) -> Dict[str, Optional[List[DetectionResult]]]:
        """
        Retrieve cached results without processing.

        Args:
            pdf_paths: Single PDF, directory of PDFs, or list of PDF paths

        Returns:
            Dict mapping PDF filename to list of DetectionResult (None if not cached)

        Example:
            # Check what's already cached
            cached = detector.get_cached_results(pdf_paths)
            missing = [pdf for pdf, result in cached.items() if result is None]
            print(f"Need to process: {len(missing)} PDFs")
        """
        # Parse input
        if isinstance(pdf_paths, Path):
            if pdf_paths.is_dir():
                pdf_list = sorted(pdf_paths.glob("*.pdf"))
            else:
                pdf_list = [pdf_paths]
        elif isinstance(pdf_paths, list):
            pdf_list = [Path(p) for p in pdf_paths]
        else:
            pdf_list = [Path(pdf_paths)]

        results = {}
        for pdf_path in pdf_list:
            cached = None
            if self.cache:
                cached = self.cache.get(
                    pdf_path,
                    model=self.model,
                    confidence=self.confidence,
                    iou=self.iou,
                    dpi=self.dpi,
                    imgsz=self.imgsz,
                )

            if cached is not None:
                # Convert cached dicts back to DetectionResult objects
                results[pdf_path.name] = [DetectionResult.from_dict(result_dict) for result_dict in cached]
            else:
                results[pdf_path.name] = None

        return results

    def clear_cache(self) -> None:
        """Clear cache."""
        if self.cache:
            self.cache.clear()
            if self.verbose:
                print("✓ Cache cleared")
        else:
            if self.verbose:
                print("ℹ Cache is disabled")
