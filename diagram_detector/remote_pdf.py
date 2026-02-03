"""
Enhanced SSH Remote Inference - PDF-Optimized with Local Caching

Optimized for:
- PDF files as processing unit (not individual images)
- Local network (gigabit speeds)
- SQLite-based caching with gzip compression (thread-safe)
- Parallel local PDF extraction (CPU bottleneck)
- Local PDF → image extraction (less network traffic)
"""

from pathlib import Path
from typing import List, Union, Optional, Dict
import tempfile
import shutil
import time
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from .models import DetectionResult
from .utils import convert_pdf_to_images, save_json
from .remote_ssh import RemoteConfig, SSHRemoteDetector
from .cache import DetectionCache


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
        'timestamp': timestamp.isoformat(),
        'num_pdfs': num_pdfs,
        'num_pages': num_pages,
        'num_detections': num_detections,
        'detections_per_page': round(detections_per_page, 3),
        'num_cached': num_cached,
        'extraction_time': round(extraction_time, 2),
        'inference_time': round(inference_time, 2),
        'total_time': round(total_time, 2),
        'pages_per_sec': round(pages_per_sec, 2),
        'extraction_pct': round(extraction_pct, 1),
        'inference_pct': round(inference_pct, 1),
        'model': model,
        'batch_size': batch_size,
        'image_batch_size': image_batch_size,
        'max_workers': max_workers,
        'tensorrt': tensorrt,
        'remote_host': remote_host,
        'gpu_mem_used_mb': gpu_mem_used_mb,
        'gpu_mem_free_mb': gpu_mem_free_mb,
    }

    # Write to CSV
    with open(log_path, 'a', newline='') as f:
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
            pdf_batch_id
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
        'pdf_name': pdf_name,
        'status': status,
        'pages_extracted': pages_extracted,
        'pages_detected': pages_detected,
        'num_diagrams': num_diagrams,
        'error_type': error_type,
        'error_message': error_message,
        'timestamp': datetime.now().isoformat(),
    }

    # Write to CSV with file locking
    with open(manifest_path, 'a', newline='') as f:
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
    from PIL import Image

    images = convert_pdf_to_images(pdf_path, dpi=dpi, verbose=False, show_progress=show_progress)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths: List[Path] = []
    for page_num, img_array in enumerate(images, start=1):
        img_path = output_dir / f"page_{page_num:04d}.jpg"
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
        use_persistent_server: bool = False,
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
            use_persistent_server: Keep model resident in GPU memory across sub-batches (~3-5s reload saved per batch)
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

        # Initialize cache with proper parameter tracking
        self.cache = DetectionCache(cache_dir=cache_dir, compression=True)

        if self.verbose:
            cache_stats = self.cache.stats()
            print(
                f"Cache: {cache_stats['num_pdfs']} PDFs, "
                f"{cache_stats['total_pages']:,} pages "
                f"({cache_stats['size_mb']:.1f} MB compressed)"
            )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup SSH resources."""
        self.cleanup()
        return False

    def __del__(self):
        """Destructor - cleanup SSH resources as safety net."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during destruction

    def cleanup(self):
        """Clean up SSH control socket and connections."""
        if hasattr(self, 'remote_detector') and self.remote_detector:
            self.remote_detector.cleanup()

    def _extract_pdf_pages(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        """Extract PDF pages to images locally (delegates to shared helper)."""
        return extract_pdf_to_jpegs(pdf_path, output_dir, self.dpi, show_progress=self.verbose)

    def _extract_pdfs_parallel(
        self, pdf_batch: List[Path], batch_dir: Path
    ) -> tuple[Dict[str, List[Path]], float]:
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
                pdf_dir = batch_dir / pdf_path.stem
                image_paths = self._extract_pdf_pages(pdf_path, pdf_dir)
                pdf_images[pdf_path.name] = image_paths
        else:
            # Parallel extraction
            logger.debug(f"[EXTRACTION] Creating ThreadPoolExecutor with {self.max_workers} workers")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit extraction tasks, skipping PDFs that exceed size limit
                future_to_pdf = {}
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

                    # Warn about large files (but still process them)
                    if pdf_size_mb > 50:
                        logger.warning(f"Large PDF file: {pdf_path.name} ({pdf_size_mb:.1f} MB)")

                    pdf_dir = batch_dir / pdf_path.stem
                    future = executor.submit(self._extract_pdf_pages, pdf_path, pdf_dir)
                    future_to_pdf[future] = pdf_path

                logger.debug(f"[EXTRACTION] Submitted {len(future_to_pdf)} extraction tasks, skipped {len(skipped_pdfs)} large PDFs")

                # Collect results as they complete with progress tracking
                from tqdm import tqdm
                completed_count = 0
                total_pages = 0
                extraction_start_time = time.time()

                if self.verbose:
                    progress = tqdm(total=len(future_to_pdf), desc="  Extracting", unit="PDF", leave=False,
                                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]')

                for future in as_completed(future_to_pdf):
                    pdf_path = future_to_pdf[future]
                    try:
                        # Timeout after 10 minutes per PDF (handles very large PDFs up to ~3000 pages)
                        # At 0.2 sec/page, 3000 pages = 600 seconds = 10 minutes
                        image_paths = future.result(timeout=600)
                        pdf_images[pdf_path.name] = image_paths
                        num_pages = len(image_paths) if image_paths else 0
                        total_pages += num_pages

                        # Warn if PDF is unusually large
                        if num_pages > 500:
                            logger.warning(f"Large PDF: {pdf_path.name} has {num_pages} pages")
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
        logger.debug(f"[EXTRACTION] Complete: {len(pdf_images)} PDFs, {total_pages_extracted} pages in {extraction_time:.1f}s ({total_pages_extracted/extraction_time:.1f} pages/s)")
        if failed_pdfs > 0:
            logger.warning(f"[EXTRACTION] {failed_pdfs}/{len(pdf_images)} PDFs failed extraction")

        if self.verbose:
            print(f"  ✓ Extraction complete: {total_pages_extracted} pages in {extraction_time:.1f}s ({total_pages_extracted/extraction_time:.1f} pages/s)")

        return pdf_images, extraction_time

    def _process_pdf_batch(
        self, pdf_batch: List[Path], batch_id: str, work_dir: Path,
        auto_git_commit: bool = False, gpu_batch_size: int = 32,
        manifest_path: Optional[Path] = None
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

        # Extract all PDFs in batch (parallel if enabled)
        pdf_images, extraction_time = self._extract_pdfs_parallel(pdf_batch, batch_dir)

        # Flatten to all images
        all_images = []
        pdf_page_counts = {}
        for pdf_name, image_paths in pdf_images.items():
            if image_paths is None:
                # Extraction failed or skipped - determine reason
                pdf_path = next((p for p in pdf_batch if p.name == pdf_name), None)
                if pdf_path:
                    pdf_size_mb = pdf_path.stat().st_size / 1024 / 1024
                    if self.max_pdf_size_mb is not None and pdf_size_mb > self.max_pdf_size_mb:
                        # Skipped due to size limit
                        logger.warning(f"Skipping {pdf_name} due to size limit ({pdf_size_mb:.1f} MB > {self.max_pdf_size_mb} MB)")
                        batch_failed[pdf_name] = "extraction_failed"
                        if manifest_path:
                            _log_pdf_status(
                                manifest_path=manifest_path,
                                pdf_name=pdf_name,
                                status="extraction_failed",
                                pages_extracted=0,
                                pages_detected=0,
                                num_diagrams=0,
                                error_type="SizeLimitExceeded",
                                error_message=f"PDF size ({pdf_size_mb:.1f} MB) exceeds limit of {self.max_pdf_size_mb} MB"
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
                                error_message="PDF extraction returned None (exception occurred)"
                            )
                else:
                    # Shouldn't happen, but handle gracefully
                    logger.warning(f"Skipping {pdf_name} due to extraction failure (PDF not found in batch)")

                continue

            if len(image_paths) == 0:
                # Extraction returned empty list - this should not happen!
                # If convert_pdf_to_images returns [], it means the PDF had pages
                # but all failed to render, which is very unusual
                logger.error(f"UNEXPECTED: Extraction returned 0 pages for {pdf_name}")
                logger.error(f"  This suggests all pages failed to render (very unusual)")
                logger.error(f"  The PDF should have been caught earlier if it had 0 pages")

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
                        error_message="PDF extraction returned 0 pages (all pages failed to render)"
                    )
                continue

            all_images.extend(image_paths)
            pdf_page_counts[pdf_name] = len(image_paths)

        if self.verbose:
            print(f"  ✓ Extraction complete: {len(all_images)} pages in {extraction_time:.1f}s ({len(all_images)/extraction_time:.1f} pages/s)")
            print(f"  Running remote inference on {len(all_images)} images...")

        # Run remote inference on all images
        inference_start = time.time()
        results = self.remote_detector.detect(
            all_images,
            output_dir=batch_dir / "results",
            cleanup=False,  # DEBUG: Disabled cleanup to inspect error logs
            auto_git_commit=auto_git_commit,
            gpu_batch_size=gpu_batch_size,
        )
        inference_time = time.time() - inference_start

        if self.verbose:
            print(f"  ✓ Inference complete: {inference_time:.1f}s ({len(all_images)/inference_time:.1f} images/s)")

        # Validate inference returned correct number of results
        if len(results) != len(all_images):
            missing_count = len(all_images) - len(results)
            logger.error(f"=" * 70)
            logger.error(f"INFERENCE MISMATCH IN BATCH {batch_id}")
            logger.error(f"=" * 70)
            logger.error(f"Expected: {len(all_images)} results")
            logger.error(f"Received: {len(results)} results")
            logger.error(f"Missing:  {missing_count} results ({missing_count/len(all_images)*100:.1f}%)")
            logger.error(f"")
            logger.error(f"This indicates inference failed for {missing_count} images.")
            logger.error(f"Check inference.err logs for details (likely OOM or model errors).")
            logger.error(f"Affected PDFs in this batch will have incomplete or empty results.")
            logger.error(f"=" * 70)

        # Group results by PDF
        pdf_results = {}
        result_idx = 0

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
                logger.error(f"  This should have been caught earlier in the pipeline!")
                logger.error(f"  Please investigate why this PDF reached this point")
                continue

            pdf_result_list = results[result_idx : result_idx + num_pages]

            # CRITICAL: Check if slice returned empty results
            # This happens when inference returned fewer results than expected
            if len(pdf_result_list) == 0:
                logger.error(f"")
                logger.error(f"INFERENCE FAILURE: No results for {pdf_path.name}")
                logger.error(f"  Pages extracted: {num_pages}")
                logger.error(f"  Pages received:  0")
                logger.error(f"  Result index:    {result_idx} (expected results at this index)")
                logger.error(f"  Total results:   {len(results)} (inference stopped before this PDF)")
                logger.error(f"")
                logger.error(f"  Likely causes:")
                logger.error(f"    - Out of memory (OOM) during inference")
                logger.error(f"    - Model crash on corrupt/unusual images")
                logger.error(f"    - Remote process killed")
                logger.error(f"")
                logger.error(f"  This PDF will be reprocessed on next run")

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
                        error_message=f"Inference stopped before this PDF (expected at index {result_idx}, but only {len(results)} results total)"
                    )

                # Don't add to pdf_results - let it be reprocessed
                continue

            # Check if inference returned fewer pages than expected
            if len(pdf_result_list) < num_pages:
                logger.warning(f"PARTIAL RESULTS: {pdf_path.name} got {len(pdf_result_list)}/{num_pages} pages")
                logger.warning(f"  Some pages failed inference")

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
                        error_message=f"Received {len(pdf_result_list)}/{num_pages} pages (some pages failed inference)"
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
                    error_message=""
                )

        # Query GPU memory after inference
        gpu_mem_used, gpu_mem_free = self.remote_detector.get_gpu_memory()

        # Log batch timing breakdown
        total_time = extraction_time + inference_time
        logger.info(f"[BATCH {batch_id}] Timing breakdown:")
        logger.info(f"  Extraction:  {extraction_time:6.1f}s ({100*extraction_time/total_time:5.1f}%)")
        logger.info(f"  Inference:   {inference_time:6.1f}s ({100*inference_time/total_time:5.1f}%)")
        logger.info(f"  Total:       {total_time:6.1f}s")
        logger.info(f"  Throughput:  {len(all_images)/total_time:.1f} pages/s")
        logger.info(f"  Results:     {len(pdf_results)} PDFs, {len(all_images)} pages, {sum(len(r) for r in pdf_results.values())} diagrams")

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
                    cached_results[pdf_path.name] = [
                        DetectionResult.from_dict(result_dict) for result_dict in cached
                    ]
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
                        batch_results, batch_extraction_time, batch_inference_time, gpu_mem_used, gpu_mem_free, batch_failed = self._process_pdf_batch(
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
                            logger.error(f"=" * 70)
                            logger.error(f"REMOTE INFERENCE ERRORS - {err_file.parent.name}")
                            logger.error(f"=" * 70)
                            with open(err_file, 'r') as f:
                                for line in f:
                                    logger.error(f"  {line.rstrip()}")
                            logger.error(f"=" * 70)

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
                                    logger.info(f"Not caching {pdf_path.name} - skipped (size limit: {pdf_size_mb:.1f} MB > {self.max_pdf_size_mb} MB)")
                                else:
                                    logger.warning(f"⚠ Not caching {pdf_path.name} - extraction/inference failed")
                                    logger.warning(f"  This PDF will be reprocessed on next run")
                                continue

                            # Convert DetectionResult objects to dicts for JSON serialization
                            results_list = batch_results[pdf_path.name]
                            results_dicts = [asdict(r) for r in results_list]

                            # cache.set() will now validate that results_dicts is not empty
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
                    batch_detections = sum(
                        sum(r.count for r in results) for results in batch_results.values()
                    )

                    if self.verbose:
                        # Calculate detailed statistics
                        batch_pages = sum(len(results) for results in batch_results.values())
                        pages_with_diagrams = sum(
                            sum(1 for r in results if r.has_diagram)
                            for results in batch_results.values()
                        )
                        pages_pct = (pages_with_diagrams / batch_pages * 100) if batch_pages > 0 else 0
                        avg_diagrams_per_page = batch_detections / batch_pages if batch_pages > 0 else 0

                        print(f"✓ Batch complete: {batch_detections} diagrams found")
                        print(f"  Pages: {batch_pages} total, {pages_with_diagrams} with diagrams ({pages_pct:.1f}%)")
                        print(f"  Average: {avg_diagrams_per_page:.2f} diagrams/page")

                        # Show failure counts if any PDFs failed in this batch
                        if batch_failed:
                            n_extraction = sum(1 for s in batch_failed.values() if s == "extraction_failed")
                            n_inference  = sum(1 for s in batch_failed.values() if s == "inference_failed")
                            if n_extraction:
                                print(f"  ⚠ Extraction failed: {n_extraction} PDF(s)")
                            if n_inference:
                                print(f"  ⚠ Inference failed:  {n_inference} PDF(s)")
                            failed_names = list(batch_failed.keys())
                            print(f"  Failed: {', '.join(failed_names[:10])}" +
                                  (f" … +{len(failed_names)-10} more" if len(failed_names) > 10 else ""))

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
            total_with_diagrams = sum(
                sum(1 for r in results if r.has_diagram) for results in all_results.values()
            )

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
            print(f"\nDiagram distribution per page:")
            for count in sorted(count_dist.keys()):
                num_pages = count_dist[count]
                pct = num_pages / total_pages * 100 if total_pages > 0 else 0
                print(f"  {count} diagrams: {num_pages:,} pages ({pct:.1f}%)")

            # Show confidence statistics
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                min_conf = min(confidences)
                max_conf = max(confidences)
                print(f"\nConfidence scores:")
                print(f"  Average: {avg_conf:.3f}")
                print(f"  Range: {min_conf:.3f} - {max_conf:.3f}")
                print(f"  Threshold: {self.confidence}")

            # Flag unusual patterns that might indicate issues
            max_diagrams = max(diagram_counts) if diagram_counts else 0
            if max_diagrams > 20:
                print(f"\n⚠ WARNING: Found page(s) with {max_diagrams} diagrams")
                print(f"  This may indicate false positives or threshold issues")

            high_diagram_pages = [c for c in diagram_counts if c > 10]
            if high_diagram_pages:
                pct_high = len(high_diagram_pages) / len(diagram_counts) * 100
                print(f"  Pages with >10 diagrams: {len(high_diagram_pages)} ({pct_high:.1f}%)")

            if use_cache:
                cache_stats = self.cache.stats()
                num_pdfs = cache_stats["num_pdfs"]
                size_mb = cache_stats["size_mb"]
                print(f"Cache: {num_pdfs} PDFs ({size_mb:.1f} MB)")

            # Failure breakdown (only shown when there were failures)
            if all_failed_pdfs:
                total_processed = len(all_results) + len(all_failed_pdfs)
                n_extraction = sum(1 for s in all_failed_pdfs.values() if s == "extraction_failed")
                n_inference  = sum(1 for s in all_failed_pdfs.values() if s == "inference_failed")
                print(f"\nFailure summary:")
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
            with open(failures_path, 'w') as f:
                f.write(f"# Detection failures — {datetime.now().isoformat()}\n")
                f.write(f"# {len(all_failed_pdfs)} failed PDF(s)\n")
                f.write(f"# status: extraction_failed | inference_failed\n\n")
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

    def get_cached_results(
        self, pdf_paths: Union[Path, List[Path]]
    ) -> Dict[str, Optional[List[DetectionResult]]]:
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
                results[pdf_path.name] = [
                    DetectionResult.from_dict(result_dict) for result_dict in cached
                ]
            else:
                results[pdf_path.name] = None

        return results

    def clear_cache(self) -> None:
        """Clear cache."""
        self.cache.clear()
        if self.verbose:
            print("✓ Cache cleared")
