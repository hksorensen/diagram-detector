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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from .models import DetectionResult
from .utils import convert_pdf_to_images, save_json
from .remote_ssh import RemoteConfig, SSHRemoteDetector
from .cache import DetectionCache


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
        image_batch_size: int = 500,  # Images per upload/inference batch
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
    ):
        """
        Initialize PDF remote detector.

        Args:
            config: Remote configuration (None = use defaults for henrikkragh.dk)
            batch_size: PDFs per batch (10 = ~100-200 pages, good for gigabit LAN)
            image_batch_size: Images per upload/inference batch (500 = good balance for gigabit LAN)
                             Increase for faster networks, decrease for slow connections
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
        """
        # Use default config for local network if not provided
        if config is None:
            config = RemoteConfig()  # Uses henrikkragh.dk defaults (auto-detects local)

        self.config = config
        self.batch_size = batch_size
        self.image_batch_size = image_batch_size
        self.model = model
        self.confidence = confidence
        self.iou = iou
        self.dpi = dpi
        self.imgsz = imgsz
        self.verbose = verbose
        self.parallel_extract = parallel_extract
        self.max_workers = max_workers

        # Initialize SSH detector for actual remote execution
        self.remote_detector = SSHRemoteDetector(
            config=config,
            batch_size=image_batch_size,  # Images per upload/inference batch
            model=model,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            verbose=self.verbose,  # Pass through verbose flag for detailed timing
            run_id=run_id,
            config_dir=config_dir,
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

    def _extract_pdf_pages(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        """
        Extract PDF pages to images locally.

        Args:
            pdf_path: Path to PDF
            output_dir: Where to save images

        Returns:
            List of image paths
        """
        if self.verbose:
            print(f"  Extracting {pdf_path.name} (DPI={self.dpi})...")

        # Convert PDF to images
        images = convert_pdf_to_images(pdf_path, dpi=self.dpi, verbose=self.verbose)

        # Save images
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []

        from PIL import Image

        for page_num, img_array in enumerate(images, start=1):
            img_path = output_dir / f"page_{page_num:04d}.jpg"
            Image.fromarray(img_array).save(img_path, "JPEG", quality=95)
            image_paths.append(img_path)

        if self.verbose:
            print(f"  ✓ {pdf_path.name}: {len(image_paths)} pages")

        return image_paths

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

        pdf_images = {}

        if not self.parallel_extract or len(pdf_batch) == 1:
            # Sequential extraction
            for pdf_path in pdf_batch:
                pdf_dir = batch_dir / pdf_path.stem
                image_paths = self._extract_pdf_pages(pdf_path, pdf_dir)
                pdf_images[pdf_path.name] = image_paths
        else:
            # Parallel extraction
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all extraction tasks
                future_to_pdf = {}
                for pdf_path in pdf_batch:
                    pdf_dir = batch_dir / pdf_path.stem
                    future = executor.submit(self._extract_pdf_pages, pdf_path, pdf_dir)
                    future_to_pdf[future] = pdf_path

                # Collect results as they complete
                for future in as_completed(future_to_pdf):
                    pdf_path = future_to_pdf[future]
                    try:
                        image_paths = future.result()
                        pdf_images[pdf_path.name] = image_paths
                    except Exception as e:
                        if self.verbose:
                            print(f"  ✗ {pdf_path.name}: {e}")
                        pdf_images[pdf_path.name] = []

        extraction_time = time.time() - start_time
        return pdf_images, extraction_time

    def _process_pdf_batch(
        self, pdf_batch: List[Path], batch_id: str, work_dir: Path, auto_git_commit: bool = False
    ) -> Dict[str, List[DetectionResult]]:
        """
        Process batch of PDFs.

        Args:
            pdf_batch: List of PDF paths
            batch_id: Batch identifier
            work_dir: Working directory
            auto_git_commit: Automatically git commit the run config

        Returns:
            Dict mapping PDF name to results
        """
        batch_dir = work_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Extract all PDFs in batch (parallel if enabled)
        pdf_images, extraction_time = self._extract_pdfs_parallel(pdf_batch, batch_dir)

        # Flatten to all images
        all_images = []
        pdf_page_counts = {}
        for pdf_name, image_paths in pdf_images.items():
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
            cleanup=True,
            auto_git_commit=auto_git_commit,
        )
        inference_time = time.time() - inference_start

        if self.verbose:
            print(f"  ✓ Inference complete: {inference_time:.1f}s ({len(all_images)/inference_time:.1f} images/s)")

        # Group results by PDF
        pdf_results = {}
        result_idx = 0

        for pdf_path in pdf_batch:
            num_pages = pdf_page_counts[pdf_path.name]
            pdf_result_list = results[result_idx : result_idx + num_pages]

            # Add page numbers
            for page_num, result in enumerate(pdf_result_list, start=1):
                result.page_number = page_num

            pdf_results[pdf_path.name] = pdf_result_list
            result_idx += num_pages

        # Cleanup batch directory
        shutil.rmtree(batch_dir, ignore_errors=True)

        return pdf_results

    def detect_pdfs(
        self,
        pdf_paths: Union[Path, List[Path]],
        output_dir: Optional[Path] = None,
        use_cache: bool = True,
        force_reprocess: bool = False,
        auto_git_commit: bool = False,
    ) -> Dict[str, List[DetectionResult]]:
        """
        Process PDFs with remote inference and local caching.

        Args:
            pdf_paths: Single PDF, directory of PDFs, or list of PDF paths
            output_dir: Where to save results (None = don't save)
            use_cache: Use cached results if available
            force_reprocess: Force reprocessing even if cached
            auto_git_commit: Automatically git commit the run config

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

        if self.verbose:
            print(f"\n{'='*60}")
            print("PDF REMOTE INFERENCE (detect_pdfs)")
            print(f"{'='*60}")
            print(f"PDFs: {len(pdf_list)}")
            print(f"Batch size: {self.batch_size} PDFs/batch")
            print(f"Model: {self.model}")
            print(f"Remote: {self.config.ssh_target}")
            print(f"Cache: {'enabled' if use_cache else 'disabled'}")
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
                    if self.verbose:
                        print(f"  ✓ {pdf_path.name} (cached)")
                else:
                    to_process.append(pdf_path)
                    if self.verbose:
                        print(f"  • {pdf_path.name} (needs processing)")
        else:
            to_process = pdf_list

        if self.verbose:
            print(f"\nProcessing {len(to_process)} PDFs ({len(cached_results)} cached)...\n")

        if not to_process:
            if self.verbose:
                print("✓ All PDFs cached, no processing needed!")
            return cached_results

        print (f"to_process: {to_process}")

        # Process in batches
        all_results = cached_results.copy()
        num_batches = (len(to_process) + self.batch_size - 1) // self.batch_size
        print (f"num_batches: {num_batches}")

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)

            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min((batch_idx + 1) * self.batch_size, len(to_process))
                batch_pdfs = to_process[batch_start:batch_end]
                batch_id = f"batch_{batch_idx:04d}"

                if self.verbose:
                    print(f"\n--- Batch {batch_idx + 1}/{num_batches} ({len(batch_pdfs)} PDFs) ---")

                # Process batch
                print (f"batch {batch_id}: {batch_start}-{batch_end}")
                batch_results = self._process_pdf_batch(batch_pdfs, batch_id, work_dir, auto_git_commit)

                # Cache results
                if use_cache:
                    for pdf_path in batch_pdfs:
                        # Convert DetectionResult objects to dicts for JSON serialization
                        results_list = batch_results[pdf_path.name]
                        results_dicts = [asdict(r) for r in results_list]

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

                if self.verbose:
                    batch_diagrams = sum(
                        sum(r.count for r in results) for results in batch_results.values()
                    )
                    print(f"✓ Batch complete: {batch_diagrams} diagrams found")

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

        # Print summary
        if self.verbose:
            total_pages = sum(len(results) for results in all_results.values())
            total_with_diagrams = sum(
                sum(1 for r in results if r.has_diagram) for results in all_results.values()
            )
            total_diagrams = sum(sum(r.count for r in results) for results in all_results.values())

            print(f"\n{'='*60}")
            print("PDF REMOTE INFERENCE COMPLETE")
            print(f"{'='*60}")
            print(f"Total PDFs: {len(all_results)}")
            print(f"Total pages: {total_pages:,}")
            pct = total_with_diagrams / total_pages * 100
            print(f"Pages with diagrams: {total_with_diagrams:,} ({pct:.1f}%)")
            print(f"Total diagrams: {total_diagrams:,}")
            if use_cache:
                cache_stats = self.cache.stats()
                num_pdfs = cache_stats["num_pdfs"]
                size_mb = cache_stats["size_mb"]
                print(f"Cache: {num_pdfs} PDFs ({size_mb:.1f} MB)")
            print(f"{'='*60}\n")

        return all_results

    def clear_cache(self) -> None:
        """Clear cache."""
        self.cache.clear()
        if self.verbose:
            print("✓ Cache cleared")
