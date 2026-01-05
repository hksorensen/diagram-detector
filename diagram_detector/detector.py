"""Main diagram detection class."""

from pathlib import Path
from typing import List, Union, Optional
import numpy as np
from tqdm import tqdm

from .models import DetectionResult, DiagramDetection
from .utils import (
    detect_device,
    download_model,
    get_model_path,
    optimize_batch_size,
    convert_pdf_to_images,
    load_image,
    save_json,
    save_csv,
    get_image_files,
    get_device_info,
)


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
    ):
        """
        Initialize detector.

        Args:
            model: Model name ('yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x')
            confidence: Confidence threshold (0.0-1.0, default: 0.20 - optimized via grid search)
            iou: IOU threshold for NMS (0.0-1.0, default: 0.30 - optimized via grid search)
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            batch_size: Batch size for inference (int or 'auto')
            verbose: Print progress information
        """
        self.model_name = model
        self.confidence = confidence
        self.iou = iou
        self.verbose = verbose

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

        # Optimize batch size
        if batch_size == "auto":
            self.batch_size = optimize_batch_size(model, self.device)
            if self.verbose:
                print(f"Auto batch size: {self.batch_size}")
        else:
            self.batch_size = batch_size

        # Download model if needed
        model_path = get_model_path(model)
        if not model_path.exists():
            if self.verbose:
                print("Model not found in cache, downloading...")
            download_model(model)

        # Load model
        if self.verbose:
            print(f"Loading {model} model...")

        from ultralytics import YOLO

        self.model = YOLO(str(model_path))

        if self.verbose:
            print("✓ Model loaded")

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
            image_paths = get_image_files(input_path)
            if not image_paths:
                raise ValueError(f"No images found in {input_path}")
        else:
            image_paths = [input_path]

        if self.verbose:
            print(f"\nProcessing {len(image_paths)} image(s)...")

        # Run batch inference
        results = []

        # Process in batches
        for i in tqdm(
            range(0, len(image_paths), self.batch_size),
            desc="Detecting",
            disable=not self.verbose,
            unit="batch",
        ):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_results = self._detect_batch(
                batch_paths, store_images=store_images or save_crops or save_visualizations
            )
            results.extend(batch_results)

        return results

    def detect_pdf(
        self,
        pdf_path: Union[str, Path],
        dpi: int = 200,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
        **kwargs,
    ) -> List[DetectionResult]:
        """
        Detect diagrams in PDF.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for PDF conversion
            first_page: First page to process (1-indexed)
            last_page: Last page to process (1-indexed)
            **kwargs: Additional arguments passed to detect()

        Returns:
            List of DetectionResult objects (one per page)
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if self.verbose:
            print(f"Processing PDF: {pdf_path.name}")

        # Convert PDF to images
        images = convert_pdf_to_images(pdf_path, dpi, first_page, last_page, verbose=self.verbose)

        if self.verbose:
            print(f"✓ Converted {len(images)} pages")

        # Run detection on all pages
        results = []

        for page_num, image in enumerate(
            tqdm(images, desc="Detecting", disable=not self.verbose, unit="page"),
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

        return results

    def _detect_batch(
        self, image_paths: List[Path], store_images: bool = False
    ) -> List[DetectionResult]:
        """Run inference on batch of images."""
        # Load images if needed
        if store_images:
            images = [load_image(p) for p in image_paths]
        else:
            images = None

        # Run YOLO inference
        yolo_results = self.model.predict(
            source=[str(p) for p in image_paths],
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
            stream=False,
        )

        # Parse results
        results = []
        for i, (yolo_result, image_path) in enumerate(zip(yolo_results, image_paths)):
            result = self._parse_yolo_result(
                yolo_result, filename=image_path.name, image=images[i] if images else None
            )
            results.append(result)

        return results

    def _detect_image(
        self, image: np.ndarray, filename: str, store_image: bool = False
    ) -> DetectionResult:
        """Run inference on single image array."""
        # Run YOLO inference
        yolo_results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        result = self._parse_yolo_result(
            yolo_results[0], filename=filename, image=image if store_image else None
        )

        return result

    def _parse_yolo_result(
        self, yolo_result, filename: str, image: Optional[np.ndarray] = None
    ) -> DetectionResult:
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
                    bbox[0] = max(0.0, min(bbox[0], float(width)))   # x1
                    bbox[1] = max(0.0, min(bbox[1], float(height)))  # y1
                    bbox[2] = max(0.0, min(bbox[2], float(width)))   # x2
                    bbox[3] = max(0.0, min(bbox[3], float(height)))  # y2

                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = yolo_result.names[cls_id]

                detection = DiagramDetection(
                    bbox=tuple(bbox), confidence=conf, class_name=cls_name, class_id=cls_id
                )
                detections.append(detection)

        return DetectionResult(
            filename=filename,
            detections=detections,
            image=image,
            image_width=width,
            image_height=height,
        )

    def save_results(
        self, results: List[DetectionResult], output_dir: Union[str, Path], format: str = "json"
    ) -> None:
        """
        Save detection results.

        Args:
            results: List of DetectionResult objects
            output_dir: Output directory
            format: Output format ('json' or 'csv')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # Save as JSON
            data = [r.to_dict() for r in results]
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

    def save_crops(
        self, results: List[DetectionResult], output_dir: Union[str, Path], padding: int = 10
    ) -> None:
        """
        Extract and save cropped diagram regions.

        Args:
            results: List of DetectionResult objects
            output_dir: Output directory
            padding: Pixels to add around bbox
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_crops = 0

        for result in tqdm(
            results, desc="Extracting crops", disable=not self.verbose, unit="image"
        ):
            if not result.has_diagram or result.image is None:
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
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for result in tqdm(
            results, desc="Creating visualizations", disable=not self.verbose, unit="image"
        ):
            if not result.has_diagram or result.image is None:
                continue

            vis_path = output_dir / result.filename
            result.save_visualization(vis_path, line_width=line_width)

        if self.verbose:
            print(f"✓ Saved {len(results)} visualizations to {output_dir}")
