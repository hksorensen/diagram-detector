"""Command-line interface for diagram-detector."""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

from . import __version__
from .detector import DiagramDetector
from .utils import list_models


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file (.yaml, .yml, or .json)

    Returns:
        Dict of configuration values

    Raises:
        ValueError: If file format not supported
        FileNotFoundError: If config file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()

    if suffix == ".json":
        with open(config_path, "r") as f:
            return json.load(f)
    elif suffix in [".yaml", ".yml"]:
        try:
            import yaml
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError(
                "PyYAML required for YAML config files. Install with: pip install pyyaml"
            )
    else:
        raise ValueError(f"Unsupported config format: {suffix}. Use .json, .yaml, or .yml")


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge config file values with CLI arguments.
    CLI arguments take precedence over config file.

    Args:
        config: Config dict from file
        args: Parsed CLI arguments

    Returns:
        Updated args namespace
    """
    # Map config keys to arg names (handle both formats)
    key_mapping = {
        "input": "input",
        "output": "output",
        "model": "model",
        "confidence": "confidence",
        "iou": "iou",
        "imgsz": "imgsz",
        "batch_size": "batch_size",
        "batch-size": "batch_size",
        "device": "device",
        "dpi": "dpi",
        "first_page": "first_page",
        "first-page": "first_page",
        "last_page": "last_page",
        "last-page": "last_page",
        "save_crops": "save_crops",
        "save-crops": "save_crops",
        "visualize": "visualize",
        "format": "format",
        "crop_padding": "crop_padding",
        "crop-padding": "crop_padding",
        "quiet": "quiet",
    }

    # Get default values from parser to detect which args were explicitly set
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--model", default="yolo11m")
    # ... (we'll use a simpler approach)

    # For each config value, set it if the CLI arg is still at default
    for config_key, value in config.items():
        arg_name = key_mapping.get(config_key, config_key)

        # Skip nested dicts (like "remote": {"host": ..., "port": ...})
        # These are metadata, not CLI arguments
        if isinstance(value, dict):
            continue

        # Only set from config if not explicitly provided via CLI
        # This is a simplified check - in practice, detecting "was it provided" is tricky
        if hasattr(args, arg_name):
            # For required args like input/output, config can provide defaults
            current_value = getattr(args, arg_name)

            # Set from config if current value looks like a default
            # (This is imperfect but works for most cases)
            if arg_name in ["input", "output"]:
                # These are required, so config must provide them if --config used alone
                if current_value is None or config_key in config:
                    setattr(args, arg_name, value)
            else:
                # For optional args, config provides defaults
                setattr(args, arg_name, value)

    return args


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description=f"diagram-detector v{__version__} - Detect diagrams in images and PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect in images
  diagram-detect --input images/ --output results/

  # Process PDF
  diagram-detect --input paper.pdf --output results/ --save-crops

  # With visualization
  diagram-detect --input paper.pdf --visualize --confidence 0.35

  # Batch processing
  diagram-detect --input papers/*.pdf --output results/ --batch-size 16

  # Specify model
  diagram-detect --input images/ --model yolo11l --output results/
        """,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Load configuration from YAML or JSON file (CLI args override config file)",
    )

    # Input/output
    parser.add_argument(
        "--input", "-i", help="Input file or directory (images or PDF)"
    )

    parser.add_argument(
        "--output", "-o", default="results", help="Output directory (default: results)"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        "-m",
        default="yolo11m",
        choices=list_models(),
        help="Model to use (default: yolo11m)",
    )

    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.35,
        help="Confidence threshold 0.0-1.0 (default: 0.35)",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.30,
        help="IoU threshold for NMS 0.0-1.0 (default: 0.30, optimal from grid search)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for preprocessing (default: 640)",
    )

    parser.add_argument(
        "--batch-size", "-b", default="auto", help="Batch size for inference (default: auto)"
    )

    parser.add_argument(
        "--device",
        "-d",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use (default: auto)",
    )

    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Use TensorRT optimization with FP16 (NVIDIA GPU only, 2-3x faster)",
    )

    # Remote inference
    parser.add_argument(
        "--remote", type=str, help="Run inference on remote GPU via SSH (format: user@host:port)"
    )

    parser.add_argument(
        "--remote-batch-size",
        type=int,
        default=1000,
        help="Images per batch for remote inference (default: 1000)",
    )

    parser.add_argument(
        "--gpu-batch-size", type=int, default=32, help="GPU batch size on remote (default: 32)"
    )

    parser.add_argument("--resume", action="store_true", help="Resume interrupted remote job")

    parser.add_argument(
        "--no-cleanup", action="store_true", help="Keep remote files after processing"
    )

    parser.add_argument(
        "--no-cache", action="store_true", help="Disable local result caching (for PDFs)"
    )

    # Output options
    parser.add_argument(
        "--save-crops", action="store_true", help="Extract and save cropped diagram regions"
    )

    parser.add_argument(
        "--visualize", action="store_true", help="Save visualizations with bounding boxes drawn"
    )

    parser.add_argument(
        "--format",
        "-f",
        default="json",
        choices=["json", "csv", "both"],
        help="Output format (default: json)",
    )

    parser.add_argument(
        "--crop-padding", type=int, default=10, help="Padding around crops in pixels (default: 10)"
    )

    # PDF options
    parser.add_argument(
        "--dpi", type=int, default=200, help="DPI for PDF conversion (default: 200)"
    )

    parser.add_argument("--first-page", type=int, help="First page to process (1-indexed)")

    parser.add_argument("--last-page", type=int, help="Last page to process (1-indexed)")

    # Other options
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    parser.add_argument(
        "--version", "-v", action="version", version=f"diagram-detector {__version__}"
    )

    args = parser.parse_args()

    # Load and merge config file if provided
    if args.config:
        try:
            config_path = Path(args.config)
            config = load_config_file(config_path)

            # Merge config with args (CLI args take precedence)
            # Track parser defaults to know what values weren't explicitly set
            parser_defaults = {
                'model': 'yolo11m',
                'confidence': 0.35,
                'output': 'results',
                'device': 'auto',
                'batch_size': None,
                'format': 'json',
            }

            for key, value in config.items():
                # Skip nested dicts (like "remote": {"host": ..., "port": ...})
                # These are metadata, not CLI arguments
                if isinstance(value, dict):
                    continue

                # Convert kebab-case to snake_case
                arg_name = key.replace("-", "_")

                # Only set if arg wasn't explicitly provided
                if hasattr(args, arg_name):
                    current = getattr(args, arg_name)
                    default = parser_defaults.get(arg_name)

                    # Set from config if it's None or still at parser default
                    if current is None or current == default:
                        setattr(args, arg_name, value)

            if not args.quiet:
                print(f"✓ Loaded configuration from {config_path}")

        except Exception as e:
            print(f"✗ Failed to load config: {e}", file=sys.stderr)
            sys.exit(1)

    # Validate inputs
    if args.input is None:
        print(f"✗ --input is required (or provide via config file)", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Parse batch size
    if args.batch_size == "auto":
        batch_size = "auto"
    else:
        try:
            batch_size = int(args.batch_size)
            if batch_size < 1:
                raise ValueError()
        except ValueError:
            print(f"✗ Invalid batch size: {args.batch_size}", file=sys.stderr)
            sys.exit(1)

    # Validate confidence
    if not 0.0 <= args.confidence <= 1.0:
        print("✗ Confidence must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Check if using remote inference
        if args.remote:
            from .remote_ssh import parse_remote_string

            if not args.quiet:
                print(f"Using remote GPU inference: {args.remote}")

            # Parse remote connection
            remote_config = parse_remote_string(args.remote)

            # Check if input is PDFs
            if input_path.is_file() and input_path.suffix.lower() == ".pdf":
                # Single PDF - use PDF remote detector
                from .remote_pdf import PDFRemoteDetector

                detector = PDFRemoteDetector(
                    config=remote_config,
                    batch_size=1,  # Single PDF
                    image_batch_size=args.remote_batch_size,  # Images per upload/inference batch
                    model=args.model,
                    confidence=args.confidence,
                    dpi=args.dpi,
                    verbose=not args.quiet,
                    tensorrt=args.tensorrt,
                )

                # Process PDF
                pdf_results = detector.detect_pdfs(
                    input_path,
                    output_dir=output_dir,
                    use_cache=not args.no_cache,
                )

                # Flatten to list of results
                results = list(pdf_results.values())[0]

            elif input_path.is_dir():
                # Directory - check if contains PDFs
                pdf_files = list(input_path.glob("*.pdf"))

                if pdf_files:
                    # PDFs found - use PDF remote detector
                    from .remote_pdf import PDFRemoteDetector

                    detector = PDFRemoteDetector(
                        config=remote_config,
                        batch_size=10,  # PDFs per batch (reasonable default)
                        image_batch_size=args.remote_batch_size,  # Images per upload/inference batch
                        model=args.model,
                        confidence=args.confidence,
                        dpi=args.dpi,
                        verbose=not args.quiet,
                        tensorrt=args.tensorrt,
                    )

                    # Process all PDFs
                    pdf_results = detector.detect_pdfs(
                        pdf_files,
                        output_dir=output_dir,
                        use_cache=not args.no_cache,
                    )

                    # Flatten to list of results
                    results = []
                    for pdf_results_list in pdf_results.values():
                        results.extend(pdf_results_list)

                else:
                    # No PDFs - use image remote detector
                    from .remote_ssh import SSHRemoteDetector

                    detector = SSHRemoteDetector(
                        config=remote_config,
                        batch_size=args.remote_batch_size,
                        model=args.model,
                        confidence=args.confidence,
                        verbose=not args.quiet,
                        tensorrt=args.tensorrt,
                    )

                    # Run remote inference on images
                    results = detector.detect(
                        input_path,
                        output_dir=output_dir,
                        gpu_batch_size=args.gpu_batch_size,
                        cleanup=not args.no_cleanup,
                        resume=args.resume,
                    )

            else:
                # Single image or list
                from .remote_ssh import SSHRemoteDetector

                detector = SSHRemoteDetector(
                    config=remote_config,
                    batch_size=args.remote_batch_size,
                    model=args.model,
                    confidence=args.confidence,
                    verbose=not args.quiet,
                    tensorrt=args.tensorrt,
                )

                results = detector.detect(
                    input_path,
                    output_dir=output_dir,
                    gpu_batch_size=args.gpu_batch_size,
                    cleanup=not args.no_cleanup,
                    resume=args.resume,
                )

        else:
            # Local inference (existing code)
            # Initialize detector
            detector = DiagramDetector(
                model=args.model,
                confidence=args.confidence,
                iou=args.iou,
                device=args.device,
                batch_size=batch_size,
                verbose=not args.quiet,
                tensorrt=args.tensorrt,
            )

            # Run detection
            if input_path.is_file() and input_path.suffix.lower() == ".pdf":
                # PDF processing
                results = detector.detect_pdf(
                    input_path,
                    dpi=args.dpi,
                    first_page=args.first_page,
                    last_page=args.last_page,
                    store_images=args.save_crops or args.visualize,
                )
            else:
                # Image processing
                results = detector.detect(
                    input_path, store_images=args.save_crops or args.visualize
                )

        # Save results
        if args.format in ["json", "both"]:
            detector.save_results(results, output_dir, format="json")

        if args.format in ["csv", "both"]:
            detector.save_results(results, output_dir, format="csv")

        # Save crops if requested
        if args.save_crops:
            crops_dir = output_dir / "crops"
            detector.save_crops(results, crops_dir, padding=args.crop_padding)

        # Save visualizations if requested
        if args.visualize:
            vis_dir = output_dir / "visualizations"
            detector.save_visualizations(results, vis_dir)

        # Print summary
        if not args.quiet:
            print_summary(results, output_dir)

    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def print_summary(results, output_dir):
    """Print summary of detection results."""
    total = len(results)
    with_diagrams = sum(1 for r in results if r.has_diagram)
    total_diagrams = sum(r.count for r in results)

    # Calculate average confidence
    if total_diagrams > 0:
        all_confidences = [d.confidence for r in results for d in r.detections]
        avg_confidence = sum(all_confidences) / len(all_confidences)
    else:
        avg_confidence = 0.0

    print(f"\n{'='*60}")
    print("DETECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total images/pages: {total}")
    print(f"  With diagrams: {with_diagrams} ({with_diagrams/total*100:.1f}%)")
    print(f"  Total diagrams detected: {total_diagrams}")

    if total_diagrams > 0:
        print(f"  Average confidence: {avg_confidence:.3f}")

    print(f"\n  Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
