"""Persistent inference server for diagram-detector.

Loads the YOLO model once and processes inference batches via a simple
stdin/stdout JSON protocol.  Designed to be kept alive over SSH across
multiple sub-batches, eliminating model-reload overhead (~3-5 s / batch).

Protocol
--------
Startup
    Server prints ``READY`` on stdout once the model is loaded.

Infer (from uploaded images)
    Client writes one JSON line to stdin::

        {"type": "infer", "input": "<dir>", "output": "<dir>"}

    Server runs ``detector.detect(<input>)``, saves results to ``<output>``,
    then prints one JSON line on stdout::

        {"status": "DONE",  "output": "<dir>", "num_results": N}

    On error::

        {"status": "ERROR", "error": "<message>"}

Infer PDFs (from remote filesystem)
    Client writes one JSON line to stdin::

        {"type": "infer_pdfs", "pdfs": ["path/to/file.pdf", ...], "output": "<dir>"}

    Server extracts PDFs from remote filesystem, runs detection, saves results::

        {"status": "DONE",  "output": "<dir>", "num_results": N, "pdfs_processed": M}

    On error::

        {"status": "ERROR", "error": "<message>"}

Shutdown
    Client writes::

        {"type": "shutdown"}

    Server prints::

        {"status": "SHUTDOWN"}

    and exits with code 0.

Usage
-----
Invoked by ``SSHRemoteDetector`` over SSH::

    python -m diagram_detector.server --model v5 --confidence 0.1 --iou 0.3 ...
"""

import json
import os
import sys
from pathlib import Path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Persistent YOLO inference server")
    parser.add_argument("--model", default="yolo11m", help="Model name or path")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32, help="GPU inference batch size")
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # fd-level stdout silence.  ``sys.stdout = sys.stderr`` only catches
    # Python-level print().  YOLO / ultralytics C extensions write directly to
    # fd 1, bypassing sys.stdout.  We silence fd 1 permanently by pointing it
    # at fd 2 (stderr), and keep a dup of the original fd 1 for protocol
    # messages only.
    # ---------------------------------------------------------------------------
    _proto_fd = os.dup(1)  # save the real stdout fd
    os.dup2(2, 1)  # fd 1 → stderr for the lifetime of the server
    # sys.stdout now also writes to stderr — safe for any detector prints

    def _send(msg: str) -> None:
        """Write one line to the protocol fd (original stdout)."""
        os.write(_proto_fd, (msg + "\n").encode())

    from .detector import DiagramDetector

    detector = DiagramDetector(
        model=args.model,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        tensorrt=args.tensorrt,
        verbose=True,  # all output goes to stderr via fd 1
        cache=False,  # caller manages caching; server is stateless between batches
    )

    # Signal ready — caller blocks on this line
    _send("READY")

    # -------------------------------------------------------------------------
    # Command loop — one JSON object per line from stdin
    # -------------------------------------------------------------------------
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            _send(json.dumps({"status": "ERROR", "error": f"Invalid JSON: {line}"}))
            continue

        cmd_type = cmd.get("type", "")

        if cmd_type == "shutdown":
            _send(json.dumps({"status": "SHUTDOWN"}))
            break

        if cmd_type == "infer":
            input_dir = Path(cmd["input"]).expanduser()
            output_dir = Path(cmd["output"]).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                results = detector.detect(input_dir, store_images=False)

                # Save results in the same JSON format cli.py uses
                detector.save_results(results, output_dir, format="json")

                _send(
                    json.dumps(
                        {
                            "status": "DONE",
                            "output": str(output_dir),
                            "num_results": len(results),
                        }
                    )
                )

            except Exception as e:
                import traceback

                traceback.print_exc(file=sys.stderr)
                _send(
                    json.dumps(
                        {
                            "status": "ERROR",
                            "error": str(e),
                        }
                    )
                )
                # Server stays alive — next batch can still be attempted

        elif cmd_type == "infer_pdfs":
            # New command: detect PDFs directly from remote filesystem
            # PDFs are already on remote, no upload needed
            pdf_paths = [Path(p).expanduser() for p in cmd["pdfs"]]
            output_dir = Path(cmd["output"]).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)

            # ORDERING LOG: Show what PDFs were received and in what order
            sys.stderr.write(f"[SERVER ORDERING] Received infer_pdfs command with {len(pdf_paths)} PDFs\n")
            if len(pdf_paths) >= 3:
                sys.stderr.write(f"[SERVER ORDERING]   First 3: {[p.name for p in pdf_paths[:3]]}\n")
                sys.stderr.write(f"[SERVER ORDERING]   Last 3: {[p.name for p in pdf_paths[-3:]]}\n")
            sys.stderr.flush()

            try:
                # Extract PDFs to temporary directory on remote
                import tempfile

                from .remote_pdf import extract_pdf_to_jpegs

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    # Extract all PDFs to temp directory
                    # CRITICAL: Each PDF gets its own subdirectory to prevent overwriting
                    sys.stderr.write(f"[SERVER] Extracting {len(pdf_paths)} PDFs to {temp_path}\n")
                    sys.stderr.flush()

                    total_images = 0
                    for idx, pdf_path in enumerate(pdf_paths):
                        if not pdf_path.exists():
                            raise FileNotFoundError(f"PDF not found on remote: {pdf_path}")

                        # Create subdirectory for this PDF (using PDF stem as folder name)
                        pdf_output_dir = temp_path / pdf_path.stem
                        pdf_output_dir.mkdir(parents=True, exist_ok=True)

                        # Extract PDF to JPEG files in its own subdirectory
                        image_paths = extract_pdf_to_jpegs(pdf_path, pdf_output_dir, dpi=200, show_progress=False)
                        if not image_paths:
                            sys.stderr.write(f"Warning: No pages extracted from {pdf_path.name}\n")
                        else:
                            total_images += len(image_paths)

                        # Progress every 20 PDFs
                        if (idx + 1) % 20 == 0 or (idx + 1) == len(pdf_paths):
                            sys.stderr.write(
                                f"[SERVER] Extracted {idx + 1}/{len(pdf_paths)} PDFs ({total_images} images)\n"
                            )
                            sys.stderr.flush()

                    sys.stderr.write(f"[SERVER] Extraction complete: {total_images} images in {temp_path}\n")

                    # CRITICAL FIX: Create manifest.txt to preserve PDF ordering
                    # Without this, detector.detect() uses glob which returns filesystem order
                    # This causes contamination (results assigned to wrong PDFs)
                    manifest_path = temp_path / "manifest.txt"
                    sys.stderr.write("[SERVER] Creating manifest.txt to preserve PDF order...\n")
                    sys.stderr.flush()

                    with open(manifest_path, "w") as f:
                        f.write("# Image processing manifest - explicit PDF ordering\n")
                        f.write(f"# PDFs: {len(pdf_paths)}\n")
                        f.write(f"# Total images: {total_images}\n")
                        f.write("#\n")
                        f.write("# Order matches pdf_paths list sent to infer_pdfs command\n")
                        f.write("#\n")

                        # Write images in PDF order (each PDF's subdirectory in sequence)
                        for pdf_path in pdf_paths:
                            pdf_output_dir = temp_path / pdf_path.stem
                            if pdf_output_dir.exists():
                                # Get all images from this PDF's subdirectory
                                # Sort by page number to ensure correct page order within each PDF
                                pdf_images = sorted(pdf_output_dir.glob("*.jpg"))
                                for img_path in pdf_images:
                                    # Write relative path (subdir/filename.jpg)
                                    relative_path = img_path.relative_to(temp_path)
                                    f.write(f"{relative_path}\n")

                    manifest_image_count = (
                        len(list(manifest_path.read_text().strip().split("\n"))) - 6
                    )  # subtract header lines
                    sys.stderr.write(f"[SERVER] Manifest created: {manifest_image_count} images in PDF order\n")
                    sys.stderr.write(f"[SERVER] Starting detection on {temp_path}\n")
                    sys.stderr.flush()

                    # Run detection on extracted images
                    # Detector will find manifest.txt and use explicit ordering
                    results = detector.detect(temp_path, store_images=False)

                    sys.stderr.write(f"[SERVER] Detection complete: {len(results)} results\n")
                    sys.stderr.flush()

                    # Save results
                    detector.save_results(results, output_dir, format="json")

                    _send(
                        json.dumps(
                            {
                                "status": "DONE",
                                "output": str(output_dir),
                                "num_results": len(results),
                                "pdfs_processed": len(pdf_paths),
                            }
                        )
                    )

            except Exception as e:
                import traceback

                traceback.print_exc(file=sys.stderr)
                _send(
                    json.dumps(
                        {
                            "status": "ERROR",
                            "error": str(e),
                        }
                    )
                )

        else:
            _send(json.dumps({"status": "ERROR", "error": f"Unknown command type: {cmd_type}"}))


if __name__ == "__main__":
    main()
