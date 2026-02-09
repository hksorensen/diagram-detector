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

            try:
                # Extract PDFs to temporary directory on remote
                import tempfile

                from .remote_pdf import extract_pdf_to_jpegs

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    # Extract all PDFs to temp directory
                    # CRITICAL: Each PDF gets its own subdirectory to prevent overwriting
                    for pdf_path in pdf_paths:
                        if not pdf_path.exists():
                            raise FileNotFoundError(f"PDF not found on remote: {pdf_path}")

                        # Create subdirectory for this PDF (using PDF stem as folder name)
                        pdf_output_dir = temp_path / pdf_path.stem
                        pdf_output_dir.mkdir(parents=True, exist_ok=True)

                        # Extract PDF to JPEG files in its own subdirectory
                        image_paths = extract_pdf_to_jpegs(pdf_path, pdf_output_dir, dpi=200, show_progress=False)
                        if not image_paths:
                            sys.stderr.write(f"Warning: No pages extracted from {pdf_path.name}\n")

                    # Run detection on extracted images
                    results = detector.detect(temp_path, store_images=False)

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
