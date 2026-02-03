"""Persistent inference server for diagram-detector.

Loads the YOLO model once and processes inference batches via a simple
stdin/stdout JSON protocol.  Designed to be kept alive over SSH across
multiple sub-batches, eliminating model-reload overhead (~3-5 s / batch).

Protocol
--------
Startup
    Server prints ``READY`` on stdout once the model is loaded.

Infer
    Client writes one JSON line to stdin::

        {"type": "infer", "input": "<dir>", "output": "<dir>"}

    Server runs ``detector.detect(<input>)``, saves results to ``<output>``,
    then prints one JSON line on stdout::

        {"status": "DONE",  "output": "<dir>", "num_results": N}

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

import sys
import json
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
    # All detector output (verbose prints, [DEBUG] lines) must go to stderr so
    # stdout stays clean for the JSON protocol.  We restore stdout only when we
    # need to write a protocol message.
    # ---------------------------------------------------------------------------
    _orig_stdout = sys.stdout
    sys.stdout = sys.stderr  # detector loading prints → stderr

    from .detector import DiagramDetector

    detector = DiagramDetector(
        model=args.model,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        tensorrt=args.tensorrt,
        verbose=True,   # goes to stderr now
        cache=False,    # caller manages caching; server is stateless between batches
    )

    # Signal ready — caller blocks on this line
    sys.stdout = _orig_stdout
    print("READY", flush=True)

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
            print(json.dumps({"status": "ERROR", "error": f"Invalid JSON: {line}"}), flush=True)
            continue

        cmd_type = cmd.get("type", "")

        if cmd_type == "shutdown":
            print(json.dumps({"status": "SHUTDOWN"}), flush=True)
            break

        if cmd_type == "infer":
            input_dir = Path(cmd["input"])
            output_dir = Path(cmd["output"])
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Redirect stdout during detection so [DEBUG] prints don't
                # corrupt the protocol stream.
                sys.stdout = sys.stderr
                results = detector.detect(input_dir, store_images=False)
                sys.stdout = _orig_stdout

                # Save results in the same JSON format cli.py uses
                detector.save_results(results, output_dir, format="json")

                print(json.dumps({
                    "status": "DONE",
                    "output": str(output_dir),
                    "num_results": len(results),
                }), flush=True)

            except Exception as e:
                sys.stdout = _orig_stdout
                import traceback
                traceback.print_exc(file=sys.stderr)
                print(json.dumps({
                    "status": "ERROR",
                    "error": str(e),
                }), flush=True)
                # Server stays alive — next batch can still be attempted

        else:
            print(json.dumps({"status": "ERROR", "error": f"Unknown command type: {cmd_type}"}), flush=True)


if __name__ == "__main__":
    main()
