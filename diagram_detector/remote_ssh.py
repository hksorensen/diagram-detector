"""
SSH Remote Inference Module

Run inference on remote GPU server via SSH with intelligent batching.
Optimized for processing large image corpora (100K+ images).
"""

from pathlib import Path
from typing import List, Union, Optional
import subprocess
import json
from dataclasses import dataclass
import tempfile
import shutil

from .models import DetectionResult, DiagramDetection
from .utils import get_image_files


@dataclass
class RemoteConfig:
    """Configuration for remote SSH server."""

    host: str = "thinkcentre.local"
    port: int = 22
    user: str = "hkragh"
    remote_work_dir: str = "~/diagram-inference"
    python_path: str = "python"  # or path to venv python

    @property
    def ssh_target(self) -> str:
        """Get SSH connection string."""
        return f"{self.user}@{self.host}"

    @property
    def ssh_port_args(self) -> List[str]:
        """Get SSH port arguments."""
        return ["-p", str(self.port)] if self.port != 22 else []


class SSHRemoteDetector:
    """
    Run inference on remote GPU server via SSH.

    Optimized for large-scale batch processing with:
    - Intelligent batching (upload → process → download in chunks)
    - Progress tracking
    - Automatic resume on failure
    - Minimal network overhead
    """

    def __init__(
        self,
        config: Union[RemoteConfig, str],
        batch_size: int = 1000,
        model: str = "yolo11m",
        confidence: float = 0.35,
        verbose: bool = True,
    ):
        """
        Initialize remote detector.

        Args:
            config: RemoteConfig or connection string (user@host:port)
            batch_size: Images per batch (1000 = ~10-20 min on GPU)
            model: Model to use on remote
            confidence: Confidence threshold
            verbose: Print progress
        """
        if isinstance(config, str):
            self.config = self._parse_connection_string(config)
        else:
            self.config = config

        self.batch_size = batch_size
        self.model = model
        self.confidence = confidence
        self.verbose = verbose

        # Verify SSH connection
        self._verify_connection()

    def _parse_connection_string(self, conn_str: str) -> RemoteConfig:
        """Parse connection string like 'user@host:port'."""
        # user@host:port format
        if "@" not in conn_str:
            raise ValueError("Connection string must be in format: user@host:port")

        user, rest = conn_str.split("@", 1)

        if ":" in rest:
            host, port = rest.rsplit(":", 1)
            port = int(port)
        else:
            host = rest
            port = 22

        return RemoteConfig(host=host, port=port, user=user)

    def _verify_connection(self) -> None:
        """Verify SSH connection works."""
        if self.verbose:
            print(f"Verifying SSH connection to {self.config.ssh_target}:{self.config.port}...")

        try:
            cmd = ["ssh"] + self.config.ssh_port_args + [self.config.ssh_target, 'echo "OK"']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                raise RuntimeError(f"SSH connection failed: {result.stderr}")

            if self.verbose:
                print("✓ SSH connection verified")

        except subprocess.TimeoutExpired:
            raise RuntimeError("SSH connection timed out")
        except Exception as e:
            raise RuntimeError(f"SSH connection failed: {e}")

    def _run_ssh_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run command on remote server."""
        cmd = ["ssh"] + self.config.ssh_port_args + [self.config.ssh_target, command]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if check and result.returncode != 0:
            raise RuntimeError(f"Remote command failed: {result.stderr}")

        return result

    def _setup_remote_workspace(self) -> None:
        """Setup remote workspace directory."""
        if self.verbose:
            print("Setting up remote workspace...")

        # Create directories
        commands = [
            f"mkdir -p {self.config.remote_work_dir}",
            f"mkdir -p {self.config.remote_work_dir}/input",
            f"mkdir -p {self.config.remote_work_dir}/output",
        ]

        for cmd in commands:
            self._run_ssh_command(cmd)

        if self.verbose:
            print("✓ Remote workspace ready")

    def _upload_batch(self, image_paths: List[Path], batch_id: str) -> None:
        """Upload batch of images via rsync."""
        if self.verbose:
            print(f"Uploading batch {batch_id} ({len(image_paths)} images)...")

        # Create temporary directory with batch images
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy images to temp directory
            for img_path in image_paths:
                shutil.copy2(img_path, temp_path / img_path.name)

            # Rsync to remote
            remote_input = f"{self.config.remote_work_dir}/input/{batch_id}/"

            cmd = (
                [
                    "rsync",
                    "-az",
                    "--progress" if self.verbose else "--quiet",
                ]
                + self.config.ssh_port_args
                + [f"{temp_path}/", f"{self.config.ssh_target}:{remote_input}"]
            )

            result = subprocess.run(cmd, capture_output=not self.verbose)

            if result.returncode != 0:
                raise RuntimeError(f"Upload failed: {result.stderr}")

        if self.verbose:
            print(f"✓ Batch {batch_id} uploaded")

    def _run_inference_batch(self, batch_id: str, gpu_batch_size: int = 32) -> None:
        """Run inference on batch on remote server."""
        if self.verbose:
            print(f"Running inference on batch {batch_id}...")

        # Build remote command
        input_dir = f"{self.config.remote_work_dir}/input/{batch_id}"
        output_dir = f"{self.config.remote_work_dir}/output/{batch_id}"

        cmd = (
            f"cd {self.config.remote_work_dir} && "
            f"{self.config.python_path} -m diagram_detector.cli "
            f"--input {input_dir} "
            f"--output {output_dir} "
            f"--model {self.model} "
            f"--confidence {self.confidence} "
            f"--batch-size {gpu_batch_size} "
            f"--format json "
            f"--quiet"
        )

        # Run inference
        self._run_ssh_command(cmd, check=True)

        if self.verbose:
            print(f"✓ Batch {batch_id} processed")

    def _download_results(self, batch_id: str, output_dir: Path) -> Path:
        """Download results from remote server."""
        if self.verbose:
            print(f"Downloading results for batch {batch_id}...")

        # Create local output directory
        batch_output = output_dir / batch_id
        batch_output.mkdir(parents=True, exist_ok=True)

        # Rsync results
        remote_output = f"{self.config.remote_work_dir}/output/{batch_id}/"

        cmd = (
            [
                "rsync",
                "-az",
                "--progress" if self.verbose else "--quiet",
            ]
            + self.config.ssh_port_args
            + [f"{self.config.ssh_target}:{remote_output}", str(batch_output)]
        )

        result = subprocess.run(cmd, capture_output=not self.verbose)

        if result.returncode != 0:
            raise RuntimeError(f"Download failed: {result.stderr}")

        if self.verbose:
            print(f"✓ Batch {batch_id} results downloaded")

        return batch_output

    def _cleanup_batch(self, batch_id: str) -> None:
        """Clean up batch files on remote server."""
        if self.verbose:
            print(f"Cleaning up batch {batch_id}...")

        # Remove batch directories
        commands = [
            f"rm -rf {self.config.remote_work_dir}/input/{batch_id}",
            f"rm -rf {self.config.remote_work_dir}/output/{batch_id}",
        ]

        for cmd in commands:
            self._run_ssh_command(cmd, check=False)  # Don't fail on cleanup

    def _parse_results(self, results_dir: Path) -> List[DetectionResult]:
        """Parse results from JSON file."""
        json_file = results_dir / "detections.json"

        if not json_file.exists():
            raise RuntimeError(f"Results file not found: {json_file}")

        with open(json_file, "r") as f:
            data = json.load(f)

        results = []
        for item in data:
            detections = [
                DiagramDetection(
                    bbox=tuple(d["bbox"]),
                    confidence=d["confidence"],
                    class_name=d.get("class", "diagram"),
                )
                for d in item.get("detections", [])
            ]

            result = DetectionResult(
                filename=item["filename"],
                detections=detections,
                image_width=item.get("image_width", 0),
                image_height=item.get("image_height", 0),
            )

            results.append(result)

        return results

    def detect(
        self,
        input_path: Union[str, Path, List[Path]],
        output_dir: Optional[Path] = None,
        gpu_batch_size: int = 32,
        cleanup: bool = True,
        resume: bool = False,
    ) -> List[DetectionResult]:
        """
        Run remote inference on images.

        Args:
            input_path: Image file, directory, or list of paths
            output_dir: Where to save results locally
            gpu_batch_size: Batch size for GPU inference (16-64 typical)
            cleanup: Clean up remote files after processing
            resume: Resume from partially completed job

        Returns:
            List of DetectionResult objects
        """
        # Parse input
        if isinstance(input_path, list):
            image_paths = [Path(p) for p in input_path]
        elif isinstance(input_path, (str, Path)):
            input_path = Path(input_path)
            if input_path.is_dir():
                image_paths = get_image_files(input_path)
            else:
                image_paths = [input_path]
        else:
            raise ValueError("input_path must be path, directory, or list of paths")

        if not image_paths:
            raise ValueError("No images found")

        # Setup output directory
        if output_dir is None:
            output_dir = Path("remote_inference_results")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"\n{'='*60}")
            print("REMOTE INFERENCE")
            print(f"{'='*60}")
            print(f"Images: {len(image_paths):,}")
            print(f"Batch size: {self.batch_size:,} images/batch")
            print(f"GPU batch size: {gpu_batch_size}")
            print(f"Model: {self.model}")
            print(f"Remote: {self.config.ssh_target}:{self.config.port}")
            print(f"{'='*60}\n")

        # Setup remote workspace
        self._setup_remote_workspace()

        # Calculate batches
        num_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        if self.verbose:
            print(f"Processing {len(image_paths):,} images in {num_batches} batch(es)...\n")

        # Process batches
        all_results = []

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            batch_id = f"batch_{batch_idx:04d}"

            if self.verbose:
                print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")

            try:
                # Check if batch already processed (resume)
                batch_output_dir = output_dir / batch_id
                if resume and batch_output_dir.exists():
                    if self.verbose:
                        print(f"✓ Batch {batch_id} already processed (resuming)")
                    results = self._parse_results(batch_output_dir)
                    all_results.extend(results)
                    continue

                # 1. Upload batch
                self._upload_batch(batch_paths, batch_id)

                # 2. Run inference
                self._run_inference_batch(batch_id, gpu_batch_size)

                # 3. Download results
                batch_results_dir = self._download_results(batch_id, output_dir)

                # 4. Parse results
                results = self._parse_results(batch_results_dir)
                all_results.extend(results)

                # 5. Cleanup (optional)
                if cleanup:
                    self._cleanup_batch(batch_id)

                if self.verbose:
                    batch_diagrams = sum(r.count for r in results)
                    print(f"✓ Batch complete: {batch_diagrams} diagrams found")

            except Exception as e:
                print(f"\n✗ Batch {batch_id} failed: {e}")
                if not resume:
                    raise
                print("Continuing with next batch (use --resume to retry failed batches)...")
                continue

        # Print summary
        if self.verbose:
            total_with_diagrams = sum(1 for r in all_results if r.has_diagram)
            total_diagrams = sum(r.count for r in all_results)

            print(f"\n{'='*60}")
            print("REMOTE INFERENCE COMPLETE")
            print(f"{'='*60}")
            print(f"Total images: {len(all_results):,}")
            pct = total_with_diagrams / len(all_results) * 100
            print(f"With diagrams: {total_with_diagrams:,} ({pct:.1f}%)")
            print(f"Total diagrams: {total_diagrams:,}")
            print(f"Results saved: {output_dir}")
            print(f"{'='*60}\n")

        return all_results


def parse_remote_string(remote_str: str) -> RemoteConfig:
    """
    Parse remote string into RemoteConfig.

    Formats supported:
    - user@host
    - user@host:port
    - ssh://user@host
    - ssh://user@host:port
    """
    # Remove ssh:// prefix if present
    if remote_str.startswith("ssh://"):
        remote_str = remote_str[6:]

    # Parse user@host:port
    if "@" not in remote_str:
        raise ValueError("Remote string must contain '@' (format: user@host:port)")

    user, rest = remote_str.split("@", 1)

    if ":" in rest:
        host, port_str = rest.rsplit(":", 1)
        port = int(port_str)
    else:
        host = rest
        port = 22

    return RemoteConfig(host=host, port=port, user=user)
