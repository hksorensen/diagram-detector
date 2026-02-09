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
import time
import threading
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from network_utils import SFTPUploader
    SFTP_AVAILABLE = True
except ImportError:
    SFTP_AVAILABLE = False
    SFTPUploader = None

from .models import DetectionResult, DiagramDetection
from .utils import get_image_files


@dataclass
class RemoteConfig:
    """
    Configuration for remote SSH server.

    All fields are required - load from YAML config file using from_yaml()
    or auto_load().
    """

    # SSH connection settings
    user: str
    endpoints: List[tuple]  # List of (host, port) tuples to try in order
    max_rsync_retries: int = 3
    connection_timeout: float = 2.0

    # Remote server paths
    remote_work_dir: str = "~/diagram-detector"
    python_path: str = "~/diagram-detector/.venv/bin/python"

    # Detection parameters (defaults, can be overridden in detector)
    model: str = "yolo11m"
    confidence: float = 0.35
    iou: float = 0.30
    imgsz: int = 640
    batch_size: int = 1000
    max_workers: int = 8
    tensorrt: bool = False  # Use TensorRT optimization (NVIDIA GPU only)

    # Auto-detected fields (set by is_remote_available)
    host: str = None  # Will be set to working endpoint
    port: int = None  # Will be set to working endpoint port

    def __post_init__(self):
        """Validate required fields."""
        if not self.user:
            raise ValueError("user is required in remote config")
        if not self.endpoints or len(self.endpoints) == 0:
            raise ValueError("endpoints list is required in remote config")

        # Set host/port from first endpoint if not already set
        if self.host is None or self.port is None:
            self.host, self.port = self.endpoints[0]

    @property
    def ssh_target(self) -> str:
        """Get SSH connection string."""
        return f"{self.user}@{self.host}"

    @property
    def ssh_port_args(self) -> List[str]:
        """Get SSH port arguments."""
        return ["-p", str(self.port)] if self.port != 22 else []

    def get_rsync_ssh_args(self, control_path: Optional[str] = None) -> List[str]:
        """
        Get rsync SSH arguments (uses -e for custom SSH command).

        Args:
            control_path: Optional SSH ControlMaster socket path for connection reuse

        Returns:
            List of rsync arguments including -e with SSH options
        """
        ssh_opts = []

        # Port
        if self.port != 22:
            ssh_opts.append(f"-p {self.port}")

        # Connection multiplexing (reuse single SSH connection for multiple rsync operations)
        if control_path:
            ssh_opts.extend([
                f"-o ControlMaster=auto",
                f"-o ControlPath={control_path}",
                f"-o ControlPersist=600",  # Keep connection alive for 10 minutes
            ])

        # Keep-alive to detect dead connections
        ssh_opts.extend([
            "-o ServerAliveInterval=15",  # Send keepalive every 15s
            "-o ServerAliveCountMax=3",    # Fail after 3 missed keepalives (45s)
        ])

        # Batch mode (don't ask for passwords/confirmations)
        ssh_opts.append("-o BatchMode=yes")

        # Combine into rsync -e argument
        if ssh_opts:
            return ["-e", f"ssh {' '.join(ssh_opts)}"]
        return []

    @property
    def rsync_ssh_args(self) -> List[str]:
        """Get rsync SSH arguments without multiplexing (backward compatibility)."""
        return self.get_rsync_ssh_args(control_path=None)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path], key: Optional[str] = None) -> "RemoteConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file
            key: Optional key to extract subsection (e.g., "detect_diagrams.remote_detection")

        Returns:
            RemoteConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required fields are missing or key not found

        Example YAML (standalone):
            user: hkragh
            endpoints:
              - [192.168.1.183, 22]
              - [thinkcentre.local, 22]
            remote_work_dir: ~/diagram-detector
            python_path: ~/diagram-detector/.venv/bin/python

        Example YAML (subsection):
            detect_diagrams:
              remote_detection:
                user: hkragh
                endpoints:
                  - [192.168.1.183, 22]
        """
        import yaml

        config_path = Path(config_path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Extract subsection if key provided
        if key:
            for key_part in key.split('.'):
                if not isinstance(data, dict) or key_part not in data:
                    raise ValueError(f"Key '{key}' not found in config file: {config_path}")
                data = data[key_part]

        # Convert endpoints from list of lists to list of tuples
        if "endpoints" in data and data["endpoints"]:
            data["endpoints"] = [tuple(e) for e in data["endpoints"]]

        # Remove host/port from data if present (will be set from endpoints)
        data.pop("host", None)
        data.pop("port", None)

        return cls(**data)

    @classmethod
    def auto_load(cls) -> "RemoteConfig":
        """
        Auto-load configuration from standard locations.

        Search order:
        1. DIAGRAM_DETECTOR_CONFIG environment variable
        2. ./pipeline/config.yaml (working directory - detect_diagrams.remote_detection)
        3. ./config.yaml (working directory - detect_diagrams.remote_detection or root)
        4. ~/.diagram-detector/config.yaml (user config)

        Returns:
            RemoteConfig instance

        Raises:
            FileNotFoundError: If no config file found in standard locations
        """
        import os

        # Check environment variable first
        env_config = os.environ.get("DIAGRAM_DETECTOR_CONFIG")
        if env_config:
            env_path = Path(env_config).expanduser()
            if env_path.exists():
                # Try subsection first, fall back to root
                try:
                    return cls.from_yaml(env_path, key="detect_diagrams.remote_detection")
                except (ValueError, KeyError):
                    return cls.from_yaml(env_path)
            else:
                raise FileNotFoundError(
                    f"Config file specified in DIAGRAM_DETECTOR_CONFIG not found: {env_path}"
                )

        # Check standard locations
        search_paths = [
            (Path.cwd() / "pipeline" / "config.yaml", "detect_diagrams.remote_detection"),
            (Path.cwd() / "config.yaml", "detect_diagrams.remote_detection"),
            (Path.cwd() / "config.yaml", None),  # Try root level too
            (Path.home() / ".diagram-detector" / "config.yaml", "detect_diagrams.remote_detection"),
            (Path.home() / ".diagram-detector" / "config.yaml", None),
        ]

        for path, key in search_paths:
            if path.exists():
                try:
                    return cls.from_yaml(path, key=key)
                except (ValueError, KeyError):
                    # Key not found, try next location
                    continue

        # No config found - provide helpful error
        raise FileNotFoundError(
            "No remote detection config found. Add to one of:\n"
            "  - ./pipeline/config.yaml (detect_diagrams.remote_detection section)\n"
            "  - ./config.yaml (detect_diagrams.remote_detection section)\n"
            "  - ~/.diagram-detector/config.yaml\n"
            "Or set DIAGRAM_DETECTOR_CONFIG environment variable.\n"
            "\n"
            "See pipeline/config.yaml for example configuration."
        )


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
        config: Optional[Union[RemoteConfig, str, Path]] = None,
        batch_size: Optional[int] = None,
        gpu_batch_size: int = 32,
        model: Optional[str] = None,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        verbose: bool = True,
        run_id: Optional[str] = None,
        config_dir: Optional[Path] = None,
        tensorrt: Optional[bool] = None,
        upload_workers: Optional[int] = None,
        use_persistent_server: bool = False,
    ):
        """
        Initialize remote detector.

        Args:
            config: RemoteConfig, connection string (user@host:port), path to YAML,
                   or None to auto-load from standard locations
            batch_size: Images per batch (defaults from config)
            model: Model to use on remote (defaults from config)
            confidence: Confidence threshold (defaults from config)
            iou: IoU threshold for NMS (defaults from config)
            imgsz: Image size for preprocessing (defaults from config)
            verbose: Print progress
            run_id: Unique run identifier (auto-generated if None)
            config_dir: Local directory to store run config YAML (for git tracking)
            tensorrt: Use TensorRT optimization on remote (defaults from config)
            upload_workers: Number of parallel rsync workers (None = auto: min(5, max(1, files//50)))
            use_persistent_server: Keep model resident in GPU memory across sub-batches
                via a long-lived SSH process running diagram_detector.server.
                Eliminates ~3-5 s model-reload overhead per sub-batch.
        """
        # Load config
        if config is None:
            self.config = RemoteConfig.auto_load()
        elif isinstance(config, (str, Path)):
            # Check if it's a file path or connection string
            config_str = str(config)
            if "@" in config_str and not Path(config_str).exists():
                # Connection string format: user@host:port
                self.config = self._parse_connection_string(config_str)
            else:
                # File path
                self.config = RemoteConfig.from_yaml(config)
        else:
            self.config = config

        # Use config defaults or explicit parameters
        self.batch_size = batch_size if batch_size is not None else self.config.batch_size
        self.gpu_batch_size = gpu_batch_size
        self.model = model if model is not None else self.config.model
        self.confidence = confidence if confidence is not None else self.config.confidence
        self.iou = iou if iou is not None else self.config.iou
        self.imgsz = imgsz if imgsz is not None else self.config.imgsz
        self.tensorrt = tensorrt if tensorrt is not None else self.config.tensorrt
        self.verbose = verbose
        self.upload_workers = upload_workers  # None = auto-calculate, int = fixed count
        self.use_persistent_server = use_persistent_server
        self._server_proc = None  # subprocess.Popen when persistent server is running

        # Generate run ID if not provided
        if run_id is None:
            from datetime import datetime
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_id = run_id

        # Config directory for git-tracked run configs
        self.config_dir = Path(config_dir) if config_dir else None
        self._run_config_path = None

        # SSH connection multiplexing (reuse single connection for multiple rsync operations)
        # Use short path to avoid Unix domain socket path length limit (104 bytes on macOS)
        # Format: /tmp/dd-{PID}-{timestamp} (e.g., /tmp/dd-12345-150827)
        import os
        timestamp = datetime.now().strftime("%H%M%S")
        socket_name = f"dd-{os.getpid()}-{timestamp}"
        self._ssh_control_path = f"/tmp/{socket_name}"

        # Persistent SFTP uploader - DISABLED (paramiko SFTP is not thread-safe)
        # Parallel uploads cause "Server connection dropped" errors
        # Using rsync instead, which is reliable even if slightly slower
        self._sftp_uploader = None
        self._sftp_connected = False

        # Verify SSH connection
        self._verify_connection()

        # Measure network throughput (useful for understanding upload bottlenecks)
        if verbose:
            self._measure_network_throughput()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup SSH control socket."""
        self.cleanup()
        return False

    def cleanup(self):
        """Clean up SSH control socket and connections."""
        # Stop persistent inference server if running
        self._stop_persistent_server()

        # Close persistent SFTP uploader if it exists
        if hasattr(self, '_sftp_uploader') and self._sftp_uploader is not None:
            try:
                self._sftp_uploader.close()
            except Exception:
                pass  # Ignore cleanup errors
            self._sftp_uploader = None

        # Close SSH control master connection if it exists
        if hasattr(self, '_ssh_control_path') and self._ssh_control_path:
            control_path = Path(self._ssh_control_path)
            if control_path.exists():
                # Ask SSH to close the control master connection
                try:
                    cmd = [
                        "ssh",
                        "-O", "exit",
                        "-o", f"ControlPath={self._ssh_control_path}",
                    ] + self.config.ssh_port_args + [self.config.ssh_target]
                    subprocess.run(cmd, capture_output=True, timeout=5)
                except Exception:
                    pass  # Ignore cleanup errors

                # Remove control socket file if it still exists
                try:
                    control_path.unlink()
                except Exception:
                    pass

    def get_gpu_memory(self) -> tuple[int, int]:
        """
        Query GPU memory usage on remote server.

        Returns:
            Tuple of (memory_used_mb, memory_free_mb)
        """
        cmd = [
            "ssh"
        ] + self.config.ssh_port_args + [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self._ssh_control_path}",
            "-o", "ControlPersist=600",
            self.config.ssh_target,
            "nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                output = result.stdout.strip()
                # Parse: "3456, 4651" → (3456, 4651)
                parts = output.split(",")
                if len(parts) == 2:
                    used_mb = int(parts[0].strip())
                    free_mb = int(parts[1].strip())
                    return used_mb, free_mb
        except Exception:
            pass

        # Return 0,0 if query fails
        return 0, 0

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

        # Create RemoteConfig with endpoints parameter (list of tuples)
        return RemoteConfig(user=user, endpoints=[(host, port)])

    def _create_run_config(self, gpu_batch_size: int = 32) -> Path:
        """
        Create run-level YAML config file (for git tracking and reproducibility).

        Config is created ONCE per run and shared across all batches.

        Args:
            gpu_batch_size: GPU batch size for this run

        Returns:
            Path to created config file (local)
        """
        if self._run_config_path is not None:
            return self._run_config_path

        # Determine where to save config
        if self.config_dir:
            config_dir = self.config_dir
        else:
            # Default: current working directory / configs
            config_dir = Path.cwd() / "configs"

        config_dir.mkdir(parents=True, exist_ok=True)

        # Create config dict
        config_data = {
            "run_id": self.run_id,
            "model": self.model,
            "confidence": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "batch_size": gpu_batch_size,
            "tensorrt": self.tensorrt,
            "format": "json",
            "use_cache": False,  # PRIMARY FIX: Force remote to reprocess, never use stale cache
            "created": datetime.now().isoformat(),
            "remote": {
                "host": self.config.host,
                "port": self.config.port,
                "user": self.config.user,
            },
        }

        # Write as YAML
        config_path = config_dir / f"{self.run_id}.yaml"

        try:
            import yaml
            with open(config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            # Fallback to JSON if PyYAML not available
            import json
            config_path = config_dir / f"{self.run_id}.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

        if self.verbose:
            print(f"✓ Run config created: {config_path}")
            print(f"  (This config is used for ALL batches in this run)")

        self._run_config_path = config_path
        return config_path

    def _commit_run_config(self, message: Optional[str] = None) -> bool:
        """
        Git commit the run config file (if in a git repo).

        Args:
            message: Commit message (auto-generated if None)

        Returns:
            True if committed successfully, False otherwise
        """
        if self._run_config_path is None:
            return False

        config_file = self._run_config_path

        # Check if in git repo
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=config_file.parent,
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError:
            if self.verbose:
                print(f"  ⚠ Not in a git repository, skipping auto-commit")
            return False

        # Add config file
        try:
            subprocess.run(
                ["git", "add", str(config_file)],
                cwd=config_file.parent,
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"  ⚠ Failed to git add config: {e}")
            return False

        # Generate commit message
        if message is None:
            message = f"Add run config: {self.run_id}\n\nParameters:\n- model: {self.model}\n- confidence: {self.confidence}\n- iou: {self.iou}"

        # Commit
        try:
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=config_file.parent,
                capture_output=True,
                check=True
            )

            if self.verbose:
                print(f"  ✓ Config committed to git: {config_file.name}")

            return True

        except subprocess.CalledProcessError:
            # Might fail if no changes (already committed) - that's ok
            return False

    def _verify_connection(self, max_retries: int = 5, initial_delay: float = 5.0) -> None:
        """Verify SSH connection works with retry logic.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds (doubles each retry)

        Raises:
            RuntimeError: If connection fails after all retries
        """
        import time

        if self.verbose:
            print(f"Verifying SSH connection to {self.config.ssh_target}:{self.config.port}...")

        delay = initial_delay
        last_error = None

        for attempt in range(max_retries):
            try:
                cmd = ["ssh"] + self.config.ssh_port_args + [self.config.ssh_target, 'echo "OK"']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                if result.returncode != 0:
                    raise RuntimeError(f"SSH connection failed: {result.stderr}")

                if self.verbose:
                    if attempt > 0:
                        print(f"✓ SSH connection verified (after {attempt + 1} attempts)")
                    else:
                        print("✓ SSH connection verified")
                return  # Success!

            except subprocess.TimeoutExpired as e:
                last_error = RuntimeError("SSH connection timed out")
            except Exception as e:
                last_error = RuntimeError(f"SSH connection failed: {e}")

            # If this wasn't the last attempt, wait and retry
            if attempt < max_retries - 1:
                retry_msg = f"⚠ SSH connection failed (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f}s..."
                if self.verbose:
                    print(retry_msg)
                else:
                    # Always log retries even if not verbose (important for unsupervised runs)
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(retry_msg)

                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                # Last attempt failed
                error_msg = f"SSH connection failed after {max_retries} attempts: {last_error}"
                if self.verbose:
                    print(f"✗ {error_msg}")
                raise RuntimeError(error_msg)

    def _measure_network_throughput(self) -> None:
        """
        Measure network upload throughput using a test file.

        Uploads a 5MB test file and reports speed in MB/s.
        """
        import tempfile
        import time

        # Create 5MB test file
        test_size_mb = 5
        test_size_bytes = test_size_mb * 1024 * 1024

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
                # Write random data
                f.write(b'0' * test_size_bytes)
                test_file = Path(f.name)

            # Remote test location
            remote_test_dir = f"{self.config.remote_work_dir}/test"
            self._run_ssh_command(f"mkdir -p {remote_test_dir}", check=True)

            # Upload test file with rsync
            print(f"Measuring network throughput ({test_size_mb}MB test upload)...", end='', flush=True)

            start_time = time.time()
            cmd = (
                [
                    "rsync",
                    "-a",
                    "--quiet",
                ]
                + self.config.get_rsync_ssh_args(self._ssh_control_path)
                + [str(test_file), f"{self.config.ssh_target}:{remote_test_dir}/"]
            )

            result = subprocess.run(cmd, capture_output=True, timeout=60)
            upload_time = time.time() - start_time

            if result.returncode == 0 and upload_time > 0:
                throughput_mbps = test_size_mb / upload_time
                print(f" {throughput_mbps:.2f} MB/s")

                # Estimate how long 250 files (~250MB typical batch) should take
                typical_batch_mb = 250 * 1  # Assume ~1MB per image
                estimated_time = typical_batch_mb / throughput_mbps
                print(f"  Estimated upload time for typical batch (250 files, ~250MB): {estimated_time:.1f}s")
            else:
                print(" (test failed)")

            # Cleanup
            test_file.unlink(missing_ok=True)
            self._run_ssh_command(f"rm -rf {remote_test_dir}", check=False)

        except Exception as e:
            print(f" (error: {e})")
            # Don't fail initialization on throughput test failure

    def _run_ssh_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run command on remote server with retry logic for transient failures."""
        cmd = ["ssh"] + self.config.ssh_port_args + [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self._ssh_control_path}",
            "-o", "ControlPersist=600",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "BatchMode=yes",
            self.config.ssh_target,
            command
        ]

        max_attempts = self.config.max_rsync_retries
        last_error = None

        for attempt in range(1, max_attempts + 1):
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return result

            # Check if this is a transient error worth retrying
            stderr = result.stderr or ""
            is_transient = any(pattern in stderr.lower() for pattern in [
                "connection closed",
                "connection unexpectedly closed",
                "connection refused",
                "connection timed out",
                "connection reset",
                "broken pipe",
                "network is unreachable",
                "no route to host",
            ])

            last_error = (
                f"SSH command failed (attempt {attempt}/{max_attempts})\n"
                f"  Command: {command[:100]}{'...' if len(command) > 100 else ''}\n"
                f"  Return code: {result.returncode}\n"
                f"  Stderr: {stderr.strip() or '(empty)'}\n"
                f"  Stdout: {(result.stdout or '').strip() or '(empty)'}"
            )

            if not check:
                return result

            if not is_transient or attempt == max_attempts:
                raise RuntimeError(f"Remote command failed after {attempt} attempts.\n  {last_error}")

            # Transient error - retry with exponential backoff
            wait_time = 2 ** attempt  # 2s, 4s, 8s, ...
            if self.verbose:
                print(f"  ⚠ SSH connection error, retrying in {wait_time}s (attempt {attempt}/{max_attempts})...")
                print(f"    Error: {stderr.strip()}")
            time.sleep(wait_time)

        return result

    def _run_ssh_command_with_spinner(self, command: str, num_images: int = 0) -> subprocess.CompletedProcess:
        """
        Run SSH command with a visual spinner to show activity.

        Args:
            command: Command to run
            num_images: Number of images being processed (for context)

        Returns:
            CompletedProcess result
        """
        # Try fancy Unicode spinner, fallback to ASCII if terminal doesn't support it
        try:
            # Test Unicode support by encoding
            test = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏".encode(sys.stdout.encoding or 'utf-8')
            spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        except (AttributeError, UnicodeEncodeError, LookupError):
            # Fallback to ASCII spinner
            spinner_chars = ["|", "/", "-", "\\"]

        result_container = []
        exception_container = []
        stop_spinner = threading.Event()

        def run_command():
            try:
                result = self._run_ssh_command(command, check=True)
                result_container.append(result)
            except Exception as e:
                exception_container.append(e)
            finally:
                stop_spinner.set()

        # Start command in background thread
        thread = threading.Thread(target=run_command, daemon=True)
        thread.start()

        # Ensure we start on a new line
        print()

        # Show spinner while command runs
        start_time = time.time()
        spinner_idx = 0

        while not stop_spinner.is_set():
            elapsed = time.time() - start_time
            elapsed_int = int(elapsed)

            # Update spinner on every iteration
            spinner = spinner_chars[spinner_idx % len(spinner_chars)]

            if num_images > 0:
                # Show progress message with image count
                msg = f"  {spinner} Processing {num_images} images on remote GPU... ({elapsed_int}s)"
            else:
                msg = f"  {spinner} Processing on remote GPU... ({elapsed_int}s)"

            # Pad message to consistent length to avoid artifacts
            msg = msg.ljust(80)
            print(f"\r{msg}", end="", flush=True)

            spinner_idx += 1
            time.sleep(0.1)  # Update 10 times per second (default .2 for smooth animation)

        # Clear spinner line and add newline
        print("\r" + " " * 80, flush=True)

        # Wait for thread to complete
        thread.join()

        # Check for exceptions
        if exception_container:
            raise exception_container[0]

        return result_container[0] if result_container else None

    def _run_rsync_with_retry(
        self,
        cmd: List[str],
        operation: str = "rsync"
    ) -> subprocess.CompletedProcess:
        """
        Run rsync command with retry logic for transient failures.

        Args:
            cmd: rsync command as list
            operation: Description of operation (for error messages)

        Returns:
            Completed subprocess result

        Raises:
            RuntimeError: If all retry attempts fail
        """
        max_attempts = self.config.max_rsync_retries

        for attempt in range(1, max_attempts + 1):
            result = subprocess.run(cmd, capture_output=True)

            if result.returncode == 0:
                return result

            # Check if this is a transient error worth retrying
            stderr = result.stderr.decode() if result.stderr else ""
            is_transient = any(pattern in stderr.lower() for pattern in [
                "connection closed",
                "connection unexpectedly closed",
                "connection refused",
                "connection timed out",
                "broken pipe",
                "network is unreachable",
                "no route to host",
            ])

            if not is_transient or attempt == max_attempts:
                # Non-transient error or final attempt - fail immediately
                raise RuntimeError(f"{operation.capitalize()} failed: {result.stderr}")

            # Transient error - retry with exponential backoff
            wait_time = 2 ** attempt  # 2s, 4s, 8s, ...
            if self.verbose:
                print(f"  Connection error, retrying in {wait_time}s (attempt {attempt}/{max_attempts})...")
            time.sleep(wait_time)

        # Should never reach here, but just in case
        raise RuntimeError(f"{operation.capitalize()} failed after {max_attempts} attempts")

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
        """Upload batch of images via SFTP (with progress bar) or rsync (fallback)."""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[UPLOAD] _upload_batch called with {len(image_paths)} image paths")
        if len(image_paths) > 0:
            logger.debug(f"[UPLOAD]   First 3: {[p.name for p in image_paths[:3]]}")
            logger.debug(f"[UPLOAD]   Last 3: {[p.name for p in image_paths[-3:]]}")

        if self.verbose:
            print(f"  Uploading batch {batch_id} ({len(image_paths)} images)...")

        # Create temporary directory with batch images
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy images to temp directory with unique names
            # Use parent directory name to avoid filename collisions (e.g., multiple PDFs with page_0001.jpg)
            missing_files = []
            copy_errors = []
            for img_path in image_paths:
                if not img_path.exists():
                    missing_files.append(img_path)
                else:
                    try:
                        # Image filenames already include PDF identifier (pdfname_page_XXXX.jpg)
                        # No need to prepend parent dir name - would cause doubling
                        shutil.copy2(img_path, temp_path / img_path.name)
                    except Exception as e:
                        copy_errors.append((img_path, str(e)))

            import logging
            logger = logging.getLogger(__name__)

            if missing_files:
                logger.warning(f"Batch {batch_id}: {len(missing_files)}/{len(image_paths)} image files don't exist!")
                logger.warning(f"  First missing: {missing_files[0]}")
                logger.warning(f"  Last missing: {missing_files[-1]}")

            if copy_errors:
                logger.error(f"Batch {batch_id}: {len(copy_errors)}/{len(image_paths)} files failed to copy!")
                logger.error(f"  First error: {copy_errors[0][0]} - {copy_errors[0][1]}")
                logger.error(f"  Last error: {copy_errors[-1][0]} - {copy_errors[-1][1]}")

            # Count files in temp directory before rsync
            import logging
            logger = logging.getLogger(__name__)
            temp_files = list(temp_path.glob("*.jpg")) + list(temp_path.glob("*.png"))
            logger.debug(f"[UPLOAD] Batch {batch_id}: {len(temp_files)} files in temp dir (expected {len(image_paths)})")

            # Clear remote input directory before upload to avoid accumulation
            remote_input = f"{self.config.remote_work_dir}/input/{batch_id}/"
            cleanup_cmd = f"rm -rf {remote_input} && mkdir -p {remote_input}"
            self._run_ssh_command(cleanup_cmd, check=True)

            # Upload via parallel rsync
            upload_start = time.time()

            # CRITICAL FIX: Preserve image_paths order instead of using glob (which returns arbitrary filesystem order)
            # Build files_to_upload in the same order as image_paths to prevent contamination
            files_to_upload = [temp_path / img_path.name for img_path in image_paths
                             if (temp_path / img_path.name).exists()]
            num_files = len(files_to_upload)

            logger.debug(f"[UPLOAD] Starting parallel rsync upload of {num_files} files (order preserved from image_paths)")

            if num_files == 0:
                logger.warning(f"No files to upload for batch {batch_id}")
            else:
                # Split files into groups for parallel rsync
                # Use 5 parallel processes (safe for typical SSH MaxSessions limit)
                if self.upload_workers is not None:
                    # Use explicit worker count
                    num_parallel = max(1, min(self.upload_workers, num_files))
                else:
                    # Auto-calculate: at least 50 files per process, max 5 workers
                    num_parallel = min(5, max(1, num_files // 50))
                files_per_group = (num_files + num_parallel - 1) // num_parallel

                file_groups = []
                for i in range(num_parallel):
                    start_idx = i * files_per_group
                    end_idx = min((i + 1) * files_per_group, num_files)
                    if start_idx < num_files:
                        file_groups.append(files_to_upload[start_idx:end_idx])

                logger.info(f"[UPLOAD] Uploading {num_files} images in {len(file_groups)} parallel rsync processes")

                def rsync_file_group(group_idx: int, files: List[Path]) -> tuple:
                    """Upload a group of files via rsync, return (success, error)."""
                    try:
                        # Create temp file list for rsync --files-from
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                            for file in files:
                                f.write(f"{file.name}\n")
                            files_list_path = f.name

                        try:
                            # rsync with --files-from reads list of files to transfer
                            cmd = (
                                [
                                    "rsync",
                                    "-a",  # Archive mode (no compression)
                                    "--whole-file",  # Skip delta algorithm (files are new)
                                    "--inplace",  # Write directly, no temp files
                                    "--files-from", files_list_path,
                                ]
                                + self.config.get_rsync_ssh_args(self._ssh_control_path)
                                + [str(temp_path) + "/", f"{self.config.ssh_target}:{remote_input}"]
                            )

                            # Adaptive timeout: 5 min base + 10 sec per file (handles slow networks)
                            # For 250 files: 5min + 42min = 47min timeout
                            adaptive_timeout = 300 + (len(files) * 10)

                            result = subprocess.run(cmd, capture_output=True, timeout=adaptive_timeout)
                            if result.returncode != 0:
                                return (False, f"rsync rc={result.returncode}, stderr={result.stderr.decode()[:200]}")
                            return (True, None)
                        finally:
                            Path(files_list_path).unlink(missing_ok=True)
                    except Exception as e:
                        return (False, str(e))

                # Upload groups in parallel
                upload_errors = []

                if self.verbose and len(file_groups) > 1:
                    from tqdm import tqdm
                    progress = tqdm(
                        total=len(file_groups),
                        desc=f"  Uploading {num_files} files",
                        unit="process",
                        leave=False,
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} processes'
                    )

                if len(file_groups) == 1:
                    # Single group - just use regular rsync
                    logger.info(f"[UPLOAD] Single group, using sequential rsync")
                    cmd = (
                        [
                            "rsync",
                            "-a",
                            "--whole-file",  # Skip delta algorithm
                            "--inplace",  # No temp files
                            "--quiet",
                        ]
                        + self.config.get_rsync_ssh_args(self._ssh_control_path)
                        + [f"{temp_path}/", f"{self.config.ssh_target}:{remote_input}"]
                    )
                    self._run_rsync_with_retry(cmd, operation="upload")
                else:
                    # Multiple groups - parallel rsync
                    with ThreadPoolExecutor(max_workers=len(file_groups)) as executor:
                        # CRITICAL FIX: Store futures in order, not as dict
                        # Wait for completion in submission order to preserve file order on remote
                        futures_in_order = [
                            (executor.submit(rsync_file_group, idx, group), idx)
                            for idx, group in enumerate(file_groups)
                        ]

                        for future, group_idx in futures_in_order:
                            success, error = future.result()  # Wait in submission order!

                            if not success:
                                logger.error(f"Rsync group {group_idx} failed: {error}")
                                upload_errors.append((group_idx, error))

                            if self.verbose and len(file_groups) > 1:
                                progress.update(1)

                    if self.verbose and len(file_groups) > 1:
                        progress.close()

                    if upload_errors:
                        raise RuntimeError(f"Parallel rsync upload failed: {upload_errors[0][1]}")

                upload_time = time.time() - upload_start
                logger.debug(f"[UPLOAD] Upload complete: {num_files} files in {upload_time:.1f}s ({num_files/upload_time:.1f} files/s)")
                if self.verbose:
                    print(f"  ✓ Upload complete: {num_files} files in {upload_time:.1f}s ({num_files/upload_time:.1f} files/s)")

            # Verify upload - check file count and sizes on remote
            # First: count files
            count_cmd = f"ls {remote_input} | wc -l"
            result = self._run_ssh_command(count_cmd, check=False)
            if result.returncode == 0:
                remote_count = int(result.stdout.strip())
                logger.info(f"[UPLOAD] Upload verification: {remote_count} files in {remote_input} (expected {len(image_paths)})")
                if remote_count != len(image_paths):
                    logger.error(f"[UPLOAD] MISMATCH: Only {remote_count}/{len(image_paths)} files uploaded!")
                    # List what files are actually there
                    ls_result = self._run_ssh_command(f"ls -la {remote_input}", check=False)
                    logger.error(f"[UPLOAD] Remote directory contents:\n{ls_result.stdout}")
                    raise RuntimeError(f"Upload verification failed: expected {len(image_paths)} files, found {remote_count}")

                # Second: verify file sizes match
                logger.info(f"[UPLOAD] Verifying file sizes...")
                size_cmd = f"du -b {remote_input}* | sort -k2"
                size_result = self._run_ssh_command(size_cmd, check=False)
                if size_result.returncode == 0:
                    remote_files = {}
                    for line in size_result.stdout.strip().split('\n'):
                        if line:
                            parts = line.split('\t')
                            if len(parts) == 2:
                                size_str, filepath = parts
                                filename = Path(filepath).name
                                remote_files[filename] = int(size_str)

                    # Compare with local files
                    local_sizes = {f.name: f.stat().st_size for f in temp_files}
                    size_mismatches = []
                    for filename, local_size in local_sizes.items():
                        remote_size = remote_files.get(filename, -1)
                        if remote_size != local_size:
                            size_mismatches.append(f"{filename}: local={local_size}B, remote={remote_size}B")

                    if size_mismatches:
                        logger.error(f"[UPLOAD] SIZE MISMATCH: {len(size_mismatches)} files have wrong size!")
                        for mismatch in size_mismatches[:5]:  # Show first 5
                            logger.error(f"  {mismatch}")
                        raise RuntimeError(f"Upload verification failed: {len(size_mismatches)} files have incorrect size")
                    else:
                        logger.info(f"[UPLOAD] ✓ Size verification passed: all {len(local_sizes)} files match")
                else:
                    logger.warning(f"[UPLOAD] Could not verify file sizes (du command failed)")

                # Third: Force filesystem sync and add small delay
                logger.info(f"[UPLOAD] Syncing filesystem...")
                self._run_ssh_command(f"sync", check=False)
                time.sleep(0.5)  # 500ms delay for filesystem to settle
                logger.info(f"[UPLOAD] ✓ Verification passed: all {remote_count} files present with correct sizes")
            else:
                logger.error(f"[UPLOAD] Failed to verify upload - ls command failed")
                logger.error(f"  Return code: {result.returncode}")
                logger.error(f"  Stderr: {result.stderr}")
                raise RuntimeError(f"Upload verification failed: ls command returned {result.returncode}")

    def _run_inference_batch(self, batch_id: str, gpu_batch_size: int = 32, num_images: int = 0) -> None:
        """Run inference on batch using run-level config."""

        # Create run-level config (only once per run)
        run_config_local = self._create_run_config(gpu_batch_size)

        # Remote paths
        input_dir = f"{self.config.remote_work_dir}/input/{batch_id}"
        output_dir = f"{self.config.remote_work_dir}/output/{batch_id}"
        remote_config_dir = f"{self.config.remote_work_dir}/configs"

        # Debug: Check what's actually in the input directory before inference
        import logging
        logger = logging.getLogger(__name__)
        debug_cmd = f"ls -la {input_dir} | head -20"
        debug_result = self._run_ssh_command(debug_cmd, check=False)
        logger.info(f"[INFERENCE] Verifying input directory before inference: {input_dir}")
        logger.info(f"[INFERENCE] Directory listing (first 20 files):\n{debug_result.stdout}")
        remote_config_file = f"{remote_config_dir}/{self.run_id}.yaml"

        # Upload run config to remote (idempotent - ok to upload multiple times)
        mkdir_cmd = f"mkdir -p {remote_config_dir}"
        self._run_ssh_command(mkdir_cmd, check=True)

        # SCP upload with retry logic for transient connection failures
        scp_cmd = [
            "scp",
            "-P", str(self.config.port),
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "BatchMode=yes",
            str(run_config_local),
            f"{self.config.user}@{self.config.host}:{remote_config_file}"
        ]

        max_attempts = self.config.max_rsync_retries
        last_error = None

        for attempt in range(1, max_attempts + 1):
            result = subprocess.run(scp_cmd, capture_output=True)

            if result.returncode == 0:
                break

            stderr = result.stderr.decode() if result.stderr else ""
            stdout = result.stdout.decode() if result.stdout else ""

            # Check if this is a transient error worth retrying
            is_transient = any(pattern in stderr.lower() for pattern in [
                "connection closed",
                "connection unexpectedly closed",
                "connection refused",
                "connection timed out",
                "broken pipe",
                "network is unreachable",
                "no route to host",
            ])

            last_error = (
                f"SCP upload failed (attempt {attempt}/{max_attempts})\n"
                f"  Command: {' '.join(scp_cmd)}\n"
                f"  Return code: {result.returncode}\n"
                f"  Stderr: {stderr.strip() or '(empty)'}\n"
                f"  Stdout: {stdout.strip() or '(empty)'}"
            )

            if not is_transient or attempt == max_attempts:
                # Non-transient error or final attempt - fail with detailed message
                # Also verify SSH connection state for better diagnostics
                ssh_check = subprocess.run(
                    ["ssh"] + self.config.ssh_port_args + [self.config.ssh_target, "echo OK"],
                    capture_output=True, text=True, timeout=10
                )
                ssh_status = "OK" if ssh_check.returncode == 0 else f"FAILED ({ssh_check.stderr.strip()})"

                raise RuntimeError(
                    f"Failed to upload config after {attempt} attempts.\n"
                    f"  {last_error}\n"
                    f"  SSH connection check: {ssh_status}"
                )

            # Transient error - retry with exponential backoff
            wait_time = 2 ** attempt  # 2s, 4s, 8s, ...
            if self.verbose:
                print(f"  ⚠ SCP connection error, retrying in {wait_time}s (attempt {attempt}/{max_attempts})...")
                print(f"    Error: {stderr.strip()}")
            time.sleep(wait_time)

        # Run inference with config file + override input/output for this batch
        # Redirect stdout/stderr to log files for debugging
        log_file = f"{output_dir}/inference.log"
        err_file = f"{output_dir}/inference.err"
        cmd = (
            f"cd {self.config.remote_work_dir} && "
            f"mkdir -p {output_dir} && "  # Ensure output dir exists for log redirection
            f"echo 'Starting inference batch {batch_id}...' > {log_file} && "  # Overwrite old log
            f"rm -f {err_file} && "  # Clean old error file to prevent stale errors
            # Set file descriptor limit to prevent exhaustion with many images
            # Must be set here since SSH non-interactive shells don't source ~/.bashrc
            f"ulimit -n 4096 && "
            # Check ulimit and GPU memory before inference
            f"echo 'File descriptor limit: '$(ulimit -n) >> {log_file} && "
            f"echo 'Processing {num_images} images' >> {log_file} && "
            f"nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits >> {log_file} 2>/dev/null && "
            f"echo '---' >> {log_file} && "
            # Removed dummy error string - let stderr create the file naturally
            f"{self.config.python_path} -u -m diagram_detector.cli "  # -u for unbuffered output
            f"--config {remote_config_file} "
            f"--input {input_dir} "
            f"--output {output_dir} "
            f"--quiet "
            f">> {log_file} 2>> {err_file}; "  # Semicolon to capture exit code
            f"EXIT_CODE=$?; "
            f"echo \"Exit code: $EXIT_CODE\" >> {log_file}; "
            f"nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits >> {log_file} 2>/dev/null; "
            f"exit $EXIT_CODE"  # Preserve original exit code
        )

        # Run inference (show spinner if verbose)
        if self.verbose and num_images > 0:
            self._run_ssh_command_with_spinner(cmd, num_images)
        else:
            self._run_ssh_command(cmd, check=True)

        if self.verbose:
            print(f"  ✓ Batch {batch_id} processed and ready for download")

    def _download_results(self, batch_id: str, output_dir: Path) -> Path:
        """Download results from remote server."""
        # if self.verbose:
        #     print(f"  Downloading results...")

        # Create local output directory (organized by run)
        run_output = output_dir / self.run_id
        batch_output = run_output / batch_id
        batch_output.mkdir(parents=True, exist_ok=True)

        # Rsync results (quiet - just show summary)
        remote_output = f"{self.config.remote_work_dir}/output/{batch_id}/"

        cmd = (
            [
                "rsync",
                "-az",
                "--quiet",  # Always quiet - no per-file output
            ]
            + self.config.get_rsync_ssh_args(self._ssh_control_path)
            + [f"{self.config.ssh_target}:{remote_output}", str(batch_output)]
        )

        # Download with retry logic for transient failures
        self._run_rsync_with_retry(cmd, operation="download")

        # Copy run config to run output directory (once per run)
        if self._run_config_path and not (run_output / "config.yaml").exists():
            import shutil
            shutil.copy2(self._run_config_path, run_output / "config.yaml")
            if self.verbose:
                print(f"  ✓ Run config copied to: {run_output / 'config.yaml'}")

        # Check for errors in the log files
        err_file = batch_output / "inference.err"
        log_file = batch_output / "inference.log"

        if err_file.exists() and err_file.stat().st_size > 0:
            import logging
            logger = logging.getLogger(__name__)

            # Read error log
            with open(err_file, 'r') as f:
                errors = f.read()

            # Filter out dummy test lines (written before inference starts)
            error_lines = []
            for line in errors.strip().split('\n'):
                # Skip dummy lines that we write for testing
                if line.startswith("Batch: "):
                    continue
                error_lines.append(line)

            # Only log if there are actual errors (not just dummy lines)
            if error_lines:
                logger.error(f"Remote inference errors for batch {batch_id}:")
                logger.error(f"{'='*60}")
                for line in error_lines[-50:]:  # Last 50 lines
                    logger.error(f"  {line}")
                logger.error(f"{'='*60}")
                logger.error(f"Full error log saved to: {err_file}")

        if self.verbose:
            print(f"✓ Batch {batch_id} results downloaded to: {batch_output}")
            if log_file.exists():
                print(f"  Log files: {log_file}, {err_file}")

        return batch_output

    def _cleanup_batch(self, batch_id: str) -> None:
        """Clean up batch files on remote server."""
        if self.verbose:
            print(f"Cleaning up batch {batch_id}...")

        # Remove batch directories (keep run config for potential resume)
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

    # -----------------------------------------------------------------------
    # Persistent inference server — keeps the model resident in GPU memory
    # across all sub-batches via a single long-lived SSH process.
    # -----------------------------------------------------------------------

    def _start_persistent_server(self) -> None:
        """Start the persistent inference server over SSH.

        Opens a subprocess with an SSH connection to the remote.  The remote
        side runs ``python -m diagram_detector.server``.  We block until we
        read ``READY`` on stdout (i.e. the model is loaded).
        """
        if self._server_proc is not None:
            return  # already running

        tensorrt_flag = "--tensorrt " if self.tensorrt else ""
        cmd = (
            ["ssh"] + self.config.ssh_port_args + [self.config.ssh_target,
            f"cd {self.config.remote_work_dir} && "
            f"ulimit -n 4096 && "
            f"{self.config.python_path} -u -m diagram_detector.server "
            f"--model {self.model} "
            f"--confidence {self.confidence} "
            f"--iou {self.iou} "
            f"--imgsz {self.imgsz} "
            f"--device auto "
            f"{tensorrt_flag}"
            f"--batch-size {self.gpu_batch_size}"]
        )

        if self.verbose:
            print("  Starting persistent inference server on remote...")
            if self.tensorrt:
                print("    (TensorRT: first run exports engine ~5 min; cached runs ~5-15 s)")
            else:
                print("    (loading model into GPU — one-time per run)")

        # stderr → DEVNULL: the server redirects all detector output (YOLO
        # progress bars, [DEBUG] lines) to stderr.  Capturing it via PIPE would
        # fill the kernel pipe buffer and deadlock while we block on stdout.
        # Diagnostics after a crash are available via remote inference.log/err.
        self._server_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        # Block until READY (model loaded on remote GPU)
        line = self._server_proc.stdout.readline().strip()
        if line != "READY":
            self._server_proc.kill()
            self._server_proc = None
            raise RuntimeError(
                f"Persistent server did not send READY (got: {line!r}). "
                f"Check remote stderr / inference.err for details."
            )

        if self.verbose:
            print("  ✓ Persistent server ready (model loaded)")

    def _run_inference_persistent(self, batch_id: str, num_images: int) -> None:
        """Send an infer command to the persistent server and wait for completion.

        Args:
            batch_id: Batch identifier (used to locate input/output dirs on remote)
            num_images: Number of images (for logging only)

        Raises:
            RuntimeError: If the server reports an error or the process died.
        """
        if self._server_proc is None or self._server_proc.poll() is not None:
            raise RuntimeError(
                "Persistent server process is not running. "
                "Call _start_persistent_server() first."
            )

        input_dir  = f"{self.config.remote_work_dir}/input/{batch_id}"
        output_dir = f"{self.config.remote_work_dir}/output/{batch_id}"  # must match _download_results

        cmd = json.dumps({"type": "infer", "input": input_dir, "output": output_dir})
        self._server_proc.stdin.write(cmd + "\n")
        self._server_proc.stdin.flush()

        # Block until we get a valid JSON response line.  Any non-JSON lines
        # (e.g. stray C-extension output that leaked past fd redirect) are
        # logged and skipped.
        while True:
            response_line = self._server_proc.stdout.readline().strip()
            if not response_line:
                # EOF — server died
                self._server_proc.wait()
                self._server_proc = None
                raise RuntimeError(
                    f"Persistent server died during inference (batch {batch_id}). "
                    f"Check remote stderr / inference.err for details."
                )
            try:
                response = json.loads(response_line)
                break  # valid JSON — proceed
            except json.JSONDecodeError:
                # Non-JSON line leaked through — skip and keep reading
                if self.verbose:
                    print(f"  [server] Skipping non-JSON stdout: {response_line[:80]!r}")
        if response["status"] == "ERROR":
            raise RuntimeError(
                f"Persistent server error on batch {batch_id}: {response['error']}"
            )

        if response["status"] != "DONE":
            raise RuntimeError(
                f"Unexpected response from persistent server: {response}"
            )

        if self.verbose:
            print(f"  ✓ Batch {batch_id} processed (persistent server, {num_images} images)")

    def _stop_persistent_server(self) -> None:
        """Send shutdown command and wait for the server process to exit."""
        if self._server_proc is None:
            return

        try:
            if self._server_proc.poll() is None:
                self._server_proc.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                self._server_proc.stdin.flush()
                self._server_proc.wait(timeout=5)
        except Exception:
            # Best-effort — kill if it won't cooperate
            try:
                self._server_proc.kill()
                self._server_proc.wait(timeout=2)
            except Exception:
                pass

        self._server_proc = None
        if self.verbose:
            print("  ✓ Persistent server stopped")

    def detect(
        self,
        input_path: Union[str, Path, List[Path]],
        output_dir: Optional[Path] = None,
        gpu_batch_size: Optional[int] = None,
        cleanup: bool = True,
        resume: bool = False,
        auto_git_commit: bool = False,
    ) -> List[DetectionResult]:
        """
        Run remote inference on images.

        Args:
            input_path: Image file, directory, or list of paths
            output_dir: Where to save results locally
            gpu_batch_size: Batch size for GPU inference (16-64 typical)
            cleanup: Clean up remote files after processing
            resume: Resume from partially completed job
            auto_git_commit: Automatically git commit the run config (if in git repo)

        Returns:
            List of DetectionResult objects
        """
        # Parse input
        if isinstance(input_path, list):
            image_paths = [Path(p) for p in input_path]
        elif isinstance(input_path, (str, Path)):
            input_path = Path(input_path)
            if input_path.is_dir():
                # get_image_files() returns filesystem order (no sorting)
                # This preserves the carefully maintained priority_list order from rsync upload
                image_paths = get_image_files(input_path)
            else:
                image_paths = [input_path]
        else:
            raise ValueError("input_path must be path, directory, or list of paths")

        if not image_paths:
            raise ValueError("No images found")

        # Use instance gpu_batch_size if not provided
        if gpu_batch_size is None:
            gpu_batch_size = self.gpu_batch_size

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

        # Create run config (before first batch)
        self._create_run_config(gpu_batch_size)

        # Auto-commit config if requested
        if auto_git_commit:
            self._commit_run_config()

        # Calculate batches
        num_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        if self.verbose:
            print(f"Processing {len(image_paths):,} images in {num_batches} batch(es)...")
            if self.use_persistent_server:
                print(f"  Mode: persistent server (model loads once)\n")
            else:
                print(f"  Mode: per-batch spawn (model reloads each batch)\n")

        # Start persistent inference server (if requested) — model loads here
        if self.use_persistent_server:
            self._start_persistent_server()

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
                    import logging
                    _logger = logging.getLogger(__name__)
                    _logger.info(
                        "[REMOTE INFERENCE] resume batch_id=%s expected=%d received=%d",
                        batch_id, len(batch_paths), len(results),
                    )
                    if len(results) != len(batch_paths):
                        _logger.error(
                            "Resumed batch result count mismatch: batch_id=%s expected=%d received=%d",
                            batch_id, len(batch_paths), len(results),
                        )
                        raise RuntimeError(
                            f"Resumed batch {batch_id} has {len(results)} results for {len(batch_paths)} images. "
                            f"Remove {batch_output_dir} and re-run without resume, or fix the batch."
                        )
                    all_results.extend(results)
                    continue

                # 1. Upload batch
                upload_start = time.time()
                self._upload_batch(batch_paths, batch_id)
                upload_time = time.time() - upload_start

                # 2. Run inference
                inference_start = time.time()
                if self.use_persistent_server:
                    self._run_inference_persistent(batch_id, num_images=len(batch_paths))
                else:
                    self._run_inference_batch(batch_id, gpu_batch_size, num_images=len(batch_paths))
                inference_time = time.time() - inference_start

                # 3. Download results
                download_start = time.time()
                batch_results_dir = self._download_results(batch_id, output_dir)
                download_time = time.time() - download_start

                # 4. Parse results and validate sub-batch result count
                results = self._parse_results(batch_results_dir)
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    "[REMOTE INFERENCE] batch_id=%s expected=%d received=%d",
                    batch_id, len(batch_paths), len(results),
                )
                if self.verbose:
                    print(f"  ✓ Parsed {len(results)} results from batch {batch_id} (expected {len(batch_paths)})")
                    if len(results) != len(batch_paths):
                        print(f"    ⚠ MISMATCH: {len(batch_paths) - len(results)} results missing!")

                # Validate: do not extend when count is wrong; retry once then fail fast
                if len(results) != len(batch_paths):
                    logger.error(
                        "Sub-batch result count mismatch: batch_id=%s expected=%d received=%d (missing=%d)",
                        batch_id, len(batch_paths), len(results), len(batch_paths) - len(results),
                    )
                    # Retry sub-batch once (inference only; images already on remote)
                    if self.verbose:
                        print(f"  Retrying inference for batch {batch_id} once...")
                    if self.use_persistent_server:
                        self._run_inference_persistent(batch_id, num_images=len(batch_paths))
                    else:
                        self._run_inference_batch(batch_id, gpu_batch_size, num_images=len(batch_paths))
                    batch_results_dir_retry = self._download_results(batch_id, output_dir)
                    results = self._parse_results(batch_results_dir_retry)
                    logger.info(
                        "[REMOTE INFERENCE] retry batch_id=%s expected=%d received=%d",
                        batch_id, len(batch_paths), len(results),
                    )
                    if len(results) != len(batch_paths):
                        raise RuntimeError(
                            f"Sub-batch {batch_id} returned {len(results)} results for {len(batch_paths)} images "
                            f"(after one retry). Failing fast instead of returning partial results. "
                            f"Check remote inference.err / inference.log for this batch."
                        )
                    if self.verbose:
                        print(f"  ✓ Retry succeeded: {len(results)} results")

                all_results.extend(results)

                # 5. Cleanup (optional)
                if cleanup:
                    self._cleanup_batch(batch_id)

                if self.verbose:
                    batch_diagrams = sum(r.count for r in results)
                    total_time = upload_time + inference_time + download_time
                    print(f"\n  Timing breakdown:")
                    print(f"    Upload:    {upload_time:6.1f}s ({upload_time/total_time*100:4.1f}%) - {len(batch_paths)/upload_time:.1f} imgs/s")
                    print(f"    Inference: {inference_time:6.1f}s ({inference_time/total_time*100:4.1f}%) - {len(batch_paths)/inference_time:.1f} imgs/s")
                    print(f"    Download:  {download_time:6.1f}s ({download_time/total_time*100:4.1f}%)")
                    print(f"    Total:     {total_time:6.1f}s - {len(batch_paths)/total_time:.1f} imgs/s")
                    print(f"\n  ✓ Batch complete: {batch_diagrams} diagrams found")

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

        # Stop persistent server (normal exit path; cleanup() handles crash path)
        if self.use_persistent_server:
            self._stop_persistent_server()

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


def is_remote_available(
    config: Optional[RemoteConfig] = None,
    timeout: float = 2.0,
    verbose: bool = False,
    try_alternates: bool = True
) -> bool:
    """
    Check if remote server is available for detection.

    Tests SSH connectivity by attempting to connect to the remote server.
    By default, tries both external (henrikkragh.dk:8022) and local (thinkcentre.local:22).

    Args:
        config: RemoteConfig to test (default: henrikkragh.dk:8022)
        timeout: Connection timeout in seconds (default: 2.0)
        verbose: Print status messages (default: False)
        try_alternates: Try alternate host/port combinations (default: True)

    Returns:
        True if remote is reachable, False otherwise

    Example:
        >>> from diagram_detector.remote_ssh import is_remote_available, RemoteConfig
        >>>
        >>> # Check default remote (tries both external and local)
        >>> if is_remote_available():
        ...     print("Can use remote detection")
        >>>
        >>> # Check specific host only
        >>> config = RemoteConfig(host="myserver.local", port=22)
        >>> if is_remote_available(config, try_alternates=False):
        ...     print("Remote available")
    """
    import socket

    # Use auto-loaded config if not provided
    if config is None:
        config = RemoteConfig.auto_load()

    # Build list of (host, port) combinations to try
    # Use endpoints from config if available, otherwise just try the primary host/port
    if try_alternates and config.endpoints:
        # Try all configured endpoints in order
        combinations = config.endpoints
    else:
        # Non-default: just try the specified host/port
        combinations = [(config.host, config.port)]

    for host, port in combinations:
        try:
            if verbose:
                print(f"Checking remote: {config.user}@{host}:{port}...")

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()

            is_available = (result == 0)

            if is_available:
                if verbose:
                    print(f"✓ Remote is available: {config.user}@{host}:{port}")
                # Update config with working host/port
                config.host = host
                config.port = port
                return True
            else:
                # Connection failed - print error if verbose
                if verbose:
                    import errno
                    error_msg = errno.errorcode.get(result, f"error {result}")
                    print(f"✗ Connection failed: {error_msg}")
                continue

        except socket.gaierror:
            # DNS resolution failed - continue to next combination
            if verbose and len(combinations) == 1:
                print(f"✗ Cannot resolve hostname: {host}")
            continue
        except Exception as e:
            # Other errors - continue to next combination
            if verbose and len(combinations) == 1:
                print(f"✗ Error checking remote: {e}")
            continue

    # All combinations failed
    if verbose:
        if len(combinations) > 1:
            print(f"✗ Remote is not reachable on any endpoint")
        else:
            print(f"✗ Remote is not reachable: {config.user}@{config.host}:{config.port}")
    return False


def get_remote_endpoint(
    config: Optional[RemoteConfig] = None,
    timeout: float = 2.0,
    verbose: bool = False
) -> Optional[RemoteConfig]:
    """
    Get working remote endpoint configuration.

    Tests connectivity and returns a RemoteConfig with the working host/port,
    or None if remote is not available.

    Args:
        config: RemoteConfig to test (default: auto-detect)
        timeout: Connection timeout in seconds (default: 2.0)
        verbose: Print status messages (default: False)

    Returns:
        RemoteConfig with working endpoint, or None if not available

    Example:
        >>> from diagram_detector.remote_ssh import get_remote_endpoint
        >>>
        >>> # Get working endpoint
        >>> endpoint = get_remote_endpoint(verbose=True)
        >>> if endpoint:
        ...     print(f"Remote: {endpoint.user}@{endpoint.host}:{endpoint.port}")
        ...     # Use endpoint.host and endpoint.port for connections
    """
    if config is None:
        config = RemoteConfig.auto_load()

    if is_remote_available(config, timeout=timeout, verbose=verbose):
        return config
    else:
        return None
