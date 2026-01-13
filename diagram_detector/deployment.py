"""
Deployment utilities for diagram-detector.

Handles deploying code and models to remote servers with git-based version tracking.
"""

import subprocess
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import datetime

from .remote_ssh import RemoteConfig


@dataclass
class DeploymentInfo:
    """Information about a deployment."""
    version: str
    timestamp: str
    git_commit: str
    git_branch: str
    git_remote: str
    models: Dict[str, str]  # model_name -> sha256
    hostname: str
    port: int


def get_git_info(repo_dir: Path) -> Tuple[str, str, str, bool]:
    """
    Get git repository information.

    Args:
        repo_dir: Path to git repository

    Returns:
        Tuple of (commit_hash, branch_name, remote_url, has_uncommitted_changes)

    Raises:
        RuntimeError: If not a git repository or git command fails
    """
    try:
        # Check if it's a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Not a git repository: {repo_dir}")

    # Get current commit hash
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True
    )
    commit_hash = result.stdout.strip()

    # Get current branch
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True
    )
    branch_name = result.stdout.strip()

    # Get remote URL
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        cwd=repo_dir,
        capture_output=True,
        text=True
    )
    remote_url = result.stdout.strip() if result.returncode == 0 else "unknown"

    # Check for uncommitted changes
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True
    )
    has_uncommitted = bool(result.stdout.strip())

    return commit_hash, branch_name, remote_url, has_uncommitted


def check_local_git_status(repo_dir: Path, strict: bool = True) -> Tuple[str, str, str]:
    """
    Check local git repository status before deployment.

    Args:
        repo_dir: Path to git repository
        strict: If True, raise error on uncommitted changes

    Returns:
        Tuple of (commit_hash, branch_name, remote_url)

    Raises:
        RuntimeError: If not a git repo, or has uncommitted changes (when strict=True)
    """
    commit_hash, branch_name, remote_url, has_uncommitted = get_git_info(repo_dir)

    if has_uncommitted and strict:
        raise RuntimeError(
            "Cannot deploy with uncommitted changes!\n"
            "Please commit your changes first:\n"
            "  git add .\n"
            "  git commit -m 'Your message'\n"
            "\n"
            "Or use strict=False to deploy anyway (not recommended)"
        )

    return commit_hash, branch_name, remote_url


def get_model_hash(model_path: Path) -> str:
    """
    Compute SHA256 hash of model file.

    Args:
        model_path: Path to .pt model file

    Returns:
        SHA256 hex digest
    """
    hasher = hashlib.sha256()

    with open(model_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_local_version() -> str:
    """Get local package version."""
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "unknown"


def get_code_hash(repo_dir: Path) -> str:
    """
    Get hash of current git commit.

    Args:
        repo_dir: Path to git repository

    Returns:
        Git commit hash (first 12 chars)
    """
    commit_hash, _, _, _ = get_git_info(repo_dir)
    return commit_hash[:12]


def get_remote_version(config: RemoteConfig) -> Optional[DeploymentInfo]:
    """
    Get deployment info from remote server.

    Args:
        config: Remote server configuration

    Returns:
        DeploymentInfo if deployed, None if not deployed or unreachable
    """
    try:
        # Try to read deployment info from remote
        cmd = [
            "ssh",
            "-p", str(config.port),
            f"{config.user}@{config.host}",
            f"cat ~/diagram-detector/.deployment_info.json 2>/dev/null || echo ''"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0 or not result.stdout.strip():
            return None

        data = json.loads(result.stdout)
        return DeploymentInfo(**data)

    except Exception:
        return None


def deploy_to_remote(
    config: Optional[RemoteConfig] = None,
    repo_dir: Optional[Path] = None,
    force: bool = False,
    strict: bool = True,
    verbose: bool = True,
    deploy_models: bool = True,
    force_model_download: bool = False
) -> bool:
    """
    Deploy diagram-detector to remote server using git.

    Handles:
    - Git-based code deployment (clone/pull)
    - Uncommitted changes check (strict mode)
    - Version tracking via git commit hash
    - Model deployment with SHA256 verification
    - Installation via pip

    Args:
        config: Remote server config (None = use defaults)
        repo_dir: Local git repository directory (None = auto-detect)
        force: Force deployment even if commits match
        strict: Fail if uncommitted changes exist (default: True)
        verbose: Print progress messages
        deploy_models: Also deploy model files
        force_model_download: Force download models from HuggingFace on remote

    Returns:
        True if deployment successful

    Example:
        >>> from diagram_detector.deployment import deploy_to_remote
        >>> from diagram_detector.remote_ssh import RemoteConfig
        >>>
        >>> # Deploy to default server
        >>> deploy_to_remote(verbose=True)
        >>>
        >>> # Deploy to specific server with force
        >>> config = RemoteConfig(host="myserver.local", port=22)
        >>> deploy_to_remote(config=config, force=True)
    """
    # Use defaults if not provided
    if config is None:
        config = RemoteConfig()

    if repo_dir is None:
        # Auto-detect: Find git root from this file
        repo_dir = Path(__file__).parent.parent
        if not (repo_dir / ".git").exists():
            # Try parent again (for editable installs)
            repo_dir = repo_dir.parent
            if not (repo_dir / ".git").exists():
                raise RuntimeError(f"Cannot find git repository at {repo_dir}")

    # Check local git status
    if verbose:
        print("=" * 70)
        print("DEPLOYING DIAGRAM-DETECTOR TO REMOTE SERVER")
        print("=" * 70)

    try:
        commit_hash, branch_name, remote_url = check_local_git_status(repo_dir, strict=strict)
    except RuntimeError as e:
        if verbose:
            print(f"✗ Git check failed: {e}")
        return False

    local_version = get_local_version()

    if verbose:
        print(f"Local version:  {local_version}")
        print(f"Git commit:     {commit_hash[:12]}")
        print(f"Git branch:     {branch_name}")
        print(f"Git remote:     {remote_url}")
        print(f"Remote:         {config.user}@{config.host}:{config.port}")
        print()

    # Check if deployment needed
    if not force:
        remote_info = get_remote_version(config)
        if remote_info:
            if verbose:
                print(f"Remote version: {remote_info.version}")
                print(f"Remote commit:  {remote_info.git_commit[:12]}")
                print(f"Remote branch:  {remote_info.git_branch}")

            if remote_info.git_commit == commit_hash:
                if verbose:
                    print("\n✓ Remote is up-to-date (same git commit)")
                    print("  Use force=True to redeploy anyway")
                return True
            elif verbose:
                print("\n→ Code has changed, deploying update...")
        elif verbose:
            print("→ No previous deployment found")

    # Test connectivity
    if verbose:
        print("\n1. Testing connectivity...")

    try:
        test_cmd = ["ssh", "-p", str(config.port),
                   f"{config.user}@{config.host}", "echo OK"]
        result = subprocess.run(test_cmd, capture_output=True, timeout=10)

        if result.returncode != 0:
            if verbose:
                print(f"✗ Cannot connect to {config.user}@{config.host}:{config.port}")
            return False

        if verbose:
            print(f"✓ Connected")
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"✗ Connection timeout")
        return False

    # Check if git repo exists on remote
    if verbose:
        print("\n2. Checking remote git repository...")

    check_repo_cmd = [
        "ssh", "-p", str(config.port),
        f"{config.user}@{config.host}",
        "test -d ~/diagram-detector/.git && echo EXISTS || echo MISSING"
    ]
    result = subprocess.run(check_repo_cmd, capture_output=True, text=True)
    repo_exists = result.stdout.strip() == "EXISTS"

    if repo_exists:
        if verbose:
            print("✓ Repository exists, will pull updates")

        # Pull latest changes
        if verbose:
            print("\n3. Pulling latest changes...")

        pull_cmd = [
            "ssh", "-p", str(config.port),
            f"{config.user}@{config.host}",
            f"cd ~/diagram-detector && git fetch origin && git checkout {branch_name} && git pull origin {branch_name}"
        ]
        result = subprocess.run(pull_cmd, capture_output=not verbose, text=True)
        if result.returncode != 0:
            if verbose:
                print("✗ Git pull failed")
                if result.stderr:
                    print(f"  Error: {result.stderr}")
            return False

        if verbose:
            print("✓ Code updated")
    else:
        if verbose:
            print("✗ Repository does not exist, will clone")
            print("\n3. Cloning repository...")

        # Clone repository
        clone_cmd = [
            "ssh", "-p", str(config.port),
            f"{config.user}@{config.host}",
            f"git clone {remote_url} ~/diagram-detector && cd ~/diagram-detector && git checkout {branch_name}"
        ]
        result = subprocess.run(clone_cmd, capture_output=not verbose, text=True)
        if result.returncode != 0:
            if verbose:
                print("✗ Git clone failed")
                if result.stderr:
                    print(f"  Error: {result.stderr}")
            return False

        if verbose:
            print("✓ Repository cloned")

    # Verify commit matches
    if verbose:
        print("\n4. Verifying git commit...")

    verify_commit_cmd = [
        "ssh", "-p", str(config.port),
        f"{config.user}@{config.host}",
        "cd ~/diagram-detector && git rev-parse HEAD"
    ]
    result = subprocess.run(verify_commit_cmd, capture_output=True, text=True)
    remote_commit = result.stdout.strip()

    if remote_commit != commit_hash:
        if verbose:
            print(f"✗ Commit mismatch!")
            print(f"  Local:  {commit_hash[:12]}")
            print(f"  Remote: {remote_commit[:12]}")
        return False

    if verbose:
        print(f"✓ Commit matches: {commit_hash[:12]}")

    # Install on remote
    if verbose:
        print("\n5. Installing package...")

    install_cmd = [
        "ssh", "-p", str(config.port),
        f"{config.user}@{config.host}",
        "cd ~/diagram-detector && python3 -m pip install --user -e . > /dev/null 2>&1"
    ]

    result = subprocess.run(install_cmd)
    if result.returncode != 0:
        if verbose:
            print("✗ Installation failed")
        return False

    if verbose:
        print("✓ Package installed (editable mode)")

    # Verify installation
    if verbose:
        print("\n6. Verifying installation...")

    verify_cmd = [
        "ssh", "-p", str(config.port),
        f"{config.user}@{config.host}",
        "python3 -c 'import diagram_detector; print(diagram_detector.__version__)'"
    ]

    result = subprocess.run(verify_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if verbose:
            print("✗ Verification failed")
        return False

    remote_version = result.stdout.strip()
    if verbose:
        print(f"✓ Remote version: {remote_version}")

    # Deploy models
    model_hashes = {}
    if deploy_models:
        if verbose:
            print("\n7. Deploying models...")

        # Create remote model directory
        mkdir_models_cmd = [
            "ssh", "-p", str(config.port),
            f"{config.user}@{config.host}",
            "mkdir -p ~/.cache/diagram-detector/models"
        ]
        subprocess.run(mkdir_models_cmd, check=True)

        model_dir = Path.home() / ".cache" / "diagram-detector" / "models"
        if model_dir.exists():
            models = list(model_dir.glob("*.pt"))

            if models:
                # Sync each model
                for model_path in models:
                    model_hash = get_model_hash(model_path)
                    model_hashes[model_path.stem] = model_hash

                    if verbose:
                        print(f"  Syncing {model_path.name} (SHA256: {model_hash[:16]}...)")

                    scp_model_cmd = [
                        "scp", "-P", str(config.port),
                        str(model_path),
                        f"{config.user}@{config.host}:~/.cache/diagram-detector/models/"
                    ]
                    subprocess.run(scp_model_cmd, check=True, capture_output=not verbose)

                if verbose:
                    print(f"✓ {len(models)} models deployed")
            else:
                if verbose:
                    print("  No models found locally")
        else:
            if verbose:
                print("  No model cache found locally")

        # Force download models from HuggingFace if requested
        if force_model_download:
            if verbose:
                print("\n  Forcing model download from HuggingFace on remote...")

            # Note: Reusing existing download_model() function
            download_cmd = [
                "ssh", "-p", str(config.port),
                f"{config.user}@{config.host}",
                "cd ~/diagram-detector && python3 -c 'from diagram_detector import download_model; download_model(\"v5\", force=True)'"
            ]
            result = subprocess.run(download_cmd, capture_output=not verbose, text=True)
            if result.returncode == 0:
                if verbose:
                    print("  ✓ Model downloaded from HuggingFace")
            else:
                if verbose:
                    print("  ⚠ Model download failed (may already exist)")

    # Save deployment info
    if verbose:
        print("\n8. Saving deployment info...")

    deployment_info = DeploymentInfo(
        version=local_version,
        timestamp=datetime.datetime.now().isoformat(),
        git_commit=commit_hash,
        git_branch=branch_name,
        git_remote=remote_url,
        models=model_hashes,
        hostname=config.host,
        port=config.port
    )

    # Write locally
    info_json = json.dumps(deployment_info.__dict__, indent=2)
    temp_info = Path("/tmp/deployment_info.json")
    temp_info.write_text(info_json)

    # Copy to remote
    scp_info_cmd = [
        "scp", "-P", str(config.port),
        str(temp_info),
        f"{config.user}@{config.host}:~/diagram-detector/.deployment_info.json"
    ]
    subprocess.run(scp_info_cmd, check=True)
    temp_info.unlink()

    if verbose:
        print("✓ Deployment info saved")

    # Final summary
    if verbose:
        print("\n" + "=" * 70)
        print("✓ DEPLOYMENT COMPLETE")
        print("=" * 70)
        print(f"Version:     {local_version}")
        print(f"Git commit:  {commit_hash[:12]}")
        print(f"Git branch:  {branch_name}")
        print(f"Timestamp:   {deployment_info.timestamp}")
        print(f"Models:      {len(model_hashes)}")
        if model_hashes:
            for model_name, model_hash in model_hashes.items():
                print(f"  - {model_name}: {model_hash[:16]}...")
        print("=" * 70)
        print()

    return True


def check_remote_deployment(
    config: Optional[RemoteConfig] = None,
    verbose: bool = True
) -> Optional[DeploymentInfo]:
    """
    Check deployment status on remote server.

    Args:
        config: Remote server config (None = use defaults)
        verbose: Print deployment info

    Returns:
        DeploymentInfo if deployed, None otherwise
    """
    if config is None:
        config = RemoteConfig()

    info = get_remote_version(config)

    if verbose:
        if info:
            print("=" * 70)
            print(f"REMOTE DEPLOYMENT: {config.user}@{config.host}:{config.port}")
            print("=" * 70)
            print(f"Version:    {info.version}")
            print(f"Code hash:  {info.code_hash}")
            print(f"Deployed:   {info.timestamp}")
            print(f"Models:     {len(info.models)}")
            if info.models:
                for model_name, model_hash in info.models.items():
                    print(f"  - {model_name}: {model_hash[:16]}...")
            print("=" * 70)
        else:
            print(f"No deployment found on {config.user}@{config.host}:{config.port}")

    return info
