#!/usr/bin/env python3
"""
Universal Model Uploader

Uploads trained models to both GitHub releases and Hugging Face Hub.
Includes model weights + metadata + training artifacts.

Usage:
    # Upload single model package
    python upload_models.py --package ~/Downloads/diagram-detector-models/yolo11m_20241224_...
    
    # Upload all models in directory
    python upload_models.py --directory ~/Downloads/diagram-detector-models/
    
    # Upload to specific destination
    python upload_models.py --package path/to/model --github-only
    python upload_models.py --package path/to/model --huggingface-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict
import shutil


def find_model_packages(directory: Path) -> List[Path]:
    """Find all model package directories."""
    packages = []
    
    for item in directory.iterdir():
        if item.is_dir() and item.name.startswith('yolo11'):
            # Check if it looks like a model package
            if (item / 'model.pt').exists() and (item / 'metadata.json').exists():
                packages.append(item)
    
    return sorted(packages)


def extract_model_info(package_dir: Path) -> Dict:
    """Extract model information from package."""
    metadata_path = package_dir / 'metadata.json'
    
    if not metadata_path.exists():
        raise ValueError(f"No metadata.json found in {package_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract key info
    info = {
        'package_name': package_dir.name,
        'architecture': metadata['model_info']['architecture'],
        'version': metadata['model_info'].get('package_version', '1.0.0'),
        'f1_score': metadata['performance'].get('f1_score'),
        'epochs': metadata['training']['command']['epochs_completed'],
        'model_path': package_dir / 'model.pt',
        'onnx_path': package_dir / 'model.onnx',
        'metadata_path': metadata_path,
        'metadata': metadata,
    }
    
    return info


def upload_to_github(
    model_info: Dict,
    release_tag: str = 'v1.0.0',
    token: Optional[str] = None
) -> bool:
    """
    Upload model to GitHub release.
    
    Args:
        model_info: Model information dict
        release_tag: GitHub release tag
        token: GitHub token (or use gh CLI)
        
    Returns:
        True if successful
    """
    try:
        import subprocess
        
        print(f"\nUploading {model_info['architecture']} to GitHub release {release_tag}...")
        
        # Check if gh CLI is available
        try:
            subprocess.run(['gh', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ GitHub CLI (gh) not found. Install with: brew install gh")
            return False
        
        # Upload model weights
        arch = model_info['architecture']
        model_path = model_info['model_path']
        
        # Rename to simple architecture name for easy access
        simple_name = f"{arch}.pt"
        
        cmd = [
            'gh', 'release', 'upload', release_tag,
            str(model_path),
            '--clobber',  # Overwrite if exists
            '--repo', 'hksorensen/diagram-detector'
        ]
        
        print(f"  Uploading {model_path.name} as {simple_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ Upload failed: {result.stderr}")
            return False
        
        # Upload metadata
        metadata_path = model_info['metadata_path']
        metadata_name = f"{arch}_metadata.json"
        
        cmd = [
            'gh', 'release', 'upload', release_tag,
            str(metadata_path),
            '--clobber',
            '--repo', 'hksorensen/diagram-detector'
        ]
        
        print(f"  Uploading {metadata_path.name} as {metadata_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠ Metadata upload failed: {result.stderr}")
        
        # Upload ONNX if available
        if model_info['onnx_path'].exists():
            onnx_name = f"{arch}.onnx"
            
            cmd = [
                'gh', 'release', 'upload', release_tag,
                str(model_info['onnx_path']),
                '--clobber',
                '--repo', 'hksorensen/diagram-detector'
            ]
            
            print(f"  Uploading {model_info['onnx_path'].name} as {onnx_name}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"⚠ ONNX upload failed: {result.stderr}")
        
        print(f"✓ {arch} uploaded to GitHub")
        return True
        
    except Exception as e:
        print(f"✗ GitHub upload failed: {e}")
        return False


def upload_to_huggingface(
    model_info: Dict,
    repo_id: str = 'hksorensen/diagram-detector-models',
    token: Optional[str] = None
) -> bool:
    """
    Upload model to Hugging Face Hub.
    
    Args:
        model_info: Model information dict
        repo_id: Hugging Face repo ID
        token: HF token (or use HF_TOKEN env var)
        
    Returns:
        True if successful
    """
    try:
        from huggingface_hub import HfApi, create_repo
        import os
        
        print(f"\nUploading {model_info['architecture']} to Hugging Face Hub...")
        
        # Get token
        if token is None:
            token = os.environ.get('HF_TOKEN')
        
        if token is None:
            print("✗ No Hugging Face token found. Set HF_TOKEN environment variable or pass --hf-token")
            return False
        
        # Initialize API
        api = HfApi(token=token)
        
        # Create repo if it doesn't exist
        try:
            create_repo(
                repo_id=repo_id,
                token=token,
                repo_type="model",
                exist_ok=True,
            )
        except Exception as e:
            print(f"⚠ Repo creation note: {e}")
        
        arch = model_info['architecture']
        
        # Upload model weights
        model_path = model_info['model_path']
        model_filename = f"{arch}.pt"
        
        print(f"  Uploading {model_path.name} as {model_filename}...")
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=model_filename,
            repo_id=repo_id,
            token=token,
        )
        
        # Upload metadata
        metadata_path = model_info['metadata_path']
        metadata_filename = f"{arch}_metadata.json"
        
        print(f"  Uploading {metadata_path.name} as {metadata_filename}...")
        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo=metadata_filename,
            repo_id=repo_id,
            token=token,
        )
        
        # Upload ONNX if available
        if model_info['onnx_path'].exists():
            onnx_filename = f"{arch}.onnx"
            
            print(f"  Uploading {model_info['onnx_path'].name} as {onnx_filename}...")
            api.upload_file(
                path_or_fileobj=str(model_info['onnx_path']),
                path_in_repo=onnx_filename,
                repo_id=repo_id,
                token=token,
            )
        
        print(f"✓ {arch} uploaded to Hugging Face")
        print(f"  View at: https://huggingface.co/{repo_id}")
        
        return True
        
    except ImportError:
        print("✗ huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"✗ Hugging Face upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_model_card(model_info: Dict, output_path: Path) -> None:
    """Create Hugging Face model card (README.md)."""
    metadata = model_info['metadata']
    arch = model_info['architecture']
    
    card = f"""---
language: en
license: mit
tags:
- diagram-detection
- computer-vision
- yolo11
- academic-papers
---

# Diagram Detector - {arch.upper()}

YOLO11-based diagram detection model for academic papers.

## Model Details

- **Architecture**: {arch.upper()}
- **Parameters**: {metadata['model_info'].get('parameters', 'N/A')}
- **Input Size**: 640x640
- **Training Epochs**: {metadata['training']['command']['epochs_completed']}
- **Batch Size**: {metadata['training']['hyperparameters']['batch_size']}

## Performance

- **F1 Score**: {metadata['performance']['f1_score']:.3f}
- **mAP50**: {metadata['performance']['mAP50']:.3f}
- **Precision**: {metadata['performance']['precision']:.3f}
- **Recall**: {metadata['performance']['recall']:.3f}

## Usage

```python
from diagram_detector import DiagramDetector

# Initialize with this model
detector = DiagramDetector(model='{arch}')

# Detect diagrams
results = detector.detect('path/to/images/')
```

## Training Details

- **Dataset Size**: {metadata['training'].get('dataset_size', 'N/A')} images
- **Training Time**: {metadata['training']['command'].get('training_time', 'N/A')}
- **Device**: {metadata['training']['hyperparameters'].get('device', 'GPU')}

## Citation

```bibtex
@software{{diagram_detector_{arch},
  title = {{diagram-detector: {arch.upper()} model}},
  author = {{Henrik Kragh Sørensen}},
  year = {{2024}},
  url = {{https://github.com/hksorensen/diagram-detector}}
}}
```

## License

MIT License - see repository for details.
"""
    
    with open(output_path, 'w') as f:
        f.write(card)


def main():
    parser = argparse.ArgumentParser(
        description='Upload models to GitHub and Hugging Face',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--package', '-p',
        type=str,
        help='Path to single model package directory'
    )
    
    parser.add_argument(
        '--directory', '-d',
        type=str,
        help='Path to directory containing multiple model packages'
    )
    
    parser.add_argument(
        '--github-only',
        action='store_true',
        help='Upload to GitHub only'
    )
    
    parser.add_argument(
        '--huggingface-only',
        action='store_true',
        help='Upload to Hugging Face only'
    )
    
    parser.add_argument(
        '--release-tag',
        default='v1.0.0',
        help='GitHub release tag (default: v1.0.0)'
    )
    
    parser.add_argument(
        '--hf-repo',
        default='hksorensen/diagram-detector-models',
        help='Hugging Face repo ID'
    )
    
    parser.add_argument(
        '--hf-token',
        help='Hugging Face token (or use HF_TOKEN env var)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be uploaded without uploading'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.package and not args.directory:
        parser.error("Must specify either --package or --directory")
    
    # Find model packages
    packages = []
    
    if args.package:
        package_dir = Path(args.package)
        if not package_dir.exists():
            print(f"✗ Package not found: {package_dir}")
            sys.exit(1)
        packages = [package_dir]
    
    if args.directory:
        dir_path = Path(args.directory)
        if not dir_path.exists():
            print(f"✗ Directory not found: {dir_path}")
            sys.exit(1)
        packages = find_model_packages(dir_path)
        
        if not packages:
            print(f"✗ No model packages found in {dir_path}")
            sys.exit(1)
    
    print(f"Found {len(packages)} model package(s)")
    
    # Extract model info
    models = []
    for package in packages:
        try:
            info = extract_model_info(package)
            models.append(info)
            print(f"  ✓ {info['architecture']}: {info['package_name']}")
        except Exception as e:
            print(f"  ✗ {package.name}: {e}")
    
    if not models:
        print("✗ No valid model packages found")
        sys.exit(1)
    
    # Determine upload targets
    upload_github = not args.huggingface_only
    upload_hf = not args.github_only
    
    print(f"\nUpload targets:")
    if upload_github:
        print(f"  - GitHub: hksorensen/diagram-detector (release {args.release_tag})")
    if upload_hf:
        print(f"  - Hugging Face: {args.hf_repo}")
    
    if args.dry_run:
        print("\n[DRY RUN - No actual uploads]")
        return
    
    # Confirm
    confirm = input("\nContinue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted")
        sys.exit(0)
    
    # Upload each model
    results = {
        'github': {},
        'huggingface': {},
    }
    
    for model_info in models:
        arch = model_info['architecture']
        
        if upload_github:
            success = upload_to_github(
                model_info,
                release_tag=args.release_tag
            )
            results['github'][arch] = success
        
        if upload_hf:
            success = upload_to_huggingface(
                model_info,
                repo_id=args.hf_repo,
                token=args.hf_token
            )
            results['huggingface'][arch] = success
    
    # Print summary
    print("\n" + "="*60)
    print("UPLOAD SUMMARY")
    print("="*60)
    
    if upload_github:
        print("\nGitHub:")
        for arch, success in results['github'].items():
            status = "✓" if success else "✗"
            print(f"  {status} {arch}")
    
    if upload_hf:
        print("\nHugging Face:")
        for arch, success in results['huggingface'].items():
            status = "✓" if success else "✗"
            print(f"  {status} {arch}")
    
    # Check if all successful
    all_success = all(results['github'].values()) if upload_github else True
    all_success = all_success and (all(results['huggingface'].values()) if upload_hf else True)
    
    if all_success:
        print("\n✓ All uploads successful!")
    else:
        print("\n⚠ Some uploads failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
