# GitHub Setup & Deployment Guide
**Package:** diagram-detector v1.0.0
**Date:** December 24, 2024

---

## 🚀 QUICK START

### Step 1: Extract Package
```bash
tar -xzf diagram-detector-complete-v1.0.0.tar.gz
cd diagram-detector-package
```

### Step 2: Test Locally
```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Test CLI
diagram-detect --help
```

### Step 3: Create GitHub Repository
Follow instructions below →

---

## 📋 GITHUB SETUP INSTRUCTIONS

### Option A: GitHub Web Interface (Easiest)

**1. Create New Repository**
- Go to: https://github.com/new
- Repository name: `diagram-detector`
- Description: "Production-ready diagram detection for academic papers using YOLO11"
- Visibility: **Public** (required for PyPI)
- **Do NOT initialize with README** (we have our own)
- Click "Create repository"

**2. Initialize Local Repository**
```bash
cd diagram-detector-package

# Initialize git
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: diagram-detector v1.0.0

- Complete Python package structure
- Core detection modules (detector, models, utils, cli)
- PyPI-ready configuration
- Docker support
- GitHub Actions CI/CD
- Comprehensive documentation
- Basic test suite"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/diagram-detector.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**3. Verify on GitHub**
- Go to: https://github.com/YOUR_USERNAME/diagram-detector
- You should see all files uploaded

---

### Option B: GitHub CLI (Alternative)

**1. Install GitHub CLI** (if not already installed)
```bash
# macOS
brew install gh

# Linux
sudo apt install gh

# Windows
winget install GitHub.cli
```

**2. Authenticate**
```bash
gh auth login
# Follow prompts
```

**3. Create Repository**
```bash
cd diagram-detector-package

# Initialize and create in one step
gh repo create diagram-detector --public --source=. --remote=origin --push

# Description and other settings
gh repo edit --description "Production-ready diagram detection for academic papers using YOLO11" \
  --homepage "https://diagram-detector.readthedocs.io"
```

---

## 🏷️ ADD TOPICS TO REPOSITORY

**Via Web:**
- Go to repository page
- Click "⚙️ Settings"
- Under "Topics", add:
  - `diagram-detection`
  - `computer-vision`
  - `yolo`
  - `academic-papers`
  - `pdf-processing`
  - `machine-learning`
  - `pytorch`

**Via CLI:**
```bash
gh repo edit --add-topic diagram-detection,computer-vision,yolo,academic-papers,pdf-processing,machine-learning,pytorch
```

---

## 🔧 SETUP GITHUB ACTIONS (CI/CD)

The `.github/workflows/ci.yml` file is already included!

**What it does:**
- ✅ Runs tests on every push/PR
- ✅ Tests on Linux, macOS, Windows
- ✅ Tests Python 3.9, 3.10, 3.11, 3.12
- ✅ Runs linting (flake8, black, mypy)
- ✅ Automatically publishes to PyPI on new tags

**First CI run will fail** because model download needs network. That's OK!

---

## 📦 SETUP PYPI (For Package Distribution)

### 1. Create PyPI Account
- Go to: https://pypi.org/account/register/
- Verify email

### 2. Create API Token
- Go to: https://pypi.org/manage/account/
- Scroll to "API tokens"
- Click "Add API token"
- Name: "diagram-detector-github"
- Scope: "Entire account" (or specific project after first upload)
- **Copy the token** (starts with `pypi-`)

### 3. Add Token to GitHub Secrets
- Go to: https://github.com/YOUR_USERNAME/diagram-detector/settings/secrets/actions
- Click "New repository secret"
- Name: `PYPI_TOKEN`
- Value: Paste your PyPI token
- Click "Add secret"

---

## 🚀 FIRST RELEASE

### 1. Test on TestPyPI First (Recommended)

**Create TestPyPI account:**
- Go to: https://test.pypi.org/account/register/

**Create token and add to GitHub as `TEST_PYPI_TOKEN`**

**Build and test:**
```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ diagram-detector
```

### 2. Release to PyPI

**Create a release tag:**
```bash
git tag -a v1.0.0 -m "Release v1.0.0

First stable release with:
- YOLO11-based diagram detection
- PDF support
- Batch processing
- CPU/CUDA/MPS support
- CLI and Python API"

git push origin v1.0.0
```

**GitHub Actions will automatically:**
1. Run all tests
2. Build the package
3. Upload to PyPI

**Or upload manually:**
```bash
python -m build
twine upload dist/*
```

### 3. Create GitHub Release

**Via Web:**
- Go to: https://github.com/YOUR_USERNAME/diagram-detector/releases/new
- Choose tag: `v1.0.0`
- Release title: `v1.0.0 - First Stable Release`
- Description: Copy from CHANGELOG.md
- Attach built packages (from `dist/`)
- Click "Publish release"

**Via CLI:**
```bash
gh release create v1.0.0 \
  --title "v1.0.0 - First Stable Release" \
  --notes "$(cat CHANGELOG.md)" \
  dist/*
```

---

## 🐳 DOCKER SETUP

### Build Docker Image

```bash
# Build
docker build -t diagram-detector:1.0.0 .

# Tag for Docker Hub (optional)
docker tag diagram-detector:1.0.0 YOUR_DOCKERHUB_USERNAME/diagram-detector:1.0.0

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/diagram-detector:1.0.0
```

### Test Docker Image

```bash
# Create test data directory
mkdir -p data/input data/output

# Copy some test images
cp test_images/* data/input/

# Run container
docker run -v $(pwd)/data:/data diagram-detector:1.0.0 \
  --input /data/input --output /data/output

# Check results
ls data/output/
```

---

## 📊 SETUP GITHUB REPOSITORY

### Enable Issues & Discussions

**Issues:**
- Go to: Settings → Features
- Check "Issues"

**Discussions:**
- Go to: Settings → Features
- Check "Discussions"

### Add Issue Templates

Create `.github/ISSUE_TEMPLATE/bug_report.md`:
```markdown
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. ...
2. ...

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11]
- Package version: [e.g., 1.0.0]

**Additional context**
Any other context about the problem.
```

Create `.github/ISSUE_TEMPLATE/feature_request.md`:
```markdown
---
name: Feature request
about: Suggest an idea
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Additional context**
Any other context or screenshots.
```

### Add README Badges

Update README.md to show build status:
```markdown
[![CI](https://github.com/YOUR_USERNAME/diagram-detector/workflows/CI/badge.svg)](https://github.com/YOUR_USERNAME/diagram-detector/actions)
[![PyPI version](https://badge.fury.io/py/diagram-detector.svg)](https://badge.fury.io/py/diagram-detector)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
```

---

## 🔐 SECURITY

### Add Security Policy

Create `SECURITY.md`:
```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please report security vulnerabilities to: YOUR_EMAIL

Do not open public issues for security vulnerabilities.
```

### Enable Security Features

- Go to: Settings → Security
- Enable "Dependency graph"
- Enable "Dependabot alerts"
- Enable "Dependabot security updates"

---

## 📖 DOCUMENTATION (Optional but Recommended)

### Setup Read the Docs

1. Go to: https://readthedocs.org/
2. Sign in with GitHub
3. Import repository
4. It will auto-build docs from `docs/` directory

### Create docs/ Structure

```bash
mkdir -p docs
cd docs

# Create conf.py for Sphinx
# Create index.rst
# Create api.rst
```

---

## 🎯 POST-RELEASE CHECKLIST

- [ ] Repository created on GitHub
- [ ] All files pushed
- [ ] Topics/tags added
- [ ] CI/CD passing
- [ ] PyPI token added to secrets
- [ ] Package published to PyPI
- [ ] GitHub release created
- [ ] Docker image built and pushed (optional)
- [ ] Issues/Discussions enabled
- [ ] Issue templates created
- [ ] Security policy added
- [ ] README badges updated
- [ ] Documentation site setup (optional)

---

## 🔄 ONGOING MAINTENANCE

### For Bug Fixes (Patch Release)

```bash
# Make fixes
git add .
git commit -m "Fix: description of fix"

# Update version in pyproject.toml (1.0.0 → 1.0.1)

# Tag and push
git tag v1.0.1
git push origin main --tags

# GitHub Actions will auto-publish to PyPI
```

### For New Features (Minor Release)

```bash
# Add features
git add .
git commit -m "Feature: description"

# Update version (1.0.0 → 1.1.0)

# Tag and push
git tag v1.1.0
git push origin main --tags
```

### For Breaking Changes (Major Release)

```bash
# Make breaking changes
git add .
git commit -m "BREAKING: description"

# Update version (1.0.0 → 2.0.0)
# Update CHANGELOG.md

# Tag and push
git tag v2.0.0
git push origin main --tags
```

---

## 💡 TIPS

### Before Publishing

1. **Test locally thoroughly:**
   ```bash
   pip install -e ".[dev]"
   pytest
   black --check .
   flake8 .
   mypy diagram_detector
   ```

2. **Test installation from source:**
   ```bash
   pip install .
   diagram-detect --help
   ```

3. **Build and inspect:**
   ```bash
   python -m build
   tar -tzf dist/diagram_detector-1.0.0.tar.gz
   ```

### Model Distribution

**Important:** The package does NOT include model weights (too large).

**Options:**
1. **GitHub Releases** (Recommended):
   - Upload trained models to GitHub releases
   - Package will download on first use

2. **Separate repository:**
   - Create `diagram-detector-models` repo
   - Host models there

3. **Hugging Face:**
   - Upload to Hugging Face Hub
   - Use `huggingface_hub` package to download

**To upload models to GitHub release:**
```bash
gh release upload v1.0.0 path/to/yolo11n.pt path/to/yolo11m.pt
```

---

## 🎓 COMPLETE EXAMPLE

```bash
# 1. Extract package
tar -xzf diagram-detector-complete-v1.0.0.tar.gz
cd diagram-detector-package

# 2. Test locally
pip install -e ".[dev]"
pytest
diagram-detect --help

# 3. Create GitHub repo
gh auth login
gh repo create diagram-detector --public --source=. --remote=origin --push

# 4. Add topics
gh repo edit --add-topic diagram-detection,computer-vision,yolo

# 5. Create PyPI token and add to GitHub secrets
# (via web interface)

# 6. Upload models to GitHub release
# (after creating first release)
gh release upload v1.0.0 models/*.pt

# 7. Create release
git tag -a v1.0.0 -m "v1.0.0"
git push origin v1.0.0

# 8. Verify
pip install diagram-detector
diagram-detect --help
```

---

## 📞 NEED HELP?

- GitHub Issues: For bugs and features
- GitHub Discussions: For questions
- Email: hks@ku.dk

---

**End of GitHub setup guide - December 24, 2024**
**Package version:** 1.0.0
**Status:** Ready for deployment ✅
