# Model Distribution Guide
**Package:** diagram-detector v1.0.0
**Date:** December 24, 2024

---

## 🎯 OVERVIEW

Models are distributed through **two channels** for robustness:

1. **Hugging Face Hub** (Primary)
   - Fast CDN delivery
   - Built-in versioning
   - Easy browsing
   - Resume downloads

2. **GitHub Releases** (Fallback)
   - No extra dependencies
   - Same repository as code
   - Simple URLs

**The package automatically tries both sources and caches locally.**

---

## 📦 LOCAL CACHING

All models are cached at:
```
~/.cache/diagram-detector/models/
├── yolo11n.pt
├── yolo11n_metadata.json
├── yolo11m.pt
├── yolo11m_metadata.json
└── ...
```

**Benefits:**
- ✅ Downloaded once, used forever
- ✅ No re-download on reinstall
- ✅ Shared across environments
- ✅ Automatic verification

**To clear cache:**
```bash
rm -rf ~/.cache/diagram-detector/
```

---

## 🚀 UPLOADING MODELS

### **Prerequisites**

**1. GitHub CLI**
```bash
# macOS
brew install gh

# Linux
sudo apt install gh

# Authenticate
gh auth login
```

**2. Hugging Face Account**
- Sign up: https://huggingface.co/join
- Create token: https://huggingface.co/settings/tokens
  - Name: "diagram-detector-upload"
  - Type: Write
  - Copy token

**3. Install Dependencies**
```bash
pip install huggingface_hub
```

---

### **Setup Hugging Face Repository**

**Create model repository:**
```bash
# Via web: https://huggingface.co/new-model
# Or via CLI:
from huggingface_hub import create_repo

create_repo(
    repo_id="hksorensen/diagram-detector-models",
    repo_type="model",
    private=False,  # Public for free downloads
)
```

**Set token:**
```bash
export HF_TOKEN=hf_...your_token...
```

---

### **Upload Models**

**Single model:**
```bash
python upload_models.py \
  --package ~/Downloads/diagram-detector-models/yolo11m_20241224_183045_e200_b16_f0742/
```

**All models in directory:**
```bash
python upload_models.py \
  --directory ~/Downloads/diagram-detector-models/
```

**Upload to specific destination:**
```bash
# GitHub only
python upload_models.py --package path/to/model --github-only

# Hugging Face only
python upload_models.py --package path/to/model --huggingface-only

# Custom HF repo
python upload_models.py \
  --package path/to/model \
  --hf-repo username/custom-repo
```

**Dry run (preview):**
```bash
python upload_models.py --directory ~/Downloads/diagram-detector-models/ --dry-run
```

---

### **What Gets Uploaded**

For each model (e.g., yolo11m):

**Files:**
- `yolo11m.pt` - PyTorch model weights
- `yolo11m_metadata.json` - Training metadata
- `yolo11m.onnx` - ONNX model (if available)

**To GitHub Release:**
- Location: `https://github.com/hksorensen/diagram-detector/releases/tag/v1.0.0`
- Direct URL: `https://github.com/.../releases/download/v1.0.0/yolo11m.pt`

**To Hugging Face:**
- Location: `https://huggingface.co/hksorensen/diagram-detector-models`
- Direct URL: `https://huggingface.co/hksorensen/diagram-detector-models/resolve/main/yolo11m.pt`

---

## 📥 DOWNLOAD BEHAVIOR

### **Automatic Download (Package)**

When using the package:
```python
from diagram_detector import DiagramDetector

# First use of yolo11m
detector = DiagramDetector(model='yolo11m')
# Downloads automatically:
# 1. Try Hugging Face (fast CDN)
# 2. Fall back to GitHub if needed
# 3. Cache to ~/.cache/diagram-detector/models/
```

**Console output:**
```
Downloading yolo11m model (49 MB)...
Downloading from Hugging Face Hub: hksorensen/diagram-detector-models/yolo11m.pt
✓ Metadata downloaded
✓ Model downloaded to /Users/you/.cache/diagram-detector/models/yolo11m.pt
```

### **Manual Download**

```python
from diagram_detector.utils import download_model

# Download specific model
model_path = download_model('yolo11m')

# Force re-download
model_path = download_model('yolo11m', force=True)

# Use specific source
model_path = download_model('yolo11m', source='huggingface')
model_path = download_model('yolo11m', source='github')
```

---

## 🔍 METADATA

### **What's in Metadata**

```json
{
  "model_info": {
    "architecture": "yolo11m",
    "package_version": "1.0.0",
    "created": "2024-12-24T18:30:45",
    "parameters": "20.1M"
  },
  "training": {
    "command": {
      "epochs_requested": 200,
      "epochs_completed": 200,
      "training_time": "2.5 hours"
    },
    "hyperparameters": {
      "batch_size": 16,
      "image_size": 640,
      "patience": 50,
      "lr0": 0.01
    },
    "dataset_size": 5685
  },
  "performance": {
    "f1_score": 0.742,
    "mAP50": 0.856,
    "precision": 0.891,
    "recall": 0.678
  }
}
```

### **Using Metadata**

```python
from diagram_detector.utils import load_model_metadata

# Load metadata
metadata = load_model_metadata('yolo11m')

if metadata:
    print(f"F1 Score: {metadata['performance']['f1_score']}")
    print(f"Training epochs: {metadata['training']['command']['epochs_completed']}")
```

---

## 🔧 TROUBLESHOOTING

### **Download Fails**

**Problem:** Both sources fail
```
RuntimeError: Failed to download yolo11m from all sources
```

**Solutions:**
1. Check internet connection
2. Verify URLs are accessible
3. Try manual download:
   ```bash
   # Download from GitHub
   wget https://github.com/hksorensen/diagram-detector/releases/download/v1.0.0/yolo11m.pt
   
   # Move to cache
   mkdir -p ~/.cache/diagram-detector/models/
   mv yolo11m.pt ~/.cache/diagram-detector/models/
   ```

### **Corrupted Download**

**Problem:** Downloaded file is 0 bytes or incomplete

**Solution:**
```bash
# Remove corrupted file
rm ~/.cache/diagram-detector/models/yolo11m.pt

# Download again
python -c "from diagram_detector.utils import download_model; download_model('yolo11m', force=True)"
```

### **Hugging Face Auth Issues**

**Problem:** 401 Unauthorized

**Solution:**
```bash
# Set token
export HF_TOKEN=hf_...your_token...

# Or use huggingface-cli
huggingface-cli login
```

### **GitHub Rate Limiting**

**Problem:** API rate limit exceeded

**Solution:**
1. Wait 1 hour (public API has hourly limits)
2. Use Hugging Face instead (no rate limits)
3. Authenticate gh CLI for higher limits

---

## 📊 MODEL SIZES & DOWNLOAD TIMES

| Model | Size | Download (Fast) | Download (Slow) |
|-------|------|-----------------|-----------------|
| yolo11n | 6 MB | < 1 sec | ~10 sec |
| yolo11s | 22 MB | ~2 sec | ~30 sec |
| yolo11m | 49 MB | ~5 sec | ~1 min |
| yolo11l | 63 MB | ~6 sec | ~1.5 min |
| yolo11x | 137 MB | ~15 sec | ~3 min |

**Fast:** Hugging Face CDN or good connection  
**Slow:** GitHub releases or limited bandwidth

---

## 🔄 UPDATING MODELS

### **New Model Version**

```bash
# 1. Train new model
python pipeline.py --train --model yolo11m --epochs 200

# 2. Package automatically created
# ~/Downloads/diagram-detector-models/yolo11m_20241225_...

# 3. Upload
python upload_models.py --package ~/Downloads/diagram-detector-models/yolo11m_20241225_...

# 4. Models are versioned by timestamp in package name
# Users get latest by re-downloading
```

### **Force Users to Re-download**

Users with cached models won't auto-update. To force:

**Option 1: Version bump**
```python
# In utils.py
MODEL_INFO = {
    'yolo11m': {
        'size_mb': 49,
        'version': '1.1.0',  # Bump version
        ...
    }
}
```

**Option 2: Checksum validation**
```python
# Add checksums to MODEL_INFO
# Verify on download
```

**Option 3: User clears cache**
```bash
rm -rf ~/.cache/diagram-detector/
```

---

## 🎯 BEST PRACTICES

### **For Model Uploading**

1. ✅ **Always test locally first**
   ```bash
   pip install -e .
   diagram-detect --input test.jpg --model yolo11m
   ```

2. ✅ **Use dry-run before real upload**
   ```bash
   python upload_models.py --directory models/ --dry-run
   ```

3. ✅ **Upload to both channels**
   - Don't use `--github-only` or `--huggingface-only` unless necessary
   - Redundancy ensures availability

4. ✅ **Keep metadata updated**
   - Metadata helps users choose models
   - Include training details, performance metrics

5. ✅ **Version control**
   - Tag GitHub releases: v1.0.0, v1.1.0, etc.
   - Keep old versions available

### **For Package Development**

1. ✅ **Test download logic**
   ```bash
   # Clear cache
   rm -rf ~/.cache/diagram-detector/
   
   # Test download
   python -c "from diagram_detector import DiagramDetector; DiagramDetector(model='yolo11n')"
   ```

2. ✅ **Handle network failures gracefully**
   - Multiple sources with fallback
   - Clear error messages
   - Resume partial downloads

3. ✅ **Verify file integrity**
   - Check file size > 0
   - Optional: SHA256 checksums

4. ✅ **Keep cache manageable**
   - Only download when needed
   - Don't bundle models with package

---

## 📋 UPLOAD CHECKLIST

Before uploading models:

- [ ] Models trained and validated
- [ ] Metadata files complete
- [ ] ONNX models exported (optional)
- [ ] GitHub CLI authenticated (`gh auth status`)
- [ ] Hugging Face token set (`echo $HF_TOKEN`)
- [ ] HF repository created
- [ ] GitHub release exists (or will be created)
- [ ] Test with dry-run
- [ ] Verify upload success
- [ ] Test download from package
- [ ] Update documentation if URLs changed

---

## 🔗 USEFUL LINKS

**Hugging Face:**
- Hub: https://huggingface.co/
- Models: https://huggingface.co/hksorensen/diagram-detector-models
- Docs: https://huggingface.co/docs/huggingface_hub

**GitHub:**
- Releases: https://github.com/hksorensen/diagram-detector/releases
- CLI: https://cli.github.com/

**Package:**
- PyPI: https://pypi.org/project/diagram-detector/
- Docs: https://diagram-detector.readthedocs.io/

---

## 💡 EXAMPLE WORKFLOW

```bash
# 1. Train model
cd diagram-detection-pipeline
python pipeline.py --clean-remote
python pipeline.py --train --model yolo11m --epochs 200

# 2. Package is auto-created
ls ~/Downloads/diagram-detector-models/
# yolo11m_20241224_183045_e200_b16_f0742/

# 3. Setup tokens
export HF_TOKEN=hf_...
gh auth login

# 4. Upload
cd diagram-detector-package
python upload_models.py \
  --directory ~/Downloads/diagram-detector-models/ \
  --dry-run  # Verify first

python upload_models.py \
  --directory ~/Downloads/diagram-detector-models/
# Uploads to both HF and GitHub

# 5. Test
pip install diagram-detector
python -c "from diagram_detector import DiagramDetector; d = DiagramDetector(model='yolo11m')"
# Should download from HF and work!

# 6. Verify URLs
curl -I https://huggingface.co/hksorensen/diagram-detector-models/resolve/main/yolo11m.pt
curl -I https://github.com/hksorensen/diagram-detector/releases/download/v1.0.0/yolo11m.pt
```

---

**End of model distribution guide - December 24, 2024**
**Status:** Ready for production use ✅
