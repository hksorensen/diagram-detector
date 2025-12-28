# Model Distribution - Quick Reference

## 🎯 TL;DR

**Models are cached locally** at `~/.cache/diagram-detector/models/`

**Download sources** (automatic fallback):
1. Hugging Face Hub (primary)
2. GitHub Releases (fallback)

---

## 📥 DOWNLOADING (User Perspective)

```python
# Just use the package - models download automatically!
from diagram_detector import DiagramDetector

detector = DiagramDetector(model='yolo11m')
# First use: Downloads from HF/GitHub
# Second use: Uses cached version
```

---

## 📤 UPLOADING (Developer Perspective)

### **Setup (Once)**

```bash
# 1. Install tools
brew install gh                    # GitHub CLI
pip install huggingface_hub        # HF uploader

# 2. Authenticate
gh auth login                      # GitHub
export HF_TOKEN=hf_...             # Hugging Face

# 3. Create HF repo (once)
# Visit: https://huggingface.co/new-model
# Name: diagram-detector-models
```

### **Upload Models**

```bash
# Upload all trained models
python upload_models.py --directory ~/Downloads/diagram-detector-models/

# Upload single model
python upload_models.py --package ~/Downloads/diagram-detector-models/yolo11m_TIMESTAMP/

# Preview (no actual upload)
python upload_models.py --directory ~/Downloads/diagram-detector-models/ --dry-run
```

### **What Gets Uploaded**

For each model (e.g., yolo11m):
- `yolo11m.pt` - Model weights
- `yolo11m_metadata.json` - Training details
- `yolo11m.onnx` - ONNX version (if exists)

**To:**
- Hugging Face: `hksorensen/diagram-detector-models`
- GitHub: `hksorensen/diagram-detector/releases/v1.0.0`

---

## 🔧 IMPLEMENTATION DETAILS

### **Robust Download (utils.py)**

```python
def download_model(model_name: str, source: str = 'auto') -> Path:
    """
    Downloads with automatic fallback:
    1. Try Hugging Face (fast CDN)
    2. Fall back to GitHub
    3. Cache locally
    4. Verify integrity
    """
```

**Features:**
- ✅ Multi-source fallback
- ✅ Resume partial downloads (HF)
- ✅ Local caching
- ✅ Metadata download
- ✅ Progress bars
- ✅ Integrity verification

### **Cache Location**

```
~/.cache/diagram-detector/
├── models/
│   ├── yolo11n.pt
│   ├── yolo11n_metadata.json
│   ├── yolo11m.pt
│   ├── yolo11m_metadata.json
│   └── ...
└── hf_cache/  (Hugging Face internal cache)
```

---

## 🚀 QUICK COMMANDS

```bash
# Upload all models
python upload_models.py -d ~/Downloads/diagram-detector-models/

# Upload one model
python upload_models.py -p ~/Downloads/diagram-detector-models/yolo11m_20241224_.../

# GitHub only
python upload_models.py -p path/to/model --github-only

# Hugging Face only
python upload_models.py -p path/to/model --huggingface-only

# Test download
python -c "from diagram_detector import DiagramDetector; DiagramDetector(model='yolo11n')"

# Clear cache
rm -rf ~/.cache/diagram-detector/

# Check cache
ls -lh ~/.cache/diagram-detector/models/

# Force re-download
python -c "from diagram_detector.utils import download_model; download_model('yolo11m', force=True)"
```

---

## 📊 MODEL INFO

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolo11n | 6 MB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Testing, mobile |
| yolo11s | 22 MB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Edge devices |
| yolo11m | 49 MB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | **Production** ✅ |
| yolo11l | 63 MB | ⚡⚡ | ⭐⭐⭐⭐⭐⭐ | High accuracy |
| yolo11x | 137 MB | ⚡ | ⭐⭐⭐⭐⭐⭐⭐ | Research |

---

## 🔗 URLS

**Hugging Face:**
- Hub: https://huggingface.co/hksorensen/diagram-detector-models
- Direct: `https://huggingface.co/hksorensen/diagram-detector-models/resolve/main/yolo11m.pt`

**GitHub:**
- Releases: https://github.com/hksorensen/diagram-detector/releases
- Direct: `https://github.com/hksorensen/diagram-detector/releases/download/v1.0.0/yolo11m.pt`

---

## ❓ TROUBLESHOOTING

**Download fails:**
```bash
# 1. Check internet
ping github.com

# 2. Try manual download
wget https://huggingface.co/hksorensen/diagram-detector-models/resolve/main/yolo11n.pt

# 3. Move to cache
mv yolo11n.pt ~/.cache/diagram-detector/models/
```

**Corrupted download:**
```bash
rm ~/.cache/diagram-detector/models/yolo11m.pt
python -c "from diagram_detector.utils import download_model; download_model('yolo11m', force=True)"
```

**Upload fails:**
```bash
# Check auth
gh auth status
echo $HF_TOKEN

# Test manually
gh release list
huggingface-cli repo info hksorensen/diagram-detector-models
```

---

## ✅ CHECKLIST

**Before uploading:**
- [ ] Models trained & tested
- [ ] Metadata complete
- [ ] `gh auth login` done
- [ ] `HF_TOKEN` set
- [ ] HF repo created
- [ ] Dry-run successful

**After uploading:**
- [ ] Verify HF: Visit repo page
- [ ] Verify GitHub: Check releases
- [ ] Test download: Clear cache & test
- [ ] Update docs if needed

---

**For full details, see MODEL_DISTRIBUTION.md**
