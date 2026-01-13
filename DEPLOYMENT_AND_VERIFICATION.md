# Deployment and Verification Issues

## Critical Findings

### 🚨 **CRITICAL: Remote Server Has No Code Deployed**

The remote server (`thinkcentre.local`) **does not have diagram_detector installed**:

```bash
# Tested on remote:
python3 -c 'import diagram_detector'  # ModuleNotFoundError
python3 -m pip list | grep diagram    # Not found
find ~ -name 'diagram_detector'       # No results
```

**Current state**: Remote detection **cannot work** because the code isn't there!

---

## Question 1: Should we add DPI as a parameter locally and remotely?

### Answer: **DPI is already correct as-is**

**Current behavior** (CORRECT):
- DPI is used for **local PDF→image conversion** only
- Images are sent to remote, not PDFs
- Remote receives pre-rendered images at specified DPI

**Why this is correct**:
```
Local:
  PDF --[DPI=200]--> Images (1654x2339 pixels @ 200 DPI)
                        ↓
                    Upload to remote
                        ↓
Remote:
  Images --> YOLO detection (no DPI needed, just pixels)
```

**Recommendation**: ✅ **No change needed** - DPI is correctly handled locally only.

---

## Question 2: How is the CLI script deployed?

### Answer: **It's NOT deployed (critical issue)**

**Current state**:
- Remote server has NO diagram_detector package
- Only has work directories: `~/diagram-inference/input` and `~/diagram-inference/output`
- No git repo, no Python package

**This means remote detection is broken!**

### Recommended Deployment Methods

#### **Option A: Git-based deployment** (Recommended for development)

1. Create git repo of diagram-detector package
2. Deploy via SSH:
   ```bash
   # On remote server
   cd ~
   git clone <your-repo>/diagram-detector.git
   cd diagram-detector
   python3 -m pip install -e .  # Editable install for development

   # Update when needed
   git pull
   ```

**Pros**: Easy updates, version control, can roll back
**Cons**: Need to maintain git repo

#### **Option B: pip install from local** (Recommended for stable releases)

1. Build package locally:
   ```bash
   cd /Users/fvb832/Downloads/diagram-detector-package\ 2
   python3 setup.py sdist bdist_wheel
   ```

2. Copy and install on remote:
   ```bash
   scp -P 22 dist/diagram_detector-1.0.0-py3-none-any.whl hkragh@thinkcentre.local:~/
   ssh -p 22 hkragh@thinkcentre.local "python3 -m pip install --upgrade ~/diagram_detector-1.0.0-py3-none-any.whl"
   ```

**Pros**: Clean, versioned, standard Python packaging
**Cons**: Need to rebuild/redeploy after changes

#### **Option C: rsync-based deployment** (Quick for development)

```bash
# Sync local code to remote
rsync -avz --delete \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  /Users/fvb832/Downloads/diagram-detector-package\ 2/diagram_detector/ \
  hkragh@thinkcentre.local:~/diagram-detector-package/diagram_detector/

# Install on remote
ssh -p 22 hkragh@thinkcentre.local "cd ~/diagram-detector-package && python3 -m pip install -e ."
```

**Pros**: Very fast updates during development
**Cons**: Manual process, no version control

### Recommended: **Hybrid Approach**

```bash
# 1. Initial setup: Git clone + editable install (do once)
ssh -p 22 hkragh@thinkcentre.local << 'EOF'
cd ~
git clone <your-repo> diagram-detector
cd diagram-detector
python3 -m pip install -e .
EOF

# 2. Quick updates during development: rsync
rsync -avz --delete \
  --exclude='__pycache__' \
  /Users/fvb832/Downloads/diagram-detector-package\ 2/diagram_detector/ \
  hkragh@thinkcentre.local:~/diagram-detector/diagram_detector/

# 3. For stable releases: pip install wheel
```

---

## Question 3: How do we ensure the model is correct beyond name?

### Answer: **Currently NO verification (security risk)**

**Current behavior**:
- Model identified by name only ("v5", "yolo11m")
- No hash verification
- No signature checking
- Only checks: file exists and size > 0

**This is a problem because**:
- Can't detect corrupted downloads
- Can't detect malicious model swaps
- No guarantee local and remote have same weights
- Cache might use wrong model

### Recommended: **SHA256 Hash Verification**

#### **Implementation Plan**:

1. **Add model registry with hashes**:
   ```python
   MODEL_REGISTRY = {
       "v5": {
           "size_mb": 5.2,
           "sha256": "abc123def456...",  # Actual hash of v5.pt
           "url_hf": "...",
           "url_github": "...",
       },
       "yolo11m": {
           "size_mb": 49,
           "sha256": "def789ghi012...",
           "url_hf": "...",
           "url_github": "...",
       },
   }
   ```

2. **Verify after download**:
   ```python
   def verify_model_hash(model_path: Path, expected_hash: str) -> bool:
       """Verify model file matches expected SHA256 hash."""
       import hashlib

       sha256 = hashlib.sha256()
       with open(model_path, 'rb') as f:
           while chunk := f.read(8192):
               sha256.update(chunk)

       actual_hash = sha256.hexdigest()
       return actual_hash == expected_hash

   # In download_model():
   if not verify_model_hash(model_path, MODEL_REGISTRY[model_name]['sha256']):
       model_path.unlink()
       raise RuntimeError(f"Model hash mismatch! File may be corrupted or tampered.")
   ```

3. **Verify before use**:
   ```python
   # In DiagramDetector.__init__():
   if not verify_model_hash(model_path, MODEL_REGISTRY[model_name]['sha256']):
       raise RuntimeError(f"Cached model hash mismatch! Redownload with force=True")
   ```

4. **Remote verification command**:
   ```python
   def verify_remote_model(config: RemoteConfig, model: str) -> bool:
       """Verify remote server has correct model."""
       cmd = f"python3 -c 'from diagram_detector.utils import verify_model_hash, get_model_path, MODEL_REGISTRY; " \
             f"print(verify_model_hash(get_model_path(\"{model}\"), MODEL_REGISTRY[\"{model}\"][\"sha256\"]))'"
       # Run via SSH, check output is "True"
   ```

### Alternative: **Model Metadata Files**

Store metadata with hash:
```json
// v5_metadata.json
{
  "name": "v5",
  "version": "1.0.0",
  "sha256": "abc123def456...",
  "trained_on": "clean_dataset_v2",
  "training_date": "2024-01-15",
  "optimal_params": {
    "confidence": 0.1,
    "iou": 0.3
  }
}
```

**Benefits**:
- Links model to optimal parameters
- Provides provenance
- Enables version checking

---

## Immediate Action Items

### 1. **Deploy diagram_detector to remote** (CRITICAL)

```bash
# Quick fix (recommended):
cd /Users/fvb832/Downloads/diagram-detector-package\ 2
tar czf diagram-detector.tar.gz diagram_detector/ setup.py README.md
scp -P 22 diagram-detector.tar.gz hkragh@thinkcentre.local:~/
ssh -p 22 hkragh@thinkcentre.local << 'EOF'
cd ~
tar xzf diagram-detector.tar.gz
cd diagram-detector-package
python3 -m pip install --user -e .
rm ~/diagram-detector.tar.gz
EOF
```

### 2. **Generate model hashes**

```bash
cd /Users/fvb832/.cache/diagram-detector/models/
sha256sum *.pt > model_hashes.txt
cat model_hashes.txt
```

### 3. **Add hash verification** (Code changes needed)

1. Add MODEL_REGISTRY with hashes to `utils.py`
2. Add `verify_model_hash()` function
3. Add verification to download and load operations
4. Update remote verification

### 4. **Test remote detection**

```bash
# After deploying code
cd /Users/fvb832/Downloads/diagram-detector-package\ 2
python3 test_batched_detection.py
```

---

## Summary

| Question | Current State | Recommended Action |
|----------|--------------|-------------------|
| **1. DPI parameter** | ✅ Correct (local only) | No action needed |
| **2. CLI deployment** | ❌ NOT deployed | Deploy via git + pip install -e |
| **3. Model verification** | ❌ No verification | Add SHA256 hash checking |

**Priority**:
1. 🔴 **URGENT**: Deploy diagram_detector to remote server
2. 🟡 **Important**: Add model hash verification
3. 🟢 **Nice-to-have**: Automated deployment script

Would you like me to:
1. Create deployment script?
2. Generate model hashes?
3. Implement hash verification?
