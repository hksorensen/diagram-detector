# Remote GPU Inference Guide

**Feature:** SSH Remote Inference
**Version:** 1.0.0
**Status:** Production Ready ✅

---

## 🎯 OVERVIEW

Process **300K+ images** on remote GPU server instead of local Mac.

**Benefits:**
- ⚡ **50-100x faster** than Mac CPU
- 🔋 **No battery drain** on Mac
- 💻 **Keep working** while processing
- 📦 **Intelligent batching** (1000 images/batch)
- 🔄 **Auto-resume** on failure
- 🧹 **Auto-cleanup** remote files

**Performance:**
```
300,000 images:
- Mac CPU: ~150-200 hours (6-8 days!)
- Remote GPU: ~5-10 hours ⚡
```

---

## 🚀 QUICK START

### **Command Line**

```bash
# Process directory on remote GPU
diagram-detect \
  --input images/ \
  --remote hkragh@henrikkragh.dk:8022 \
  --model yolo11m \
  --output results/

# Process with custom batching
diagram-detect \
  --input images/ \
  --remote hkragh@henrikkragh.dk:8022 \
  --remote-batch-size 500 \
  --gpu-batch-size 64 \
  --output results/
```

### **Python API**

```python
from diagram_detector import SSHRemoteDetector, RemoteConfig

# Setup remote connection
config = RemoteConfig(
    host='henrikkragh.dk',
    port=8022,
    user='hkragh'
)

# Initialize detector
detector = SSHRemoteDetector(
    config=config,
    batch_size=1000,  # Images per batch
    model='yolo11m',
    confidence=0.35
)

# Run inference
results = detector.detect('images/', output_dir='results/')

# Results are same format as local!
for result in results:
    if result.has_diagram:
        print(f"{result.filename}: {result.count} diagrams")
```

---

## 📊 HOW IT WORKS

### **Intelligent Batching**

```
300K images → Split into 300 batches (1000 each)

For each batch:
1. Upload images (rsync)      ~2-3 min
2. Run inference (GPU)         ~5-10 min
3. Download results (rsync)    ~1-2 min
4. Cleanup remote files        ~10 sec

Total: ~8-15 min per batch
300 batches: ~40-75 hours (with network)
```

**Why batching?**
- Prevents network bottleneck
- Allows resume on failure
- Manages disk space
- Progress tracking

---

## 🔧 CONFIGURATION

### **Batch Sizes**

**Remote batch size** (`--remote-batch-size`):
- How many images to upload at once
- Default: 1000
- Larger = fewer uploads, more disk space
- Smaller = more frequent progress

**GPU batch size** (`--gpu-batch-size`):
- Inference batch on GPU
- Default: 32
- Depends on GPU memory
- RTX 4090: 64-128
- RTX 3090: 32-64

### **Connection String Formats**

```bash
# Simple
--remote user@host

# With port
--remote user@host:port

# With ssh:// prefix
--remote ssh://user@host:port

# Examples
--remote hkragh@henrikkragh.dk:8022
--remote ssh://user@server.com
```

---

## 📋 COMMAND LINE REFERENCE

### **Basic Options**

```bash
--remote USER@HOST:PORT        # Enable remote inference
--model MODEL_NAME             # yolo11n, yolo11m, etc.
--confidence FLOAT             # Threshold (default: 0.35)
--output DIR                   # Output directory
```

### **Batching Options**

```bash
--remote-batch-size INT        # Images per upload batch (default: 1000)
--gpu-batch-size INT           # GPU inference batch (default: 32)
```

### **Advanced Options**

```bash
--resume                       # Resume interrupted job
--no-cleanup                   # Keep remote files
--quiet                        # Suppress progress output
```

### **Output Options** (same as local)

```bash
--save-crops                   # Extract diagram regions
--visualize                    # Draw bounding boxes
--format json|csv|both         # Output format
```

---

## 💡 USAGE EXAMPLES

### **Example 1: Process Large Directory**

```bash
# 100,000 images in ~/papers/figures/
diagram-detect \
  --input ~/papers/figures/ \
  --remote hkragh@henrikkragh.dk:8022 \
  --output ~/results/ \
  --model yolo11m
```

**What happens:**
1. Finds 100,000 images
2. Splits into 100 batches (1000 each)
3. Uploads batch 1 → Processes → Downloads results
4. Uploads batch 2 → Processes → Downloads results
5. ... continues until done
6. Final results in `~/results/`

**Time:** ~13-25 hours (depends on network)

---

### **Example 2: Resume Interrupted Job**

```bash
# Job interrupted at batch 47/100

# Just add --resume!
diagram-detect \
  --input ~/papers/figures/ \
  --remote hkragh@henrikkragh.dk:8022 \
  --output ~/results/ \
  --resume

# Skips batches 1-46 (already processed)
# Continues from batch 47
```

---

### **Example 3: Custom Batching**

```bash
# Faster network? Use larger batches
diagram-detect \
  --input images/ \
  --remote hkragh@henrikkragh.dk:8022 \
  --remote-batch-size 2000 \
  --gpu-batch-size 64 \
  --output results/

# Slower network? Use smaller batches
diagram-detect \
  --input images/ \
  --remote hkragh@henrikkragh.dk:8022 \
  --remote-batch-size 500 \
  --gpu-batch-size 32 \
  --output results/
```

---

### **Example 4: Process with Visualization**

```bash
diagram-detect \
  --input images/ \
  --remote hkragh@henrikkragh.dk:8022 \
  --visualize \
  --save-crops \
  --output results/

# Results:
# results/
# ├── detections.json
# ├── visualizations/  (with bounding boxes)
# └── crops/           (extracted diagrams)
```

---

### **Example 5: Python API with Custom Config**

```python
from diagram_detector import SSHRemoteDetector, RemoteConfig
from pathlib import Path

# Custom remote configuration
config = RemoteConfig(
    host='henrikkragh.dk',
    port=8022,
    user='hkragh',
    remote_work_dir='~/inference-work',
    python_path='~/venv/bin/python'  # Use venv
)

# Initialize with large batches (fast network)
detector = SSHRemoteDetector(
    config=config,
    batch_size=2000,
    model='yolo11l',  # Larger model
    confidence=0.4,
)

# Process images
image_dir = Path('~/corpus/images')
results = detector.detect(
    image_dir,
    output_dir='results/',
    gpu_batch_size=64,
    cleanup=True,
    resume=False,
)

# Analyze results
total_diagrams = sum(r.count for r in results)
print(f"Found {total_diagrams:,} diagrams in {len(results):,} images")
```

---

## 🔍 MONITORING PROGRESS

### **Command Line Output**

```bash
$ diagram-detect --input images/ --remote hkragh@henrikkragh.dk:8022

============================================================
REMOTE INFERENCE
============================================================
Images: 300,000
Batch size: 1,000 images/batch
GPU batch size: 32
Model: yolo11m
Remote: hkragh@henrikkragh.dk:8022
============================================================

Processing 300,000 images in 300 batch(es)...

--- Batch 1/300 ---
Uploading batch batch_0000 (1000 images)...
✓ Batch batch_0000 uploaded
Running inference on batch batch_0000...
✓ Batch batch_0000 processed
Downloading results for batch batch_0000...
✓ Batch batch_0000 results downloaded
Cleaning up batch batch_0000...
✓ Batch complete: 342 diagrams found

--- Batch 2/300 ---
...
```

### **Check Progress**

```bash
# In another terminal, check results directory
ls -lh results/

# Count completed batches
ls results/ | wc -l

# Example: 47 batches done, 253 remaining
```

---

## 🛠️ TROUBLESHOOTING

### **Issue 1: SSH Connection Failed**

```
Error: SSH connection failed: Permission denied
```

**Solution:**
```bash
# Test SSH manually
ssh -p 8022 hkragh@henrikkragh.dk

# If fails, check:
# 1. Correct hostname
# 2. Correct port
# 3. SSH key added
ssh-add ~/.ssh/id_rsa

# 4. Server accessible
ping henrikkragh.dk
```

---

### **Issue 2: Package Not Installed on Remote**

```
Error: Remote command failed: diagram_detector.cli not found
```

**Solution:**
```bash
# SSH to remote and install
ssh -p 8022 hkragh@henrikkragh.dk

# On remote:
pip install diagram-detector

# Or use venv (recommended):
python -m venv ~/venv
source ~/venv/bin/activate
pip install diagram-detector

# Then specify python path:
diagram-detect --remote ... --python-path ~/venv/bin/python
```

Or in Python:
```python
config = RemoteConfig(
    host='henrikkragh.dk',
    port=8022,
    user='hkragh',
    python_path='~/venv/bin/python'
)
```

---

### **Issue 3: Out of Disk Space on Remote**

```
Error: No space left on device
```

**Solutions:**

**A. Use smaller batches:**
```bash
--remote-batch-size 500  # Instead of 1000
```

**B. Enable cleanup (default):**
```bash
# This is default, but if you used --no-cleanup before:
diagram-detect --remote ... # (no --no-cleanup flag)
```

**C. Manually clean remote:**
```bash
ssh -p 8022 hkragh@henrikkragh.dk
rm -rf ~/diagram-inference/*
```

---

### **Issue 4: Interrupted Job**

```
# Job stopped at batch 47/100
^C
```

**Solution:**
```bash
# Just add --resume!
diagram-detect \
  --input images/ \
  --remote hkragh@henrikkragh.dk:8022 \
  --output results/ \
  --resume

# Automatically skips completed batches
```

---

### **Issue 5: Slow Upload/Download**

**Symptoms:** Hours per batch

**Solutions:**

**A. Check network:**
```bash
# Test upload speed
rsync -az --progress test.jpg hkragh@henrikkragh.dk:~/test/

# Test download speed
rsync -az --progress hkragh@henrikkragh.dk:~/test.jpg ./
```

**B. Use compression (already enabled):**
```bash
# rsync uses -z flag automatically
```

**C. Process on remote directly:**
```bash
# If images already on remote, SSH and run locally:
ssh -p 8022 hkragh@henrikkragh.dk

# On remote:
diagram-detect --input ~/remote-images/ --output ~/results/
```

---

## 📊 PERFORMANCE EXPECTATIONS

### **Network Impact**

| Network Speed | 1000 images | 10,000 images | 100,000 images |
|---------------|-------------|---------------|----------------|
| Fast (100 Mbps) | 8-10 min | 1.5-2 hours | 15-20 hours |
| Medium (50 Mbps) | 12-15 min | 2-3 hours | 20-30 hours |
| Slow (10 Mbps) | 20-30 min | 3-5 hours | 30-50 hours |

**Note:** Most time is upload/download, not inference!

### **GPU Speed**

| GPU | 1000 images | 10,000 images | 100,000 images |
|-----|-------------|---------------|----------------|
| RTX 4090 | ~3-5 min | ~30-50 min | ~5-8 hours |
| RTX 3090 | ~5-8 min | ~50-80 min | ~8-13 hours |
| RTX 2080 Ti | ~8-12 min | ~80-120 min | ~13-20 hours |

**Inference only** (excluding network time)

### **300K Images Timeline**

**Scenario:** 300,000 images, yolo11m, RTX 4090, 50 Mbps network

```
Total batches: 300
Per batch: ~15 min (upload + inference + download)
Total: 300 × 15 = 4,500 min = 75 hours = ~3 days

With interruptions/resume: ~3.5-4 days
```

**Faster options:**
- Use larger batches: ~50-60 hours
- Process directly on remote: ~8-10 hours
- Use yolo11n (faster model): ~40-50 hours

---

## 🎯 BEST PRACTICES

### **For 300K Images**

**1. Test first:**
```bash
# Test with 1000 images
diagram-detect \
  --input test-images/ \
  --remote hkragh@henrikkragh.dk:8022 \
  --output test-results/

# Check time, adjust batching
```

**2. Use resume-friendly setup:**
```bash
# Save to specific directory
--output ~/corpus-results-2024-12-24/

# Can resume anytime:
diagram-detect --input images/ --remote ... --output ~/corpus-results-2024-12-24/ --resume
```

**3. Monitor disk space:**
```bash
# Check remote disk before starting
ssh -p 8022 hkragh@henrikkragh.dk df -h ~/
```

**4. Consider direct processing:**
```bash
# If images already on remote:
# 1. Upload images once (rsync)
# 2. SSH to remote
# 3. Run inference locally on remote (much faster!)

rsync -az images/ hkragh@henrikkragh.dk:~/corpus/
ssh -p 8022 hkragh@henrikkragh.dk
diagram-detect --input ~/corpus/ --output ~/results/ --model yolo11m
```

---

## 💰 COST ESTIMATE

**Assuming:**
- 300K images
- Network: 50 Mbps
- GPU: RTX 4090
- Electricity: $0.15/kWh
- GPU power: 450W

**Time:** ~75 hours (3 days)
**Energy:** 75h × 0.45kW = 33.75 kWh
**Cost:** 33.75 × $0.15 = **~$5**

**Compare to Mac CPU:**
- Time: ~200 hours (8 days)
- Energy: 200h × 0.05kW = 10 kWh
- Cost: 10 × $0.15 = **~$1.50**

**But:** 
- Mac is unusable for 8 days
- Remote: Mac free to use!
- Time saved = $$$

---

## 📋 CHECKLIST

**Before starting large job:**

- [ ] Test SSH connection works
- [ ] Package installed on remote
- [ ] Test with small batch (100 images)
- [ ] Check remote disk space
- [ ] Verify GPU available (nvidia-smi)
- [ ] Choose batch sizes
- [ ] Setup output directory
- [ ] Bookmark this guide!

**During processing:**

- [ ] Monitor progress occasionally
- [ ] Check disk space if slow
- [ ] Note which batch you're on
- [ ] Keep Mac connected (don't sleep!)

**If interrupted:**

- [ ] Use `--resume` to continue
- [ ] Check last completed batch
- [ ] Resume from where left off

---

## 🚀 READY TO START?

```bash
# The command:
diagram-detect \
  --input ~/your-300k-images/ \
  --remote hkragh@henrikkragh.dk:8022 \
  --model yolo11m \
  --confidence 0.35 \
  --output ~/results/ \
  --format json \
  --resume

# Let it run for ~3-4 days
# Or split into sessions with --resume
```

---

**End of remote inference guide**
**Good luck with your 300K images! 🎉**
