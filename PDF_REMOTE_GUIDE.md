# PDF Remote Inference Guide - Local Network Optimized

**Optimized for:** Processing PDFs on local gigabit network with caching
**Your setup:** Mac → thinkcentre.local (gigabit LAN)

---

## 🎯 KEY FEATURES

### **PDF-Level Processing**
- ✅ PDFs as processing unit (not individual images)
- ✅ Results per PDF (one JSON per PDF)
- ✅ Clean organization

### **Local Extraction + Remote Inference**
- ✅ Extract PDF pages locally (Mac)
- ✅ Downsample locally = less network traffic
- ✅ Send images to remote GPU
- ✅ Get results back

### **Smart Caching**
- ✅ Cache results by PDF filename
- ✅ Skip already processed PDFs
- ✅ Resume-friendly
- ✅ Fast re-runs

### **Gigabit LAN Optimized**
- ✅ Batch 10 PDFs at once (~100-200 pages)
- ✅ Fast transfer over local network
- ✅ No internet bottleneck

---

## 🚀 QUICK START

### **Default Setup (thinkcentre.local)**

```bash
# Process directory of PDFs
diagram-detect \
  --input ~/pdfs/ \
  --remote hkragh@thinkcentre.local \
  --output results/

# That's it! Defaults optimized for your setup.
```

**What it does:**
1. Finds all PDFs in `~/pdfs/`
2. Checks cache (skips already processed)
3. For each new PDF:
   - Extracts pages locally (DPI=200)
   - Sends images to thinkcentre.local
   - Runs inference on GPU
   - Gets results back
   - Caches results
4. Saves one JSON per PDF

---

## 💻 PYTHON API

```python
from diagram_detector import PDFRemoteDetector

# Initialize (uses thinkcentre.local defaults)
detector = PDFRemoteDetector(
    batch_size=10,     # Process 10 PDFs at a time
    model='yolo11m',
    dpi=200,          # PDF extraction quality
)

# Process PDFs
results = detector.detect_pdfs(
    '~/pdfs/',
    output_dir='results/',
    use_cache=True,    # Skip already processed
)

# Results organized by PDF
for pdf_name, page_results in results.items():
    diagrams = sum(r.count for r in page_results)
    print(f"{pdf_name}: {len(page_results)} pages, {diagrams} diagrams")
```

---

## 📊 HOW IT WORKS

### **Architecture**

```
Mac (Local)                    thinkcentre.local (GPU)
┌────────────────┐            ┌──────────────────┐
│ 1. PDF Files   │            │                  │
│    paper1.pdf  │            │                  │
│    paper2.pdf  │            │                  │
│                │            │                  │
│ 2. Check Cache │            │                  │
│    ✓ paper1    │            │                  │
│    ✗ paper2    │            │                  │
│                │            │                  │
│ 3. Extract     │            │                  │
│    paper2.pdf  │            │                  │
│    → 20 images │            │                  │
│                │  rsync     │                  │
│ 4. Upload      │───────────►│ Receive images   │
│    20 images   │            │                  │
│                │  ssh       │                  │
│ 5. Request     │───────────►│ Run inference    │
│    inference   │            │ (GPU)            │
│                │  rsync     │                  │
│ 6. Download    │◄───────────│ Send results     │
│    results     │            │                  │
│                │            │                  │
│ 7. Cache       │            │                  │
│    paper2      │            │                  │
│    results     │            │                  │
└────────────────┘            └──────────────────┘
```

### **Why This Approach?**

**Local extraction:**
- You control DPI/quality
- Less network traffic (downsampled)
- Works offline for extraction

**Remote inference:**
- Fast GPU (50-100x faster than Mac CPU)
- Mac stays usable
- Gigabit LAN = no bottleneck

**Caching:**
- Re-run = instant (already cached)
- Process new PDFs only
- Resume-friendly

---

## 🔧 CONFIGURATION

### **Default Settings (Optimized for You)**

```python
RemoteConfig(
    host='thinkcentre.local',  # Your local GPU server
    port=22,                    # Standard SSH
    user='hkragh',              # Your username
)

PDFRemoteDetector(
    batch_size=10,     # 10 PDFs per batch (100-200 pages)
    dpi=200,          # Good quality, reasonable size
    model='yolo11m',   # Best balance
)
```

### **Custom Configuration**

```python
from diagram_detector import PDFRemoteDetector, RemoteConfig

# Custom remote
config = RemoteConfig(
    host='192.168.1.100',  # IP address
    port=2222,             # Custom SSH port
    user='username',
)

# Custom detector
detector = PDFRemoteDetector(
    config=config,
    batch_size=5,      # Smaller batches
    dpi=150,           # Lower quality = faster
    model='yolo11l',   # Higher accuracy
    cache_dir='~/.my-cache',
)
```

---

## 💡 USAGE EXAMPLES

### **Example 1: Process New PDFs Only**

```bash
# First run - processes all
diagram-detect --input ~/papers/ --remote hkragh@thinkcentre.local --output results/
# Processes: paper1.pdf, paper2.pdf, paper3.pdf

# Add more PDFs later
# Second run - only processes new ones!
diagram-detect --input ~/papers/ --remote hkragh@thinkcentre.local --output results/
# Cached: paper1.pdf, paper2.pdf, paper3.pdf
# Processes: paper4.pdf (new!)
```

### **Example 2: Force Reprocess**

```bash
# Need to reprocess everything (changed model, confidence, etc.)
diagram-detect \
  --input ~/papers/ \
  --remote hkragh@thinkcentre.local \
  --no-cache \
  --output results/
```

### **Example 3: Custom DPI**

```bash
# Higher quality extraction (slower, larger files)
diagram-detect \
  --input ~/papers/ \
  --remote hkragh@thinkcentre.local \
  --dpi 300 \
  --output results/

# Lower quality (faster, smaller files)
diagram-detect \
  --input ~/papers/ \
  --remote hkragh@thinkcentre.local \
  --dpi 150 \
  --output results/
```

### **Example 4: Python API with Cache Management**

```python
from diagram_detector import PDFRemoteDetector

detector = PDFRemoteDetector()

# Check cache stats
stats = detector.cache.stats()
print(f"Cached: {stats['num_cached_pdfs']} PDFs ({stats['size_mb']:.1f} MB)")

# Process PDFs
results = detector.detect_pdfs('~/papers/', use_cache=True)

# Clear cache if needed
# detector.clear_cache()
```

### **Example 5: Single PDF**

```python
from diagram_detector import PDFRemoteDetector

detector = PDFRemoteDetector()

# Process single PDF
results = detector.detect_pdfs(
    '~/important-paper.pdf',
    output_dir='results/'
)

# Get results for this PDF
pdf_results = results['important-paper.pdf']

# Analyze
for page_result in pdf_results:
    if page_result.has_diagram:
        print(f"Page {page_result.page_number}: {page_result.count} diagrams")
```

---

## 📋 CACHE MANAGEMENT

### **Cache Location**

```
~/.cache/diagram-detector/remote-results/
├── cache_index.json              # Index of cached PDFs
├── abc123def456...json           # Cached results for PDF 1
├── 789ghi012jkl...json           # Cached results for PDF 2
└── ...
```

### **Cache Key**

Cache key based on:
- Filename
- File size
- Modification time

**Result:** If PDF unchanged, cache hit!

### **Python API**

```python
from diagram_detector import PDFRemoteDetector

detector = PDFRemoteDetector()

# Check cache
stats = detector.cache.stats()
print(stats)
# {'num_cached_pdfs': 142, 'cache_dir': '~/.cache/...', 'size_mb': 15.3}

# Get cached result manually
from pathlib import Path
cached = detector.cache.get(Path('~/paper.pdf'))
if cached:
    print("Already processed!")
else:
    print("Need to process")

# Clear cache
detector.clear_cache()
```

### **Command Line**

```bash
# Skip cache (force reprocess)
diagram-detect --input ~/papers/ --remote ... --no-cache

# Cache is automatic otherwise
```

---

## ⚡ PERFORMANCE

### **Your Setup (Gigabit LAN)**

**Single PDF (20 pages, yolo11m):**
- Extract locally: ~10 sec
- Upload (gigabit): ~2 sec
- Inference (GPU): ~5 sec
- Download: ~1 sec
- **Total: ~18 sec**

**Batch (10 PDFs, ~200 pages):**
- Extract: ~2 min
- Upload: ~20 sec
- Inference: ~50 sec
- Download: ~10 sec
- **Total: ~4 min per batch**

**Large corpus (1000 PDFs, ~20K pages):**
- Batches: 100 (10 PDFs each)
- Time per batch: ~4 min
- **Total: ~6-7 hours**

**With caching (second run, 100 new PDFs):**
- Cached: 900 PDFs (instant!)
- Process: 100 PDFs (~40 min)
- **Total: ~40 min**

### **Compare to Alternatives**

| Approach | 1000 PDFs (~20K pages) | Cached (100 new) |
|----------|------------------------|------------------|
| **Local Mac CPU** | ~100-150 hours | Same (no cache) |
| **Remote (no cache)** | ~6-7 hours | ~6-7 hours |
| **Remote (cached)** ✅ | ~6-7 hours | ~40 min ⚡ |

---

## 🔍 MONITORING

### **Command Line Output**

```bash
$ diagram-detect --input ~/papers/ --remote hkragh@thinkcentre.local

============================================================
PDF REMOTE INFERENCE
============================================================
PDFs: 142
Batch size: 10 PDFs/batch
Model: yolo11m
Remote: hkragh@thinkcentre.local
Cache: enabled
============================================================

Checking cache...
  ✓ paper001.pdf (cached)
  ✓ paper002.pdf (cached)
  ...
  ✓ paper100.pdf (cached)
  • paper101.pdf (needs processing)
  • paper102.pdf (needs processing)
  ...

Processing 42 PDFs (100 cached)...

--- Batch 1/5 (10 PDFs) ---
  Extracting pages from paper101.pdf (DPI=200)...
  ✓ Extracted 15 pages
  Extracting pages from paper102.pdf (DPI=200)...
  ✓ Extracted 22 pages
  ...
  Total pages in batch: 187
  Running inference on remote...
✓ Batch complete: 89 diagrams found

--- Batch 2/5 (10 PDFs) ---
...

============================================================
PDF REMOTE INFERENCE COMPLETE
============================================================
Total PDFs: 142
Total pages: 2,847
Pages with diagrams: 1,203 (42.3%)
Total diagrams: 1,567
Cache: 142 PDFs (18.7 MB)
============================================================
```

---

## 🛠️ TROUBLESHOOTING

### **Issue 1: Cache Not Working**

```python
# Check cache stats
from diagram_detector import PDFRemoteDetector

detector = PDFRemoteDetector()
print(detector.cache.stats())

# If shows 0 cached but you processed before:
# Cache key changed (PDF modified?) or cache cleared

# Solution: Check PDF modification time
from pathlib import Path
pdf = Path('paper.pdf')
print(f"Modified: {pdf.stat().st_mtime}")
```

### **Issue 2: Wrong Results Cached**

```bash
# Changed model/confidence but cache has old results

# Solution: Force reprocess
diagram-detect --input ~/papers/ --remote ... --no-cache

# Or clear cache manually
python -c "from diagram_detector import PDFRemoteDetector; PDFRemoteDetector().clear_cache()"
```

### **Issue 3: Out of Disk Space (Cache)**

```python
# Check cache size
from diagram_detector import PDFRemoteDetector

detector = PDFRemoteDetector()
stats = detector.cache.stats()
print(f"Cache using {stats['size_mb']:.1f} MB")

# Clear if too large
detector.clear_cache()
```

### **Issue 4: Can't Connect to thinkcentre.local**

```bash
# Test connection
ping thinkcentre.local
ssh hkragh@thinkcentre.local

# If fails, use IP address instead
diagram-detect --input ~/papers/ --remote hkragh@192.168.1.100

# Or in Python:
from diagram_detector import RemoteConfig, PDFRemoteDetector

config = RemoteConfig(host='192.168.1.100')
detector = PDFRemoteDetector(config=config)
```

---

## 🎯 BEST PRACTICES

### **For Large Corpus**

**1. Test first:**
```bash
# Test with small subset
diagram-detect \
  --input ~/papers-subset/ \
  --remote hkragh@thinkcentre.local \
  --output test-results/
```

**2. Use caching:**
```bash
# Always leave caching enabled (default)
# Don't use --no-cache unless necessary
```

**3. Organize PDFs:**
```bash
# Keep PDFs in one directory for easier processing
~/corpus/
├── paper001.pdf
├── paper002.pdf
└── ...
```

**4. Incremental processing:**
```python
# Process in stages
from pathlib import Path
from diagram_detector import PDFRemoteDetector

detector = PDFRemoteDetector()

# Process by subdirectory
for subdir in Path('~/corpus').iterdir():
    if subdir.is_dir():
        print(f"Processing {subdir.name}...")
        detector.detect_pdfs(subdir, output_dir=f'results/{subdir.name}/')
```

---

## 📊 STORAGE REQUIREMENTS

### **Cache Storage**

**Per PDF:**
- Results: ~1-10 KB per PDF (depends on diagrams found)

**For 1000 PDFs:**
- Cache: ~5-50 MB total

**Cache location:**
```bash
du -sh ~/.cache/diagram-detector/remote-results/
```

### **Temporary Storage**

During processing:
- Mac: Temporary images (~1-2 GB for batch of 200 pages)
- Remote: Temporary images (~same, auto-cleaned)

---

## 🚀 READY TO START?

### **Quick Test**

```bash
diagram-detect \
  --input ~/test-pdfs/ \
  --remote hkragh@thinkcentre.local \
  --output test-results/
```

### **Production Run**

```bash
diagram-detect \
  --input ~/all-papers/ \
  --remote hkragh@thinkcentre.local \
  --model yolo11m \
  --output results/
```

### **Check Results**

```bash
# Count PDFs processed
ls results/*_results.json | wc -l

# View sample
cat results/paper001_results.json | jq .
```

---

## 📝 CHEAT SHEET

```bash
# Basic usage
diagram-detect --input ~/pdfs/ --remote hkragh@thinkcentre.local

# Force reprocess (ignore cache)
diagram-detect --input ~/pdfs/ --remote ... --no-cache

# Custom DPI
diagram-detect --input ~/pdfs/ --remote ... --dpi 300

# Different remote
diagram-detect --input ~/pdfs/ --remote user@other-host:port

# Python API
python -c "
from diagram_detector import PDFRemoteDetector
d = PDFRemoteDetector()
d.detect_pdfs('~/pdfs/', output_dir='results/')
"

# Check cache
python -c "
from diagram_detector import PDFRemoteDetector
print(PDFRemoteDetector().cache.stats())
"

# Clear cache
python -c "
from diagram_detector import PDFRemoteDetector
PDFRemoteDetector().clear_cache()
"
```

---

**Perfect for your use case! PDFs on gigabit LAN with smart caching.** 🎉
