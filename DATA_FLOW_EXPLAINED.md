# Remote Detection Data Flow Explained

## TL;DR: Yes, Results Return to Local!

**Results storage:**
- ✅ **Local cache**: `~/.cache/diagram-detector/detection_cache.db` (on your Mac)
- ✅ **Remote cache**: `~/.cache/diagram-detector/detection_cache.db` (on thinkcentre)
- ✅ **Your code**: Gets results as Python objects

**Cache purpose:**
- **Local cache**: Avoids re-downloading from remote
- **Remote cache**: Avoids re-running inference on GPU

## Complete Data Flow

### First Time Processing a PDF

```
┌─────────────────────────────────────────────────────────────┐
│ YOUR MACHINE (Mac)                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. Your code:                                               │
│    detector.detect_pdfs(["paper1.pdf", "paper2.pdf"])     │
│                                                             │
│ 2. Check LOCAL cache:                                       │
│    ~/.cache/diagram-detector/detection_cache.db            │
│    → Cache miss (first time)                                │
│                                                             │
│ 3. Convert PDFs to images (locally):                        │
│    PDF --[pdf2image @ 200 DPI]--> PNG images               │
│    /tmp/batch_0000/                                         │
│      paper1_page_001.png                                    │
│      paper1_page_002.png                                    │
│      paper2_page_001.png                                    │
│      ...                                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ rsync/scp (upload images)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ REMOTE SERVER (thinkcentre.local)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 4. Images arrive at:                                        │
│    ~/diagram-inference/input/batch_0000/*.png              │
│                                                             │
│ 5. Check REMOTE cache:                                      │
│    ~/.cache/diagram-detector/detection_cache.db            │
│    → Cache miss (first time)                                │
│                                                             │
│ 6. Run inference on GPU:                                    │
│    python3 -m diagram_detector.cli \                       │
│      --input ~/diagram-inference/input/batch_0000 \        │
│      --output ~/diagram-inference/output/batch_0000 \      │
│      --model v5 \                                           │
│      --confidence 0.1 \                                     │
│      --iou 0.3                                              │
│                                                             │
│ 7. YOLO model processes images:                             │
│    [GPU] → Detections for each image                        │
│                                                             │
│ 8. Save results as JSON:                                    │
│    ~/diagram-inference/output/batch_0000/                  │
│      paper1_page_001.json                                   │
│      paper1_page_002.json                                   │
│      paper2_page_001.json                                   │
│      ...                                                    │
│                                                             │
│ 9. Cache results (for next time):                           │
│    ~/.cache/diagram-detector/detection_cache.db            │
│    (Stores by PDF hash + model + params)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ rsync/scp (download JSON)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ YOUR MACHINE (Mac)                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 10. JSON files downloaded to:                               │
│     /tmp/batch_0000/results/*.json                         │
│                                                             │
│ 11. Parse JSON → Python objects:                            │
│     List[DetectionResult] with:                             │
│       - .has_diagram (bool)                                 │
│       - .count (int)                                        │
│       - .detections (List[DiagramDetection])                │
│       - .confidence (float)                                 │
│                                                             │
│ 12. Cache results locally:                                  │
│     ~/.cache/diagram-detector/detection_cache.db           │
│     (Gzip compressed, ~70-90% space savings)                │
│                                                             │
│ 13. Return to your code:                                    │
│     results = {                                             │
│       "paper1.pdf": [DetectionResult(...), ...],           │
│       "paper2.pdf": [DetectionResult(...), ...]            │
│     }                                                       │
│                                                             │
│ 14. Cleanup temp files:                                     │
│     rm -rf /tmp/batch_0000                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Second Time Processing Same PDF

```
┌─────────────────────────────────────────────────────────────┐
│ YOUR MACHINE (Mac)                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. Your code:                                               │
│    detector.detect_pdfs(["paper1.pdf"])                    │
│                                                             │
│ 2. Check LOCAL cache:                                       │
│    ~/.cache/diagram-detector/detection_cache.db            │
│    Cache key = hash(paper1.pdf + v5 + 0.1 + 0.3 + 200)    │
│    → ✓ CACHE HIT!                                          │
│                                                             │
│ 3. Return cached results immediately:                       │
│    results = {"paper1.pdf": [DetectionResult(...), ...]}  │
│                                                             │
│ ⚡ NO network call! NO GPU usage! Instant!                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Cache Locations

### Local Cache (Your Mac)
```bash
~/.cache/diagram-detector/
├── detection_cache.db        # SQLite database
├── detection_cache.db-shm     # Shared memory file
└── detection_cache.db-wal     # Write-ahead log

# Check size:
du -sh ~/.cache/diagram-detector/
# Current: ~2.1 MB (compressed) for 1435 PDFs, 57,601 pages

# Inspect cache:
sqlite3 ~/.cache/diagram-detector/detection_cache.db \
  "SELECT COUNT(*), SUM(LENGTH(results_compressed))/1024/1024 AS mb
   FROM detection_cache"
```

### Remote Cache (thinkcentre.local)
```bash
~/.cache/diagram-detector/
├── detection_cache.db        # SQLite database
└── (same structure)

# Purpose: Avoids re-running inference if you process same PDF again
```

## What Gets Cached?

### Cache Key Components
```python
cache_key = hash(
    pdf_name,
    pdf_size,
    pdf_mtime,       # Last modified time
    model,           # "v5"
    confidence,      # 0.1
    iou,             # 0.3
    dpi,             # 200
    imgsz            # 640
)
```

**Important**: If you change ANY parameter, cache is invalidated!

### Cached Data (Per PDF)
```json
{
  "cache_key": "abc123def456...",
  "pdf_name": "paper1.pdf",
  "model": "v5",
  "confidence": 0.1,
  "iou": 0.3,
  "dpi": 200,
  "results_compressed": "<gzipped JSON>",  // ~70-90% smaller
  "created": "2026-01-12T10:30:00",
  "last_accessed": "2026-01-12T11:45:00",
  "access_count": 3
}
```

## Cache Hierarchy (Two-Level)

```
Your Code
    ↓
┌─────────────────────────────────────┐
│ LOCAL CACHE (L1 - Fast)             │
│ ~/.cache/diagram-detector/          │
│ • Stores final results               │
│ • Avoids re-downloading from remote  │
│ • Instant retrieval                  │
└─────────────────────────────────────┘
    ↓ (if cache miss)
┌─────────────────────────────────────┐
│ REMOTE CACHE (L2 - Still Fast)      │
│ ~/.cache/diagram-detector/          │
│ • Stores GPU inference results       │
│ • Avoids re-running YOLO             │
│ • Saves GPU time                     │
└─────────────────────────────────────┘
    ↓ (if cache miss)
┌─────────────────────────────────────┐
│ GPU INFERENCE (Slow)                 │
│ • Actually runs YOLO model           │
│ • Most expensive operation           │
│ • ~2-5 seconds per page on GPU       │
└─────────────────────────────────────┘
```

## Performance Comparison

| Scenario | Local Cache | Remote Cache | GPU Inference | Total Time |
|----------|-------------|--------------|---------------|------------|
| **First run** | Miss | Miss | ✓ Runs | ~3-5s/page |
| **Second run (same machine)** | ✓ Hit | - | - | ~0.001s/page |
| **Different machine** | Miss | ✓ Hit | - | ~0.5s/page |
| **Different params** | Miss | Miss | ✓ Runs | ~3-5s/page |

## Storage Locations Summary

| What | Where | Size | Purpose |
|------|-------|------|---------|
| **Your results** | Python objects in memory | N/A | What you work with |
| **Local cache** | `~/.cache/diagram-detector/` (Mac) | ~2.1 MB | Avoid re-downloading |
| **Remote cache** | `~/.cache/diagram-detector/` (remote) | ~2.1 MB | Avoid re-inference |
| **Models** | `~/.cache/diagram-detector/models/` | 5-50 MB | YOLO weights |

## Cache Management

### Clear Local Cache
```python
from diagram_detector import DetectionCache

cache = DetectionCache()
cache.clear()  # Clear all cached results
```

### Clear Remote Cache
```bash
ssh -p 22 hkragh@thinkcentre.local \
  "rm -rf ~/.cache/diagram-detector/detection_cache.db*"
```

### Check Cache Stats
```python
from diagram_detector import DetectionCache

cache = DetectionCache()
stats = cache.stats()
print(f"PDFs cached: {stats['num_pdfs']}")
print(f"Pages cached: {stats['total_pages']}")
print(f"Cache size: {stats['size_mb']:.1f} MB")
```

## Key Insights

1. **Results always return to local** ✅
   - You work with Python objects
   - Stored in local cache
   - No data stays "stuck" on remote

2. **Two-level caching is smart** ✅
   - Local cache: fastest (no network)
   - Remote cache: fast (no GPU)
   - GPU: slow (only when necessary)

3. **Remote cache benefits everyone** ✅
   - If your colleague processes same PDF, uses your cached results
   - If you process on different machine, reuses remote cache
   - Shared GPU compute resource

4. **Cache invalidation works correctly** ✅
   - Change model → cache miss
   - Change confidence → cache miss
   - Change any parameter → cache miss
   - Ensures results match parameters

## Example Usage

```python
from pathlib import Path
from diagram_detector import PDFRemoteDetector, get_remote_endpoint

# Get endpoint
endpoint = get_remote_endpoint()

# Create detector
detector = PDFRemoteDetector(
    config=endpoint,
    model="v5",
    confidence=0.1,
    iou=0.3,
    dpi=200,
    verbose=True
)

# Process PDFs (uses cache automatically)
pdfs = list(Path("pdfs/").glob("*.pdf"))
results = detector.detect_pdfs(pdfs, use_cache=True)

# Results are now in memory on YOUR machine:
for pdf_name, page_results in results.items():
    diagrams = sum(r.count for r in page_results)
    print(f"{pdf_name}: {diagrams} diagrams")

# Results are also cached locally for instant retrieval next time!
```

## Summary

✅ **Results return to local**: Always
✅ **Local cache location**: `~/.cache/diagram-detector/` on your Mac
✅ **Remote cache purpose**: Avoid re-running GPU inference
✅ **Your storage**: Python objects + local cache (you control this)
✅ **Remote storage**: Just a cache (can be cleared anytime)

The remote server is **purely computational** - it runs inference and returns results. All long-term storage happens on your local machine.
