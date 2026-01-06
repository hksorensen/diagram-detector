# Configuration File Guide

The diagram-detector package supports YAML configuration files for reproducible, configurable detection workflows.

## Basic Usage

```python
from diagram_detector import DiagramDetector

# Load detector from config
detector = DiagramDetector.from_config("config.yaml")

# Or run complete workflow from config
results = DiagramDetector.run_from_config("config.yaml")
```

## Configuration Schema

### Complete Example

```yaml
# config.yaml - Complete configuration example

detector:
  model: v5                    # Model name (v5, yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
  confidence: 0.20             # Confidence threshold (0.0-1.0)
  iou: 0.30                    # IoU threshold for NMS (0.0-1.0)
  device: auto                 # Device (auto, cpu, cuda, mps)
  batch_size: auto             # Batch size (auto or integer)
  verbose: true                # Print progress information

paths:
  input: /path/to/pdfs         # Input directory or file (required)
  output: /path/to/results     # Output directory (default: results)
  detections: detections.json  # Detections filename (optional)
  crops: crops/                # Crops subdirectory (optional, relative to output)
  visualizations: viz/         # Visualizations subdirectory (optional, relative to output)

options:
  dpi: 300                     # DPI for PDF conversion (default: 200)
  save_crops: true             # Extract diagram crops (default: false)
  save_visualizations: false   # Save visualizations with bboxes (default: false)
  crop_padding: 10             # Padding around crops in pixels (default: 10)
  format: json                 # Output format: json, csv, or both (default: json)
```

## Section Details

### `detector` Section (Required)

Configures the detection model and inference parameters.

**Fields:**

- `model` (string, default: `yolo11m`)
  - Available models: `v5`, `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`
  - Larger models are more accurate but slower

- `confidence` (float, default: `0.20`)
  - Minimum confidence threshold for detections
  - Range: 0.0 to 1.0
  - Recommended: 0.20 (optimized via grid search)

- `iou` (float, default: `0.30`)
  - IoU threshold for non-maximum suppression
  - Range: 0.0 to 1.0
  - Recommended: 0.30 (optimized via grid search)

- `device` (string, default: `auto`)
  - Compute device for inference
  - Options: `auto`, `cpu`, `cuda`, `mps`
  - `auto` automatically detects best available device

- `batch_size` (int or string, default: `auto`)
  - Number of images to process in parallel
  - `auto` optimizes based on device and model
  - Manual values: 1, 8, 16, 32, etc.

- `verbose` (boolean, default: `true`)
  - Print progress information and warnings
  - Set to `false` for quiet operation

### `paths` Section (Required for `run_from_config()`)

Specifies input and output paths.

**Fields:**

- `input` (string, required)
  - Path to input PDF file, image file, or directory
  - Can be absolute or relative path

- `output` (string, default: `results`)
  - Output directory for all results
  - Created automatically if doesn't exist
  - **Warning:** Relative paths use current working directory

- `detections` (string, optional)
  - Filename for detection results
  - Default: `detections.json`
  - Ignored - always saved to output directory

- `crops` (string, default: `crops`)
  - Subdirectory for extracted diagram crops
  - Relative to `output` directory unless absolute path
  - Only used if `save_crops: true`

- `visualizations` (string, default: `visualizations`)
  - Subdirectory for visualization images
  - Relative to `output` directory unless absolute path
  - Only used if `save_visualizations: true`

### `options` Section (Optional)

Configures detection and output options.

**Fields:**

- `dpi` (int, default: `200`)
  - DPI for PDF to image conversion
  - Higher = better quality but slower
  - Recommended: 200-300

- `save_crops` (boolean, default: `false`)
  - Extract and save cropped diagram regions
  - Requires images to be stored in memory
  - Crops saved to `{output}/{crops}/`

- `save_visualizations` (boolean, default: `false`)
  - Save images with bounding boxes drawn
  - Requires images to be stored in memory
  - Visualizations saved to `{output}/{visualizations}/`

- `crop_padding` (int, default: `10`)
  - Pixels to add around bbox when cropping
  - Only used if `save_crops: true`

- `format` (string, default: `json`)
  - Output format for detection results
  - Options: `json`, `csv`, `both`

## Usage Examples

### Example 1: Basic Detection

```yaml
# basic_config.yaml
detector:
  model: v5
  confidence: 0.20

paths:
  input: papers/
  output: results/
```

```python
from diagram_detector import DiagramDetector

results = DiagramDetector.run_from_config("basic_config.yaml")
print(f"Detected {len(results)} images")
```

### Example 2: High-Quality Crop Extraction

```yaml
# crop_extraction.yaml
detector:
  model: yolo11l           # Larger, more accurate model
  confidence: 0.35         # Higher confidence threshold
  device: cuda             # Use GPU

paths:
  input: /data/pdfs
  output: /data/extracted
  crops: high_quality_crops/

options:
  dpi: 300                 # High resolution
  save_crops: true         # Extract crops
  crop_padding: 20         # Extra padding
```

### Example 3: Fast Batch Processing

```yaml
# fast_batch.yaml
detector:
  model: yolo11n           # Smallest, fastest model
  confidence: 0.20
  device: cuda
  batch_size: 64           # Large batch for speed

paths:
  input: /data/corpus
  output: /data/results

options:
  format: csv              # CSV for easy analysis
  save_crops: false        # Skip crops for speed
  save_visualizations: false
```

### Example 4: Debugging/Development

```yaml
# debug_config.yaml
detector:
  model: v5
  confidence: 0.15          # Lower threshold to catch more
  device: auto
  verbose: true             # Detailed output

paths:
  input: test_pdfs/
  output: debug_results/
  crops: crops/
  visualizations: viz/

options:
  dpi: 200
  save_crops: true
  save_visualizations: true  # Visualize all detections
  format: both              # Both JSON and CSV
```

## Path Resolution Rules

1. **Absolute paths**: Used as-is
   ```yaml
   paths:
     output: /absolute/path/results
     crops: /absolute/path/crops       # → /absolute/path/crops
   ```

2. **Relative subdirs**: Relative to output directory
   ```yaml
   paths:
     output: results
     crops: my_crops                    # → results/my_crops
   ```

3. **Current working directory**: For relative output
   ```yaml
   paths:
     output: results                    # → {CWD}/results
   ```
   **Warning:** Will show warning about relative path

## Output Structure

With this config:

```yaml
paths:
  output: /data/results
  crops: crops/
  visualizations: viz/
options:
  save_crops: true
  save_visualizations: true
  format: both
```

Output structure:

```
/data/results/
├── detections.json          # Detection results (JSON)
├── detections.csv           # Detection results (CSV)
├── crops/                   # Extracted diagrams
│   ├── paper1_diagram1.jpg
│   ├── paper1_diagram2.jpg
│   └── paper2_diagram1.jpg
└── viz/                     # Visualizations
    ├── paper1_page1.jpg
    ├── paper1_page2.jpg
    └── paper2_page1.jpg
```

## Caching System

The diagram-detector package includes a built-in SQLite-based caching system that dramatically speeds up repeated processing of the same PDFs with the same parameters.

### How Caching Works

**Cache Key**: Results are cached using a composite key that includes:
- PDF metadata: filename, file size, modification time
- Detection parameters: model, confidence, iou, dpi

This means:
- ✅ Same PDF + same parameters = cache hit (fast!)
- ❌ Same PDF + different parameters = cache miss (reprocesses)
- ❌ Modified PDF (even same name) = cache miss (detects change)

**Example:**

```python
detector = DiagramDetector(model="v5", confidence=0.20)

# First run: processes PDF, saves to cache
results1 = detector.detect_pdf("paper.pdf")  # ~30 seconds

# Second run: loads from cache
results2 = detector.detect_pdf("paper.pdf")  # ~0.1 seconds ⚡

# Different parameters: cache miss, reprocesses
detector2 = DiagramDetector(model="yolo11l", confidence=0.35)
results3 = detector2.detect_pdf("paper.pdf")  # ~30 seconds (cache miss)
```

### Configuring Cache Behavior

**Programmatic Usage:**

```python
from diagram_detector import DiagramDetector, DetectionCache

# Option 1: Use default cache (recommended)
detector = DiagramDetector(cache=True)  # Default

# Option 2: Disable cache
detector = DiagramDetector(cache=False)

# Option 3: Custom cache location
custom_cache = DetectionCache(
    cache_dir="/path/to/my/cache",
    compression=True,
    auto_cleanup=True,
    max_size_mb=1000  # 1GB limit
)
detector = DiagramDetector(cache=custom_cache)

# Option 4: Shared cache across multiple detectors
cache = DetectionCache()
detector1 = DiagramDetector(model="v5", cache=cache)
detector2 = DiagramDetector(model="yolo11l", cache=cache)
```

**Config File:**

```yaml
# Caching is enabled by default
detector:
  model: v5
  confidence: 0.20
  # cache: true  # Implicit default

# To disable caching
detector:
  model: v5
  cache: false
```

**Per-Call Override:**

```python
# Disable cache for one call (e.g., force reprocessing)
results = detector.detect_pdf("paper.pdf", use_cache=False)

# Force reprocessing even if cached
# (Not currently supported - use use_cache=False instead)
```

### Cache Location

**Default Locations:**

- macOS: `~/Library/Caches/diagram-detector/`
- Linux: `~/.cache/diagram-detector/`
- Windows: `%LOCALAPPDATA%\diagram-detector\cache\`

**Custom Location:**

```python
cache = DetectionCache(cache_dir="/my/project/cache")
detector = DiagramDetector(cache=cache)
```

**Important:** All detectors sharing the same cache directory will share cached results (as long as parameters match).

### Cache Statistics

```python
from diagram_detector import DiagramDetector

detector = DiagramDetector()

# Get cache stats
stats = detector.cache.stats()
print(f"Cached PDFs: {stats['num_pdfs']}")
print(f"Total pages: {stats['total_pages']:,}")
print(f"Cache size: {stats['size_mb']:.1f} MB")
print(f"Cache compression: ~{stats['compression_ratio']:.1f}x")

# Example output:
# Cached PDFs: 1,247
# Total pages: 18,392
# Cache size: 89.3 MB
# Cache compression: ~12x
```

### Cache Management

**Clear Cache:**

```python
# Clear entire cache
detector.cache.clear()

# Or manually
from diagram_detector import DetectionCache
cache = DetectionCache()
cache.clear()
```

**Automatic Cleanup:**

The cache supports automatic size-based cleanup using LRU (Least Recently Used) eviction:

```python
cache = DetectionCache(
    auto_cleanup=True,
    max_size_mb=1000  # Keep cache under 1GB
)

# When cache exceeds 1GB, least recently accessed entries are removed
```

**Manual Inspection:**

```bash
# Find cache database
ls -lh ~/Library/Caches/diagram-detector/

# Inspect with sqlite3
sqlite3 ~/Library/Caches/diagram-detector/detection_cache.db

sqlite> SELECT
    pdf_name,
    model,
    confidence,
    num_pages,
    datetime(cached_at) as cached,
    access_count
FROM detection_cache
ORDER BY last_accessed DESC
LIMIT 10;
```

### Cache Database Schema

The cache uses SQLite with the following schema:

```sql
CREATE TABLE detection_cache (
    -- Composite cache key (SHA-256)
    cache_key TEXT PRIMARY KEY,

    -- PDF metadata
    pdf_name TEXT NOT NULL,
    pdf_size INTEGER NOT NULL,
    pdf_mtime REAL NOT NULL,

    -- Detection parameters
    model TEXT NOT NULL,
    confidence REAL NOT NULL,
    iou REAL NOT NULL,
    dpi INTEGER NOT NULL,

    -- Results
    num_pages INTEGER NOT NULL,
    results_compressed BLOB NOT NULL,  -- Gzipped JSON
    compressed_size INTEGER NOT NULL,

    -- Cache metadata
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0
);

-- Indexes for efficient lookups
CREATE INDEX idx_pdf_name ON detection_cache(pdf_name);
CREATE INDEX idx_last_accessed ON detection_cache(last_accessed);
```

### When Cache is Invalidated

The cache automatically detects changes and invalidates entries when:

1. **PDF modified**: File modification time changes
2. **PDF size changes**: File size different
3. **Parameters change**: Different model, confidence, iou, or dpi

**Example:**

```python
detector = DiagramDetector(model="v5", confidence=0.20)

# First run: caches results
results = detector.detect_pdf("paper.pdf")

# Edit the PDF file
# ... modify paper.pdf ...

# Next run: detects change, reprocesses
results = detector.detect_pdf("paper.pdf")  # Cache miss due to mtime change
```

### Performance Characteristics

**Without Cache:**
- PDF extraction: ~20-50ms per page
- Model inference: ~50-200ms per page
- **Total: ~70-250ms per page**

**With Cache (hit):**
- SQLite lookup: ~1-5ms
- Gzip decompression: ~10-30ms
- **Total: ~11-35ms per page (10-20x faster!) ⚡**

**Cache Overhead (miss):**
- Gzip compression: ~5-10ms
- SQLite insert: ~2-5ms
- **Total overhead: ~7-15ms (negligible)**

**Storage Efficiency:**

Typical compression ratios with gzip:
- Detection results (JSON): ~10-15x compression
- Large PDFs (100+ pages): ~12-18x compression
- Average: **~70-90% space savings**

Example:
- 1,000 PDFs, 15,000 pages
- Uncompressed: ~1.2 GB
- Compressed cache: **~120 MB**

### Thread Safety

The cache is **fully thread-safe** for concurrent access:

```python
from concurrent.futures import ThreadPoolExecutor
from diagram_detector import DiagramDetector

# Shared cache across threads
detector = DiagramDetector()

pdf_files = list(Path("pdfs").glob("*.pdf"))

# Process in parallel - cache handles concurrent access safely
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(detector.detect_pdf, pdf_files))
```

**Implementation:**
- Thread-local SQLite connections
- WAL (Write-Ahead Logging) mode enabled
- Automatic connection pooling
- No manual locking required

### Remote Detection Caching

The cache also works with remote detection:

```python
from diagram_detector import PDFRemoteDetector

# Remote detector with caching
remote = PDFRemoteDetector(
    config=None,  # Uses default thinkcentre.local
    model="yolo11m",
    confidence=0.35
)

# First run: extracts locally, runs inference remotely, caches results
results = remote.detect_pdfs("papers/", use_cache=True)

# Second run: loads from cache (no remote call!)
results = remote.detect_pdfs("papers/", use_cache=True)  # ⚡ Instant
```

### Best Practices

1. **Enable caching by default**
   ```python
   # Good (default)
   detector = DiagramDetector()

   # Avoid disabling unless testing
   detector = DiagramDetector(cache=False)
   ```

2. **Use shared cache for related projects**
   ```python
   # All scripts in project share cache
   CACHE_DIR = Path(__file__).parent / "cache"
   cache = DetectionCache(cache_dir=CACHE_DIR)
   detector = DiagramDetector(cache=cache)
   ```

3. **Set size limits for long-running processes**
   ```python
   # Prevent unbounded growth
   cache = DetectionCache(
       auto_cleanup=True,
       max_size_mb=2000  # 2GB limit
   )
   ```

4. **Clear cache when changing model weights**
   ```python
   # After retraining model
   detector.cache.clear()
   ```

5. **Monitor cache statistics**
   ```python
   # Periodically check cache health
   stats = detector.cache.stats()
   if stats['size_mb'] > 5000:  # >5GB
       print("⚠️  Large cache, consider cleanup")
   ```

6. **Use appropriate cache location**
   - **Development**: Project-local cache for isolation
   - **Production**: System cache for sharing across pipelines
   - **Cluster/HPC**: Shared network cache (future: MySQL backend)

### Debugging Cache Issues

**Check if cache is being used:**

```python
detector = DiagramDetector(verbose=True)

# First run:
results = detector.detect_pdf("paper.pdf")
# Output: "Processing 15 pages..." (no cache message)

# Second run:
results = detector.detect_pdf("paper.pdf")
# Output: "✓ Loaded 15 pages from cache" (cache hit!)
```

**Force cache bypass:**

```python
# Temporarily disable cache
results = detector.detect_pdf("paper.pdf", use_cache=False)
```

**Inspect cache for specific PDF:**

```python
from pathlib import Path

pdf_path = Path("paper.pdf")
cached_results = detector.cache.get(
    pdf_path,
    model=detector.model_name,
    confidence=detector.confidence,
    iou=detector.iou,
    dpi=200
)

if cached_results:
    print(f"✓ Found in cache: {len(cached_results)} pages")
else:
    print("✗ Not in cache")
```

**Common Issues:**

1. **"Cache always misses"**
   - Check parameters match exactly
   - Verify PDF hasn't been modified
   - Check if `use_cache=True` (default)

2. **"Cache using too much space"**
   - Enable auto-cleanup
   - Set `max_size_mb` limit
   - Run periodic `cache.clear()`

3. **"Different results with cache"**
   - This shouldn't happen! Cache key includes all parameters
   - Report as bug if reproducible

## Error Messages

### Missing `detector` section

```
ValueError: Config file missing 'detector' section.
Expected format:
detector:
  model: v5
  confidence: 0.20
  ...
```

**Fix:** Add `detector:` section to config file

### Missing `paths` section (for `run_from_config()`)

```
ValueError: Config file missing 'paths' section
```

**Fix:** Add `paths:` section with at least `input:` field

### Config file not found

```
FileNotFoundError: Config file not found: config.yaml
```

**Fix:** Check file path and ensure file exists

### Cannot save crops without stored images

```
ValueError: Cannot save crops: 3 result(s) have diagrams but images were not stored.
Fix: Pass store_images=True when calling detect() or detect_pdf()
```

**Fix:** This shouldn't happen with `run_from_config()`, but if using `from_config()` + manual workflow, ensure `store_images=True` when calling `detect()`

### Output directory not writable

```
PermissionError: Cannot create or write to output directory: /protected/path
Error: [Errno 13] Permission denied
Fix: Check directory permissions or choose a different output path.
```

**Fix:** Choose writable output directory or fix permissions

### Relative path warning

```
⚠️  Using relative output path: results (CWD: /current/directory)
```

**Fix:** Use absolute paths in production to avoid confusion:

```yaml
paths:
  output: /absolute/path/results
```

## Best Practices

1. **Use absolute paths in production**
   - Avoids confusion about current working directory
   - Makes configs portable across environments

2. **Commit configs to version control**
   - Reproducible experiments
   - Easy to share with collaborators

3. **Use descriptive names**
   ```yaml
   # Good
   paths:
     output: /data/v5_model_arxiv_corpus_20260105

   # Avoid
   paths:
     output: results
   ```

4. **Set appropriate confidence thresholds**
   - Lower (0.15-0.20) for high recall
   - Higher (0.35-0.50) for high precision

5. **Enable crops/viz for debugging only**
   - Significantly increases memory usage
   - Slower processing
   - Use for validation, disable for production

## Integration with Pipeline Scripts

Example pipeline script:

```python
# pipeline.py
from diagram_detector import DiagramDetector
from pathlib import Path

def run_detection_pipeline(config_file: str):
    """Run detection with config file."""
    print(f"Running detection from {config_file}")

    # Run complete workflow
    results = DiagramDetector.run_from_config(config_file)

    # Post-processing
    diagrams = [r for r in results if r.has_diagram]

    print(f"✓ Processed {len(results)} pages")
    print(f"✓ Found diagrams in {len(diagrams)} pages")

    return results

if __name__ == "__main__":
    import sys
    run_detection_pipeline(sys.argv[1])
```

Usage:

```bash
python pipeline.py production_config.yaml
```

## See Also

- `README.md` - Package overview and installation
- `examples/` - Example scripts and notebooks
- API documentation - Full method documentation
