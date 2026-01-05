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
