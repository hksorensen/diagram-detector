# Remote Detection Parameter Transparency

## Summary

Remote detection is now **stable and transparent**. All detection parameters (model, confidence, IoU, imgsz) are correctly passed through the entire pipeline and executed on the remote server exactly as specified.

## What Was Fixed

### 1. **Missing IoU Parameter** (Critical)
- **Problem**: IoU threshold was not passed to remote CLI, using default 0.30 always
- **Impact**: Could not use optimal IoU from grid search (0.3 was lucky match!)
- **Fix**: Added `--iou` parameter to CLI and entire remote pipeline

### 2. **Missing imgsz Parameter**
- **Problem**: Image size parameter was stored but not passed to remote
- **Impact**: Remote used default 640 (happened to match)
- **Fix**: Added `--imgsz` parameter to CLI and remote command

### 3. **Hardcoded Cache Keys**
- **Problem**: Cache used `iou=0.30` hardcoded instead of actual parameter
- **Impact**: Cache wouldn't detect parameter changes
- **Fix**: Cache now uses `self.iou` from detector configuration

### 4. **Auto Host/Port Detection**
- **Problem**: Different hostname/port for VPN vs local network
- **Impact**: `is_remote_available()` failed from VPN
- **Fix**: Auto-tries both `henrikkragh.dk:8022` (VPN) and `thinkcentre.local:22` (local)

## Parameter Flow (Now Complete)

```
Your Code
  ↓
PDFRemoteDetector(model="v5", confidence=0.1, iou=0.3, imgsz=640)
  ↓
SSHRemoteDetector(model="v5", confidence=0.1, iou=0.3, imgsz=640)
  ↓
Remote CLI Command:
  python3 -m diagram_detector.cli \
    --model v5 \
    --confidence 0.1 \
    --iou 0.3 \
    --imgsz 640 \
    --batch-size 32
  ↓
Remote DiagramDetector(model="v5", confidence=0.1, iou=0.3, imgsz=640)
  ↓
Remote YOLO model.predict(conf=0.1, iou=0.3, imgsz=640)
```

**Every step passes all parameters correctly!**

## Usage Examples

### 1. **Check if remote is available**
```python
from diagram_detector import is_remote_available

# Simple check (works from VPN or local network)
if is_remote_available():
    print("Can use remote detection")

# With details
if is_remote_available(verbose=True):
    print("Remote is ready")
```

### 2. **Get working endpoint**
```python
from diagram_detector import get_remote_endpoint

# Get auto-detected endpoint
endpoint = get_remote_endpoint(verbose=True)
if endpoint:
    print(f"Remote: {endpoint.host}:{endpoint.port}")
    # Use endpoint for detection
```

### 3. **Remote detection with optimal parameters**
```python
from diagram_detector import PDFRemoteDetector, get_remote_endpoint

# Get working endpoint
endpoint = get_remote_endpoint()

if endpoint:
    # Use remote with optimal parameters
    detector = PDFRemoteDetector(
        config=endpoint,
        model="v5",
        confidence=0.1,  # Optimal from grid search
        iou=0.3,         # Optimal from grid search
        dpi=200,
        imgsz=640,
        batch_size=10,
        verbose=True
    )

    results = detector.detect_pdfs(pdf_list, use_cache=True)
else:
    # Fall back to local
    from diagram_detector import DiagramDetector
    detector = DiagramDetector(
        model="v5",
        confidence=0.1,
        iou=0.3,
        verbose=True
    )
    results = {p.name: detector.detect_pdf(p) for p in pdf_list}
```

### 4. **Verify parameters are used correctly**
```python
from diagram_detector import PDFRemoteDetector

detector = PDFRemoteDetector(
    model="v5",
    confidence=0.1,
    iou=0.3,
    imgsz=640
)

# Check what will be sent to remote
print(f"Model: {detector.model}")              # v5
print(f"Confidence: {detector.confidence}")    # 0.1
print(f"IoU: {detector.iou}")                  # 0.3
print(f"Imgsz: {detector.imgsz}")              # 640

# Remote server will use exactly these values!
```

## Verification

Run the parameter transparency test:
```bash
cd /Users/fvb832/Downloads/diagram-detector-package\ 2
python3 test_parameter_transparency.py
```

All tests should pass:
- ✓ Parameter initialization
- ✓ CLI command building
- ✓ Optimal parameters

## Optimal Parameters from Grid Search

Based on `binary_f1_optimization.json` (562 images, 100% page-level F1):

```python
optimal_params = {
    'model': 'v5',
    'confidence': 0.1,
    'iou': 0.3,
    'dpi': 200,
    'imgsz': 640,
}
```

These parameters are **now guaranteed** to be used on the remote server!

## Cache Behavior

The cache now correctly tracks all parameters:
- Model name
- Confidence threshold
- **IoU threshold** (now tracked!)
- DPI
- Image size

If you change any parameter, the cache will correctly reprocess PDFs.

## Network Detection

The `is_remote_available()` function now auto-detects:

**From VPN/External:**
- Tries: `henrikkragh.dk:8022` ✓
- Falls back to: `thinkcentre.local:22`

**From Local Network:**
- Tries: `henrikkragh.dk:8022`
- Falls back to: `thinkcentre.local:22` ✓

Works automatically from both locations!

## What This Means for You

1. **Reproducible Results**: Remote and local detection will produce identical results with same parameters

2. **Optimal Performance**: Can now use optimal parameters from grid search on remote GPU

3. **Transparent Behavior**: Always know exactly what parameters are being used

4. **Correct Caching**: Cache correctly tracks parameter changes

5. **Network Flexibility**: Works automatically from VPN or local network

## Files Modified

1. `diagram_detector/cli.py` - Added `--iou` and `--imgsz` arguments
2. `diagram_detector/remote_ssh.py` - Added IoU/imgsz parameters, auto host detection
3. `diagram_detector/remote_pdf.py` - Added IoU parameter, fixed cache
4. `diagram_detector/__init__.py` - Exported `is_remote_available`, `get_remote_endpoint`

## Testing

All tests pass:
- ✓ Parameter transparency test
- ✓ Batched detection test (local and remote)
- ✓ Remote availability check

Remote detection is now **production-ready**!
