# diagram-detector - TODO & Future Enhancements

**Version:** 1.0.0
**Last Updated:** December 24, 2024

---

## 🎯 CURRENT STATUS

✅ **Complete:**
- Core detection (detector, models, utils, CLI)
- Multi-source download (HF Hub + GitHub)
- Local caching
- Model upload tools
- PyPI-ready package
- GitHub CI/CD
- Docker support
- Comprehensive documentation

⏳ **In Progress:**
- Testing on real data
- Model uploads to HF/GitHub
- PyPI publishing

---

## 🔥 HIGH PRIORITY

### **1. Remote GPU Inference** ✅ COMPLETE

**Status:** Implemented (SSH with intelligent batching)

**Features:**
- ✅ SSH-based remote execution
- ✅ Intelligent batching (1000 images/batch)
- ✅ Progress tracking
- ✅ Auto-resume on failure
- ✅ Auto-cleanup
- ✅ Optimized for 300K+ images

**Usage:**
```bash
diagram-detect --input images/ --remote user@host:port --output results/
```

**Performance:**
- 300,000 images: ~3-4 days (vs 8 days on Mac CPU!)
- Batch size: 1000 images (customizable)
- GPU batch size: 32 (customizable)

**Documentation:**
- Guide: `REMOTE_INFERENCE_GUIDE.md`
- Module: `diagram_detector/remote_ssh.py`

**Future Enhancement (Optional):**
- **Phase 2:** FastAPI Server (1 week) 🎓 LEARNING PROJECT
- **Phase 3:** Web UI (3-5 days) 🚀 ENHANCEMENT
- Only needed if you want web interface or multi-user support

---

### **2. Upload Models to Distribution Channels**

**Status:** Tools ready, needs execution

**Tasks:**
- [ ] Create Hugging Face repository
  - Name: `hksorensen/diagram-detector-models`
  - Type: Model
  - Visibility: Public
  
- [ ] Get HF token
  - Visit: https://huggingface.co/settings/tokens
  - Create write token
  - `export HF_TOKEN=hf_...`

- [ ] Upload models
  ```bash
  python upload_models.py --directory ~/Downloads/diagram-detector-models/
  ```

- [ ] Verify downloads work
  ```bash
  rm -rf ~/.cache/diagram-detector/
  python -c "from diagram_detector import DiagramDetector; DiagramDetector(model='yolo11n')"
  ```

- [ ] Update README with actual download URLs

**Priority:** High (needed for PyPI release)
**Time:** 1 hour

---

### **3. Publish to PyPI**

**Status:** Package ready, needs testing & upload

**Tasks:**
- [ ] Test on TestPyPI first
  ```bash
  python -m build
  twine upload --repository testpypi dist/*
  pip install --index-url https://test.pypi.org/simple/ diagram-detector
  ```

- [ ] Test installation
  ```bash
  diagram-detect --help
  python -c "from diagram_detector import DiagramDetector; print('OK')"
  ```

- [ ] Upload to PyPI
  ```bash
  twine upload dist/*
  ```

- [ ] Create GitHub release
  ```bash
  git tag v1.0.0
  git push origin v1.0.0
  gh release create v1.0.0 --notes "$(cat CHANGELOG.md)"
  ```

**Priority:** High (main deliverable)
**Time:** 2-3 hours

---

## 📊 MEDIUM PRIORITY

### **4. Additional Tests**

**Status:** Basic tests exist, need more coverage

**Needed:**
- [ ] Integration tests (full pipeline)
- [ ] PDF processing tests
- [ ] Remote download tests
- [ ] Error handling tests
- [ ] Performance benchmarks

**Files:**
- `tests/test_integration.py`
- `tests/test_pdf.py`
- `tests/test_download.py`

**Priority:** Medium (important for CI/CD)
**Time:** 2-3 days

---

### **5. Documentation Improvements**

**Status:** Good docs exist, could be better

**Needed:**
- [ ] Read the Docs setup
- [ ] API reference (Sphinx)
- [ ] More usage examples
- [ ] Tutorial notebooks
- [ ] Video walkthrough

**Priority:** Medium (helps adoption)
**Time:** 3-5 days

---

### **6. Performance Optimizations**

**Ideas:**
- [ ] ONNX quantization (INT8) for faster CPU inference
- [ ] Batch size auto-tuning based on GPU memory
- [ ] Multi-GPU support
- [ ] Streaming inference for huge PDFs
- [ ] Result caching (avoid re-processing same files)

**Priority:** Medium (nice to have)
**Time:** Varies (1-5 days per item)

---

## 🎨 LOW PRIORITY / FUTURE ENHANCEMENTS

### **7. Advanced Features**

**Diagram Classification:**
- [ ] Classify diagram types (flowchart, plot, schematic, etc.)
- [ ] Use CLIP embeddings
- [ ] Zero-shot classification

**Diagram Extraction:**
- [ ] Better cropping (tight bounds)
- [ ] Remove background
- [ ] Enhance quality

**OCR Integration:**
- [ ] Extract text from diagrams
- [ ] Caption detection
- [ ] Label extraction

**Priority:** Low (v2.0.0 features)
**Time:** 1-2 weeks per feature

---

### **8. GUI Application**

**Status:** CLI works, but GUI would be nice

**Options:**
- Electron app (cross-platform)
- Qt/PyQt (native)
- Web UI (if FastAPI server exists)

**Features:**
- Drag-drop PDF upload
- Real-time progress
- View results inline
- Batch processing
- Export options

**Priority:** Low (nice to have)
**Time:** 1-2 weeks

---

### **9. Cloud Deployment**

**Options:**
- Docker container on cloud (AWS, GCP, Azure)
- Hugging Face Spaces (free tier!)
- Gradio demo
- Replicate.com model hosting

**Priority:** Low (after FastAPI server)
**Time:** 2-3 days

---

### **10. Mobile Support**

**Options:**
- iOS app (CoreML)
- Android app (TensorFlow Lite)
- Progressive Web App

**Priority:** Low (niche use case)
**Time:** 2-4 weeks

---

## 🐛 KNOWN ISSUES

**None currently!** 🎉

---

## 📝 DOCUMENTATION TODO

**Needed:**
- [ ] Contributing guide (CONTRIBUTING.md)
- [ ] Code of conduct (CODE_OF_CONDUCT.md)
- [ ] Issue templates (bug, feature)
- [ ] Pull request template
- [ ] Development setup guide
- [ ] Architecture overview
- [ ] Performance benchmarks

**Priority:** Medium
**Time:** 1 day

---

## 🔧 TECHNICAL DEBT

**None currently!** Clean slate ✨

---

## 🎓 LEARNING PROJECTS

**Want to learn while building?**

### **FastAPI** (if remote inference)
- Build REST API
- Background tasks
- WebSocket progress
- Authentication
- Deployment

### **React/Vue** (if web UI)
- Modern frontend
- Real-time updates
- File upload
- Results visualization

### **Docker & Kubernetes**
- Containerization
- Orchestration
- Scaling

### **MLOps**
- Model versioning
- A/B testing
- Monitoring
- Metrics

---

## 🎯 ROADMAP

### **v1.0.0** (Current) ✅
- Core detection
- PDF support
- CLI & Python API
- Model distribution
- PyPI package

### **v1.1.0** (Next Minor)
- Remote inference (if decided yes)
- More tests
- Performance improvements
- Better documentation

### **v1.2.0**
- Advanced features (classification, OCR)
- Better error handling
- More output formats

### **v2.0.0** (Future Major)
- FastAPI server (if not in v1.1.0)
- Web UI
- Multi-user support
- Cloud deployment

---

## 💬 QUESTIONS FOR USER

**Remote Inference:**
- [ ] Do you want remote GPU inference?
- [ ] If yes, which phase? (SSH / FastAPI / Both)
- [ ] Single-user or multi-user?
- [ ] Need web UI?

**Priorities:**
- [ ] What's most important right now?
- [ ] What can wait for v1.1.0?
- [ ] Any other features needed?

**Use Cases:**
- [ ] How will you use the package?
- [ ] How often large-scale inference?
- [ ] Any specific paper corpus to process?

---

## 📋 IMMEDIATE NEXT STEPS

**To get v1.0.0 released:**

1. **Upload models** (1 hour)
   ```bash
   python upload_models.py --directory ~/Downloads/diagram-detector-models/
   ```

2. **Test locally** (30 min)
   ```bash
   pip install -e ".[dev]"
   pytest
   diagram-detect --input test.jpg --model yolo11n
   ```

3. **Publish to PyPI** (1 hour)
   ```bash
   python -m build
   twine upload --repository testpypi dist/*  # Test first
   twine upload dist/*  # Then real PyPI
   ```

4. **Create GitHub release** (30 min)
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   gh release create v1.0.0
   ```

5. **Announce!** (30 min)
   - Tweet/share
   - Post on Reddit (r/MachineLearning, r/Python)
   - Hacker News
   - Your institution

**Total time:** ~3.5 hours to published package! 🚀

---

## 🎉 SUCCESS METRICS

**v1.0.0 Release:**
- [ ] Published on PyPI
- [ ] Installable: `pip install diagram-detector`
- [ ] CLI works: `diagram-detect --help`
- [ ] Models download automatically
- [ ] Tests passing on CI/CD
- [ ] Documentation complete
- [ ] GitHub release created

**Community:**
- [ ] First external user
- [ ] First GitHub star
- [ ] First issue/PR
- [ ] First citation

---

## 🔗 REFERENCES

**Analysis Documents:**
- `REMOTE_INFERENCE_OPTIONS.md` - Remote inference analysis
- `MODEL_DISTRIBUTION.md` - Distribution guide
- `GITHUB_SETUP.md` - GitHub & PyPI setup
- `IMPLEMENTATION_GUIDE.md` - Development guide

**External:**
- FastAPI: https://fastapi.tiangolo.com/
- Hugging Face: https://huggingface.co/docs
- PyPI: https://packaging.python.org/

---

## 💡 IDEAS INBOX

*Quick ideas to evaluate later:*

- Diagram similarity search
- Automatic caption extraction
- LaTeX figure generation
- Integration with Zotero/Mendeley
- Jupyter notebook widgets
- VS Code extension
- Browser extension (detect diagrams in web papers)
- Diagram dataset creation tool

---

**Status:** Ready for v1.0.0 release
**Blockers:** None
**Next decision:** Remote inference (yes/no, which phase)

---

**End of TODO - December 24, 2024**
