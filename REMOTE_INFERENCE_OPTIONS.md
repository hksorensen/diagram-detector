# Remote Inference Options - Analysis & Recommendations

**Date:** December 24, 2024
**Status:** To Be Decided

---

## 🎯 THE QUESTION

**Should we build remote inference capability?**

Currently:
- ✅ Training runs on remote GPU (henrikkragh.dk:8022) via SSH/rsync
- ❌ Inference runs locally on Mac (CPU, slow for large batches)

**Proposed:**
- Run inference on remote GPU server
- Keep Mac as client (submit jobs, get results)

---

## 🔍 USE CASE ANALYSIS

### **When Remote Inference Makes Sense**

**1. Large-Scale Processing**
```python
# Processing 10,000+ pages
# Local Mac: 2-3 hours
# Remote GPU: 10-20 minutes
```

**2. Batch Jobs**
- Process entire paper corpus overnight
- Submit job, get results later
- Mac can sleep/disconnect

**3. Consistent Performance**
- Always fast GPU inference
- No battery drain on Mac
- No thermal throttling

### **When Local Inference is Better**

**1. Small Jobs**
- Single paper (10-50 pages)
- Local is "good enough"
- No network latency

**2. Interactive Work**
- Quick tests
- Immediate feedback
- No queue waiting

**3. Offline/Travel**
- No internet needed
- Private documents
- Portable workflow

---

## 🏗️ ARCHITECTURE OPTIONS

### **Option 1: FastAPI Web Service** ⭐ RECOMMENDED

**Architecture:**
```
Mac (Client)                 Remote Server (henrikkragh.dk:8022)
┌─────────────┐             ┌──────────────────────────┐
│             │   HTTPS     │  FastAPI Server          │
│ Python API  │◄───────────►│  ┌────────────────────┐  │
│ or CLI      │             │  │ Upload Endpoint    │  │
│             │             │  │ Process Queue      │  │
└─────────────┘             │  │ GPU Inference      │  │
                            │  │ Download Results   │  │
                            │  └────────────────────┘  │
                            └──────────────────────────┘
```

**Pros:**
- ✅ Modern, clean API design
- ✅ Easy to use from any language (Python, curl, browser)
- ✅ Built-in documentation (Swagger/ReDoc)
- ✅ Async support (handle multiple requests)
- ✅ Progress monitoring via WebSocket
- ✅ Can add web UI later
- ✅ Industry standard
- ✅ **You want to learn FastAPI!** 🎓

**Cons:**
- Requires server setup (nginx, SSL, authentication)
- Need to expose port (firewall config)
- Need authentication mechanism
- More complex than SSH

**Implementation Complexity:** Medium

---

### **Option 2: SSH + File Transfer** (Current Approach)

**Architecture:**
```
Mac (Client)                 Remote Server
┌─────────────┐             ┌──────────────────────────┐
│             │   rsync     │                          │
│ Upload PDF  │────────────►│  File → Inference        │
│ via rsync   │             │  Result → File           │
│ Download    │◄────────────│                          │
│ results     │             │                          │
└─────────────┘             └──────────────────────────┘
```

**Pros:**
- ✅ Already working for training
- ✅ Simple, no server setup
- ✅ Uses existing SSH access
- ✅ Reliable file transfer
- ✅ No new dependencies

**Cons:**
- Manual workflow (upload, run, download)
- No queuing system
- No progress monitoring
- Hard to scale

**Implementation Complexity:** Low (extend existing code)

---

### **Option 3: Message Queue (Celery + Redis)**

**Architecture:**
```
Mac (Client)                 Message Broker           Workers
┌─────────────┐             ┌─────────────┐         ┌──────────┐
│             │   Submit    │             │  Fetch  │ GPU      │
│ Python API  │────────────►│   Redis     │────────►│ Worker 1 │
│             │   Get       │   Queue     │         │ Worker 2 │
│             │◄────────────│             │         │ ...      │
└─────────────┘             └─────────────┘         └──────────┘
```

**Pros:**
- ✅ True distributed system
- ✅ Multiple workers
- ✅ Automatic retry on failure
- ✅ Task scheduling
- ✅ Built-in monitoring (Flower)

**Cons:**
- Complex setup (Redis, Celery, workers)
- Overkill for single-user system
- More moving parts to maintain

**Implementation Complexity:** High

---

### **Option 4: Hybrid Approach** ⭐ PRAGMATIC

**Architecture:**
```
Mac                          Remote Server
┌─────────────┐             ┌──────────────────────────┐
│             │   Option 1  │                          │
│ diagram-    │   SSH+rsync │  CLI Tool                │
│ detect-     │◄───────────►│  (batch mode)            │
│ remote CLI  │   Option 2  │                          │
│             │   FastAPI   │  FastAPI Server          │
│             │◄───────────►│  (web API)               │
└─────────────┘             └──────────────────────────┘
```

**Two modes:**
1. **SSH mode** (default, simple)
2. **API mode** (optional, when server running)

**Pros:**
- ✅ Best of both worlds
- ✅ Start simple, upgrade later
- ✅ Flexible for different use cases

**Cons:**
- Two code paths to maintain

**Implementation Complexity:** Medium-High

---

## 💡 RECOMMENDATIONS

### **Recommended Approach: FastAPI with Phased Rollout**

**Phase 1: SSH Extension (Quick Win)** ⏱️ 1-2 days
```bash
# Extend current SSH approach for inference
diagram-detect-remote --input paper.pdf --model yolo11m --remote henrikkragh.dk:8022
```

**What it does:**
1. Upload PDF via rsync
2. SSH execute: `diagram-detect --input uploaded.pdf`
3. Download results via rsync
4. Clean up remote files

**Code:**
- Extend existing `pipeline.py` SSH patterns
- Simple wrapper around `diagram-detect`
- No server changes needed

**Phase 2: FastAPI Server (Learning & Production)** ⏱️ 1 week
```python
# Python API
from diagram_detector.remote import RemoteDetector

remote = RemoteDetector(server='https://henrikkragh.dk:8443', api_key='...')
results = remote.detect_pdf('paper.pdf')
```

**What it does:**
- FastAPI server on remote
- Upload via HTTP multipart
- Queue system (background tasks)
- WebSocket for progress
- Download results

**Phase 3: Web UI (Optional)** ⏱️ 3-5 days
- Drag-drop PDF upload
- Real-time progress
- Download results
- View visualizations
- Browse history

---

## 🎓 WHY FASTAPI?

**1. Modern & Fast**
- Built on Starlette (async)
- Pydantic validation (type safety)
- Automatic OpenAPI docs

**2. Easy to Learn**
```python
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/detect/")
async def detect_pdf(file: UploadFile):
    # Save file
    # Run inference
    # Return results
    return {"status": "processing", "job_id": "123"}

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    # Return results
    return FileResponse(f"results/{job_id}.json")
```

**3. Production Ready**
- Used by Netflix, Microsoft, Uber
- Great documentation
- Active community

**4. Extensible**
- Add authentication (OAuth, API keys)
- Add rate limiting
- Add caching
- Add monitoring

---

## 🚀 PROPOSED IMPLEMENTATION

### **Phase 1: SSH Remote Inference** (START HERE)

**New module:** `diagram_detector/remote_ssh.py`

```python
class SSHRemoteDetector:
    """Run inference on remote GPU via SSH."""
    
    def __init__(self, host: str, port: int, user: str):
        self.host = host
        self.port = port
        self.user = user
    
    def detect_pdf(self, pdf_path: Path) -> List[DetectionResult]:
        """
        1. Upload PDF via rsync
        2. SSH: diagram-detect --input uploaded.pdf
        3. Download results
        4. Parse and return
        """
        pass
```

**CLI:**
```bash
diagram-detect --input paper.pdf --remote ssh://henrikkragh.dk:8022
```

**Pros:**
- Quick to implement (2 days)
- Uses existing infrastructure
- No server setup
- Good for batch jobs

**Cons:**
- Still manual workflow
- No real-time progress
- No queuing

---

### **Phase 2: FastAPI Server** (NEXT)

**New module:** `diagram_detector_server/` (separate package)

```python
# server/main.py
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import uuid

app = FastAPI(title="Diagram Detector API")

# In-memory job storage (upgrade to Redis later)
jobs = {}

@app.post("/api/v1/detect/upload")
async def upload_pdf(
    file: UploadFile,
    model: str = "yolo11m",
    confidence: float = 0.35
):
    """Upload PDF and start processing."""
    job_id = str(uuid.uuid4())
    
    # Save file
    file_path = f"/tmp/{job_id}.pdf"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Queue job (background task)
    jobs[job_id] = {"status": "queued", "model": model}
    
    # Start processing in background
    background_tasks.add_task(process_job, job_id, file_path, model, confidence)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/v1/detect/status/{job_id}")
async def get_status(job_id: str):
    """Check job status."""
    if job_id not in jobs:
        return {"error": "Job not found"}
    return jobs[job_id]

@app.get("/api/v1/detect/results/{job_id}")
async def get_results(job_id: str):
    """Download results."""
    if job_id not in jobs:
        return {"error": "Job not found"}
    
    if jobs[job_id]["status"] != "completed":
        return {"error": "Job not completed"}
    
    return FileResponse(f"/tmp/{job_id}_results.json")

async def process_job(job_id: str, file_path: str, model: str, confidence: float):
    """Background task to process PDF."""
    jobs[job_id]["status"] = "processing"
    
    try:
        # Run inference
        detector = DiagramDetector(model=model, confidence=confidence)
        results = detector.detect_pdf(file_path)
        
        # Save results
        detector.save_results(results, f"/tmp/{job_id}_results.json")
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["results_url"] = f"/api/v1/detect/results/{job_id}"
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
```

**Client:**
```python
# diagram_detector/remote_api.py
import requests

class RemoteDetector:
    """Client for FastAPI remote inference."""
    
    def __init__(self, server_url: str, api_key: str = None):
        self.server_url = server_url
        self.api_key = api_key
    
    def detect_pdf(self, pdf_path: Path) -> List[DetectionResult]:
        """Submit PDF and wait for results."""
        
        # 1. Upload
        with open(pdf_path, 'rb') as f:
            response = requests.post(
                f"{self.server_url}/api/v1/detect/upload",
                files={"file": f},
                params={"model": "yolo11m"}
            )
        
        job_id = response.json()["job_id"]
        
        # 2. Poll for completion
        while True:
            status = requests.get(
                f"{self.server_url}/api/v1/detect/status/{job_id}"
            ).json()
            
            if status["status"] == "completed":
                break
            elif status["status"] == "failed":
                raise RuntimeError(status["error"])
            
            time.sleep(2)  # Poll every 2 seconds
        
        # 3. Download results
        results = requests.get(
            f"{self.server_url}/api/v1/detect/results/{job_id}"
        ).json()
        
        return parse_results(results)
```

**Usage:**
```python
from diagram_detector.remote_api import RemoteDetector

# Connect to server
remote = RemoteDetector('https://henrikkragh.dk:8443')

# Run inference
results = remote.detect_pdf('paper.pdf')

# Results are same format as local!
for result in results:
    if result.has_diagram:
        print(f"Page {result.page_number}: {result.count} diagrams")
```

---

## 🔐 SECURITY CONSIDERATIONS

**Authentication:**
- API keys (simple)
- OAuth2 (production)
- IP whitelist (your Mac only)

**Encryption:**
- HTTPS/TLS (required)
- Let's Encrypt (free SSL)

**Rate Limiting:**
- Prevent abuse
- FastAPI-Limiter

**File Validation:**
- Check file types
- Size limits
- Virus scanning (optional)

---

## 📊 PERFORMANCE COMPARISON

**Test case:** 1,000 page PDF

| Approach | Time | Pros | Cons |
|----------|------|------|------|
| **Local Mac (CPU)** | 2-3 hours | Simple, no setup | Slow, blocks Mac |
| **SSH Remote** | 15-20 min | Fast GPU, simple | Manual workflow |
| **FastAPI Remote** | 15-20 min | Fast GPU, queuing | Setup required |

**For 10,000 pages:**

| Approach | Time |
|----------|------|
| Local Mac | 20-30 hours |
| SSH Remote | 2-3 hours (manual batching) |
| FastAPI Remote | 2-3 hours (automatic queuing) |

---

## 💰 IMPLEMENTATION EFFORT

### **Phase 1: SSH Extension** ⏱️ 1-2 days

**Files to create:**
- `diagram_detector/remote_ssh.py` (200 lines)
- Update CLI to support `--remote` flag
- Tests

**Total:** ~300 lines of code

---

### **Phase 2: FastAPI Server** ⏱️ 1 week

**Files to create:**
- `diagram_detector_server/main.py` (API endpoints)
- `diagram_detector_server/jobs.py` (job management)
- `diagram_detector/remote_api.py` (client)
- Deployment scripts (systemd, nginx)
- Tests

**Total:** ~800 lines of code + server setup

---

### **Phase 3: Web UI** ⏱️ 3-5 days

**Files to create:**
- React frontend (or Vue, Svelte)
- Drag-drop upload
- Progress indicators
- Results viewer

**Total:** ~1000 lines of frontend code

---

## ✅ TODO LIST

### **Immediate Decision Needed**

- [ ] **Decide: Build remote inference?**
  - If yes → Start with Phase 1 (SSH)
  - If no → Stay with local inference

### **If YES to Remote Inference:**

**Phase 1: SSH Remote (Quick Win)**
- [ ] Create `diagram_detector/remote_ssh.py`
- [ ] Add `--remote` flag to CLI
- [ ] Test with henrikkragh.dk:8022
- [ ] Document usage
- [ ] **Timeline:** 1-2 days

**Phase 2: FastAPI Server (Learning Project)**
- [ ] Learn FastAPI basics (tutorials)
- [ ] Create `diagram_detector_server` package
- [ ] Implement upload/process/download endpoints
- [ ] Add background task processing
- [ ] Create Python client (`remote_api.py`)
- [ ] Setup server (nginx, systemd, SSL)
- [ ] Add authentication
- [ ] Test end-to-end
- [ ] Document deployment
- [ ] **Timeline:** 1 week

**Phase 3: Web UI (Optional Enhancement)**
- [ ] Choose framework (React/Vue/Svelte)
- [ ] Build upload interface
- [ ] Add progress indicators
- [ ] Create results viewer
- [ ] Deploy frontend
- [ ] **Timeline:** 3-5 days

---

## 🎯 MY RECOMMENDATION

**Start with Phase 1 (SSH Remote), then evaluate.**

**Why:**
1. **Quick win** - 1-2 days to working remote inference
2. **Uses existing infrastructure** - No server setup
3. **Solves immediate need** - Fast GPU inference for large batches
4. **Low risk** - Easy to implement and test

**Then decide on FastAPI:**
- If you like SSH approach → Done!
- If you want to learn FastAPI → Phase 2
- If you want web UI → Phase 3

**Best of both worlds:**
- Keep SSH for simple batch jobs
- Add FastAPI for interactive/web use
- Both use same inference code

---

## 🤔 QUESTIONS TO CONSIDER

**1. How often will you do large-scale inference?**
- Daily → FastAPI worth it
- Weekly → SSH is fine
- Rarely → Local is OK

**2. Will others use the system?**
- Yes → FastAPI + Web UI
- No → SSH is simpler

**3. Do you want to learn FastAPI?**
- Yes → Great learning project! 🎓
- No → Stick with SSH

**4. Security requirements?**
- Public access → Need auth, SSL, rate limiting
- Private (you only) → Simple API key

**5. Time available?**
- 2 days → SSH only
- 1 week → SSH + FastAPI
- 2 weeks → SSH + FastAPI + Web UI

---

## 📝 PROPOSED NEXT STEPS

**Option A: Start Small**
1. Implement Phase 1 (SSH remote)
2. Use for actual work
3. Evaluate if Phase 2 needed

**Option B: Go Big** (if excited about FastAPI)
1. Skip Phase 1 OR do it quickly
2. Jump to Phase 2 (FastAPI)
3. Learn FastAPI properly
4. Build production system

**Option C: Hold Off**
1. Use local inference for now
2. Revisit when need arises
3. Focus on other priorities

---

## 🎓 LEARNING RESOURCES (If FastAPI)

**Official Tutorial:**
- https://fastapi.tiangolo.com/tutorial/

**Great for beginners:**
- https://testdriven.io/blog/fastapi-crud/

**Background tasks:**
- https://fastapi.tiangolo.com/tutorial/background-tasks/

**File uploads:**
- https://fastapi.tiangolo.com/tutorial/request-files/

**Deployment:**
- https://fastapi.tiangolo.com/deployment/

---

## 💭 FINAL THOUGHTS

**My vote:** Start with Phase 1 (SSH), keep Phase 2 (FastAPI) as future enhancement.

**Reasoning:**
1. SSH gives you 80% benefit for 20% effort
2. FastAPI is valuable learning but not urgent
3. Can always add API layer later
4. Your immediate need is fast inference, not web UI

**But if you're excited about FastAPI:**
- Go for it! Great learning project
- Real-world use case
- You'll use it again for other projects
- Modern skill to have

---

**Decision point:** Do you want remote inference? If yes, which phase to start with?

**Add to TODO:**
```markdown
## Remote Inference (To Be Decided)

### Question
Should we build remote GPU inference capability?

### Options Analyzed
1. SSH Extension (Phase 1) - 1-2 days ⭐ QUICK WIN
2. FastAPI Server (Phase 2) - 1 week 🎓 LEARNING
3. Web UI (Phase 3) - 3-5 days 🚀 ENHANCEMENT

### Recommendation
Start with Phase 1 (SSH), evaluate Phase 2 later

### Files to Create (if proceeding)
- diagram_detector/remote_ssh.py
- diagram_detector/remote_api.py (Phase 2)
- diagram_detector_server/ (Phase 2)

### References
- This document: REMOTE_INFERENCE_OPTIONS.md
- FastAPI docs: https://fastapi.tiangolo.com/
```

---

**Status:** Awaiting decision
**Priority:** Medium (nice to have, not urgent)
**Complexity:** Low (Phase 1) to Medium (Phase 2)
**Learning Value:** High (if doing FastAPI)

---

**End of analysis - December 24, 2024**
