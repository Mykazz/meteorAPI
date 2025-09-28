from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, Dict
from uuid import uuid4
import asyncio

# ---------- FastAPI app ----------
app = FastAPI(title="Impactor API (bare bones)", version="0.0.1")



from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True, "version": "0.0.1"}

# ---------- Sync stub (accepts a payload; does nothing yet) ----------
class SimulationRequest(BaseModel):
    scenario_id: Optional[str] = None
    # you can add: elements, entry_params, aoi, etc. later

class SimulationAck(BaseModel):
    ok: bool = True
    note: str = "stub"

@app.post("/v1/simulate", response_model=SimulationAck)
def simulate_stub(req: SimulationRequest):
    # TODO: plug your physics later
    return SimulationAck(ok=True, note="stub: received payload")

# ---------- Async job pattern (queue → poll) ----------
class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued","running","done","error"] = "queued"
    message: Optional[str] = None
    # later you can add: czml_url, geojson_url, raster_url, etc.

_JOBS: Dict[str, JobStatus] = {}

@app.post("/v1/jobs", status_code=202)
async def create_job(req: SimulationRequest):
    job_id = str(uuid4())
    _JOBS[job_id] = JobStatus(job_id=job_id, status="queued", message="queued")
    asyncio.create_task(_fake_worker(job_id))
    return {"job_id": job_id}

@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    js = _JOBS.get(job_id)
    if not js:
        raise HTTPException(status_code=404, detail="job not found")
    return js

async def _fake_worker(job_id: str):
    try:
        _JOBS[job_id].status = "running"
        _JOBS[job_id].message = "running"
        await asyncio.sleep(1.0)  # pretend work
        _JOBS[job_id].status = "done"
        _JOBS[job_id].message = "done (stub)"
    except Exception as e:
        _JOBS[job_id].status = "error"
        _JOBS[job_id].message = f"error: {e}"


