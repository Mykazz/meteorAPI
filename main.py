from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal, Dict, List
from uuid import uuid4
import asyncio
import os

# ---------- FastAPI app ----------
app = FastAPI(title="Asteroid Simulation API", version="0.0.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True, "version": "0.0.5"}

# ---------- Input Models ----------
class OrbitRequest(BaseModel):
    fullName: str
    epochMjd: float
    a: float      # semi-major axis [AU]
    e: float      # eccentricity
    i: float      # inclination [deg]
    node: float   # longitude of ascending node [deg]
    peri: float   # argument of perihelion [deg]
    M: float      # mean anomaly [deg]
    success: bool = True

# ---------- Three.js-compatible response ----------
class ThreeJSObject(BaseModel):
    object: str
    position: dict
    material: dict
    radius: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None

class SimulationResponse(BaseModel):
    ok: bool
    input_data: dict          # 👈 echo back request
    objects: List[ThreeJSObject]

# ---------- Sync simulation ----------
@app.post("/v1/simulate", response_model=SimulationResponse)
def simulate_orbit(req: OrbitRequest):
    """
    For now this just builds a sphere representing the asteroid at epoch.
    """
   

    return SimulationResponse(ok=True, input_data=req.dict(), objects=[])

# ---------- Async job pattern ----------
class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued","running","done","error"] = "queued"
    message: Optional[str] = None
    input_data: Optional[dict] = None
    result: Optional[SimulationResponse] = None

_JOBS: Dict[str, JobStatus] = {}

async def skaiciavimai(job_id: str, req: OrbitRequest):
    try:
        _JOBS[job_id].status = "running"
        await asyncio.sleep(2)  # simulate computation
        res = simulate_orbit(req)
        _JOBS[job_id].status = "done"
        _JOBS[job_id].result = res
    except Exception as e:
        _JOBS[job_id].status = "error"
        _JOBS[job_id].message = str(e)

@app.post("/v1/jobs", status_code=202)
async def create_job(req: OrbitRequest):
    job_id = str(uuid4())
    _JOBS[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        message="queued",
        input_data=req.dict()
    )
    asyncio.create_task(skaiciavimai(job_id, req))
    return {"job_id": job_id}

@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    js = _JOBS.get(job_id)
    if not js:
        raise HTTPException(status_code=404, detail="job not found")
    return js


# ---------- Run with Uvicorn ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
