from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal, Dict, List
from uuid import uuid4
import asyncio
import os

# ---------- FastAPI app ----------
app = FastAPI(title="Impactor API", version="0.0.3")

# Allow CORS for all origins (loosen later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: restrict later in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True, "version": "0.0.3"}

# ---------- Input Models ----------
class SimulationRequest(BaseModel):
    scenario_id: Optional[str] = None
    velocity: float   # m/s
    angle: float      # degrees
    density: float    # kg/m^3
    diameter: float   # m
    altitude: float   # m
    target_material: str = "rock"

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
    objects: List[ThreeJSObject]

# ---------- Sync simulation ----------
@app.post("/v1/simulate", response_model=SimulationResponse)
def simulate_stub(req: SimulationRequest):
    # Example: sphere for impactor
    sphere = ThreeJSObject(
        object="sphere",
        radius=req.diameter / 2,
        position={"x": 0, "y": req.altitude, "z": 0},
        material={"color": "#ff0000"}
    )

    # Example: ground plane
    ground = ThreeJSObject(
        object="plane",
        width=500,
        height=500,
        position={"x": 0, "y": 0, "z": 0},
        material={"color": "#00ff00"}
    )

    return SimulationResponse(ok=True, objects=[sphere, ground])

# ---------- Async job pattern ----------
class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued","running","done","error"] = "queued"
    message: Optional[str] = None
    input_data: Optional[dict] = None
    result: Optional[SimulationResponse] = None  # store JSON result

_JOBS: Dict[str, JobStatus] = {}

@app.post("/v1/jobs", status_code=202)
async def create_job(req: SimulationRequest):
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
