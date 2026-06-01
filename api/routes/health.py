import time
import torch
from fastapi import APIRouter
from api.schemas import HealthResponse

router = APIRouter()

@router.get("/health",response_model=HealthResponse)
async def health():
    t0 = time.perf_counter()
    from api.routes.predict import model_state

    gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu else None
    loaded = model_state["model"] is not None
    latency_ms = round((time.perf_counter() - t0)*1000,2)

    return HealthResponse(
        status= "ok" if loaded else "model_not_loaded",
        model_loaded=loaded,
        model_name = model_state.get("model_name", "google/muril-base-cased"),
        gpu_available=gpu,
        gpu_name=gpu_name,
        latency_ms=latency_ms
    )