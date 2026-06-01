import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.getLogger(__name__).info("OCULA API starting...")
    yield
    logging.getLogger(__name__).info("OCULA API shutting down.")

app = FastAPI(
    title       = "OCULA Inference API",
    description = "Multilingual Hate Speech Detection",
    version     = "1.0.0",
    lifespan    = lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)

from api.routes import health, predict, explain
app.include_router(health.router,  tags=["health"])
app.include_router(predict.router, tags=["predict"])
app.include_router(explain.router, tags=["explain"])

@app.get("/")
async def root():
    return {"name": "OCULA API", "version": "1.0.0", "status": "running"}