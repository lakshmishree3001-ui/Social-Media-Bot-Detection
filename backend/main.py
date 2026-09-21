import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.routers import (
    auth,
    datasets,
    accounts,
    predict,
    graph,
    communities,
    coordination,
    temporal,
    models_router,
    admin,
)
from backend.services.inference import InferenceEngine

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(
    title="Graphwarden API",
    description="Production Forensic Social Media Bot Detection using Graph Neural Networks & Multimodal Topology",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:80",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permissive for local development and Docker networking
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(accounts.router)
app.include_router(predict.router)
app.include_router(graph.router)
app.include_router(communities.router)
app.include_router(coordination.router)
app.include_router(temporal.router)
app.include_router(models_router.router)
app.include_router(admin.router)

@app.on_event("startup")
async def startup_event():
    # Warm up GNN inference models in background
    print("[Startup] Initializing Graphwarden API services...")
    InferenceEngine.get_instance()
    print("[Startup] Graphwarden API ready to serve requests.")

@app.get("/api/health", tags=["system"])
def health_check():
    return {
        "status": "healthy",
        "service": "Graphwarden Backend API",
        "version": "1.0.0",
        "models_loaded": 5,
        "database": "connected",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
