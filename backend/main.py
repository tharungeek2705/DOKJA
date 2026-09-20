import os
import sys
import yaml
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from configs.logging_config import setup_logger
from backend.database.memory_store import MemoryObservationStore
from backend.database.sqlite_store import SQLiteObservationStore
from gis.camera_registry import CameraGISRegistry
from backend.services.stream_manager import StreamManager
from backend.api.routes_stream import router as stream_router
from backend.api.routes_cameras import router as camera_router
from backend.api.routes_analytics import router as analytics_router
from backend.api.routes_investigation import router as investigation_router

logger = setup_logger("main")

# Load configuration
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Core Platform Singletons
memory_store = MemoryObservationStore()
sqlite_store = SQLiteObservationStore()
gis_registry = CameraGISRegistry()
stream_manager = StreamManager(
    config=config,
    store=memory_store,
    gis_registry=gis_registry,
    sqlite_store=sqlite_store
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing DOKJA Platform Core Services...")
    stream_manager.start_all()
    yield
    logger.info("Shutting down DOKJA Platform Core Services...")
    stream_manager.stop_all()

app = FastAPI(
    title="DOKJA AI Traffic Intelligence Platform",
    description="Cross-Camera Vehicle Intelligence, Real-Time Detection, Tracking & Investigation System",
    version="0.1.0-alpha",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(stream_router, prefix="/api", tags=["Streaming & WebSockets"])
app.include_router(camera_router, prefix="/api", tags=["Cameras & GIS"])
app.include_router(analytics_router, prefix="/api", tags=["Analytics & Observations"])
app.include_router(investigation_router, prefix="/api", tags=["Investigation & ANPR"])

# Mount Evidence Crops Directory
CROPS_DIR = PROJECT_ROOT / "data" / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/crops", StaticFiles(directory=str(CROPS_DIR)), name="crops")

# Mount Frontend Assets
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def root_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "DOKJA API operational. Frontend not found."}

@app.get("/upload")
async def upload_index():
    upload_file = FRONTEND_DIR / "upload.html"
    if upload_file.exists():
        return FileResponse(upload_file)
    return {"message": "Upload page not found."}

@app.get("/investigate")
async def investigate_index():
    investigate_file = FRONTEND_DIR / "investigate.html"
    if investigate_file.exists():
        return FileResponse(investigate_file)
    return {"message": "Investigation console not found."}

@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "cameras_active": len(stream_manager.workers),
        "vision_engine": config.get("vision", {}).get("model_name", "yolov8n.pt"),
        "tracker": config.get("tracking", {}).get("tracker_type", "bytetrack")
    }

if __name__ == "__main__":
    import uvicorn
    app_cfg = config.get("app", {})
    uvicorn.run(
        "backend.main:app",
        host=app_cfg.get("host", "0.0.0.0"),
        port=app_cfg.get("port", 8000),
        reload=False
    )
