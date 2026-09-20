from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
from backend.models.schemas import CameraInfo
from gis.camera_registry import CameraGISRegistry


class CameraSourceUpdate(BaseModel):
    source: str
    source_type: str = "synthetic"


class CameraEnrollment(BaseModel):
    id: str
    name: str
    location: str
    latitude: float
    longitude: float
    source: str
    source_type: str = "rtsp"
    fps: int = 25

router = APIRouter()

def get_gis_registry() -> CameraGISRegistry:
    from backend.main import gis_registry
    return gis_registry

@router.get("/cameras", response_model=List[CameraInfo])
def list_all_cameras():
    """
    Returns metadata and current operational status for all registered cameras.
    """
    gis = get_gis_registry()
    return gis.list_cameras()

@router.get("/cameras/geojson")
def get_cameras_geojson():
    """
    Returns GeoJSON feature collection for Leaflet GIS map rendering.
    """
    gis = get_gis_registry()
    return gis.to_geojson()

@router.get("/cameras/{camera_id}", response_model=CameraInfo)
def get_camera_by_id(camera_id: str):
    """
    Returns specific camera details by ID.
    """
    gis = get_gis_registry()
    cam = gis.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return cam


@router.post("/cameras", response_model=CameraInfo, status_code=201)
def enroll_camera(payload: CameraEnrollment):
    """Enroll an RTSP/IP, webcam, video-file, or demo camera into the live grid."""
    source_type = payload.source_type.lower().strip()
    if source_type not in {"rtsp", "webcam", "video", "synthetic"}:
        raise HTTPException(status_code=422, detail="source_type must be rtsp, webcam, video, or synthetic.")
    if source_type == "rtsp" and not payload.source.lower().startswith(("rtsp://", "rtsps://")):
        raise HTTPException(status_code=422, detail="RTSP cameras require an rtsp:// or rtsps:// source URL.")

    from backend.main import stream_manager
    camera = CameraInfo(**payload.model_dump(), source_type=source_type)
    try:
        return stream_manager.register_camera(camera)
    except KeyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/cameras/{camera_id}/source")
def update_camera_source(camera_id: str, payload: CameraSourceUpdate):
    """Changes the input source of a camera between synthetic/demo, local video, RTSP, or webcam."""
    from backend.main import stream_manager
    try:
        camera = stream_manager.update_camera_source(
            camera_id=camera_id,
            source=payload.source,
            source_type=payload.source_type
        )
        return {"status": "updated", "camera": camera.model_dump()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
