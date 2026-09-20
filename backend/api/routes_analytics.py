from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
from backend.models.schemas import TrafficTelemetry, VehicleTrack, ObservationRecord, TrafficForecast
from analytics.traffic.traffic_analyzer import TrafficForecastAnalyzer
from backend.database.memory_store import MemoryObservationStore

router = APIRouter()

def get_store() -> MemoryObservationStore:
    from backend.main import memory_store
    return memory_store

@router.get("/analytics/telemetry", response_model=Dict[str, Optional[TrafficTelemetry]])
def get_all_telemetry():
    """
    Returns latest traffic telemetry across all camera junctions.
    """
    store = get_store()
    return store.get_all_telemetry()

@router.get("/analytics/telemetry/{camera_id}", response_model=TrafficTelemetry)
def get_camera_telemetry(camera_id: str):
    """
    Returns latest telemetry for a specific camera junction.
    """
    store = get_store()
    telem = store.get_telemetry(camera_id)
    if not telem:
        raise HTTPException(status_code=404, detail=f"No telemetry available for '{camera_id}'.")
    return telem

@router.get("/analytics/forecast/{camera_id}", response_model=TrafficForecast)
def get_camera_forecast(camera_id: str):
    """Returns a short-horizon congestion forecast for a camera junction."""
    store = get_store()
    telem = store.get_telemetry(camera_id)
    if not telem:
        raise HTTPException(status_code=404, detail=f"No telemetry available for '{camera_id}'.")
    return TrafficForecastAnalyzer().predict(telem)

@router.get("/analytics/forecast", response_model=Dict[str, TrafficForecast])
def get_all_camera_forecasts():
    """Returns short-horizon forecasts for all active camera junctions."""
    store = get_store()
    analyzer = TrafficForecastAnalyzer()
    return {
        camera_id: analyzer.predict(telem)
        for camera_id, telem in store.get_all_telemetry().items()
    }

@router.get("/analytics/tracks/{camera_id}", response_model=List[VehicleTrack])
def get_camera_tracks(camera_id: str):
    """
    Returns currently active vehicle tracks for a specific camera.
    """
    store = get_store()
    return store.get_active_tracks(camera_id)

@router.get("/analytics/observations", response_model=List[ObservationRecord])
def get_recent_observations(
    limit: int = Query(default=30, ge=1, le=200),
    camera_id: Optional[str] = None
):
    """
    Returns chronological timeline log of detected vehicles.
    """
    store = get_store()
    return store.get_recent_observations(limit=limit, camera_id=camera_id)
