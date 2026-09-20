from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from backend.models.schemas import ObservationRecord
from backend.database.sqlite_store import SQLiteObservationStore
from backend.agents.investigation_agent import InvestigationAgent
from backend.services.trajectory_service import TrajectoryBuilder
from configs.logging_config import setup_logger

logger = setup_logger("routes_investigation")
router = APIRouter()


class WatchlistEntry(BaseModel):
    plate_text: str = Field(min_length=3, max_length=20)
    label: str = Field(default="Flagged vehicle", max_length=120)
    reason: str = Field(default="", max_length=500)

def get_sqlite_store() -> SQLiteObservationStore:
    from backend.main import sqlite_store
    return sqlite_store


def get_agent() -> InvestigationAgent:
    return InvestigationAgent(get_sqlite_store())


@router.post("/watchlist", status_code=201)
def add_watchlist_entry(payload: WatchlistEntry):
    """Add a vehicle to the local authorised-investigator watchlist."""
    try:
        return get_sqlite_store().add_watchlist_entry(payload.plate_text, payload.label, payload.reason)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/watchlist")
def list_watchlist_entries():
    return {"entries": get_sqlite_store().list_watchlist()}


@router.get("/alerts")
def list_alerts(status: Optional[str] = Query(None), limit: int = Query(50, ge=1, le=200)):
    return {"alerts": get_sqlite_store().list_alerts(status=status, limit=limit)}


@router.get("/investigation/search", response_model=List[ObservationRecord])
def search_observations(
    plate: Optional[str] = Query(None, description="Exact or partial license plate string"),
    camera_id: Optional[str] = Query(None, description="Camera ID filter"),
    vehicle_type: Optional[str] = Query(None, description="Vehicle category (car, bus, truck, motorcycle)"),
    color: Optional[str] = Query(None, description="Vehicle body color"),
    start_time: Optional[str] = Query(None, description="Start ISO timestamp"),
    end_time: Optional[str] = Query(None, description="End ISO timestamp"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum plate confidence"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """
    Structured investigation query endpoint searching historical vehicle observations,
    license plates, vehicle attributes, and geospatial timestamps.
    """
    store = get_sqlite_store()
    records = store.search_observations(
        plate_query=plate,
        camera_id=camera_id,
        vehicle_type=vehicle_type,
        vehicle_color=color,
        start_time=start_time,
        end_time=end_time,
        min_plate_conf=min_confidence,
        limit=limit,
        offset=offset
    )
    return records

@router.get("/investigation/plates/recent", response_model=List[ObservationRecord])
def get_recent_plates(limit: int = Query(25, ge=1, le=100)):
    """
    Returns recently recognized license plates with confidence metrics and evidence snapshot URLs.
    """
    store = get_sqlite_store()
    return store.get_recent_plates(limit=limit)

@router.get("/investigation/observations/{obs_id}", response_model=ObservationRecord)
def get_observation_details(obs_id: str):
    """
    Retrieves full audit and evidence details for a specific observation ID.
    """
    store = get_sqlite_store()
    record = store.get_observation_by_id(obs_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Observation '{obs_id}' not found.")
    return record


@router.get("/investigation/agent")
def run_investigation_agent(query: str = Query(..., description="Natural-language investigation request")):
    """Runs the lightweight DOKJA investigation agent against verified observation data."""
    agent = get_agent()
    return agent.execute(query)


@router.get("/investigation/trajectory")
def get_vehicle_trajectory(
    plate: Optional[str] = Query(None, description="Plate number to reconstruct route for"),
    camera_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = Query(20, ge=1, le=200),
):
    """Returns a time-ordered trajectory from stored observations and GeoJSON route data."""
    store = get_sqlite_store()
    builder = TrajectoryBuilder(store)
    if plate:
        return builder.build_for_plate(
            plate=plate,
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
    if camera_id:
        return builder.build_for_camera(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
    return {"plate": plate, "camera_id": camera_id, "count": 0, "trajectory": [], "geojson": {"type": "FeatureCollection", "features": []}}


@router.get("/investigation/matches")
def find_possible_matches(
    plate: Optional[str] = Query(None, description="Plate to compare against other observations"),
    threshold: float = Query(0.6, ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=100),
):
    """Finds likely same-vehicle observations using explainable evidence rules."""
    store = get_sqlite_store()
    source = store.search_observations(plate_query=plate, limit=limit, offset=0)
    if not source:
        return {"matches": [], "plate": plate, "count": 0}

    agent = get_agent()
    matches = agent.find_matches([
        {
            "plate_text": obs.plate_text,
            "vehicle_type": obs.vehicle_type,
            "vehicle_color": obs.vehicle_color,
            "bbox_size": tuple(obs.bbox[2:]) if len(obs.bbox) >= 4 else None,
            "timestamp": obs.timestamp,
        } for obs in source
    ], threshold=threshold)
    return {"matches": matches, "plate": plate, "count": len(matches)}
