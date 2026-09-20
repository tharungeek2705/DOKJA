from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.database.sqlite_store import SQLiteObservationStore
from backend.models.schemas import ObservationRecord


class TrajectoryBuilder:
    """Builds time-ordered vehicle trajectories and GeoJSON route overlays."""

    def __init__(self, store: SQLiteObservationStore):
        self.store = store

    def build_for_plate(
        self,
        plate: Optional[str] = None,
        camera_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        records = self.store.search_observations(
            plate_query=plate,
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=0,
        )
        ordered = sorted(records, key=lambda obs: obs.timestamp)
        return {
            "plate": plate,
            "camera_id": camera_id,
            "count": len(ordered),
            "trajectory": [obs.model_dump() for obs in ordered],
            "geojson": self._build_geojson(ordered, plate=plate),
        }

    def build_for_camera(
        self,
        camera_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        records = self.store.search_observations(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=0,
        )
        ordered = sorted(records, key=lambda obs: obs.timestamp)
        return {
            "camera_id": camera_id,
            "count": len(ordered),
            "trajectory": [obs.model_dump() for obs in ordered],
            "geojson": self._build_geojson(ordered, plate=None, camera_id=camera_id),
        }

    def _build_geojson(self, records: List[ObservationRecord], plate: Optional[str] = None, camera_id: Optional[str] = None) -> Dict[str, Any]:
        points = [
            [float(obs.longitude), float(obs.latitude)]
            for obs in records
            if obs.latitude is not None and obs.longitude is not None
        ]

        if not points:
            return {"type": "FeatureCollection", "features": []}

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": points,
            },
            "properties": {
                "plate": plate,
                "camera_id": camera_id,
                "point_count": len(points),
                "confidence": max((obs.plate_confidence if obs.plate_text else obs.confidence) for obs in records) if records else 0.0,
            },
        }
        return {"type": "FeatureCollection", "features": [feature]}
