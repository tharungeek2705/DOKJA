from typing import Dict, List, Optional
from backend.models.schemas import CameraInfo

class CameraGISRegistry:
    """
    Manages geospatial locations, coverage sectors, and metadata for all city cameras.
    Produces GeoJSON-compatible data for interactive map rendering.
    """
    def __init__(self):
        self._cameras: Dict[str, CameraInfo] = {}

    def register_camera(self, camera: CameraInfo):
        self._cameras[camera.id] = camera

    def get_camera(self, camera_id: str) -> Optional[CameraInfo]:
        return self._cameras.get(camera_id)

    def list_cameras(self) -> List[CameraInfo]:
        return list(self._cameras.values())

    def update_vehicle_count(self, camera_id: str, count: int):
        if camera_id in self._cameras:
            self._cameras[camera_id].active_vehicles_count = count

    def to_geojson(self) -> dict:
        features = []
        for cam in self._cameras.values():
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [cam.longitude, cam.latitude]
                },
                "properties": {
                    "id": cam.id,
                    "name": cam.name,
                    "location": cam.location,
                    "status": cam.status,
                    "fps": cam.fps,
                    "active_vehicles": cam.active_vehicles_count
                }
            })
        return {
            "type": "FeatureCollection",
            "features": features
        }
