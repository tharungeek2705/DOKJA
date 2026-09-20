from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Optional
from datetime import datetime

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

class VehicleDetection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

class VehicleTrack(BaseModel):
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    centroid: Tuple[float, float]
    speed_px_s: float = 0.0
    direction_deg: float = 0.0
    heading_cardinal: str = "N"
    dwell_time_seconds: float = 0.0
    trajectory: List[Tuple[float, float]] = Field(default_factory=list)
    # Phase 2 Additions
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    vehicle_color: Optional[str] = None
    vehicle_crop_url: Optional[str] = None
    plate_crop_url: Optional[str] = None

class CameraInfo(BaseModel):
    id: str
    name: str
    location: str
    latitude: float
    longitude: float
    source: str
    source_type: str = "synthetic"  # synthetic, video, rtsp, webcam
    status: str = "active" # active, offline, error
    fps: int = 25
    active_vehicles_count: int = 0

class TrafficTelemetry(BaseModel):
    camera_id: str
    timestamp: str
    total_active_vehicles: int
    class_counts: Dict[str, int]
    density_score: float # 0.0 to 1.0
    density_status: str # "Light", "Moderate", "Heavy", "Severe Congestion"
    avg_speed_px_s: float
    flow_rate_per_min: int

class TrafficForecast(BaseModel):
    camera_id: str
    forecast_time_horizon_minutes: int = 5
    predicted_total_vehicles: int
    predicted_density: float
    forecast_status: str
    forecast_confidence: float
    reasons: List[str]
    timestamp: str

class ObservationRecord(BaseModel):
    observation_id: str
    camera_id: str
    timestamp: str
    track_id: int
    vehicle_type: str
    confidence: float
    bbox: List[float]
    latitude: float
    longitude: float
    # Phase 2 Additions
    vehicle_color: Optional[str] = "unknown"
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    speed_px_s: float = 0.0
    heading: str = "N"
    vehicle_crop_url: Optional[str] = None
    plate_crop_url: Optional[str] = None
