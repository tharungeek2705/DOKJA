import threading
from typing import Dict, List, Optional
from collections import deque
from backend.models.schemas import ObservationRecord, VehicleTrack, TrafficTelemetry

class MemoryObservationStore:
    """
    Thread-safe in-memory store for vehicle observations, active tracks,
    and real-time traffic telemetry during Phase 1.
    """
    def __init__(self, max_records: int = 1000):
        self._lock = threading.Lock()
        self._observations: deque = deque(maxlen=max_records)
        self._active_tracks: Dict[str, List[VehicleTrack]] = {} # camera_id -> tracks
        self._latest_telemetry: Dict[str, TrafficTelemetry] = {} # camera_id -> telemetry

    def record_observation(self, observation: ObservationRecord):
        with self._lock:
            self._observations.append(observation)

    def set_active_tracks(self, camera_id: str, tracks: List[VehicleTrack]):
        with self._lock:
            self._active_tracks[camera_id] = tracks

    def get_active_tracks(self, camera_id: str) -> List[VehicleTrack]:
        with self._lock:
            return list(self._active_tracks.get(camera_id, []))

    def set_telemetry(self, camera_id: str, telemetry: TrafficTelemetry):
        with self._lock:
            self._latest_telemetry[camera_id] = telemetry

    def get_telemetry(self, camera_id: str) -> Optional[TrafficTelemetry]:
        with self._lock:
            return self._latest_telemetry.get(camera_id)

    def get_all_telemetry(self) -> Dict[str, TrafficTelemetry]:
        with self._lock:
            return dict(self._latest_telemetry)

    def get_recent_observations(self, limit: int = 50, camera_id: Optional[str] = None) -> List[ObservationRecord]:
        with self._lock:
            records = list(self._observations)
            if camera_id:
                records = [r for r in records if r.camera_id == camera_id]
            return records[-limit:][::-1] # Newest first
