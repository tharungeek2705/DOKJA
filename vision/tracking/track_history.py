import time
import math
from typing import Dict, List, Tuple, Optional
from collections import deque

def calculate_heading(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, str]:
    """
    Computes angle in degrees from p1 to p2 and maps to cardinal direction.
    Image coordinates: x goes right, y goes down.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if abs(dx) < 1.0 and abs(dy) < 1.0:
        return 0.0, "Stationary"

    # Math angle in standard Cartesian: invert dy because image y is inverted
    angle_rad = math.atan2(-dy, dx)
    angle_deg = (math.degrees(angle_rad) + 360) % 360

    # Map angle (0 = East, 90 = North, 180 = West, 270 = South)
    cardinals = [
        (22.5, 67.5, "NE"),
        (67.5, 112.5, "N"),
        (112.5, 157.5, "NW"),
        (157.5, 202.5, "W"),
        (202.5, 247.5, "SW"),
        (247.5, 292.5, "S"),
        (292.5, 337.5, "SE")
    ]
    for low, high, card in cardinals:
        if low <= angle_deg < high:
            return angle_deg, card
    return angle_deg, "E"

class TrackTrajectory:
    def __init__(self, track_id: int, initial_point: Tuple[float, float], max_points: int = 50):
        self.track_id = track_id
        self.max_points = max_points
        self.points: deque = deque(maxlen=max_points) # (x, y, timestamp)
        self.first_seen = time.time()
        self.last_seen = self.first_seen

        now = self.first_seen
        self.points.append((initial_point[0], initial_point[1], now))
        self.speed_px_s: float = 0.0
        self.heading_deg: float = 0.0
        self.cardinal: str = "N"

    def update(self, point: Tuple[float, float], timestamp: Optional[float] = None) -> None:
        now = timestamp if timestamp is not None else time.time()
        self.last_seen = now

        if self.points:
            last_x, last_y, last_t = self.points[-1]
            dt = max(0.001, now - last_t)
            dist = math.hypot(point[0] - last_x, point[1] - last_y)

            # Instantaneous speed with EMA smoothing
            inst_speed = dist / dt
            self.speed_px_s = 0.7 * self.speed_px_s + 0.3 * inst_speed if self.speed_px_s > 0 else inst_speed

            # Compute direction if vehicle moved noticeably (> 3px)
            if dist > 3.0:
                self.heading_deg, self.cardinal = calculate_heading((last_x, last_y), point)

        self.points.append((point[0], point[1], now))

    @property
    def dwell_time(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def get_trajectory_points(self) -> List[Tuple[float, float]]:
        return [(p[0], p[1]) for p in self.points]

class TrackHistoryManager:
    def __init__(self, max_points: int = 50, track_ttl_seconds: float = 5.0):
        self.max_points = max_points
        self.track_ttl_seconds = track_ttl_seconds
        self.tracks: Dict[int, TrackTrajectory] = {}

    def update_track(self, track_id: int, centroid: Tuple[float, float], timestamp: Optional[float] = None) -> TrackTrajectory:
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackTrajectory(track_id, centroid, self.max_points)
        else:
            self.tracks[track_id].update(centroid, timestamp)
        return self.tracks[track_id]

    def cleanup_stale_tracks(self) -> None:
        now = time.time()
        stale_ids = [
            tid for tid, trk in self.tracks.items()
            if (now - trk.last_seen) > self.track_ttl_seconds
        ]
        for tid in stale_ids:
            del self.tracks[tid]
