import time
from typing import List, Dict
from datetime import datetime
from backend.models.schemas import VehicleTrack, TrafficTelemetry, TrafficForecast

class TrafficFlowAnalyzer:
    """
    Computes real-time traffic statistics from active vehicle tracks:
    density classification, class distribution, flow rate, and average velocities.
    """
    def __init__(self, camera_id: str, capacity_threshold: int = 15):
        self.camera_id = camera_id
        self.capacity_threshold = capacity_threshold
        self.observed_track_ids = set()
        self.last_minute_tracks: List[float] = [] # Timestamps of seen tracks

    def analyze(self, tracks: List[VehicleTrack]) -> TrafficTelemetry:
        now = time.time()
        now_iso = datetime.now().isoformat()

        # Count vehicle categories
        class_counts: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        speeds = []

        for trk in tracks:
            cname = trk.class_name.lower()
            class_counts[cname] = class_counts.get(cname, 0) + 1
            if trk.speed_px_s > 0:
                speeds.append(trk.speed_px_s)

            if trk.track_id not in self.observed_track_ids:
                self.observed_track_ids.add(trk.track_id)
                self.last_minute_tracks.append(now)

        # Retain only tracks from the last 60 seconds for flow rate
        self.last_minute_tracks = [t for t in self.last_minute_tracks if now - t <= 60.0]
        flow_rate = len(self.last_minute_tracks)

        total_vehicles = len(tracks)
        density_ratio = min(1.0, total_vehicles / max(1, self.capacity_threshold))

        if density_ratio < 0.25:
            density_status = "Light"
        elif density_ratio < 0.55:
            density_status = "Moderate"
        elif density_ratio < 0.85:
            density_status = "Heavy"
        else:
            density_status = "Severe Congestion"

        avg_speed = float(sum(speeds) / len(speeds)) if speeds else 0.0

        return TrafficTelemetry(
            camera_id=self.camera_id,
            timestamp=now_iso,
            total_active_vehicles=total_vehicles,
            class_counts=class_counts,
            density_score=round(density_ratio, 2),
            density_status=density_status,
            avg_speed_px_s=round(avg_speed, 1),
            flow_rate_per_min=flow_rate
        )


class TrafficForecastAnalyzer:
    """Simple predictive traffic forecasting based on density, speed, and flow."""

    def predict(self, telemetry: TrafficTelemetry, horizon_minutes: int = 5) -> TrafficForecast:
        density_score = max(0.0, min(1.0, float(telemetry.density_score)))
        flow_factor = min(1.0, float(telemetry.flow_rate_per_min) / 40.0)
        speed_factor = max(0.0, min(1.0, (40.0 - float(telemetry.avg_speed_px_s)) / 40.0))

        risk_score = min(1.0, (0.55 * density_score) + (0.25 * flow_factor) + (0.20 * speed_factor))

        if risk_score >= 0.8:
            forecast_status = "Severe Congestion"
        elif risk_score >= 0.6:
            forecast_status = "Heavy"
        elif risk_score >= 0.38:
            forecast_status = "Moderate"
        else:
            forecast_status = "Light"

        predicted_total = max(
            telemetry.total_active_vehicles,
            int(round(telemetry.total_active_vehicles * (1.0 + (risk_score * 0.45))))
        )

        reasons = []
        if telemetry.density_status in {"Heavy", "Severe Congestion"}:
            reasons.append("Current density remains elevated above the typical operating threshold.")
        if telemetry.avg_speed_px_s < 12:
            reasons.append("Vehicle throughput is slowing, which typically expands queue lengths.")
        if telemetry.flow_rate_per_min >= 25:
            reasons.append("Inbound flow is high enough to sustain crowding over the next few minutes.")
        if not reasons:
            reasons.append("Current flow is stable and should remain manageable for the next window.")

        forecast = TrafficForecast(
            camera_id=telemetry.camera_id,
            forecast_time_horizon_minutes=horizon_minutes,
            predicted_total_vehicles=predicted_total,
            predicted_density=round(risk_score, 2),
            forecast_status=forecast_status,
            forecast_confidence=round(min(0.99, 0.6 + (risk_score * 0.4)), 2),
            reasons=reasons[:3],
            timestamp=datetime.now().isoformat(),
        )
        return forecast
