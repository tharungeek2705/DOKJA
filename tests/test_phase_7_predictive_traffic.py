import unittest

from analytics.traffic.traffic_analyzer import TrafficForecastAnalyzer
from backend.models.schemas import TrafficTelemetry


class TestPhase7PredictiveTraffic(unittest.TestCase):
    def test_predicts_congestion_risk_for_heavy_junction(self):
        analyzer = TrafficForecastAnalyzer()
        telemetry = TrafficTelemetry(
            camera_id="CAM-01",
            timestamp="2026-09-20T00:00:00",
            total_active_vehicles=20,
            class_counts={"car": 16, "motorcycle": 2, "bus": 1, "truck": 1},
            density_score=0.9,
            density_status="Severe Congestion",
            avg_speed_px_s=6.0,
            flow_rate_per_min=34,
        )

        forecast = analyzer.predict(telemetry)

        self.assertEqual(forecast.camera_id, "CAM-01")
        self.assertGreater(forecast.forecast_confidence, 0.5)
        self.assertIn(forecast.forecast_status, ["Severe Congestion", "Heavy", "Moderate"])
        self.assertGreaterEqual(forecast.predicted_total_vehicles, telemetry.total_active_vehicles)
        self.assertGreater(len(forecast.reasons), 0)


if __name__ == "__main__":
    unittest.main()
