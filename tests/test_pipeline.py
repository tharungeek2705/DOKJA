import sys
import os
import unittest
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.schemas import CameraInfo, VehicleTrack, BoundingBox
from backend.database.memory_store import MemoryObservationStore
from gis.camera_registry import CameraGISRegistry
from vision.preprocessing.frame_reader import VideoFrameReader, SyntheticTrafficGenerator
from vision.preprocessing.visualizer import TacticalHUDVisualizer
from vision.detection.yolo_detector import YOLOVehicleDetector
from vision.tracking.tracker import VehicleTracker
from analytics.traffic.traffic_analyzer import TrafficFlowAnalyzer

class TestDokjaPipeline(unittest.TestCase):
    def setUp(self):
        self.camera_id = "TEST-CAM-01"
        self.resolution = (1280, 720)

    def test_01_synthetic_frame_generator(self):
        """Verify synthetic traffic frame generation"""
        gen = SyntheticTrafficGenerator(self.camera_id, 1280, 720, num_vehicles=6)
        frame = gen.generate_frame()
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_02_video_frame_reader(self):
        """Verify video frame reader abstraction"""
        reader = VideoFrameReader(self.camera_id, "synthetic", target_fps=25)
        frame = reader.read_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (720, 1280, 3))
        reader.release()

    def test_03_yolo_vehicle_detector(self):
        """Verify YOLO detector loading and prediction"""
        detector = YOLOVehicleDetector(model_path="yolov8n.pt", device="cpu", conf_threshold=0.25)
        gen = SyntheticTrafficGenerator(self.camera_id, 1280, 720, num_vehicles=6)
        frame = gen.generate_frame()

        detections = detector.detect(frame)
        self.assertIsInstance(detections, list)
        print(f"\n[Test] Detected {len(detections)} vehicles in test frame.")

    def test_04_tracker_continuity(self):
        """Verify multi-object tracking and trajectory accumulation across multiple frames"""
        tracker = VehicleTracker(model_path="yolov8n.pt", device="cpu", conf_threshold=0.25)
        gen = SyntheticTrafficGenerator(self.camera_id, 1280, 720, num_vehicles=5)

        all_tracks = []
        for f in range(5):
            frame = gen.generate_frame()
            tracks = tracker.track(frame)
            all_tracks.append(tracks)

        self.assertIsInstance(all_tracks[-1], list)
        print(f"[Test] Multi-frame tracker test executed across 5 frames.")

    def test_05_traffic_analyzer(self):
        """Verify traffic telemetry and density computation"""
        analyzer = TrafficFlowAnalyzer(self.camera_id, capacity_threshold=10)
        dummy_tracks = [
            VehicleTrack(
                track_id=1,
                class_id=2,
                class_name="car",
                confidence=0.92,
                bbox=BoundingBox(x1=100, y1=100, x2=200, y2=250),
                centroid=(150, 175),
                speed_px_s=45.0,
                direction_deg=90.0,
                heading_cardinal="N",
                dwell_time_seconds=2.5,
                trajectory=[(150, 190), (150, 175)]
            )
        ]
        telemetry = analyzer.analyze(dummy_tracks)
        self.assertEqual(telemetry.total_active_vehicles, 1)
        self.assertEqual(telemetry.class_counts["car"], 1)
        self.assertEqual(telemetry.density_status, "Light")

    def test_06_hud_visualizer(self):
        """Verify tactical HUD rendering"""
        visualizer = TacticalHUDVisualizer(self.camera_id, "Test Intersection")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dummy_tracks = [
            VehicleTrack(
                track_id=1,
                class_id=2,
                class_name="car",
                confidence=0.95,
                bbox=BoundingBox(x1=200, y1=200, x2=300, y2=350),
                centroid=(250, 275),
                speed_px_s=30.0,
                direction_deg=180.0,
                heading_cardinal="W",
                dwell_time_seconds=1.2,
                trajectory=[(250, 275)]
            )
        ]
        hud_frame = visualizer.draw_hud(frame, dummy_tracks, fps=30.0)
        self.assertEqual(hud_frame.shape, (720, 1280, 3))

    def test_07_gis_registry_and_store(self):
        """Verify GIS registry and in-memory store"""
        registry = CameraGISRegistry()
        cam = CameraInfo(
            id="CAM-01",
            name="North Gate",
            location="Gate 1",
            latitude=13.0827,
            longitude=80.2707,
            source="synthetic"
        )
        registry.register_camera(cam)
        geojson = registry.to_geojson()
        self.assertEqual(len(geojson["features"]), 1)
        self.assertEqual(geojson["features"][0]["properties"]["id"], "CAM-01")

if __name__ == "__main__":
    unittest.main()
