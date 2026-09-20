import unittest
from datetime import datetime, timezone

from backend.database.sqlite_store import SQLiteObservationStore
from backend.models.schemas import ObservationRecord
from vision.anpr.anpr_pipeline import ANPRPipeline
from vision.reid.reid_matcher import VehicleReIDMatcher


class TestPhase23(unittest.TestCase):
    def test_anpr_pipeline_handles_synthetic_vehicle(self):
        pipeline = ANPRPipeline(enable_ocr=False)
        vehicle_crop = __import__('numpy').zeros((220, 520, 3), dtype=__import__('numpy').uint8)
        vehicle_crop[:, :, 2] = 220
        result = pipeline.process(vehicle_crop, camera_id='CAM-02', track_id=7, save_crops=False)
        self.assertIsInstance(result.plate_confidence, float)
        self.assertIn(result.vehicle_color, ['white', 'silver', 'black', 'blue', 'red', 'unknown'])

    def test_observation_search_tracks_plate_and_camera_filters(self):
        store = SQLiteObservationStore(db_path='data/test_observations.db')
        obs = ObservationRecord(
            observation_id='OBS-SEARCH-001',
            camera_id='CAM-02',
            timestamp=datetime.now(timezone.utc).isoformat(),
            track_id=12,
            vehicle_type='car',
            confidence=0.93,
            bbox=[100.0, 120.0, 260.0, 320.0],
            latitude=13.084,
            longitude=80.271,
            vehicle_color='white',
            plate_text='TN01AB1234',
            plate_confidence=0.96,
            speed_px_s=12.5,
            heading='E',
        )
        store.insert_observation(obs)
        records = store.search_observations(plate_query='TN01AB1234', camera_id='CAM-02', limit=10)
        self.assertTrue(any(r.observation_id == obs.observation_id for r in records))

    def test_reid_matcher_scores_same_vehicle_highly(self):
        matcher = VehicleReIDMatcher()
        obs_a = {
            'vehicle_type': 'car',
            'vehicle_color': 'white',
            'plate_text': 'TN01AB1234',
            'plate_confidence': 0.96,
            'bbox_size': (180, 80),
            'camera_id': 'CAM-02',
            'timestamp': '2026-09-19T18:00:00'
        }
        obs_b = {
            'vehicle_type': 'car',
            'vehicle_color': 'white',
            'plate_text': 'TN01AB1234',
            'plate_confidence': 0.94,
            'bbox_size': (176, 82),
            'camera_id': 'CAM-03',
            'timestamp': '2026-09-19T18:04:00'
        }
        score = matcher.match(obs_a, obs_b)
        self.assertGreaterEqual(score['score'], 0.8)
        self.assertIn('Likely match', score['reason'])


if __name__ == '__main__':
    unittest.main()
