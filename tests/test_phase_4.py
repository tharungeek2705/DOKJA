import unittest
from datetime import datetime, timezone

from backend.database.sqlite_store import SQLiteObservationStore
from backend.models.schemas import ObservationRecord
from backend.services.trajectory_service import TrajectoryBuilder


class TestPhase4Trajectory(unittest.TestCase):
    def test_builds_geojson_route_for_plate(self):
        store = SQLiteObservationStore(db_path='data/test_trajectory.db')
        now = datetime.now(timezone.utc)
        obs_a = ObservationRecord(
            observation_id='OBS-TRAJ-001',
            camera_id='CAM-02',
            timestamp=(now).isoformat(),
            track_id=101,
            vehicle_type='car',
            confidence=0.91,
            bbox=[10.0, 20.0, 110.0, 200.0],
            latitude=13.0827,
            longitude=80.2707,
            vehicle_color='white',
            plate_text='TN01AB1234',
            plate_confidence=0.97,
            speed_px_s=12.5,
            heading='E',
        )
        obs_b = ObservationRecord(
            observation_id='OBS-TRAJ-002',
            camera_id='CAM-03',
            timestamp=(now).isoformat(),
            track_id=102,
            vehicle_type='car',
            confidence=0.89,
            bbox=[20.0, 30.0, 120.0, 210.0],
            latitude=13.0900,
            longitude=80.2800,
            vehicle_color='white',
            plate_text='TN01AB1234',
            plate_confidence=0.95,
            speed_px_s=13.2,
            heading='E',
        )
        store.insert_observation(obs_a)
        store.insert_observation(obs_b)

        builder = TrajectoryBuilder(store)
        route = builder.build_for_plate('TN01AB1234', limit=10)
        self.assertEqual(route['count'], 2)
        self.assertEqual(route['geojson']['features'][0]['geometry']['type'], 'LineString')
        self.assertGreater(len(route['geojson']['features'][0]['geometry']['coordinates']), 1)


if __name__ == '__main__':
    unittest.main()
