import unittest
from datetime import datetime, timezone

from backend.agents.investigation_agent import InvestigationAgent
from backend.database.sqlite_store import SQLiteObservationStore
from backend.models.schemas import ObservationRecord


class TestPhase5Investigation(unittest.TestCase):
    def test_last_known_location_query_is_evidence_based(self):
        store = SQLiteObservationStore(db_path='data/test_investigation.db')
        base = datetime(2026, 9, 19, 18, 0, tzinfo=timezone.utc)
        records = [
            ObservationRecord(
                observation_id='OBS-INV-001',
                camera_id='CAM-01',
                timestamp=(base).isoformat(),
                track_id=1,
                vehicle_type='car',
                confidence=0.92,
                bbox=[1, 2, 3, 4],
                latitude=13.0800,
                longitude=80.2600,
                vehicle_color='white',
                plate_text='TN01AB1234',
                plate_confidence=0.96,
                speed_px_s=10,
                heading='N',
            ),
            ObservationRecord(
                observation_id='OBS-INV-002',
                camera_id='CAM-02',
                timestamp=(base.replace(hour=19)).isoformat(),
                track_id=2,
                vehicle_type='car',
                confidence=0.91,
                bbox=[1, 2, 3, 4],
                latitude=13.0850,
                longitude=80.2700,
                vehicle_color='white',
                plate_text='TN01AB1234',
                plate_confidence=0.95,
                speed_px_s=12,
                heading='E',
            ),
            ObservationRecord(
                observation_id='OBS-INV-003',
                camera_id='CAM-03',
                timestamp=(base.replace(hour=20)).isoformat(),
                track_id=3,
                vehicle_type='car',
                confidence=0.90,
                bbox=[1, 2, 3, 4],
                latitude=13.0900,
                longitude=80.2800,
                vehicle_color='white',
                plate_text='TN01AB1234',
                plate_confidence=0.94,
                speed_px_s=15,
                heading='SE',
            ),
        ]
        for record in records:
            store.insert_observation(record)

        agent = InvestigationAgent(store)
        result = agent.execute('Find the last known location of TN01AB1234.')

        self.assertEqual(result['intent'], 'last_known_location')
        self.assertEqual(result['last_seen']['camera_id'], 'CAM-03')
        self.assertIn('last seen', result['summary'].lower())
        self.assertTrue(len(result['evidence']) >= 1)


if __name__ == '__main__':
    unittest.main()
