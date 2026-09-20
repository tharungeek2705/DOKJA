import unittest

from backend.database.memory_store import MemoryObservationStore
from backend.models.schemas import CameraInfo
from backend.services.stream_manager import StreamManager
from gis.camera_registry import CameraGISRegistry


class TestPhase6SourceConfig(unittest.TestCase):
    def test_camera_source_type_is_preserved(self):
        cam = CameraInfo(
            id='CAM-UI-01',
            name='Demo Cam',
            location='Test',
            latitude=13.0,
            longitude=80.0,
            source='0',
            source_type='webcam'
        )
        self.assertEqual(cam.source_type, 'webcam')

    def test_stream_manager_updates_camera_source(self):
        config = {
            'cameras': [{
                'id': 'CAM-UPDATE-01',
                'name': 'Test',
                'location': 'Loc',
                'latitude': 13.0,
                'longitude': 80.0,
                'source': 'synthetic',
                'source_type': 'synthetic',
                'fps': 25,
            }],
            'vision': {'model_name': 'yolov8n.pt'},
            'tracking': {'tracker_type': 'bytetrack'}
        }
        mgr = StreamManager(config=config, store=MemoryObservationStore(), gis_registry=CameraGISRegistry())
        updated = mgr.update_camera_source('CAM-UPDATE-01', 'data/samples/sample.mp4', 'video')
        self.assertEqual(updated.source, 'data/samples/sample.mp4')
        self.assertEqual(updated.source_type, 'video')


if __name__ == '__main__':
    unittest.main()
