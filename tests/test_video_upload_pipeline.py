import os
import tempfile
import unittest

import cv2
import numpy as np

from backend.api.routes_stream import process_uploaded_video


class TestVideoUploadPipeline(unittest.TestCase):
    def test_process_uploaded_video_returns_plate_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = os.path.join(tmp_dir, 'sample_upload.mp4')
            width, height = 320, 240
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
            for i in range(10):
                frame = np.full((height, width, 3), 40, dtype=np.uint8)
                cv2.rectangle(frame, (60, 80), (180, 180), (50, 50, 255), -1)
                cv2.putText(frame, 'ABC123', (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                writer.write(frame)
            writer.release()

            result = process_uploaded_video(video_path)
            self.assertIn('plates', result)
            self.assertTrue(isinstance(result['plates'], list))


if __name__ == '__main__':
    unittest.main()
