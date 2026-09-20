import os
import cv2
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.preprocessing.frame_reader import SyntheticTrafficGenerator

def generate_sample_video(output_path: str, duration_sec: int = 10, fps: int = 25):
    """
    Generates a high-resolution sample traffic video file for local testing.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    gen = SyntheticTrafficGenerator(camera_id="SAMPLE-CAM", width=width, height=height, num_vehicles=8)
    total_frames = duration_sec * fps

    print(f"Generating {duration_sec}s sample traffic video at {output_path}...")
    for _ in range(total_frames):
        frame = gen.generate_frame()
        out.write(frame)

    out.release()
    print(f"Generated sample video ({total_frames} frames) successfully!")

if __name__ == "__main__":
    sample_path = str(PROJECT_ROOT / "data" / "samples" / "sample_traffic_01.mp4")
    generate_sample_video(sample_path)
