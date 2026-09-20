import cv2
import time
import math
import random
import numpy as np
from typing import Optional, Tuple
from configs.logging_config import setup_logger

logger = setup_logger("frame_reader")

class SyntheticTrafficGenerator:
    """
    Generates high-resolution realistic simulated traffic road scenes with moving vehicles,
    road markings, asphalt texture, lane dividers, and vehicle shapes.
    Used for instant testing and multi-camera simulation without external video files.
    """
    def __init__(self, camera_id: str, width: int = 1280, height: int = 720, num_vehicles: int = 7):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.num_vehicles = num_vehicles
        self.vehicles = []
        self._init_vehicles()

    def _init_vehicles(self):
        # 3 lanes: Lane 0 (left-downwards), Lane 1 (center-downwards), Lane 2 (right-upwards)
        lanes = [
            {"x": self.width * 0.28, "direction": 1, "speed_range": (3.0, 5.5)},
            {"x": self.width * 0.44, "direction": 1, "speed_range": (3.5, 6.0)},
            {"x": self.width * 0.60, "direction": -1, "speed_range": (3.0, 5.0)},
            {"x": self.width * 0.76, "direction": -1, "speed_range": (4.0, 6.5)},
        ]

        types = [
            {"type": "car", "w": 90, "h": 160, "color": (210, 210, 210)},    # Silver car
            {"type": "car", "w": 85, "h": 155, "color": (40, 40, 180)},      # Red car
            {"type": "car", "w": 88, "h": 158, "color": (190, 120, 30)},     # Blue car
            {"type": "bus", "w": 115, "h": 260, "color": (30, 180, 210)},    # Yellow bus
            {"type": "truck", "w": 110, "h": 240, "color": (140, 140, 140)}, # Gray truck
            {"type": "car", "w": 85, "h": 150, "color": (30, 30, 30)},       # Black car
            {"type": "motorcycle", "w": 45, "h": 90, "color": (50, 180, 50)} # Green bike
        ]

        plates = [
            "TN01AB1234",
            "TN07DX9900",
            "DL03CB8821",
            "MH02BQ7711",
            "KA05MH2021",
            "TN22CK5544",
            "TS09EA4321"
        ]

        for i in range(self.num_vehicles):
            lane = lanes[i % len(lanes)]
            vtype = types[i % len(types)]
            speed = random.uniform(*lane["speed_range"]) * lane["direction"]
            y = random.uniform(50, self.height - 150)
            self.vehicles.append({
                "x": lane["x"] + random.uniform(-10, 10),
                "y": y,
                "lane": lane,
                "w": vtype["w"],
                "h": vtype["h"],
                "color": vtype["color"],
                "type": vtype["type"],
                "speed": speed,
                "direction": lane["direction"],
                "plate": plates[i % len(plates)]
            })

    def generate_frame(self) -> np.ndarray:
        # Asphalt background
        frame = np.full((self.height, self.width, 3), (45, 45, 48), dtype=np.uint8)

        # Sidewalks
        cv2.rectangle(frame, (0, 0), (int(self.width * 0.18), self.height), (75, 75, 75), -1)
        cv2.rectangle(frame, (int(self.width * 0.86), 0), (self.width, self.height), (75, 75, 75), -1)

        # Road curbs
        cv2.line(frame, (int(self.width * 0.18), 0), (int(self.width * 0.18), self.height), (220, 220, 220), 4)
        cv2.line(frame, (int(self.width * 0.86), 0), (int(self.width * 0.86), self.height), (220, 220, 220), 4)

        # Double yellow center divider
        center_x = int(self.width * 0.52)
        cv2.line(frame, (center_x - 5, 0), (center_x - 5, self.height), (0, 215, 255), 3)
        cv2.line(frame, (center_x + 5, 0), (center_x + 5, self.height), (0, 215, 255), 3)

        # Dashed white lane dividers
        dash_len = 35
        gap_len = 25
        lane_dividers = [int(self.width * 0.35), int(self.width * 0.69)]
        for div_x in lane_dividers:
            y = 0
            while y < self.height:
                cv2.line(frame, (div_x, y), (div_x, min(y + dash_len, self.height)), (240, 240, 240), 2)
                y += dash_len + gap_len

        # Update & draw vehicles
        for v in self.vehicles:
            v["y"] += v["speed"]
            # Wrap around screen edges
            if v["direction"] > 0 and v["y"] > self.height + 100:
                v["y"] = -v["h"] - 50
                v["x"] = v["lane"]["x"] + random.uniform(-10, 10)
            elif v["direction"] < 0 and v["y"] < -v["h"] - 100:
                v["y"] = self.height + 50
                v["x"] = v["lane"]["x"] + random.uniform(-10, 10)

            vx = int(v["x"])
            vy = int(v["y"])
            vw = int(v["w"])
            vh = int(v["h"])

            # Vehicle shadow
            cv2.ellipse(frame, (vx, vy + vh // 2), (vw // 2 + 8, vh // 2 + 10), 0, 0, 360, (20, 20, 20), -1)

            # Vehicle body (rounded rectangle)
            top_left = (vx - vw // 2, vy - vh // 2)
            bottom_right = (vx + vw // 2, vy + vh // 2)
            cv2.rectangle(frame, top_left, bottom_right, v["color"], -1)
            cv2.rectangle(frame, top_left, bottom_right, (15, 15, 15), 2)

            # Windshield & rear glass
            glass_color = (60, 60, 60)
            windshield_y = vy - vh // 4 if v["direction"] > 0 else vy + vh // 4
            cv2.rectangle(frame, (vx - vw // 2 + 8, windshield_y - 12), (vx + vw // 2 - 8, windshield_y + 12), glass_color, -1)

            # License Plate (White background, black border & text)
            plate_w, plate_h = min(vw - 16, 72), 18
            plate_y = vy + vh // 2 - 16 if v["direction"] > 0 else vy + vh // 2 - 16
            cv2.rectangle(frame, (vx - plate_w // 2, plate_y - plate_h // 2), (vx + plate_w // 2, plate_y + plate_h // 2), (245, 245, 245), -1)
            cv2.rectangle(frame, (vx - plate_w // 2, plate_y - plate_h // 2), (vx + plate_w // 2, plate_y + plate_h // 2), (10, 10, 10), 1)

            # Plate Text
            p_text = v["plate"]
            (tw, th), _ = cv2.getTextSize(p_text, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
            cv2.putText(frame, p_text, (vx - tw // 2, plate_y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1, cv2.LINE_AA)

            # Headlights / Taillights
            if v["direction"] > 0:
                # Moving down: headlights at bottom, taillights at top
                cv2.circle(frame, (vx - vw // 2 + 12, vy + vh // 2 - 4), 6, (200, 255, 255), -1)
                cv2.circle(frame, (vx + vw // 2 - 12, vy + vh // 2 - 4), 6, (200, 255, 255), -1)
                cv2.circle(frame, (vx - vw // 2 + 12, vy - vh // 2 + 4), 5, (0, 0, 220), -1)
                cv2.circle(frame, (vx + vw // 2 - 12, vy - vh // 2 + 4), 5, (0, 0, 220), -1)
            else:
                # Moving up: headlights at top, taillights at bottom
                cv2.circle(frame, (vx - vw // 2 + 12, vy - vh // 2 + 4), 6, (200, 255, 255), -1)
                cv2.circle(frame, (vx + vw // 2 - 12, vy - vh // 2 + 4), 6, (200, 255, 255), -1)
                cv2.circle(frame, (vx - vw // 2 + 12, vy + vh // 2 - 4), 5, (0, 0, 220), -1)
                cv2.circle(frame, (vx + vw // 2 - 12, vy + vh // 2 - 4), 5, (0, 0, 220), -1)

        return frame

class VideoFrameReader:
    """
    Robust video stream reader supporting files, webcam feeds, RTSP links, and synthetic streams.
    Ensures stable FPS and automatic reconnection.
    """
    def __init__(self, camera_id: str, source: str, target_fps: int = 25, resolution: Tuple[int, int] = (1280, 720)):
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        self.frame_delay = 1.0 / max(1, target_fps)
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_synthetic = (source.lower() == "synthetic")
        self.synthetic_gen = SyntheticTrafficGenerator(camera_id, resolution[0], resolution[1]) if self.is_synthetic else None
        self.last_frame_time = 0.0

        if not self.is_synthetic:
            self._open_capture()

    def _open_capture(self) -> bool:
        try:
            # Check if source is integer (webcam device index)
            if self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                logger.warning(f"Failed to open video source '{self.source}' for camera {self.camera_id}. Falling back to synthetic.")
                self.is_synthetic = True
                self.synthetic_gen = SyntheticTrafficGenerator(self.camera_id, self.resolution[0], self.resolution[1])
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            logger.info(f"Connected to stream {self.source} for camera {self.camera_id}")
            return True
        except Exception as e:
            logger.error(f"Error opening capture for camera {self.camera_id}: {e}")
            self.is_synthetic = True
            self.synthetic_gen = SyntheticTrafficGenerator(self.camera_id, self.resolution[0], self.resolution[1])
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Retrieves the next video frame, maintaining the configured target FPS.
        """
        elapsed = time.time() - self.last_frame_time
        sleep_needed = self.frame_delay - elapsed
        if sleep_needed > 0:
            time.sleep(sleep_needed)
        self.last_frame_time = time.time()

        if self.is_synthetic:
            return self.synthetic_gen.generate_frame()

        if self.cap is None or not self.cap.isOpened():
            if not self._open_capture():
                return self.synthetic_gen.generate_frame()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Loop file if ended
            if not self.source.isdigit() and not self.source.startswith("rtsp://"):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    return None
            else:
                logger.warning(f"Stream interrupted on {self.camera_id}. Attempting reconnect...")
                time.sleep(0.5)
                self._open_capture()
                return None

        if (frame.shape[1], frame.shape[0]) != self.resolution:
            frame = cv2.resize(frame, self.resolution)

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
