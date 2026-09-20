import cv2
import time
import uuid
import threading
import numpy as np
from typing import Dict, Optional, Generator
from datetime import datetime

from backend.models.schemas import CameraInfo, ObservationRecord
from backend.database.memory_store import MemoryObservationStore
from backend.database.sqlite_store import SQLiteObservationStore
from gis.camera_registry import CameraGISRegistry
from vision.preprocessing.frame_reader import VideoFrameReader
from vision.preprocessing.visualizer import TacticalHUDVisualizer
from vision.tracking.tracker import VehicleTracker
from vision.anpr.anpr_pipeline import ANPRPipeline
from analytics.traffic.traffic_analyzer import TrafficFlowAnalyzer
from configs.logging_config import setup_logger

logger = setup_logger("stream_manager")

class CameraPipelineWorker:
    """
    Dedicated worker thread per camera that continuously executes:
    Video Ingestion -> Detection & Tracking -> ANPR & Color -> Telemetry Analysis -> HUD -> JPEG Encoding.
    """
    def __init__(
        self,
        camera: CameraInfo,
        vision_cfg: dict,
        tracking_cfg: dict,
        store: MemoryObservationStore,
        sqlite_store: Optional[SQLiteObservationStore],
        gis_registry: CameraGISRegistry
    ):
        self.camera = camera
        self.store = store
        self.sqlite_store = sqlite_store
        self.gis_registry = gis_registry

        # Dedicated tracker instance per camera feed for thread safety and track ID continuity
        self.tracker = VehicleTracker(
            model_path=vision_cfg.get("model_name", "yolov8n.pt"),
            device=vision_cfg.get("device", "auto"),
            conf_threshold=vision_cfg.get("confidence_threshold", 0.35),
            iou_threshold=vision_cfg.get("iou_threshold", 0.45),
            target_classes=vision_cfg.get("target_classes", [1, 2, 3, 5, 7]),
            tracker_config=f"{tracking_cfg.get('tracker_type', 'bytetrack')}.yaml",
            max_trajectory_points=tracking_cfg.get("max_trajectory_points", 50)
        )

        # ANPR & Appearance Pipeline
        self.anpr_pipeline = ANPRPipeline(enable_ocr=True)
        self.track_metadata: Dict[int, dict] = {} # track_id -> metadata cache

        self.reader = VideoFrameReader(
            camera_id=camera.id,
            source=camera.source,
            target_fps=camera.fps
        )
        self.analyzer = TrafficFlowAnalyzer(camera.id)
        self.visualizer = TacticalHUDVisualizer(camera.id, camera.name)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_hud_jpeg: Optional[bytes] = None
        self._latest_raw_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._measured_fps = float(camera.fps)
        self._last_frame_timestamp = time.time()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name=f"Worker-{self.camera.id}")
        self._thread.start()
        logger.info(f"Pipeline worker for camera {self.camera.id} started.")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.reader.release()
        logger.info(f"Pipeline worker for camera {self.camera.id} stopped.")

    def _run_loop(self):
        frame_counter = 0
        fps_timer = time.time()

        while self._running:
            try:
                frame = self.reader.read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                now = time.time()
                frame_counter += 1
                if now - fps_timer >= 1.0:
                    self._measured_fps = frame_counter / (now - fps_timer)
                    frame_counter = 0
                    fps_timer = now

                # 1. Multi-object tracking (YOLO + ByteTrack)
                tracks = self.tracker.track(frame)

                # 2. Run ANPR, Appearance Color Extraction & Cache Best Reading
                fh, fw = frame.shape[:2]
                for trk in tracks:
                    meta = self.track_metadata.get(trk.track_id)
                    # If high confidence plate already acquired, keep it
                    if meta and meta.get("plate_conf", 0.0) >= 0.85:
                        trk.plate_text = meta.get("plate_text")
                        trk.plate_confidence = meta.get("plate_conf", 0.0)
                        trk.vehicle_color = meta.get("color")
                        trk.vehicle_crop_url = meta.get("v_url")
                        trk.plate_crop_url = meta.get("p_url")
                    else:
                        x1 = max(0, int(trk.bbox.x1))
                        y1 = max(0, int(trk.bbox.y1))
                        x2 = min(fw, int(trk.bbox.x2))
                        y2 = min(fh, int(trk.bbox.y2))
                        
                        if (x2 - x1) >= 40 and (y2 - y1) >= 40:
                            # Run ANPR on new tracks or periodically
                            if frame_counter % 4 == 0 or trk.track_id not in self.track_metadata:
                                v_crop = frame[y1:y2, x1:x2]
                                anpr_res = self.anpr_pipeline.process(
                                    vehicle_crop=v_crop,
                                    camera_id=self.camera.id,
                                    track_id=trk.track_id,
                                    save_crops=True
                                )
                                prev_conf = meta.get("plate_conf", 0.0) if meta else 0.0
                                if anpr_res.plate_confidence >= prev_conf:
                                    self.track_metadata[trk.track_id] = {
                                        "plate_text": anpr_res.plate_text if anpr_res.plate_text else (meta.get("plate_text") if meta else None),
                                        "plate_conf": anpr_res.plate_confidence if anpr_res.plate_text else prev_conf,
                                        "color": anpr_res.vehicle_color,
                                        "v_url": anpr_res.vehicle_crop_url if anpr_res.vehicle_crop_url else (meta.get("v_url") if meta else None),
                                        "p_url": anpr_res.plate_crop_url if anpr_res.plate_crop_url else (meta.get("p_url") if meta else None)
                                    }

                        cur_meta = self.track_metadata.get(trk.track_id, {})
                        trk.plate_text = cur_meta.get("plate_text")
                        trk.plate_confidence = cur_meta.get("plate_conf", 0.0)
                        trk.vehicle_color = cur_meta.get("color", "unknown")
                        trk.vehicle_crop_url = cur_meta.get("v_url")
                        trk.plate_crop_url = cur_meta.get("p_url")

                # 3. Traffic Flow & Telemetry Analysis
                telemetry = self.analyzer.analyze(tracks)
                self.store.set_telemetry(self.camera.id, telemetry)
                self.store.set_active_tracks(self.camera.id, tracks)
                self.gis_registry.update_vehicle_count(self.camera.id, len(tracks))

                # 4. Log significant observations to in-memory store and SQLite database
                for trk in tracks:
                    if frame_counter % 15 == 0:
                        obs = ObservationRecord(
                            observation_id=f"OBS-{uuid.uuid4().hex[:8].upper()}",
                            camera_id=self.camera.id,
                            timestamp=datetime.now().isoformat(),
                            track_id=trk.track_id,
                            vehicle_type=trk.class_name,
                            vehicle_color=trk.vehicle_color or "unknown",
                            plate_text=trk.plate_text,
                            plate_confidence=trk.plate_confidence,
                            confidence=trk.confidence,
                            bbox=[trk.bbox.x1, trk.bbox.y1, trk.bbox.x2, trk.bbox.y2],
                            speed_px_s=trk.speed_px_s,
                            heading=trk.heading_cardinal,
                            latitude=self.camera.latitude,
                            longitude=self.camera.longitude,
                            vehicle_crop_url=trk.vehicle_crop_url,
                            plate_crop_url=trk.plate_crop_url
                        )
                        self.store.record_observation(obs)
                        if self.sqlite_store:
                            self.sqlite_store.insert_observation(obs)

                # 4. Render Tactical HUD
                hud_frame = self.visualizer.draw_hud(frame, tracks, fps=self._measured_fps)

                # 5. JPEG Encode for live MJPEG streaming
                # Balance compression speed and quality (quality=75 for fast streaming)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                _, hud_buffer = cv2.imencode('.jpg', hud_frame, encode_param)
                hud_bytes = hud_buffer.tobytes()

                _, raw_buffer = cv2.imencode('.jpg', frame, encode_param)
                raw_bytes = raw_buffer.tobytes()

                with self._lock:
                    self._latest_hud_jpeg = hud_bytes
                    self._latest_raw_jpeg = raw_bytes

            except Exception as e:
                logger.error(f"Error in pipeline loop for {self.camera.id}: {e}", exc_info=True)
                time.sleep(0.1)

    def get_latest_jpeg(self, raw: bool = False) -> Optional[bytes]:
        with self._lock:
            return self._latest_raw_jpeg if raw else self._latest_hud_jpeg


class StreamManager:
    """
    Central orchestration service managing camera pipeline workers,
    life-cycle, and video MJPEG distribution.
    """
    def __init__(
        self,
        config: dict,
        store: MemoryObservationStore,
        gis_registry: CameraGISRegistry,
        sqlite_store: Optional[SQLiteObservationStore] = None
    ):
        self.config = config
        self.store = store
        self.sqlite_store = sqlite_store
        self.gis_registry = gis_registry
        self.workers: Dict[str, CameraPipelineWorker] = {}

        self._initialize_cameras()

    def _initialize_cameras(self):
        cameras_cfg = self.config.get("cameras", [])
        vision_cfg = self.config.get("vision", {})
        tracking_cfg = self.config.get("tracking", {})

        for cam_data in cameras_cfg:
            camera = CameraInfo(
                id=cam_data["id"],
                name=cam_data.get("name", cam_data["id"]),
                location=cam_data.get("location", "Urban Node"),
                latitude=cam_data.get("latitude", 13.0827),
                longitude=cam_data.get("longitude", 80.2707),
                source=str(cam_data.get("source", "synthetic")),
                source_type=str(cam_data.get("source_type", "synthetic")),
                fps=cam_data.get("fps", 25)
            )
            self.gis_registry.register_camera(camera)
            worker = CameraPipelineWorker(
                camera=camera,
                vision_cfg=vision_cfg,
                tracking_cfg=tracking_cfg,
                store=self.store,
                sqlite_store=self.sqlite_store,
                gis_registry=self.gis_registry
            )
            self.workers[camera.id] = worker

    def update_camera_source(self, camera_id: str, source: str, source_type: str = "synthetic") -> CameraInfo:
        camera = self.gis_registry.get_camera(camera_id)
        if not camera:
            raise KeyError(f"Camera '{camera_id}' not found.")

        if camera_id in self.workers:
            self.workers[camera_id].stop()
            del self.workers[camera_id]

        camera.source = source
        camera.source_type = source_type.lower()
        vision_cfg = self.config.get("vision", {})
        tracking_cfg = self.config.get("tracking", {})

        worker = CameraPipelineWorker(
            camera=camera,
            vision_cfg=vision_cfg,
            tracking_cfg=tracking_cfg,
            store=self.store,
            sqlite_store=self.sqlite_store,
            gis_registry=self.gis_registry
        )
        self.workers[camera_id] = worker
        worker.start()
        return camera

    def register_camera(self, camera: CameraInfo) -> CameraInfo:
        """Add a new camera feed to the live processing network.

        This is intentionally kept in the stream manager so a newly enrolled
        RTSP camera is immediately available to the dashboard, MJPEG endpoint,
        telemetry socket, and Tamil Nadu GIS layer without restarting the app.
        """
        if camera.id in self.workers:
            raise KeyError(f"Camera '{camera.id}' already exists.")

        self.gis_registry.register_camera(camera)
        worker = CameraPipelineWorker(
            camera=camera,
            vision_cfg=self.config.get("vision", {}),
            tracking_cfg=self.config.get("tracking", {}),
            store=self.store,
            sqlite_store=self.sqlite_store,
            gis_registry=self.gis_registry,
        )
        self.workers[camera.id] = worker
        worker.start()
        return camera

    def start_all(self):
        logger.info(f"Starting all {len(self.workers)} camera pipelines...")
        for worker in self.workers.values():
            worker.start()

    def stop_all(self):
        logger.info("Stopping all camera pipelines...")
        for worker in self.workers.values():
            worker.stop()

    def get_mjpeg_stream(self, camera_id: str, raw: bool = False) -> Generator[bytes, None, None]:
        """
        MJPEG generator yielding multipart HTTP stream frames.
        """
        worker = self.workers.get(camera_id)
        if not worker:
            return

        while True:
            frame_bytes = worker.get_latest_jpeg(raw=raw)
            if frame_bytes is not None:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
            time.sleep(0.035) # ~28-30 FPS streaming yield
