import torch
import numpy as np
from typing import List, Optional
from ultralytics import YOLO

from backend.models.schemas import VehicleTrack, BoundingBox
from vision.detection.yolo_detector import COCO_VEHICLE_MAP
from vision.tracking.track_history import TrackHistoryManager
from configs.logging_config import setup_logger

logger = setup_logger("vehicle_tracker")

class VehicleTracker:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        tracker_config: str = "bytetrack.yaml",
        max_trajectory_points: int = 50
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or list(COCO_VEHICLE_MAP.keys())
        self.tracker_config = tracker_config

        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing VehicleTracker with ByteTrack on {self.device}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.history_manager = TrackHistoryManager(
            max_points=max_trajectory_points,
            track_ttl_seconds=4.0
        )
        self._fallback_id_counter = 10000

    def track(self, frame: np.ndarray) -> List[VehicleTrack]:
        """
        Processes a video frame with ByteTrack, returning structured VehicleTrack objects
        with persistent IDs, velocity, heading, and trajectory histories.
        """
        if frame is None or frame.size == 0:
            return []

        try:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_config,
                classes=self.target_classes,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Error during tracking execution: {e}")
            return []

        active_tracks: List[VehicleTrack] = []
        if not results or len(results) == 0:
            self.history_manager.cleanup_stale_tracks()
            return active_tracks

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            self.history_manager.cleanup_stale_tracks()
            return active_tracks

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        track_ids = None
        if r.boxes.id is not None:
            track_ids = r.boxes.id.cpu().numpy().astype(int)

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
            if track_ids is not None and i < len(track_ids):
                track_id = int(track_ids[i])
            else:
                self._fallback_id_counter += 1
                track_id = self._fallback_id_counter

            class_name = COCO_VEHICLE_MAP.get(int(cls_id), self.model.names.get(int(cls_id), "vehicle"))
            bbox = BoundingBox(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3])
            )
            centroid = bbox.center

            # Update historical trajectory & calculate telemetry
            traj_record = self.history_manager.update_track(track_id, centroid)

            active_tracks.append(
                VehicleTrack(
                    track_id=track_id,
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=bbox,
                    centroid=centroid,
                    speed_px_s=round(traj_record.speed_px_s, 1),
                    direction_deg=round(traj_record.heading_deg, 1),
                    heading_cardinal=traj_record.cardinal,
                    dwell_time_seconds=round(traj_record.dwell_time, 1),
                    trajectory=traj_record.get_trajectory_points()
                )
            )

        self.history_manager.cleanup_stale_tracks()
        return active_tracks
