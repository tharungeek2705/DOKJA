import torch
import numpy as np
from typing import List, Dict, Optional
from ultralytics import YOLO

from backend.models.schemas import VehicleDetection, BoundingBox
from vision.detection.base_detector import BaseVehicleDetector
from configs.logging_config import setup_logger

logger = setup_logger("yolo_detector")

# Standard COCO Vehicle mapping
COCO_VEHICLE_MAP: Dict[int, str] = {
    1: "two_wheeler",
    2: "car",
    3: "two_wheeler",
    5: "bus",
    7: "truck"
}

class YOLOVehicleDetector(BaseVehicleDetector):
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or list(COCO_VEHICLE_MAP.keys())

        # Determine target device
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading YOLO model '{model_path}' on device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        logger.info(f"YOLO detector initialized. Tracking classes: {[COCO_VEHICLE_MAP.get(c, str(c)) for c in self.target_classes]}")

    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        """
        Runs object detection on a single frame and returns filtered vehicle detections.
        """
        if frame is None or frame.size == 0:
            return []

        try:
            results = self.model.predict(
                source=frame,
                classes=self.target_classes,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Inference error during detection: {e}")
            return []

        detections: List[VehicleDetection] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, classes):
            class_name = COCO_VEHICLE_MAP.get(int(cls_id), self.model.names.get(int(cls_id), "vehicle"))
            detections.append(
                VehicleDetection(
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=BoundingBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3])
                    )
                )
            )

        return detections

    def get_supported_classes(self) -> List[str]:
        return [COCO_VEHICLE_MAP.get(c, f"class_{c}") for c in self.target_classes]
