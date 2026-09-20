from abc import ABC, abstractmethod
from typing import List
import numpy as np
from backend.models.schemas import VehicleDetection

class BaseVehicleDetector(ABC):
    """
    Abstract interface for vehicle detectors in DOKJA.
    Enables plug-and-play swapping of YOLOv8, YOLO11, RT-DETR, or custom models.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        """
        Processes a single BGR video frame and outputs a list of vehicle detections.
        """
        pass

    @abstractmethod
    def get_supported_classes(self) -> List[str]:
        """
        Returns list of detectable vehicle category names.
        """
        pass
