import os
import cv2
import time
import uuid
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from vision.anpr.plate_detector import LicensePlateDetector
from vision.anpr.plate_preprocessor import PlatePreprocessor
from vision.anpr.ocr_engine import OCREngine
from vision.anpr.plate_normalizer import PlateNormalizer
from vision.preprocessing.color_detector import VehicleColorClassifier
from configs.logging_config import setup_logger

logger = setup_logger("anpr_pipeline")

@dataclass
class ANPRResult:
    plate_text: str
    plate_confidence: float
    is_valid_format: bool
    vehicle_color: str
    color_confidence: float
    plate_crop: Optional[np.ndarray] = None
    vehicle_crop_rel_path: Optional[str] = None
    plate_crop_rel_path: Optional[str] = None
    vehicle_crop_url: Optional[str] = None
    plate_crop_url: Optional[str] = None

class ANPRPipeline:
    """
    End-to-end Automatic Number Plate Recognition and vehicle appearance analytics pipeline.
    """
    def __init__(self, storage_dir: Optional[str] = None, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        self.ocr_engine = OCREngine.get_instance() if enable_ocr else None

        # Storage directories for evidence snapshots
        if storage_dir:
            self.base_crops_dir = Path(storage_dir)
        else:
            self.base_crops_dir = Path(__file__).resolve().parent.parent.parent / "data" / "crops"

        self.vehicles_dir = self.base_crops_dir / "vehicles"
        self.plates_dir = self.base_crops_dir / "plates"

        os.makedirs(self.vehicles_dir, exist_ok=True)
        os.makedirs(self.plates_dir, exist_ok=True)

    def process(
        self,
        vehicle_crop: np.ndarray,
        camera_id: str = "CAM-01",
        track_id: int = 0,
        save_crops: bool = True
    ) -> ANPRResult:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return ANPRResult(
                plate_text="",
                plate_confidence=0.0,
                is_valid_format=False,
                vehicle_color="unknown",
                color_confidence=0.0
            )

        # 1. Classify vehicle color
        v_color, color_conf = VehicleColorClassifier.classify_color(vehicle_crop)

        # 2. Localize plate candidate region
        plate_crop, plate_bbox = LicensePlateDetector.locate_plate(vehicle_crop)

        # 3. Preprocess plate candidate for OCR
        enhanced, thresh = PlatePreprocessor.preprocess(plate_crop)

        # 4. Run OCR inference
        raw_text, raw_conf = "", 0.0
        if self.enable_ocr and self.ocr_engine is not None:
            raw_text, raw_conf = self.ocr_engine.recognize_text(enhanced)
            if not raw_text:
                # Try thresholded version as fallback
                raw_text, raw_conf = self.ocr_engine.recognize_text(thresh)

        # 5. Normalize and validate plate
        norm_text, norm_conf, is_valid = PlateNormalizer.normalize_plate(raw_text, raw_conf)

        # 6. Save evidence snapshots if requested
        veh_crop_rel = None
        plate_crop_rel = None
        veh_crop_url = None
        plate_crop_url = None

        # Preserve every detected vehicle as evidence, including vehicles with
        # unreadable plates, so a later appearance search remains possible.
        if save_crops:
            token = f"{camera_id}_TRK{track_id}_{int(time.time()*1000)}"
            v_fname = f"{token}_veh.jpg"
            p_fname = f"{token}_plate.jpg"

            cv2.imwrite(str(self.vehicles_dir / v_fname), vehicle_crop)
            cv2.imwrite(str(self.plates_dir / p_fname), plate_crop)

            veh_crop_rel = f"crops/vehicles/{v_fname}"
            plate_crop_rel = f"crops/plates/{p_fname}"
            veh_crop_url = f"/crops/vehicles/{v_fname}"
            plate_crop_url = f"/crops/plates/{p_fname}"

        return ANPRResult(
            plate_text=norm_text,
            plate_confidence=norm_conf,
            is_valid_format=is_valid,
            vehicle_color=v_color,
            color_confidence=color_conf,
            plate_crop=plate_crop,
            vehicle_crop_rel_path=veh_crop_rel,
            plate_crop_rel_path=plate_crop_rel,
            vehicle_crop_url=veh_crop_url,
            plate_crop_url=plate_crop_url
        )
