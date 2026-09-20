import torch
import numpy as np
from typing import Tuple, List, Optional
import easyocr

from configs.logging_config import setup_logger

logger = setup_logger("ocr_engine")

class OCREngine:
    """
    Lightweight OCR inference engine powered by EasyOCR.
    Initialized with English alphanumeric support and PyTorch acceleration.
    """
    _instance: Optional['OCREngine'] = None

    def __init__(self, gpu: Optional[bool] = None):
        use_gpu = torch.cuda.is_available() if gpu is None else gpu
        logger.info(f"Initializing EasyOCR Engine (GPU={use_gpu})...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        self.allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        logger.info("EasyOCR Engine initialized successfully.")

    @classmethod
    def get_instance(cls) -> 'OCREngine':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def recognize_text(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """
        Runs OCR on a preprocessed plate image.
        Returns: (extracted_text, average_confidence)
        """
        if plate_img is None or plate_img.size == 0:
            return "", 0.0

        try:
            results = self.reader.readtext(
                plate_img,
                allowlist=self.allowlist,
                paragraph=False,
                detail=1
            )

            if not results:
                return "", 0.0

            texts = []
            confs = []
            for bbox, text, conf in results:
                cleaned = text.strip()
                if cleaned:
                    texts.append(cleaned)
                    confs.append(float(conf))

            combined_text = "".join(texts)
            avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
            return combined_text, round(avg_conf, 2)

        except Exception as e:
            logger.error(f"Error during OCR extraction: {e}")
            return "", 0.0
