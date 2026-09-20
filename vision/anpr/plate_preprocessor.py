import cv2
import numpy as np
from typing import Tuple

class PlatePreprocessor:
    """
    Image preprocessing pipeline for license plate crops to optimize OCR character recognition.
    Includes contrast equalization (CLAHE), edge-preserving denoising, and adaptive binarization.
    """
    @staticmethod
    def preprocess(img: np.ndarray, target_height: int = 70) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses a license plate candidate image.
        Returns (enhanced_grayscale, binary_thresholded).
        """
        if img is None or img.size == 0:
            blank = np.zeros((target_height, 140), dtype=np.uint8)
            return blank, blank

        # 1. Resize to standardized height maintaining aspect ratio
        h, w = img.shape[:2]
        aspect = w / max(1, h)
        target_width = int(target_height * aspect)
        resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        # 2. Convert to Grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        # 3. Bilateral Filter to remove noise while keeping edges sharp
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # 4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 5. Otsu's Adaptive Thresholding
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if text is white-on-dark (most plates are dark text on light background)
        # If mean intensity is low, background is dark, so invert
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)

        return enhanced, thresh
