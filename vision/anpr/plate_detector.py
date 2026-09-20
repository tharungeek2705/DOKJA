import cv2
import numpy as np
from typing import Optional, Tuple

class LicensePlateDetector:
    """
    Detects and localizes candidate license plate bounding regions within a vehicle crop
    using edge detection, morphological character grouping, and aspect-ratio contour filtering.
    """
    @staticmethod
    def locate_plate(vehicle_crop: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Locates the license plate in a vehicle crop.
        Returns: (plate_crop_image, (px1, py1, px2, py2)) relative to vehicle crop coordinates.
        """
        vh, vw = vehicle_crop.shape[:2]
        if vh < 40 or vw < 40:
            return vehicle_crop, (0, 0, vw, vh)

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

        # Focus predominantly on the lower 60% of the vehicle where plates reside
        roi_y_start = int(vh * 0.40)
        roi_gray = gray[roi_y_start:, :]
        roi_h, roi_w = roi_gray.shape

        # Morphological gradient to highlight vertical character edges
        grad_x = cv2.Sobel(roi_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        if max_val - min_val > 0:
            grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
        grad_x = grad_x.astype("uint8")

        # Blur and morphological close with horizontal kernel to group plate letters
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel)

        # Otsu thresholding
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find candidate contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_rect = None
        max_score = -1

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h) if h > 0 else 0
            area = w * h

            # Standard plates have aspect ratio roughly between 2.0 and 5.5
            if 2.0 <= aspect <= 6.0 and 600 <= area <= (roi_w * roi_h * 0.35):
                # Prefer candidates centered horizontally in the lower vehicle region
                center_dist = abs((x + w / 2) - (roi_w / 2)) / (roi_w / 2)
                score = area * (1.0 - 0.5 * center_dist)
                if score > max_score:
                    max_score = score
                    # Adjust y back to full vehicle crop coordinates
                    best_rect = (x, y + roi_y_start, x + w, y + roi_y_start + h)

        if best_rect is not None:
            px1, py1, px2, py2 = best_rect
            # Add padding
            pad_x = int((px2 - px1) * 0.08)
            pad_y = int((py2 - py1) * 0.15)
            px1 = max(0, px1 - pad_x)
            py1 = max(0, py1 - pad_y)
            px2 = min(vw, px2 + pad_x)
            py2 = min(vh, py2 + pad_y)
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            return plate_crop, (px1, py1, px2, py2)

        # Fallback heuristic: Center-bottom 25% of vehicle
        fx1 = int(vw * 0.20)
        fx2 = int(vw * 0.80)
        fy1 = int(vh * 0.60)
        fy2 = int(vh * 0.88)
        plate_crop = vehicle_crop[fy1:fy2, fx1:fx2]
        return plate_crop, (fx1, fy1, fx2, fy2)
