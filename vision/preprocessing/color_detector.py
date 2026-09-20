import cv2
import numpy as np
from typing import Tuple

class VehicleColorClassifier:
    """
    Classifies the dominant external paint color of a vehicle from a bounding box crop
    using HSV color space thresholding and region-of-interest sampling.
    """
    @staticmethod
    def classify_color(vehicle_crop: np.ndarray) -> Tuple[str, float]:
        """
        Analyzes vehicle body crop and returns (color_name, confidence).
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "unknown", 0.0

        vh, vw = vehicle_crop.shape[:2]

        # Sample the center body region (hood/roof area) to avoid road, tires, and windshields
        roi_x1 = int(vw * 0.20)
        roi_x2 = int(vw * 0.80)
        roi_y1 = int(vh * 0.25)
        roi_y2 = int(vh * 0.75)

        roi = vehicle_crop[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            roi = vehicle_crop

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0:
            return "unknown", 0.0

        # Define color masks in OpenCV HSV (H: 0-180, S: 0-255, V: 0-255)
        # 1. Black: Very low brightness
        mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 55]))

        # 2. White: Low saturation, high brightness
        mask_white = cv2.inRange(hsv, np.array([0, 0, 195]), np.array([180, 45, 255]))

        # 3. Silver / Gray: Low saturation, medium brightness
        mask_silver = cv2.inRange(hsv, np.array([0, 0, 56]), np.array([180, 50, 194]))

        # 4. Red: Wraps around 0 and 180
        mask_red1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 60, 60]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 5. Blue
        mask_blue = cv2.inRange(hsv, np.array([95, 60, 50]), np.array([135, 255, 255]))

        # 6. Yellow
        mask_yellow = cv2.inRange(hsv, np.array([20, 60, 70]), np.array([35, 255, 255]))

        # 7. Green
        mask_green = cv2.inRange(hsv, np.array([36, 50, 50]), np.array([85, 255, 255]))

        # 8. Orange
        mask_orange = cv2.inRange(hsv, np.array([11, 60, 70]), np.array([20, 255, 255]))

        color_counts = {
            "white": cv2.countNonZero(mask_white),
            "black": cv2.countNonZero(mask_black),
            "silver": cv2.countNonZero(mask_silver),
            "red": cv2.countNonZero(mask_red),
            "blue": cv2.countNonZero(mask_blue),
            "yellow": cv2.countNonZero(mask_yellow),
            "green": cv2.countNonZero(mask_green),
            "orange": cv2.countNonZero(mask_orange)
        }

        dominant_color = max(color_counts, key=color_counts.get)
        count = color_counts[dominant_color]
        confidence = float(count / total_pixels)

        # If highest count is negligible, default to silver/neutral
        if confidence < 0.12:
            return "silver", 0.40

        return dominant_color, round(min(1.0, confidence * 1.5), 2)
