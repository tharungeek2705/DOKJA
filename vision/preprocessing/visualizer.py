import cv2
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict
from backend.models.schemas import VehicleTrack

# Tactical Cyber Color Palette (BGR)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (255, 191, 0),        # Cyan / Electric Blue
    "motorcycle": (50, 255, 120),# Neon Green
    "bus": (0, 215, 255),        # Amber Gold
    "truck": (255, 80, 180),     # Magenta / Pink
    "vehicle": (200, 200, 200)   # Neutral Gray
}

class TacticalHUDVisualizer:
    """
    Renders military/intelligence grade Tactical HUD overlays on video frames,
    including vehicle bounding boxes, persistent track IDs, velocity vectors,
    trajectory motion paths, and live telemetry telemetry headers.
    """
    def __init__(self, camera_id: str, camera_name: str = ""):
        self.camera_id = camera_id
        self.camera_name = camera_name

    def draw_corner_rect(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 2, corner_len: int = 14):
        x1, y1 = pt1
        x2, y2 = pt2

        # Draw semi-transparent bounding box outline
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Draw highlighted reinforced corners
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)

    def draw_hud(
        self,
        frame: np.ndarray,
        tracks: List[VehicleTrack],
        fps: float = 25.0,
        show_trajectories: bool = True
    ) -> np.ndarray:
        output = frame.copy()
        h, w, _ = output.shape

        # 1. Top HUD Bar
        hud_bar_height = 42
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_bar_height), (15, 18, 24), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.line(output, (0, hud_bar_height), (w, hud_bar_height), (0, 230, 255), 1)

        # Header Info
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cam_label = f"DOKJA // {self.camera_id} [{self.camera_name or 'TRAFFIC NODE'}]"
        cv2.putText(output, cam_label, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA)

        telemetry_label = f"SYS STATUS: LIVE | FPS: {fps:.1f} | TRACKS: {len(tracks)} | TIME: {time_str}"
        t_size = cv2.getTextSize(telemetry_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        cv2.putText(output, telemetry_label, (w - t_size[0] - 16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 225, 230), 1, cv2.LINE_AA)

        # 2. Render Trajectories and Vehicle Tracks
        for track in tracks:
            color = CLASS_COLORS.get(track.class_name.lower(), CLASS_COLORS["vehicle"])
            x1, y1 = int(track.bbox.x1), int(track.bbox.y1)
            x2, y2 = int(track.bbox.x2), int(track.bbox.y2)

            # Draw trajectory path
            if show_trajectories and len(track.trajectory) > 1:
                points = track.trajectory
                for idx in range(1, len(points)):
                    p1 = (int(points[idx - 1][0]), int(points[idx - 1][1]))
                    p2 = (int(points[idx][0]), int(points[idx][1]))
                    alpha = idx / len(points)
                    traj_color = (
                        int(color[0] * alpha),
                        int(color[1] * alpha),
                        int(color[2] * alpha)
                    )
                    thickness = max(1, int(3 * alpha))
                    cv2.line(output, p1, p2, traj_color, thickness, cv2.LINE_AA)

            # Draw Tactical Bounding Box
            self.draw_corner_rect(output, (x1, y1), (x2, y2), color, thickness=2, corner_len=12)

            # Centroid Reticle
            cx, cy = int(track.centroid[0]), int(track.centroid[1])
            cv2.circle(output, (cx, cy), 3, color, -1, cv2.LINE_AA)

            # Label Badge with Vehicle Color & License Plate
            color_prefix = f"{track.vehicle_color.upper()} " if track.vehicle_color and track.vehicle_color != 'unknown' else ""
            if track.plate_text:
                label = f"TRK-{track.track_id} | {color_prefix}{track.class_name.upper()} | {track.plate_text} [{int(track.plate_confidence * 100)}%]"
                badge_border_color = (0, 240, 255) # Electric gold/cyan highlight for plates
            else:
                label = f"TRK-{track.track_id} | {color_prefix}{track.class_name.upper()} {int(track.confidence * 100)}%"
                badge_border_color = color

            speed_label = f"{int(track.speed_px_s)} px/s [{track.heading_cardinal}]"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            badge_y1 = max(hud_bar_height + 4, y1 - lh - 8)
            badge_y2 = badge_y1 + lh + 6

            # Background for label badge
            cv2.rectangle(output, (x1, badge_y1), (x1 + lw + 8, badge_y2), (18, 22, 30), -1)
            cv2.rectangle(output, (x1, badge_y1), (x1 + lw + 8, badge_y2), badge_border_color, 1)
            cv2.putText(output, label, (x1 + 4, badge_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, badge_border_color, 1, cv2.LINE_AA)

            # Sub-badge with velocity/direction
            if track.speed_px_s > 4:
                (sw, sh), _ = cv2.getTextSize(speed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                sub_y1 = badge_y2 + 2
                sub_y2 = sub_y1 + sh + 4
                cv2.rectangle(output, (x1, sub_y1), (x1 + sw + 6, sub_y2), (10, 14, 20), -1)
                cv2.putText(output, speed_label, (x1 + 3, sub_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

        return output
