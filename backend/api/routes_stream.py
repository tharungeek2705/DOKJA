import asyncio
import os
import uuid
from pathlib import Path

import cv2
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from typing import Set

from backend.database.memory_store import MemoryObservationStore
from gis.camera_registry import CameraGISRegistry
from configs.logging_config import setup_logger
from vision.anpr.anpr_pipeline import ANPRPipeline
from vision.detection.yolo_detector import YOLOVehicleDetector
from backend.models.schemas import ObservationRecord

logger = setup_logger("routes_stream")
router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Active WebSocket connections
connected_websockets: Set[WebSocket] = set()

def process_uploaded_video(video_path: str, camera_id: str = "CAM-UPLOAD", max_frames: int = 40) -> dict:
    """Runs a lightweight ANPR scan on a user-uploaded local video file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Uploaded video not found: {video_path}")

    detector = YOLOVehicleDetector(model_path="yolov8n.pt", device="cpu", conf_threshold=0.35)
    anpr = ANPRPipeline(storage_dir=str(Path(__file__).resolve().parent.parent.parent / "data" / "crops"))
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open uploaded video: {video_path}")

    results = []
    frames_seen = 0
    max_plates = 10
    vehicle_detections = 0
    vehicle_classes = {}
    vehicle_records = []
    annotated_frames = []
    annotated_dir = Path(__file__).resolve().parent.parent.parent / "data" / "crops" / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    box_colors = {
        "car": (255, 93, 70), "two_wheeler": (30, 200, 80),
        "bus": (255, 170, 0), "truck": (200, 70, 255),
    }
    sqlite_store = None
    try:
        from backend.main import sqlite_store as app_sqlite_store
        sqlite_store = app_sqlite_store
    except Exception:
        pass

    while capture.isOpened() and frames_seen < max_frames:
        ret, frame = capture.read()
        if not ret or frame is None:
            break

        frames_seen += 1
        if frames_seen % 2 != 0:
            continue

        detections = detector.detect(frame)
        vehicle_detections += len(detections)
        annotated = frame.copy()
        for detection in detections:
            vehicle_classes[detection.class_name] = vehicle_classes.get(detection.class_name, 0) + 1
        for det_index, det in enumerate(detections[:12]):
            x1 = max(0, int(det.bbox.x1))
            y1 = max(0, int(det.bbox.y1))
            x2 = min(frame.shape[1], int(det.bbox.x2))
            y2 = min(frame.shape[0], int(det.bbox.y2))
            if x2 <= x1 or y2 <= y1:
                continue

            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            evidence_id = len(vehicle_records) + 1
            anpr_result = anpr.process(vehicle_crop, camera_id=camera_id, track_id=evidence_id, save_crops=True)
            color = anpr_result.vehicle_color if anpr_result else "unknown"
            plate = anpr_result.plate_text if anpr_result else ""
            plate_confidence = float(anpr_result.plate_confidence) if anpr_result else 0.0
            record = {
                "observation_id": f"UPLOAD-{uuid.uuid4().hex[:10].upper()}",
                "camera_id": camera_id,
                "vehicle_type": det.class_name,
                "vehicle_color": color,
                "plate_text": plate,
                "plate_confidence": round(plate_confidence, 3),
                "confidence": round(float(det.confidence), 3),
                "vehicle_crop_url": anpr_result.vehicle_crop_url if anpr_result else None,
                "plate_crop_url": anpr_result.plate_crop_url if anpr_result else None,
                "frame": frames_seen,
            }
            vehicle_records.append(record)
            label = f"{det.class_name.replace('_', ' ')} · {color} · {int(det.confidence * 100)}%"
            if plate:
                label += f" · {plate}"
            box_color = box_colors.get(det.class_name, (130, 130, 130))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

            if sqlite_store:
                sqlite_store.insert_observation(ObservationRecord(
                    observation_id=record["observation_id"], camera_id=camera_id,
                    timestamp=datetime.now().isoformat(), track_id=evidence_id,
                    vehicle_type=det.class_name, vehicle_color=color, plate_text=plate,
                    plate_confidence=plate_confidence, confidence=float(det.confidence),
                    bbox=[x1, y1, x2, y2], latitude=0.0, longitude=0.0,
                    vehicle_crop_url=record["vehicle_crop_url"], plate_crop_url=record["plate_crop_url"],
                ))

            if anpr_result and anpr_result.plate_text:
                results.append({
                    "camera_id": camera_id,
                    "plate_text": anpr_result.plate_text,
                    "plate_confidence": round(float(anpr_result.plate_confidence), 3),
                    "vehicle_color": anpr_result.vehicle_color,
                    "vehicle_crop_url": anpr_result.vehicle_crop_url,
                    "plate_crop_url": anpr_result.plate_crop_url,
                })
                if len(results) >= max_plates:
                    # Continue saving annotated frames and non-plate vehicle
                    # evidence, but avoid spending OCR time on more plates.
                    results = results[:max_plates]

        if detections and len(annotated_frames) < 12:
            frame_name = f"upload_{uuid.uuid4().hex[:10]}_frame_{frames_seen}.jpg"
            cv2.imwrite(str(annotated_dir / frame_name), annotated)
            annotated_frames.append(f"/crops/annotated/{frame_name}")

    capture.release()
    return {
        "status": "processed",
        "file_name": os.path.basename(video_path),
        "frames_processed": frames_seen,
        "plates": results,
        "vehicles": vehicle_records,
        "annotated_frames": annotated_frames,
        "summary": {
            "total_detections": len(results),
            "vehicle_detections": vehicle_detections,
            "vehicle_classes": vehicle_classes,
            "camera_id": camera_id,
        },
    }

def get_stream_manager():
    from backend.main import stream_manager
    return stream_manager

def get_store() -> MemoryObservationStore:
    from backend.main import memory_store
    return memory_store

def get_gis_registry() -> CameraGISRegistry:
    from backend.main import gis_registry
    return gis_registry

@router.get("/stream/{camera_id}/live")
def stream_camera_live(camera_id: str):
    """
    Streams live tactical video with HUD detection boxes and tracking trajectories
    via standard MJPEG multipart/x-mixed-replace.
    """
    sm = get_stream_manager()
    if camera_id not in sm.workers:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    return StreamingResponse(
        sm.get_mjpeg_stream(camera_id, raw=False),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/stream/{camera_id}/raw")
def stream_camera_raw(camera_id: str):
    """
    Streams raw unprocessed camera video.
    """
    sm = get_stream_manager()
    if camera_id not in sm.workers:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    return StreamingResponse(
        sm.get_mjpeg_stream(camera_id, raw=True),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.post("/upload/video")
async def upload_local_video(
    file: UploadFile = File(...),
    camera_id: str = Form("CAM-UPLOAD")
):
    """Uploads a local video file and runs a lightweight ANPR scan on it."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    allowed = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    name = file.filename.lower()
    if not name.endswith(allowed):
        raise HTTPException(status_code=400, detail="Unsupported video format. Use mp4, avi, mov, mkv, or webm.")

    target_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    with open(target_path, "wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    try:
        return process_uploaded_video(str(target_path), camera_id=camera_id)
    except Exception as exc:
        logger.error(f"Video upload processing failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@router.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """
    Real-time bidirectional WebSocket feed broadcasting active vehicle tracks,
    traffic telemetry, and camera status to the Operations Dashboard.
    """
    await websocket.accept()
    connected_websockets.add(websocket)
    logger.info(f"Dashboard client connected to telemetry WebSocket. Total: {len(connected_websockets)}")

    store = get_store()
    gis = get_gis_registry()

    try:
        while True:
            # Package state payload
            payload = {
                "cameras": [cam.model_dump() for cam in gis.list_cameras()],
                "telemetry": {
                    cam_id: telem.model_dump() if telem else None
                    for cam_id, telem in store.get_all_telemetry().items()
                },
                "active_tracks": {
                    cam_id: [t.model_dump() for t in store.get_active_tracks(cam_id)]
                    for cam_id in [c.id for c in gis.list_cameras()]
                },
                "recent_observations": [
                    obs.model_dump() for obs in store.get_recent_observations(limit=20)
                ]
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.5) # 2 Hz telemetry update rate
    except WebSocketDisconnect:
        connected_websockets.discard(websocket)
        logger.info(f"Dashboard client disconnected. Remaining: {len(connected_websockets)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connected_websockets.discard(websocket)
