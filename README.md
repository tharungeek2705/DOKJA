# DOKJA — AI-Powered Smart Traffic & Vehicle Intelligence Platform

> **Turn city-wide CCTV footage into searchable, explainable, real-time vehicle intelligence.**

---

## Overview

**DOKJA** is an AI-powered smart traffic management, vehicle intelligence, and investigation platform. Designed to convert passive CCTV / traffic-camera infrastructure into an active multi-camera perception network, DOKJA processes video feeds in real time, performs vehicle detection, tracks vehicles with persistent identities, analyzes traffic congestion, and provides a military-grade AI operations dashboard.

---

## Phase 1 Architecture (Implemented)

```text
CCTV / Recorded Video / Webcams / Synthetic Streams
                        │
                        ▼
                 Video Ingestion
          (VideoFrameReader / Reconnection)
                        │
                        ▼
                 Vehicle Detection
       (YOLOv8 — Car, Motorcycle, Bus, Truck)
                        │
                        ▼
              Multi-Object Tracking
      (ByteTrack + Persistent Trajectories)
                        │
                        ▼
             Traffic Telemetry Engine
      (Density, Class Breakdown, Flow Rates)
                        │
                        ▼
            FastAPI Ingestion & WebSockets
        (MJPEG Video Stream + 2Hz Telemetry)
                        │
                        ▼
         Tactical AI Operations Dashboard
       (Cyber HUD, Leaflet GIS Map, Matrix)
```

---

## Key Features in Phase 1

1. **Modular Vehicle Detection**: Swappable detector interface (`BaseVehicleDetector`) powered by pretrained YOLOv8 (`yolov8n.pt`).
2. **ByteTrack Multi-Object Tracking**: Persistent track IDs (`TRK-XXX`), instantaneous and smoothed velocity calculation, heading angle, and trajectory histories.
3. **Multi-Source Ingestion**: Resilient video capture supporting local video files (`.mp4`), RTSP streams, webcams, and built-in synthetic realistic traffic stream simulation for immediate multi-camera testing.
4. **Traffic Flow Analytics**: Real-time junction load scoring, density index (`Light`, `Moderate`, `Heavy`), flow rate (`vehicles/min`), and vehicle type distribution.
5. **Tactical Operations Dashboard**: Professional dark cyber-intelligence dashboard with:
   - Live multi-camera switching matrix (`CAM-01`, `CAM-02`, `CAM-03`)
   - High-definition video stream with toggleable tactical AI HUD (bounding boxes, trajectories, direction badges)
   - Interactive GIS City Map (Leaflet) with tactical radar markers and vehicle counts
   - Real-time traffic flow telemetry & class breakdown meters
   - Active track inspector
   - Chronological vehicle observation timeline

---

## Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU recommended (e.g., RTX 3050 6GB), CPU fully supported out of the box.

### 2. Install Dependencies
```bash
cd d:\Projects\dokja
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python tests/test_pipeline.py
```

### 4. Launch the Platform
```bash
python scripts/run_server.py
```
Open your browser at:
```text
http://localhost:8000
```

---

## API Endpoints

- `GET /` — Operations Dashboard UI
- `GET /health` — Platform health check and active vision engine metadata
- `GET /api/cameras` — List all registered camera nodes
- `GET /api/cameras/geojson` — GeoJSON feature collection for GIS mapping
- `GET /api/stream/{camera_id}/live` — Live MJPEG video stream with tactical HUD
- `GET /api/stream/{camera_id}/raw` — Raw MJPEG video stream without overlay
- `GET /api/analytics/telemetry` — Current traffic telemetry for all junctions
- `GET /api/analytics/tracks/{camera_id}` — Active vehicle tracks for camera
- `GET /api/analytics/observations` — Chronological detection event timeline
- `WebSocket /api/ws/telemetry` — Live real-time bidirectional telemetry socket

---

## Roadmap

- **Phase 1 (Completed)**: Video Ingestion, Vehicle Detection, ByteTrack Tracking, Live Operations Dashboard.
- **Phase 2 (Upcoming)**: ANPR / License Plate Recognition (PaddleOCR / EasyOCR) & Vehicle Observation Database.
- **Phase 3**: Vehicle Appearance Feature Extraction, Re-ID Embeddings & Cross-Camera Matching.
- **Phase 4**: Trajectory Reconstruction & GIS Map Replay.
- **Phase 5**: Natural Language & Multi-Modal Investigation Search Engine.
- **Phase 6**: Autonomous AI Investigation Agent (Tool orchestration, evidence reasoning).
- **Phase 7**: Predictive Traffic Analytics & Congestion Forecasting.
- **Phase 8**: Emergency Vehicle / Green Corridor Simulation.
- **Phase 9**: Multimodal Video LLM Analysis.
- **Phase 10**: Cloud, Docker & Scaled Edge Deployment.
