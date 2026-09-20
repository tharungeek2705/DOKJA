import sqlite3
import json
import threading
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from backend.models.schemas import ObservationRecord
from configs.logging_config import setup_logger

logger = setup_logger("sqlite_store")

class SQLiteObservationStore:
    """
    Persistent SQLite storage engine for vehicle observations, ANPR evidence,
    and audit trails. Thread-safe with query indexes for investigation searches.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path(__file__).resolve().parent.parent.parent / "data" / "dokja.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    vehicle_type TEXT NOT NULL,
                    vehicle_color TEXT DEFAULT 'unknown',
                    plate_text TEXT,
                    plate_confidence REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    bbox_json TEXT,
                    speed_px_s REAL DEFAULT 0.0,
                    heading TEXT DEFAULT 'N',
                    latitude REAL,
                    longitude REAL,
                    vehicle_crop_url TEXT,
                    plate_crop_url TEXT
                );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_plate ON observations(plate_text);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_cam ON observations(camera_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_time ON observations(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_track ON observations(track_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_color ON observations(vehicle_color);")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    plate_text TEXT PRIMARY KEY,
                    label TEXT NOT NULL DEFAULT 'Flagged vehicle',
                    reason TEXT NOT NULL DEFAULT '',
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    observation_id TEXT NOT NULL,
                    plate_text TEXT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(observation_id) REFERENCES observations(observation_id)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_status ON alerts(status, created_at);")

            conn.commit()
            conn.close()
            logger.info(f"SQLite Observation Database initialized at: {self.db_path}")

    def insert_observation(self, obs: ObservationRecord):
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            bbox_str = json.dumps(obs.bbox)
            cursor.execute("""
                INSERT OR REPLACE INTO observations (
                    observation_id, camera_id, timestamp, track_id, vehicle_type,
                    vehicle_color, plate_text, plate_confidence, confidence,
                    bbox_json, speed_px_s, heading, latitude, longitude,
                    vehicle_crop_url, plate_crop_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                obs.observation_id,
                obs.camera_id,
                obs.timestamp,
                obs.track_id,
                obs.vehicle_type,
                obs.vehicle_color,
                obs.plate_text,
                obs.plate_confidence,
                obs.confidence,
                bbox_str,
                obs.speed_px_s,
                obs.heading,
                obs.latitude,
                obs.longitude,
                obs.vehicle_crop_url,
                obs.plate_crop_url
            ))
            self._create_watchlist_alert(cursor, obs)
            conn.commit()
            conn.close()

    def _create_watchlist_alert(self, cursor: sqlite3.Cursor, obs: ObservationRecord):
        """Create one auditable alert for a confident match on the active watchlist."""
        plate = (obs.plate_text or "").strip().upper().replace("-", "").replace(" ", "")
        if not plate or obs.plate_confidence < 0.55:
            return
        cursor.execute("SELECT label, reason FROM watchlist WHERE plate_text = ? AND active = 1", (plate,))
        match = cursor.fetchone()
        if not match:
            return
        cursor.execute("SELECT 1 FROM alerts WHERE observation_id = ? AND alert_type = 'watchlist'", (obs.observation_id,))
        if cursor.fetchone():
            return
        import uuid
        message = f"{match['label']}: {plate} detected at {obs.camera_id}."
        if match["reason"]:
            message += f" Reason: {match['reason']}"
        cursor.execute(
            "INSERT INTO alerts (alert_id, observation_id, plate_text, alert_type, severity, message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"ALT-{uuid.uuid4().hex[:10].upper()}", obs.observation_id, plate, "watchlist", "high", message, datetime.now().isoformat()),
        )

    def add_watchlist_entry(self, plate_text: str, label: str = "Flagged vehicle", reason: str = "") -> Dict[str, Any]:
        plate = plate_text.strip().upper().replace("-", "").replace(" ", "")
        if not plate:
            raise ValueError("A vehicle number plate is required.")
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "INSERT OR REPLACE INTO watchlist (plate_text, label, reason, active, created_at) VALUES (?, ?, ?, 1, ?)",
                (plate, label.strip() or "Flagged vehicle", reason.strip(), datetime.now().isoformat()),
            )
            conn.commit()
            conn.close()
        return {"plate_text": plate, "label": label.strip() or "Flagged vehicle", "reason": reason.strip(), "active": True}

    def list_watchlist(self) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute("SELECT plate_text, label, reason, active, created_at FROM watchlist ORDER BY created_at DESC").fetchall()
            conn.close()
        return [dict(row) for row in rows]

    def list_alerts(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        query, params = "SELECT * FROM alerts", []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute(query, params).fetchall()
            conn.close()
        return [dict(row) for row in rows]

    def _row_to_record(self, row: sqlite3.Row) -> ObservationRecord:
        bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else [0, 0, 0, 0]
        return ObservationRecord(
            observation_id=row["observation_id"],
            camera_id=row["camera_id"],
            timestamp=row["timestamp"],
            track_id=row["track_id"],
            vehicle_type=row["vehicle_type"],
            vehicle_color=row["vehicle_color"],
            plate_text=row["plate_text"],
            plate_confidence=row["plate_confidence"],
            confidence=row["confidence"],
            bbox=bbox,
            speed_px_s=row["speed_px_s"],
            heading=row["heading"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            vehicle_crop_url=row["vehicle_crop_url"],
            plate_crop_url=row["plate_crop_url"]
        )

    def search_observations(
        self,
        plate_query: Optional[str] = None,
        camera_id: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        vehicle_color: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        min_plate_conf: float = 0.0,
        limit: int = 50,
        offset: int = 0
    ) -> List[ObservationRecord]:
        conditions = []
        params = []

        if plate_query:
            conditions.append("plate_text LIKE ?")
            params.append(f"%{plate_query.strip().upper()}%")

        if camera_id:
            conditions.append("camera_id = ?")
            params.append(camera_id)

        if vehicle_type:
            conditions.append("LOWER(vehicle_type) = LOWER(?)")
            params.append(vehicle_type)

        if vehicle_color:
            conditions.append("LOWER(vehicle_color) = LOWER(?)")
            params.append(vehicle_color)

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        if min_plate_conf > 0.0:
            conditions.append("plate_confidence >= ?")
            params.append(min_plate_conf)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT * FROM observations
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            records = [self._row_to_record(r) for r in rows]
            conn.close()
            return records

    def get_observation_by_id(self, obs_id: str) -> Optional[ObservationRecord]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM observations WHERE observation_id = ?", (obs_id,))
            row = cursor.fetchone()
            conn.close()
            return self._row_to_record(row) if row else None

    def get_recent_plates(self, limit: int = 25) -> List[ObservationRecord]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM observations
                WHERE plate_text IS NOT NULL AND plate_text != ''
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            records = [self._row_to_record(r) for r in rows]
            conn.close()
            return records
