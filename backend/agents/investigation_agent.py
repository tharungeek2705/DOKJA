from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.database.sqlite_store import SQLiteObservationStore
from vision.reid.reid_matcher import VehicleReIDMatcher


class InvestigationAgent:
    """A lightweight orchestration layer for DOKJA investigations.

    It translates simple natural-language investigation queries into structured SQLite lookups,
    then adds explainable reasoning about the evidence returned.
    """

    def __init__(self, store: SQLiteObservationStore):
        self.store = store
        self.reid_matcher = VehicleReIDMatcher()

    def execute(self, query: str) -> Dict[str, Any]:
        text = (query or "").strip()
        intent = self._detect_intent(text)
        params = self._build_params(text, intent)
        evidence = self.store.search_observations(**params)

        if intent == "last_known_location":
            return self._last_known_location_result(text, evidence)

        if intent == "first_seen":
            return self._first_seen_result(text, evidence)

        if intent == "trajectory":
            route = sorted(evidence, key=lambda obs: obs.timestamp)
            return {
                "intent": intent,
                "query": text,
                "evidence": [obs.model_dump() for obs in route[:20]],
                "summary": self._summarize_route(route, params.get("plate_query")),
                "inference": "Trajectory reconstructed from camera observations ordered by time; this is evidence-based and not a claim of certainty beyond the stored observations.",
                "confidence": 0.82 if route else 0.0,
                "route": [obs.model_dump() for obs in route[:20]],
            }

        if intent == "vehicle_search":
            return {
                "intent": intent,
                "query": text,
                "evidence": [obs.model_dump() for obs in evidence[:20]],
                "summary": self._generic_summary(text, evidence),
                "inference": "The agent searched the verified observation database for matching plate and camera evidence.",
                "confidence": 0.86 if evidence else 0.0,
            }

        return {
            "intent": intent,
            "query": text,
            "evidence": [obs.model_dump() for obs in evidence[:20]],
            "summary": self._generic_summary(text, evidence),
            "inference": "The system returned the most relevant observations for the request using structured retrieval and timestamp filters.",
            "confidence": 0.75 if evidence else 0.0,
        }

    def _detect_intent(self, text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ["last known location", "last seen", "latest confirmed location", "where is this vehicle now"]):
            return "last_known_location"
        if any(token in lowered for token in ["first seen", "earliest observation", "where was this vehicle first seen"]):
            return "first_seen"
        if any(token in lowered for token in ["trajectory", "route", "trace", "show the route", "reconstruct"]):
            return "trajectory"
        if any(token in lowered for token in ["match", "similar", "reid", "possible matches"]):
            return "vehicle_search"
        return "vehicle_search"

    def _build_params(self, text: str, intent: str) -> Dict[str, Any]:
        plate = None
        camera_id = None
        vehicle_type = None
        color = None
        start_time = None
        end_time = None

        text_upper = text.upper()
        tokens = [part.strip() for part in text.split() if part.strip()]
        for token in tokens:
            upper = token.upper()
            cleaned_upper = re.sub(r'[^A-Z0-9]', '', upper)
            if upper.startswith("CAM"):
                camera_id = upper
            if cleaned_upper.startswith("TN") or (len(cleaned_upper) >= 4 and any(ch.isdigit() for ch in cleaned_upper) and not upper.startswith("CAM")):
                plate = cleaned_upper
            if token.lower() in {"car", "bus", "truck", "motorcycle"}:
                vehicle_type = token.lower()
            if token.lower() in {"white", "black", "silver", "red", "blue", "yellow", "green", "orange"}:
                color = token.lower()

        time_match = re.search(r"(\d{1,2})\s*(?:pm|am)", text_lower := text.lower())
        if time_match:
            start_time = self._time_to_iso(time_match.group(1), is_pm=("pm" in text_lower))
        end_match = re.search(r"to\s+(\d{1,2})\s*(?:pm|am)", text_lower)
        if end_match:
            end_time = self._time_to_iso(end_match.group(1), is_pm=("pm" in text_lower))

        return {
            "plate_query": plate,
            "camera_id": camera_id,
            "vehicle_type": vehicle_type,
            "vehicle_color": color,
            "start_time": start_time,
            "end_time": end_time,
            "min_plate_conf": 0.0,
            "limit": 20,
            "offset": 0,
        }

    def _time_to_iso(self, hour: str, is_pm: bool) -> str:
        try:
            h = int(hour)
        except ValueError:
            return None
        if is_pm and h < 12:
            h += 12
        if not is_pm and h == 12:
            h = 0
        return datetime.today().replace(hour=h, minute=0, second=0, microsecond=0).isoformat()

    def _last_known_location_result(self, query: str, evidence: List[Any]) -> Dict[str, Any]:
        if not evidence:
            return {
                "intent": "last_known_location",
                "query": query,
                "evidence": [],
                "summary": "No verified observations matched the query, so there is no confirmed last-known location.",
                "last_seen": None,
                "inference": "No evidence was found in the structured database for the requested vehicle.",
                "confidence": 0.0,
            }

        latest = max(evidence, key=lambda obs: obs.timestamp)
        summary = (
            f"Last seen for {latest.plate_text or 'the vehicle'} was at camera {latest.camera_id} "
            f"at {latest.timestamp}. The latest confirmed location is {latest.latitude}, {latest.longitude}."
        )
        return {
            "intent": "last_known_location",
            "query": query,
            "evidence": [obs.model_dump() for obs in evidence[:20]],
            "summary": summary,
            "last_seen": {
                "camera_id": latest.camera_id,
                "timestamp": latest.timestamp,
                "latitude": latest.latitude,
                "longitude": latest.longitude,
                "plate_text": latest.plate_text,
                "vehicle_color": latest.vehicle_color,
                "confidence": latest.plate_confidence if latest.plate_text else latest.confidence,
            },
            "inference": "The latest observation in the verified database is treated as the current best evidence for the last known location.",
            "confidence": 0.9 if latest.plate_confidence else 0.8,
        }

    def _first_seen_result(self, query: str, evidence: List[Any]) -> Dict[str, Any]:
        if not evidence:
            return {
                "intent": "first_seen",
                "query": query,
                "evidence": [],
                "summary": "No verified observations matched the request, so the first-seen evidence is unavailable.",
                "first_seen": None,
                "inference": "No evidence was returned by the structured query.",
                "confidence": 0.0,
            }

        first = min(evidence, key=lambda obs: obs.timestamp)
        summary = (
            f"The earliest confirmed sighting for {first.plate_text or 'the vehicle'} was at camera {first.camera_id} "
            f"at {first.timestamp}."
        )
        return {
            "intent": "first_seen",
            "query": query,
            "evidence": [obs.model_dump() for obs in evidence[:20]],
            "summary": summary,
            "first_seen": {
                "camera_id": first.camera_id,
                "timestamp": first.timestamp,
                "latitude": first.latitude,
                "longitude": first.longitude,
                "plate_text": first.plate_text,
            },
            "inference": "The earliest timestamped observation in the verified database is taken as the first confirmed sighting.",
            "confidence": 0.85 if first.plate_confidence else 0.75,
        }

    def _generic_summary(self, query: str, evidence: List[Any]) -> str:
        if not evidence:
            return f"No verified observations matched the query '{query}'."
        latest = max(evidence, key=lambda obs: obs.timestamp)
        return (
            f"The database returned {len(evidence)} evidence records for '{query}'. "
            f"The most recent confirmed sighting was at camera {latest.camera_id} at {latest.timestamp}."
        )

    def _summarize_route(self, route: List[Any], plate: Optional[str]) -> str:
        if not route:
            return "No route evidence found for the selected vehicle."
        first = min(route, key=lambda obs: obs.timestamp)
        last = max(route, key=lambda obs: obs.timestamp)
        return (
            f"Vehicle {plate or 'tracking target'} was observed across {len(route)} records from "
            f"{first.camera_id} to {last.camera_id}, spanning {first.timestamp} to {last.timestamp}."
        )

    def find_matches(self, observations: List[Dict[str, Any]], threshold: float = 0.6) -> List[Dict[str, Any]]:
        if not observations:
            return []
        source = observations[0]
        matches = []
        for candidate in observations[1:]:
            result = self.reid_matcher.match(source, candidate)
            if result["score"] >= threshold:
                matches.append({**candidate, "match_score": result["score"], "reason": result["reason"]})
        matches.sort(key=lambda item: item["match_score"], reverse=True)
        return matches
