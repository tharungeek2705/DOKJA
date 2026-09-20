from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


class VehicleReIDMatcher:
    """Lightweight vehicle Re-ID heuristic for the DOKJA MVP.

    This matcher intentionally uses explainable evidence rather than opaque embeddings,
    making it suitable for local development and for demonstration in a final-year project.
    """

    def _normalize_plate(self, plate: Optional[str]) -> str:
        if not plate:
            return ""
        return plate.strip().upper().replace("-", "").replace(" ", "")

    def _parse_ts(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _plate_similarity(self, plate_a: str, plate_b: str) -> float:
        if not plate_a or not plate_b:
            return 0.0
        if plate_a == plate_b:
            return 1.0
        if plate_a[:4] == plate_b[:4]:
            return 0.85
        if plate_a[:2] == plate_b[:2] and plate_a[-4:] == plate_b[-4:]:
            return 0.6
        return 0.0

    def _type_similarity(self, type_a: Optional[str], type_b: Optional[str]) -> float:
        if not type_a or not type_b:
            return 0.5
        return 1.0 if type_a.lower() == type_b.lower() else 0.3

    def _color_similarity(self, color_a: Optional[str], color_b: Optional[str]) -> float:
        if not color_a or not color_b:
            return 0.5
        if color_a.lower() == color_b.lower():
            return 1.0
        return 0.2

    def _size_similarity(self, size_a: Optional[Tuple[float, float]], size_b: Optional[Tuple[float, float]]) -> float:
        if size_a is None or size_b is None:
            return 0.5
        w1, h1 = size_a
        w2, h2 = size_b
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return 0.5
        width_ratio = 1.0 - abs(w1 - w2) / max(w1, w2)
        height_ratio = 1.0 - abs(h1 - h2) / max(h1, h2)
        return round(max(0.0, min(1.0, (width_ratio + height_ratio) / 2.0)), 2)

    def _time_similarity(self, ts_a: Optional[str], ts_b: Optional[str]) -> float:
        dt_a = self._parse_ts(ts_a)
        dt_b = self._parse_ts(ts_b)
        if dt_a is None or dt_b is None:
            return 0.5
        delta_seconds = abs((dt_a - dt_b).total_seconds())
        if delta_seconds <= 180:
            return 1.0
        if delta_seconds <= 3600:
            return max(0.0, 1.0 - (delta_seconds / 3600.0) * 0.5)
        return max(0.0, 1.0 - (delta_seconds / 86400.0) * 0.8)

    def match(self, obs_a: Dict[str, Any], obs_b: Dict[str, Any]) -> Dict[str, Any]:
        plate_a = self._normalize_plate(obs_a.get("plate_text"))
        plate_b = self._normalize_plate(obs_b.get("plate_text"))
        plate_score = self._plate_similarity(plate_a, plate_b)
        type_score = self._type_similarity(obs_a.get("vehicle_type"), obs_b.get("vehicle_type"))
        color_score = self._color_similarity(obs_a.get("vehicle_color"), obs_b.get("vehicle_color"))
        size_score = self._size_similarity(obs_a.get("bbox_size"), obs_b.get("bbox_size"))
        time_score = self._time_similarity(obs_a.get("timestamp"), obs_b.get("timestamp"))

        # Weighted score with explicit evidence-based combination.
        confidence = (
            0.42 * plate_score
            + 0.2 * type_score
            + 0.15 * color_score
            + 0.13 * size_score
            + 0.10 * time_score
        )
        confidence = max(0.0, min(1.0, confidence))

        evidence = []
        if plate_score >= 0.8:
            evidence.append("same plate text")
        elif plate_score > 0.0:
            evidence.append("compatible plate pattern")
        if type_score >= 0.8:
            evidence.append("same vehicle type")
        if color_score >= 0.8:
            evidence.append("compatible color")
        if size_score >= 0.7:
            evidence.append("similar vehicle geometry")
        if time_score >= 0.8:
            evidence.append("plausible travel time")

        if not evidence:
            evidence.append("limited visual evidence")

        if confidence >= 0.8:
            reason = "Likely match: " + "; ".join(evidence) + "."
        elif confidence >= 0.6:
            reason = "Possible match: " + "; ".join(evidence) + "."
        else:
            reason = "Unlikely match: " + "; ".join(evidence) + "."

        return {
            "score": round(confidence, 2),
            "reason": reason,
            "likely_match": confidence >= 0.6,
        }

    def rank_matches(self, source_obs: Dict[str, Any], candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for cand in candidates:
            score = self.match(source_obs, cand)
            result = dict(cand)
            result["match_score"] = score["score"]
            result["reason"] = score["reason"]
            result["likely_match"] = score["likely_match"]
            results.append(result)
        results.sort(key=lambda item: item["match_score"], reverse=True)
        return results
