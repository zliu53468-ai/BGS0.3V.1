"""BGS 預測績效紀錄器。

每次模型輸出後先建立 pending prediction；使用者回報下一個實際結果 B/P/T
時，再把該筆預測結算。校準與自適應權重只讀取已結算資料，避免使用未來資訊。
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Sequence
import json
import math
import os
import secrets
import time


BASE_DIR = Path(__file__).resolve().parent
PERFORMANCE_DATA_FILE = Path(
    os.getenv(
        "PERFORMANCE_DATA_FILE",
        str(BASE_DIR / "data" / "prediction_performance.json"),
    )
)
PERFORMANCE_MAX_RECORDS = max(
    1000,
    min(200_000, int(os.getenv("PERFORMANCE_MAX_RECORDS", "30000") or "30000")),
)
_LOCK = RLock()
_OUTCOMES = ("B", "P", "T")


def _now() -> int:
    return int(time.time())


def _uid_key(user_id: str) -> str:
    return sha256(str(user_id or "").encode("utf-8")).hexdigest()[:24]


def _normalize_probabilities(values: Mapping[str, Any]) -> Dict[str, float]:
    raw = {key: max(1e-12, float(values.get(key, 0.0) or 0.0)) for key in _OUTCOMES}
    total = sum(raw.values())
    if total <= 0:
        return {"B": 0.458597, "P": 0.446247, "T": 0.095156}
    return {key: raw[key] / total for key in _OUTCOMES}


def _read_unlocked() -> Dict[str, Any]:
    try:
        with PERFORMANCE_DATA_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError
    except Exception:
        data = {}
    records = data.get("records")
    pending = data.get("pending")
    return {
        "records": records if isinstance(records, list) else [],
        "pending": pending if isinstance(pending, dict) else {},
        "updated_at": int(data.get("updated_at", 0) or 0),
    }


def _write_unlocked(data: Dict[str, Any]) -> None:
    PERFORMANCE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["records"] = list(data.get("records") or [])[-PERFORMANCE_MAX_RECORDS:]
    data["updated_at"] = _now()
    temporary = PERFORMANCE_DATA_FILE.with_suffix(PERFORMANCE_DATA_FILE.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary.replace(PERFORMANCE_DATA_FILE)


def _component_probabilities(prediction: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    components: Dict[str, Dict[str, float]] = {}
    candidates = {
        "main_final": prediction.get("probabilities"),
        "raw_core": prediction.get("raw_probabilities"),
        "hypergeometric": prediction.get("hypergeometric_probabilities"),
        "monte_carlo": prediction.get("monte_carlo_probabilities"),
        "particle": prediction.get("particle_probabilities"),
        "sequence": prediction.get("sequence_probabilities"),
        "core_before_road": prediction.get("core_probabilities_before_road"),
    }
    for name, values in candidates.items():
        if isinstance(values, Mapping):
            try:
                components[name] = _normalize_probabilities(values)
            except Exception:
                continue
    return components


def record_prediction(
    user_id: str,
    prediction: Mapping[str, Any],
    *,
    venue: str = "",
    room: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """保存等待實際結果的預測，回傳 prediction_id。"""
    uid = _uid_key(user_id)
    final_probs = _normalize_probabilities(
        prediction.get("calibrated_probabilities")
        if isinstance(prediction.get("calibrated_probabilities"), Mapping)
        else prediction.get("probabilities")
        if isinstance(prediction.get("probabilities"), Mapping)
        else {
            "B": float(prediction.get("banker_rate", 0.0) or 0.0) / 100.0,
            "P": float(prediction.get("player_rate", 0.0) or 0.0) / 100.0,
            "T": float(prediction.get("tie_rate", 0.0) or 0.0) / 100.0,
        }
    )
    prediction_id = f"{_now():x}{secrets.token_hex(6)}"
    record = {
        "prediction_id": prediction_id,
        "uid_key": uid,
        "created_at": _now(),
        "resolved_at": 0,
        "venue": str(venue or "").upper().strip(),
        "room": str(room or "").strip(),
        "model_version": str(prediction.get("model_version") or prediction.get("engine") or ""),
        "probabilities": final_probs,
        "raw_probabilities": _normalize_probabilities(
            prediction.get("raw_probabilities")
            if isinstance(prediction.get("raw_probabilities"), Mapping)
            else prediction.get("probabilities")
            if isinstance(prediction.get("probabilities"), Mapping)
            else final_probs
        ),
        "components": _component_probabilities(prediction),
        "recommend": str(prediction.get("recommend") or "").upper(),
        "action": str(prediction.get("action") or "O").upper(),
        "quality_score": float(prediction.get("quality_score", 0.0) or 0.0),
        "calibration": dict(prediction.get("calibration") or {}),
        "adaptive_ensemble": dict(prediction.get("adaptive_ensemble") or {}),
        "metadata": dict(metadata or {}),
        "actual_outcome": "",
    }
    with _LOCK:
        data = _read_unlocked()
        data["records"].append(record)
        data["pending"][uid] = prediction_id
        _write_unlocked(data)
    return prediction_id


def resolve_latest_prediction(
    user_id: str,
    actual_outcome: str,
    *,
    venue: str = "",
    room: str = "",
) -> Optional[Dict[str, Any]]:
    """用最新實際 B/P/T 結算該 UID 尚未結算的上一筆預測。"""
    actual = str(actual_outcome or "").upper().strip()
    if actual not in _OUTCOMES:
        raise ValueError("actual_outcome must be B, P or T")
    uid = _uid_key(user_id)
    with _LOCK:
        data = _read_unlocked()
        pending_id = str(data["pending"].get(uid) or "")
        target: Optional[Dict[str, Any]] = None
        for record in reversed(data["records"]):
            if str(record.get("uid_key") or "") != uid:
                continue
            if record.get("actual_outcome"):
                continue
            if pending_id and str(record.get("prediction_id") or "") != pending_id:
                continue
            target = record
            break
        if target is None:
            return None

        probs = _normalize_probabilities(dict(target.get("probabilities") or {}))
        target["actual_outcome"] = actual
        target["resolved_at"] = _now()
        if venue:
            target["venue"] = str(venue).upper().strip()
        if room:
            target["room"] = str(room).strip()
        target["log_loss"] = -math.log(max(1e-12, probs[actual]))
        target["brier_score"] = sum(
            (probs[key] - (1.0 if key == actual else 0.0)) ** 2
            for key in _OUTCOMES
        )
        target["top1_correct"] = max(probs, key=probs.get) == actual
        action = str(target.get("action") or "O").upper()
        target["action_correct"] = action == actual if action in _OUTCOMES else None
        data["pending"].pop(uid, None)
        _write_unlocked(data)
        return dict(target)


def get_resolved_records(
    *,
    venue: str = "",
    room: str = "",
    limit: int = 5000,
) -> List[Dict[str, Any]]:
    venue_key = str(venue or "").upper().strip()
    room_key = str(room or "").strip()
    with _LOCK:
        records = list(_read_unlocked().get("records") or [])
    result: List[Dict[str, Any]] = []
    for record in reversed(records):
        if str(record.get("actual_outcome") or "") not in _OUTCOMES:
            continue
        if venue_key and str(record.get("venue") or "").upper() != venue_key:
            continue
        if room_key and str(record.get("room") or "") != room_key:
            continue
        result.append(dict(record))
        if len(result) >= max(1, int(limit)):
            break
    result.reverse()
    return result


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sample_count = len(records)
    counts = {key: 0 for key in _OUTCOMES}
    mean_predicted = {key: 0.0 for key in _OUTCOMES}
    brier_values: List[float] = []
    log_losses: List[float] = []
    component_scores: Dict[str, List[float]] = {}

    for record in records:
        actual = str(record.get("actual_outcome") or "").upper()
        if actual not in _OUTCOMES:
            continue
        counts[actual] += 1
        probs = _normalize_probabilities(dict(record.get("probabilities") or {}))
        for key in _OUTCOMES:
            mean_predicted[key] += probs[key]
        brier_values.append(sum(
            (probs[key] - (1.0 if key == actual else 0.0)) ** 2
            for key in _OUTCOMES
        ))
        log_losses.append(-math.log(max(1e-12, probs[actual])))
        for name, component in dict(record.get("components") or {}).items():
            if not isinstance(component, Mapping):
                continue
            component_probs = _normalize_probabilities(component)
            score = sum(
                (component_probs[key] - (1.0 if key == actual else 0.0)) ** 2
                for key in _OUTCOMES
            )
            component_scores.setdefault(str(name), []).append(score)

    valid = max(1, sum(counts.values()))
    empirical = {key: counts[key] / valid for key in _OUTCOMES}
    mean_predicted = {key: mean_predicted[key] / valid for key in _OUTCOMES}
    return {
        "sample_count": sample_count,
        "outcome_counts": counts,
        "empirical_probabilities": empirical,
        "mean_predicted_probabilities": mean_predicted,
        "brier_score": sum(brier_values) / len(brier_values) if brier_values else None,
        "log_loss": sum(log_losses) / len(log_losses) if log_losses else None,
        "component_brier_scores": {
            name: sum(values) / len(values)
            for name, values in component_scores.items()
            if values
        },
        "component_sample_counts": {
            name: len(values) for name, values in component_scores.items()
        },
    }


def get_performance_summary(
    *,
    venue: str = "",
    room: str = "",
    limit: int = 5000,
) -> Dict[str, Any]:
    return summarize_records(get_resolved_records(venue=venue, room=room, limit=limit))


__all__ = [
    "get_performance_summary",
    "get_resolved_records",
    "record_prediction",
    "resolve_latest_prediction",
    "summarize_records",
]
