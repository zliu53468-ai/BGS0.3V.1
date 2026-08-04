"""BGS cMAB 預測績效紀錄器。

每次 cMAB 輸出後建立 pending prediction；使用者回報實際結果時：
- B/P：以 1 或 0 reward 更新被選擇的 Arm。
- T：和局不更新 B/P Arm。
- prediction_id 去重，避免同一局重複學習。
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

from contextual_bandit import update_bandit

BASE_DIR = Path(__file__).resolve().parent
_LOCK = RLock()
_OUTCOMES = ("B", "P", "T")
_GUARD_SECONDS = max(5, min(300, int(os.getenv("CMAB_DUPLICATE_GUARD_SECONDS", "90") or "90")))
PERFORMANCE_MAX_RECORDS = max(1000, min(200000, int(os.getenv("PERFORMANCE_MAX_RECORDS", "30000") or "30000")))


def _resolve_performance_file() -> Path:
    configured = Path(os.getenv("PERFORMANCE_DATA_FILE", str(BASE_DIR / "data" / "prediction_performance.json"))).expanduser()
    candidates = [configured, BASE_DIR / "data" / "prediction_performance.json", Path("/tmp/bgs_prediction_performance.json")]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / f".performance_write_test_{os.getpid()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            if candidate != configured:
                print(f"PERFORMANCE_DATA_FILE fallback: {configured} -> {candidate}")
            return candidate
        except OSError as exc:
            print(f"PERFORMANCE_DATA_FILE unavailable: {candidate}: {exc}")
    raise RuntimeError("No writable PERFORMANCE_DATA_FILE path is available")


PERFORMANCE_DATA_FILE = _resolve_performance_file()


def _now() -> int:
    return int(time.time())


def _uid_key(user_id: str) -> str:
    return sha256(str(user_id or "").encode("utf-8")).hexdigest()[:24]


def _normalize_probabilities(values: Mapping[str, Any]) -> Dict[str, float]:
    raw = {key: max(1e-12, float(values.get(key, 0.0) or 0.0)) for key in _OUTCOMES}
    total = sum(raw.values()) or 1.0
    return {key: raw[key] / total for key in _OUTCOMES}


def _prune_guards_unlocked(data: Dict[str, Any]) -> None:
    now = _now()
    guards = dict(data.get("duplicate_guards") or {})
    data["duplicate_guards"] = {
        uid: guard for uid, guard in guards.items()
        if isinstance(guard, Mapping) and now - int(guard.get("created_at", 0) or 0) <= _GUARD_SECONDS
    }


def _read_unlocked() -> Dict[str, Any]:
    try:
        data = json.loads(PERFORMANCE_DATA_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError
    except Exception:
        data = {}
    result = {
        "records": data.get("records") if isinstance(data.get("records"), list) else [],
        "pending": data.get("pending") if isinstance(data.get("pending"), dict) else {},
        "duplicate_guards": data.get("duplicate_guards") if isinstance(data.get("duplicate_guards"), dict) else {},
        "updated_at": int(data.get("updated_at", 0) or 0),
    }
    _prune_guards_unlocked(result)
    return result


def _write_unlocked(data: Dict[str, Any]) -> None:
    _prune_guards_unlocked(data)
    data["records"] = list(data.get("records") or [])[-PERFORMANCE_MAX_RECORDS:]
    data["updated_at"] = _now()
    PERFORMANCE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = PERFORMANCE_DATA_FILE.with_suffix(PERFORMANCE_DATA_FILE.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(PERFORMANCE_DATA_FILE)


def _prediction_probabilities(prediction: Mapping[str, Any]) -> Dict[str, float]:
    if isinstance(prediction.get("probabilities"), Mapping):
        return _normalize_probabilities(dict(prediction.get("probabilities") or {}))
    return _normalize_probabilities({
        "B": float(prediction.get("banker_rate", 0.0) or 0.0) / 100.0,
        "P": float(prediction.get("player_rate", 0.0) or 0.0) / 100.0,
        "T": float(prediction.get("tie_rate", 0.0) or 0.0) / 100.0,
    })


def record_prediction(user_id: str, prediction: Mapping[str, Any], *, venue: str = "", room: str = "",
                      metadata: Optional[Mapping[str, Any]] = None) -> str:
    uid = _uid_key(user_id)
    prediction_id = f"{_now():x}{secrets.token_hex(6)}"
    selected_arm = str(prediction.get("selected_arm") or prediction.get("action") or prediction.get("recommend") or "").upper().strip()
    record = {
        "prediction_id": prediction_id,
        "uid_key": uid,
        "created_at": _now(),
        "resolved_at": 0,
        "venue": str(venue or "").upper().strip(),
        "room": str(room or "").strip(),
        "model_version": str(prediction.get("model_version") or prediction.get("engine") or ""),
        "probabilities": _prediction_probabilities(prediction),
        "recommend": str(prediction.get("recommend") or selected_arm).upper(),
        "action": str(prediction.get("action") or selected_arm).upper(),
        "selected_arm": selected_arm,
        "context_vector": list(prediction.get("bandit_context") or prediction.get("context_vector") or []),
        "context_feature_names": list(prediction.get("context_feature_names") or []),
        "bandit_scores": dict(prediction.get("bandit_scores") or {}),
        "quality_score": float(prediction.get("quality_score", 0.0) or 0.0),
        "unknown_region_active": bool(prediction.get("unknown_region_active", False)),
        "uncertainty_braking": dict(prediction.get("uncertainty_braking") or {}),
        "short_term_trend_buffer": dict(prediction.get("short_term_trend_buffer") or {}),
        "few_shot_update_weight": float(prediction.get("few_shot_update_weight", 1.0) or 1.0),
        "metadata": dict(metadata or {}),
        "actual_outcome": "",
        "reward": None,
        "bandit_update": {},
    }
    with _LOCK:
        data = _read_unlocked()
        data["records"].append(record)
        data["pending"][uid] = prediction_id
        _write_unlocked(data)
    return prediction_id


def resolve_latest_prediction(user_id: str, actual_outcome: str, *, venue: str = "", room: str = "",
                              mark_duplicate_guard: bool = False) -> Optional[Dict[str, Any]]:
    actual = str(actual_outcome or "").upper().strip()
    if actual not in _OUTCOMES:
        raise ValueError("actual_outcome must be B, P or T")
    uid = _uid_key(user_id)
    with _LOCK:
        data = _read_unlocked()
        guard = dict(data["duplicate_guards"].get(uid) or {})
        if guard and str(guard.get("actual_outcome") or "") == actual and _now() - int(guard.get("created_at", 0) or 0) <= _GUARD_SECONDS:
            data["duplicate_guards"].pop(uid, None)
            _write_unlocked(data)
            return None

        pending_id = str(data["pending"].get(uid) or "")
        target: Optional[Dict[str, Any]] = None
        for record in reversed(data["records"]):
            if str(record.get("uid_key") or "") != uid or record.get("actual_outcome"):
                continue
            if pending_id and str(record.get("prediction_id") or "") != pending_id:
                continue
            target = record
            break

        if target is None:
            if mark_duplicate_guard:
                data["duplicate_guards"][uid] = {"actual_outcome": actual, "created_at": _now(), "resolved_prediction_id": ""}
                _write_unlocked(data)
            return None

        probabilities = _normalize_probabilities(dict(target.get("probabilities") or {}))
        selected_arm = str(target.get("selected_arm") or target.get("action") or target.get("recommend") or "").upper().strip()
        context = list(target.get("context_vector") or [])
        reward: Optional[float]
        if actual == "T":
            reward = None
        elif selected_arm in {"B", "P"}:
            reward = 1.0 if selected_arm == actual else 0.0
        else:
            reward = None

        update_weight = (
            max(1.0, min(12.0, float(target.get("few_shot_update_weight", 1.0) or 1.0)))
            if bool(target.get("unknown_region_active"))
            else 1.0
        )

        if reward is not None and context:
            bandit_update = update_bandit(
                user_id=user_id,
                context=context,
                selected_arm=selected_arm,
                reward=reward,
                event_id=str(target.get("prediction_id") or ""),
                actual_outcome=actual,
                update_weight=update_weight,
            )
        elif actual == "T":
            bandit_update = {"updated": False, "reason": "tie_not_used_for_bp_arms", "actual_outcome": actual}
        else:
            bandit_update = {"updated": False, "reason": "missing_context_or_arm", "actual_outcome": actual}

        target["actual_outcome"] = actual
        target["resolved_at"] = _now()
        target["reward"] = reward
        target["bandit_update"] = bandit_update
        target["applied_update_weight"] = (
            float(bandit_update.get("update_weight", 0.0) or 0.0)
            if isinstance(bandit_update, Mapping)
            else 0.0
        )
        target["few_shot_boost_applied"] = bool(
            isinstance(bandit_update, Mapping)
            and bandit_update.get("few_shot_boost_applied")
        )
        if venue:
            target["venue"] = str(venue).upper().strip()
        if room:
            target["room"] = str(room).strip()
        target["log_loss"] = -math.log(max(1e-12, probabilities[actual]))
        target["brier_score"] = sum((probabilities[key] - (1.0 if key == actual else 0.0)) ** 2 for key in _OUTCOMES)
        target["top1_correct"] = max(probabilities, key=probabilities.get) == actual
        target["action_correct"] = selected_arm == actual if actual in {"B", "P"} and selected_arm in {"B", "P"} else None
        data["pending"].pop(uid, None)
        if mark_duplicate_guard:
            data["duplicate_guards"][uid] = {
                "actual_outcome": actual,
                "created_at": _now(),
                "resolved_prediction_id": str(target.get("prediction_id") or ""),
            }
        _write_unlocked(data)
        return dict(target)


def get_resolved_records(*, venue: str = "", room: str = "", limit: int = 5000) -> List[Dict[str, Any]]:
    venue_key, room_key = str(venue or "").upper().strip(), str(room or "").strip()
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
    valid = [dict(record) for record in records if str(record.get("actual_outcome") or "") in _OUTCOMES]
    counts = {key: 0 for key in _OUTCOMES}
    brier_values: List[float] = []
    log_losses: List[float] = []
    decided = correct = 0
    rewards: List[float] = []
    unknown_region_count = 0
    boosted_update_count = 0
    for record in valid:
        actual = str(record.get("actual_outcome") or "").upper()
        counts[actual] += 1
        try:
            brier_values.append(float(record.get("brier_score")))
        except Exception:
            pass
        try:
            log_losses.append(float(record.get("log_loss")))
        except Exception:
            pass
        if record.get("action_correct") is not None:
            decided += 1
            correct += int(bool(record.get("action_correct")))
        if record.get("reward") is not None:
            rewards.append(float(record.get("reward") or 0.0))
        unknown_region_count += int(bool(record.get("unknown_region_active")))
        boosted_update_count += int(bool(record.get("few_shot_boost_applied")))
    total = max(1, len(valid))
    return {
        "sample_count": len(valid),
        "outcome_counts": counts,
        "empirical_probabilities": {key: counts[key] / total for key in _OUTCOMES},
        "decision_count": decided,
        "correct_count": correct,
        "accuracy": correct / max(1, decided),
        "mean_reward": sum(rewards) / max(1, len(rewards)),
        "unknown_region_count": unknown_region_count,
        "few_shot_boosted_update_count": boosted_update_count,
        "mean_brier_score": sum(brier_values) / len(brier_values) if brier_values else None,
        "mean_log_loss": sum(log_losses) / len(log_losses) if log_losses else None,
        "component_brier_scores": {},
        "component_sample_counts": {},
        "model": "CMAB-LINUCB-V1",
    }


def get_performance_summary(*, venue: str = "", room: str = "", limit: int = 5000) -> Dict[str, Any]:
    return summarize_records(get_resolved_records(venue=venue, room=room, limit=limit))


__all__ = ["get_performance_summary", "get_resolved_records", "record_prediction",
           "resolve_latest_prediction", "summarize_records"]
