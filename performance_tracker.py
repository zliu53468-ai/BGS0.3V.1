"""BGS cMAB 預測績效紀錄器。

每次 cMAB 輸出後建立 pending prediction；使用者回報實際結果時：
- B/P：使用預測當下的 cMAB B/P 分數產生信心加權獎懲。
- T：和局不更新 B/P Arm。
- prediction_id 去重，避免同一局重複學習。
- 同時保存 raw round 與 B/P round index，避免和局造成時間軸錯位。
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Sequence
import json
import math
import secrets
import time

from contextual_bandit import update_bandit

BASE_DIR = Path(__file__).resolve().parent
_LOCK = RLock()
_OUTCOMES = ("B", "P", "T")
# 這些是新 cMAB 版本的固定資料保護參數；不受 Render 舊環境變數覆寫。
_GUARD_SECONDS = 90
PERFORMANCE_MAX_RECORDS = 30000


def _resolve_performance_file() -> Path:
    """使用 V3 績效檔，保留舊資料作為歷史稽核而不混入新模型。"""
    configured = Path("/var/data/prediction_performance_v3.json")
    candidates = [
        configured,
        BASE_DIR / "data" / "prediction_performance_v3.json",
        Path("/tmp/bgs_prediction_performance_v3.json"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / f".performance_write_test_{time.time_ns()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            if candidate != configured:
                print(f"Performance V3 fallback: {configured} -> {candidate}")
            return candidate
        except OSError as exc:
            print(f"Performance V3 unavailable: {candidate}: {exc}")
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


def _wilson_lower_bound(correct: int, total: int, z: float = 1.2815515655) -> float:
    """方向命中率的一側 90% Wilson 下限；小樣本不會冒充穩定優勢。"""
    if total <= 0:
        return 0.0
    n = float(total)
    p = max(0.0, min(1.0, float(correct) / n))
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return max(0.0, min(1.0, (center - radius) / denominator))


def _direction_performance(correct: int, total: int) -> Dict[str, Any]:
    # Beta(12.5, 12.5) 將小樣本溫和收縮回 50%，避免假高勝率。
    posterior = (float(correct) + 12.5) / (float(total) + 25.0)
    return {
        "sample_count": int(total),
        "correct_count": int(correct),
        "accuracy": float(correct / max(1, total)),
        "posterior_accuracy": float(posterior),
        "wilson_lower_bound_90": float(
            _wilson_lower_bound(correct, total)
        ),
    }


def record_prediction(user_id: str, prediction: Mapping[str, Any], *, venue: str = "", room: str = "",
                      metadata: Optional[Mapping[str, Any]] = None) -> str:
    uid = _uid_key(user_id)
    prediction_id = f"{_now():x}{secrets.token_hex(6)}"
    prediction_fingerprint = str(
        prediction.get("prediction_fingerprint") or ""
    ).strip()
    selected_arm = str(prediction.get("selected_arm") or prediction.get("action") or prediction.get("recommend") or "").upper().strip()
    record = {
        "prediction_id": prediction_id,
        "uid_key": uid,
        "created_at": _now(),
        "resolved_at": 0,
        "venue": str(venue or "").upper().strip(),
        "room": str(room or "").strip(),
        "model_version": str(prediction.get("model_version") or prediction.get("engine") or ""),
        "shoe_id": str(prediction.get("shoe_id") or ""),
        "bandit_learning_user_id": str(
            prediction.get("bandit_learning_user_id") or ""
        ),
        "prediction_fingerprint": prediction_fingerprint,
        "probabilities": _prediction_probabilities(prediction),
        "bandit_learning_probabilities": _normalize_probabilities(
            dict(
                prediction.get("bandit_learning_probabilities")
                or prediction.get("pre_braking_probabilities")
                or _prediction_probabilities(prediction)
            )
        ),
        "recommend": str(prediction.get("recommend") or selected_arm).upper(),
        "action": str(prediction.get("action") or selected_arm).upper(),
        "selected_arm": selected_arm,
        "context_vector": list(prediction.get("bandit_context") or prediction.get("context_vector") or []),
        "context_feature_names": list(prediction.get("context_feature_names") or []),
        "timeline_alignment": dict(
            prediction.get("timeline_alignment") or {}
        ),
        "bandit_scores": dict(prediction.get("bandit_scores") or {}),
        "direction_source": str(prediction.get("direction_source") or ""),
        "component_champion": dict(
            prediction.get("component_champion") or {}
        ),
        "decision_validation": dict(
            prediction.get("decision_validation") or {}
        ),
        "component_probabilities": dict(
            prediction.get("component_probabilities")
            or dict(prediction.get("road_support") or {}).get(
                "component_probabilities", {}
            )
            or {}
        ),
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
        pending_id = str(data["pending"].get(uid) or "")
        if pending_id and prediction_fingerprint:
            for existing in reversed(data["records"]):
                if str(existing.get("prediction_id") or "") != pending_id:
                    continue
                if (
                    not existing.get("actual_outcome")
                    and str(existing.get("uid_key") or "") == uid
                    and str(existing.get("prediction_fingerprint") or "")
                    == prediction_fingerprint
                ):
                    # 同一歷史重試時沿用 pending，不重複建立訓練事件。
                    return pending_id
                break
        data["records"].append(record)
        data["pending"][uid] = prediction_id
        _write_unlocked(data)
    return prediction_id


def resolve_latest_prediction(user_id: str, actual_outcome: str, *, venue: str = "", room: str = "",
                              prediction_id: str = "",
                              mark_duplicate_guard: bool = False) -> Optional[Dict[str, Any]]:
    actual = str(actual_outcome or "").upper().strip()
    if actual not in _OUTCOMES:
        raise ValueError("actual_outcome must be B, P or T")
    uid = _uid_key(user_id)
    with _LOCK:
        data = _read_unlocked()
        expected_prediction_id = str(prediction_id or "").strip()
        guard = dict(data["duplicate_guards"].get(uid) or {})
        if (
            not expected_prediction_id
            and guard
            and str(guard.get("actual_outcome") or "") == actual
            and _now() - int(guard.get("created_at", 0) or 0)
            <= _GUARD_SECONDS
        ):
            data["duplicate_guards"].pop(uid, None)
            _write_unlocked(data)
            return None

        pending_id = str(data["pending"].get(uid) or "")
        target: Optional[Dict[str, Any]] = None
        for record in reversed(data["records"]):
            if str(record.get("uid_key") or "") != uid or record.get("actual_outcome"):
                continue
            record_id = str(record.get("prediction_id") or "")
            if expected_prediction_id and record_id != expected_prediction_id:
                continue
            if not expected_prediction_id and pending_id and record_id != pending_id:
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
        public_action = str(target.get("action") or target.get("recommend") or "O").upper().strip()
        context = list(target.get("context_vector") or [])
        reward: Optional[float]
        if actual == "T":
            reward = None
        elif selected_arm in {"B", "P"}:
            reward = 1.0 if selected_arm == actual else 0.0
        else:
            reward = None

        # 統計混沌區也只是一筆新觀測；禁止把單局結果複製成 4～5 筆
        # 證據，避免 Few-shot Boosting 對隨機雜訊過擬合。
        update_weight = 1.0

        if reward is not None and context:
            bandit_update = update_bandit(
                user_id=str(
                    target.get("bandit_learning_user_id") or user_id
                ),
                context=context,
                selected_arm=selected_arm,
                reward=reward,
                event_id=str(target.get("prediction_id") or ""),
                actual_outcome=actual,
                update_weight=update_weight,
                prediction_probabilities=dict(
                    target.get("bandit_learning_probabilities")
                    or probabilities
                ),
            )
        elif actual == "T":
            bandit_update = {"updated": False, "reason": "tie_not_used_for_bp_arms", "actual_outcome": actual}
        else:
            bandit_update = {"updated": False, "reason": "missing_context_or_arm", "actual_outcome": actual}

        target["actual_outcome"] = actual
        target["resolved_at"] = _now()
        target["reward"] = reward
        target["bandit_update"] = bandit_update
        target["bp_timeline_advanced"] = actual in {"B", "P"}
        target["tie_skipped_for_structural_learning"] = actual == "T"
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
        # 正式勝率只能計算前端真正開放的下注動作；No Bet 時保留的
        # selected_arm 只是被動學習標籤，不能再被算成一筆正式輸贏。
        target["action_correct"] = (
            public_action == actual
            if public_action in _OUTCOMES
            else None
        )
        target["learning_arm_correct"] = (
            selected_arm == actual
            if actual in {"B", "P"} and selected_arm in {"B", "P"}
            else None
        )
        target["no_bet"] = public_action == "O"
        if not pending_id or pending_id == str(target.get("prediction_id") or ""):
            data["pending"].pop(uid, None)
        if mark_duplicate_guard:
            data["duplicate_guards"][uid] = {
                "actual_outcome": actual,
                "created_at": _now(),
                "resolved_prediction_id": str(target.get("prediction_id") or ""),
            }
        _write_unlocked(data)
        return dict(target)


def get_resolved_records(*, venue: str = "", room: str = "", shoe_id: str = "",
                         limit: int = 5000) -> List[Dict[str, Any]]:
    venue_key, room_key = str(venue or "").upper().strip(), str(room or "").strip()
    shoe_key = str(shoe_id or "").strip()
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
        if shoe_key and str(record.get("shoe_id") or "") != shoe_key:
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
    learning_decided = learning_correct = 0
    no_bet_count = 0
    rewards: List[float] = []
    unknown_region_count = 0
    boosted_update_count = 0
    predicted_totals = {key: 0.0 for key in _OUTCOMES}
    component_brier_values: Dict[str, List[float]] = {}
    component_direction_counts: Dict[str, List[int]] = {}
    source_direction_counts: Dict[str, List[int]] = {}
    for record in valid:
        actual = str(record.get("actual_outcome") or "").upper()
        counts[actual] += 1
        probabilities = _normalize_probabilities(
            dict(record.get("probabilities") or {})
        )
        for key in _OUTCOMES:
            predicted_totals[key] += probabilities[key]
        try:
            brier_values.append(float(record.get("brier_score")))
        except Exception:
            pass
        try:
            log_losses.append(float(record.get("log_loss")))
        except Exception:
            pass
        public_action = str(
            record.get("action") or record.get("recommend") or "O"
        ).upper().strip()
        public_action_correct = (
            public_action == actual
            if public_action in _OUTCOMES
            else None
        )
        if public_action_correct is not None:
            decided += 1
            correct += int(bool(public_action_correct))
        if actual in {"B", "P"} and public_action in {"B", "P"}:
            source = str(record.get("direction_source") or "unknown")
            source_values = source_direction_counts.setdefault(source, [0, 0])
            source_values[0] += int(public_action == actual)
            source_values[1] += 1
        if record.get("learning_arm_correct") is not None:
            learning_decided += 1
            learning_correct += int(bool(record.get("learning_arm_correct")))
        no_bet_count += int(public_action == "O")
        if record.get("reward") is not None:
            rewards.append(float(record.get("reward") or 0.0))
        unknown_region_count += int(bool(record.get("unknown_region_active")))
        boosted_update_count += int(bool(record.get("few_shot_boost_applied")))
        for name, values in dict(record.get("component_probabilities") or {}).items():
            if not isinstance(values, Mapping):
                continue
            component = _normalize_probabilities(values)
            score = sum(
                (component[key] - (1.0 if key == actual else 0.0)) ** 2
                for key in _OUTCOMES
            )
            component_brier_values.setdefault(str(name), []).append(score)
            if actual in {"B", "P"}:
                component_direction = (
                    "B" if component["B"] >= component["P"] else "P"
                )
                direction_values = component_direction_counts.setdefault(
                    str(name), [0, 0]
                )
                direction_values[0] += int(component_direction == actual)
                direction_values[1] += 1
    total = max(1, len(valid))
    mean_brier = sum(brier_values) / len(brier_values) if brier_values else None
    mean_log_loss = sum(log_losses) / len(log_losses) if log_losses else None
    return {
        "sample_count": len(valid),
        "outcome_counts": counts,
        "empirical_probabilities": {key: counts[key] / total for key in _OUTCOMES},
        "decision_count": decided,
        "correct_count": correct,
        "accuracy": correct / max(1, decided),
        "decision_coverage": decided / total,
        "no_bet_count": no_bet_count,
        "no_bet_rate": no_bet_count / total,
        "learning_arm_decision_count": learning_decided,
        "learning_arm_correct_count": learning_correct,
        "learning_arm_accuracy": learning_correct / max(1, learning_decided),
        "mean_reward": sum(rewards) / max(1, len(rewards)),
        "unknown_region_count": unknown_region_count,
        "few_shot_boosted_update_count": boosted_update_count,
        "mean_predicted_probabilities": {
            key: predicted_totals[key] / total for key in _OUTCOMES
        },
        "mean_brier_score": mean_brier,
        "mean_log_loss": mean_log_loss,
        # 舊校準器讀取這兩個名稱；保留 mean_* 並補齊相容別名。
        "brier_score": mean_brier,
        "log_loss": mean_log_loss,
        "component_brier_scores": {
            name: sum(values) / len(values)
            for name, values in component_brier_values.items()
            if values
        },
        "component_sample_counts": {
            name: len(values)
            for name, values in component_brier_values.items()
        },
        "component_direction_performance": {
            name: _direction_performance(values[0], values[1])
            for name, values in component_direction_counts.items()
        },
        "source_direction_performance": {
            name: _direction_performance(values[0], values[1])
            for name, values in source_direction_counts.items()
        },
        "model": "CMAB-LINUCB-V1",
    }


def get_performance_summary(*, venue: str = "", room: str = "", shoe_id: str = "",
                            limit: int = 5000) -> Dict[str, Any]:
    return summarize_records(
        get_resolved_records(
            venue=venue, room=room, shoe_id=shoe_id, limit=limit
        )
    )


__all__ = ["get_performance_summary", "get_resolved_records", "record_prediction",
           "resolve_latest_prediction", "summarize_records"]
