"""BGS 純牌路預測績效紀錄器。

每次正式牌路預測後建立 pending prediction；使用者回報實際結果時：
- B/P：只結算正式方向的稽核資料，不更新任何 Bandit Arm。
- T：只記錄結果，不進行模型學習。
- prediction_id 去重，避免同一局重複寫入績效紀錄。
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

BASE_DIR = Path(__file__).resolve().parent
_LOCK = RLock()
_OUTCOMES = ("B", "P", "T")
# 固定資料保護參數；不受 Render 舊環境變數覆寫。
_GUARD_SECONDS = 90
PERFORMANCE_MAX_RECORDS = 30000
# 正式模式固定關閉 UCB 線上學習。這個常數寫在程式內，Render 舊環境變數
# 無法重新啟用；績效檔僅供後續人工檢視，不回灌任何預測權重。
CONTEXTUAL_BANDIT_ONLINE_UPDATE_ENABLED = False


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
    # 純牌路模式不再存在 Bandit Arm。正式 action 仍完整保存並可統計命中，
    # 但 selected_arm 保持空白，避免 resolve 時將按鈕結果回灌成 UCB 學習。
    selected_arm = ""
    record = {
        "prediction_id": prediction_id,
        "uid_key": uid,
        "created_at": _now(),
        "resolved_at": 0,
        "venue": str(venue or "").upper().strip(),
        "room": str(room or "").strip(),
        "model_version": str(prediction.get("model_version") or prediction.get("engine") or ""),
        "model_variant": str(prediction.get("model_variant") or "").strip(),
        "shoe_id": str(prediction.get("shoe_id") or ""),
        "bandit_learning_user_id": "",
        "contextual_bandit_enabled": False,
        "prediction_fingerprint": prediction_fingerprint,
        "probabilities": _prediction_probabilities(prediction),
        "bandit_learning_probabilities": {},
        "recommend": str(prediction.get("recommend") or selected_arm).upper(),
        "action": str(prediction.get("action") or selected_arm).upper(),
        "selected_arm": selected_arm,
        # 純牌路模式只有正式 Adaptive 方向；不保存 UCB 影子方向。
        "adaptive_only_direction": str(
            prediction.get("adaptive_only_direction")
            or prediction.get("action")
            or prediction.get("recommend")
            or ""
        ).upper().strip(),
        "ucb_shadow_direction": "",
        "shadow_comparison_eligible": False,
        "context_vector": [],
        "context_feature_names": [],
        "timeline_alignment": dict(
            prediction.get("timeline_alignment") or {}
        ),
        "bandit_scores": {},
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
        selected_arm = ""
        public_action = str(target.get("action") or target.get("recommend") or "O").upper().strip()
        reward: Optional[float]
        if actual == "T":
            reward = None
        elif public_action in {"B", "P"}:
            # 僅記錄正式方向的測試結果；reward 不會傳入任何學習器。
            reward = 1.0 if public_action == actual else 0.0
        else:
            reward = None

        # 不論 B/P/T，絕不觸發 update_bandit。這避免按鈕輸入的單局結果
        # 累積到 contextual_bandit_state_v3.json，並保證它無法影響後續預測。
        bandit_update = {
            "updated": False,
            "reason": "contextual_bandit_disabled_road_only_mode",
            "actual_outcome": actual,
        }

        target["actual_outcome"] = actual
        target["resolved_at"] = _now()
        target["reward"] = reward
        target["bandit_update"] = bandit_update
        target["bp_timeline_advanced"] = actual in {"B", "P"}
        target["tie_skipped_for_structural_learning"] = actual == "T"
        target["applied_update_weight"] = 0.0
        target["few_shot_boost_applied"] = False
        if venue:
            target["venue"] = str(venue).upper().strip()
        if room:
            target["room"] = str(room).strip()
        target["log_loss"] = -math.log(max(1e-12, probabilities[actual]))
        target["brier_score"] = sum((probabilities[key] - (1.0 if key == actual else 0.0)) ** 2 for key in _OUTCOMES)
        target["top1_correct"] = max(probabilities, key=probabilities.get) == actual
        # 正式勝率只計算對外輸出的牌路方向；本模式沒有 Bandit learning arm。
        target["action_correct"] = (
            public_action == actual
            if public_action in _OUTCOMES
            else None
        )
        target["learning_arm_correct"] = None
        adaptive_only_direction = str(
            target.get("adaptive_only_direction") or public_action
        ).upper().strip()
        target["adaptive_only_correct"] = (
            adaptive_only_direction == actual
            if actual in {"B", "P"} and adaptive_only_direction in {"B", "P"}
            else None
        )
        target["ucb_shadow_correct"] = None
        target["adaptive_ucb_agree"] = None
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
                         model_variant: str = "", limit: int = 5000) -> List[Dict[str, Any]]:
    venue_key, room_key = str(venue or "").upper().strip(), str(room or "").strip()
    shoe_key = str(shoe_id or "").strip()
    variant_key = str(model_variant or "").strip()
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
        if variant_key and str(record.get("model_variant") or "") != variant_key:
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
    shadow_adaptive_counts = [0, 0]
    shadow_ucb_counts = [0, 0]
    shadow_agreement_count = 0
    shadow_disagreement_count = 0
    shadow_adaptive_wins_when_different = 0
    shadow_ucb_wins_when_different = 0
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
        # 只統計新模式建立的 paired prediction；舊紀錄沒有同時保存兩個
        # 預測，不能拿來冒充 Adaptive vs UCB 的公平比較樣本。
        if bool(record.get("shadow_comparison_eligible")) and actual in {"B", "P"}:
            adaptive_direction = str(
                record.get("adaptive_only_direction") or ""
            ).upper().strip()
            ucb_direction = str(
                record.get("ucb_shadow_direction") or ""
            ).upper().strip()
            if adaptive_direction in {"B", "P"} and ucb_direction in {"B", "P"}:
                shadow_adaptive_counts[0] += int(adaptive_direction == actual)
                shadow_adaptive_counts[1] += 1
                shadow_ucb_counts[0] += int(ucb_direction == actual)
                shadow_ucb_counts[1] += 1
                if adaptive_direction == ucb_direction:
                    shadow_agreement_count += 1
                else:
                    shadow_disagreement_count += 1
                    shadow_adaptive_wins_when_different += int(
                        adaptive_direction == actual
                    )
                    shadow_ucb_wins_when_different += int(
                        ucb_direction == actual
                    )
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
        "adaptive_ucb_shadow_comparison": {
            "eligible_bp_sample_count": int(shadow_adaptive_counts[1]),
            "adaptive_ensemble_road_primary": _direction_performance(
                shadow_adaptive_counts[0], shadow_adaptive_counts[1]
            ),
            "contextual_bandit_shadow": _direction_performance(
                shadow_ucb_counts[0], shadow_ucb_counts[1]
            ),
            "agreement_count": int(shadow_agreement_count),
            "disagreement_count": int(shadow_disagreement_count),
            "adaptive_wins_when_different": int(
                shadow_adaptive_wins_when_different
            ),
            "ucb_wins_when_different": int(shadow_ucb_wins_when_different),
            "comparison_rule": (
                "只結算預測建立後的下一個實際 B/P；T 不計勝負，舊紀錄不混入。"
            ),
        },
        "model": "CMAB-LINUCB-V1",
    }


def get_performance_summary(*, venue: str = "", room: str = "", shoe_id: str = "",
                            model_variant: str = "", limit: int = 5000) -> Dict[str, Any]:
    summary = summarize_records(
        get_resolved_records(
            venue=venue, room=room, shoe_id=shoe_id,
            model_variant=model_variant, limit=limit,
        )
    )
    summary["model_variant_filter"] = str(model_variant or "").strip()
    return summary


__all__ = ["get_performance_summary", "get_resolved_records", "record_prediction",
           "resolve_latest_prediction", "summarize_records"]
