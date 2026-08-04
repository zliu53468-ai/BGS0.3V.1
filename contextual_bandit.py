"""BGS cMAB 主模型：LinUCB 上下文相關多臂老虎機。

主要方向只有兩個 Arm：B（莊）與 P（閒）。
本模組不使用粒子濾波、超幾何分布、蒙地卡羅或 Stacking。
每次預測保存上下文向量；使用者回報實際結果後，再以 reward 更新 Arm。
和局不更新 B/P Arm。

banker_rate / player_rate 是方向分數正規化結果，不是真實開出機率。
"""
from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import json
import math
import os
import time

import numpy as np

ARMS = ("B", "P")
MODEL_VERSION = "CMAB-LINUCB-V1"
FEATURE_NAMES = (
    "bias", "history_maturity", "global_banker_balance",
    "recent5_banker_balance", "recent10_banker_balance",
    "recent20_banker_balance", "recent40_banker_balance",
    "current_streak_direction", "current_streak_length",
    "alternation5", "alternation10", "alternation20",
    "last_outcome_direction", "previous_outcome_direction",
    "observed_tie_rate", "road_planning_balance",
    "road_recent_balance", "road_confidence",
    "road_planning_reliability", "road_recent_reliability",
    "road_agreement", "markov1_balance", "markov2_balance",
    "markov3_balance",
)
CONTEXT_DIM = len(FEATURE_NAMES)
BASE_DIR = Path(__file__).resolve().parent
_LOCK = RLock()


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


CMAB_ALPHA = _env_float("CMAB_ALPHA", 0.80, 0.0, 5.0)
CMAB_L2 = _env_float("CMAB_L2", 1.0, 0.05, 100.0)
CMAB_SCORE_TEMPERATURE = _env_float("CMAB_SCORE_TEMPERATURE", 1.35, 0.10, 10.0)
CMAB_MAX_EVENT_IDS = max(100, min(20000, int(os.getenv("CMAB_MAX_EVENT_IDS", "5000") or "5000")))


def _resolve_state_file() -> Path:
    configured = Path(os.getenv("CMAB_STATE_FILE", str(BASE_DIR / "data" / "contextual_bandit_state.json"))).expanduser()
    candidates = [configured, BASE_DIR / "data" / "contextual_bandit_state.json", Path("/tmp/bgs_contextual_bandit_state.json")]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / f".cmab_write_test_{os.getpid()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            if candidate != configured:
                print(f"CMAB_STATE_FILE fallback: {configured} -> {candidate}")
            return candidate
        except OSError as exc:
            print(f"CMAB_STATE_FILE unavailable: {candidate}: {exc}")
    raise RuntimeError("No writable CMAB_STATE_FILE path is available")


CMAB_STATE_FILE = _resolve_state_file()


def _clean_history(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            result.append(value)
    return result[-2000:]


def _clip(value: Any, minimum: float = -1.0, maximum: float = 1.0) -> float:
    try:
        return max(minimum, min(maximum, float(value)))
    except Exception:
        return 0.0


def _prob_balance(probability: Any) -> float:
    try:
        return _clip((float(probability) - 0.5) * 2.0)
    except Exception:
        return 0.0


def _banker_balance(sequence: Sequence[str], size: Optional[int] = None) -> float:
    values = list(sequence[-size:] if size else sequence)
    if not values:
        return 0.0
    banker = sum(value == "B" for value in values)
    return _clip((banker / len(values) - 0.5) * 2.0)


def _alternation(sequence: Sequence[str], size: int) -> float:
    values = list(sequence[-size:])
    if len(values) < 2:
        return 0.0
    rate = sum(a != b for a, b in zip(values, values[1:])) / (len(values) - 1)
    return _clip((rate - 0.5) * 2.0)


def _streak(sequence: Sequence[str]) -> tuple[str, int]:
    if not sequence:
        return "", 0
    direction = sequence[-1]
    length = 1
    for value in reversed(sequence[:-1]):
        if value != direction:
            break
        length += 1
    return direction, length


def _model_probability(road_context: Mapping[str, Any], model_name: str, fallback: float = 0.5) -> float:
    models = road_context.get("models")
    if not isinstance(models, Mapping):
        return fallback
    model = models.get(model_name)
    if not isinstance(model, Mapping):
        return fallback
    try:
        return float(model.get("banker_probability", fallback) or fallback)
    except Exception:
        return fallback


def build_context_vector(history: Iterable[Any], *, road_context: Optional[Mapping[str, Any]] = None) -> List[float]:
    """建立固定 24 維上下文；所有值限制在 [-1, 1]。"""
    raw = _clean_history(history)
    bp = [value for value in raw if value in ARMS]
    road = dict(road_context or {})
    streak_direction, streak_length = _streak(bp)
    last_direction = 1.0 if bp and bp[-1] == "B" else -1.0 if bp else 0.0
    previous_direction = 1.0 if len(bp) >= 2 and bp[-2] == "B" else -1.0 if len(bp) >= 2 else 0.0
    streak_sign = 1.0 if streak_direction == "B" else -1.0 if streak_direction == "P" else 0.0
    tie_rate = sum(value == "T" for value in raw) / max(1, len(raw))
    confidence = _clip(road.get("confidence_score", 0.0), 0.0, 1.0)
    planning_reliability = _clip(road.get("planning_reliability", 0.0), 0.0, 1.0)
    recent_reliability = _clip(road.get("recent_reliability", 0.0), 0.0, 1.0)
    disagreement = _clip(road.get("recent_model_disagreement", road.get("model_disagreement", 0.20)), 0.0, 1.0)
    agreement = _clip(1.0 - min(1.0, disagreement / 0.20), 0.0, 1.0)
    vector = [
        1.0,
        min(1.0, len(bp) / 60.0),
        _banker_balance(bp), _banker_balance(bp, 5), _banker_balance(bp, 10),
        _banker_balance(bp, 20), _banker_balance(bp, 40),
        streak_sign, min(1.0, streak_length / 8.0),
        _alternation(bp, 5), _alternation(bp, 10), _alternation(bp, 20),
        last_direction, previous_direction, _clip(tie_rate / 0.20, 0.0, 1.0),
        _prob_balance(road.get("planning_probability", 0.5)),
        _prob_balance(road.get("recent_probability", 0.5)),
        confidence, planning_reliability, recent_reliability, agreement,
        _prob_balance(_model_probability(road, "markov1")),
        _prob_balance(_model_probability(road, "markov2")),
        _prob_balance(_model_probability(road, "markov3")),
    ]
    if len(vector) != CONTEXT_DIM:
        raise RuntimeError(f"CMAB context dimension mismatch: {len(vector)} != {CONTEXT_DIM}")
    return [round(_clip(value), 10) for value in vector]


def _new_state() -> Dict[str, Any]:
    identity = (np.eye(CONTEXT_DIM, dtype=np.float64) * CMAB_L2).tolist()
    zeros = np.zeros(CONTEXT_DIM, dtype=np.float64).tolist()
    return {
        "version": MODEL_VERSION,
        "context_dim": CONTEXT_DIM,
        "feature_names": list(FEATURE_NAMES),
        "alpha": CMAB_ALPHA,
        "l2": CMAB_L2,
        "arms": {arm: {"A": identity, "b": zeros, "updates": 0, "reward_sum": 0.0} for arm in ARMS},
        "applied_event_ids": [],
        "total_updates": 0,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }


def _read_state_unlocked() -> Dict[str, Any]:
    try:
        data = json.loads(CMAB_STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or int(data.get("context_dim", 0) or 0) != CONTEXT_DIM:
            raise ValueError("invalid state")
        arms = data.get("arms")
        if not isinstance(arms, dict) or any(arm not in arms for arm in ARMS):
            raise ValueError("missing arms")
        for arm in ARMS:
            A = np.asarray(arms[arm].get("A"), dtype=np.float64)
            b = np.asarray(arms[arm].get("b"), dtype=np.float64)
            if A.shape != (CONTEXT_DIM, CONTEXT_DIM) or b.shape != (CONTEXT_DIM,):
                raise ValueError("invalid matrix shape")
        data["applied_event_ids"] = list(data.get("applied_event_ids") or [])[-CMAB_MAX_EVENT_IDS:]
        return data
    except Exception:
        return _new_state()


def _write_state_unlocked(state: Dict[str, Any]) -> None:
    state["updated_at"] = int(time.time())
    state["applied_event_ids"] = list(state.get("applied_event_ids") or [])[-CMAB_MAX_EVENT_IDS:]
    CMAB_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = CMAB_STATE_FILE.with_suffix(CMAB_STATE_FILE.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(CMAB_STATE_FILE)


def _arm_metrics(state: Mapping[str, Any], arm: str, context: np.ndarray) -> Dict[str, float]:
    arm_state = dict(dict(state.get("arms") or {}).get(arm) or {})
    A = np.asarray(arm_state.get("A"), dtype=np.float64)
    b = np.asarray(arm_state.get("b"), dtype=np.float64)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
    theta = A_inv @ b
    estimate = float(theta @ context)
    uncertainty = float(math.sqrt(max(0.0, float(context @ A_inv @ context))))
    exploration = float(CMAB_ALPHA * uncertainty)
    return {
        "estimate": estimate,
        "uncertainty": uncertainty,
        "exploration": exploration,
        "score": estimate + exploration,
        "updates": int(arm_state.get("updates", 0) or 0),
        "reward_sum": float(arm_state.get("reward_sum", 0.0) or 0.0),
    }


def _softmax_two(score_b: float, score_p: float) -> Dict[str, float]:
    values = np.asarray([score_b, score_p], dtype=np.float64) / max(0.10, CMAB_SCORE_TEMPERATURE)
    values -= float(np.max(values))
    exp_values = np.exp(np.clip(values, -40.0, 40.0))
    total = float(exp_values.sum()) or 1.0
    return {"B": float(exp_values[0] / total), "P": float(exp_values[1] / total), "T": 0.0}


def _fallback_direction(history: Sequence[str], road_context: Mapping[str, Any]) -> str:
    for key in ("direction", "planning_direction", "recent_direction"):
        value = str(road_context.get(key) or "").upper().strip()
        if value in ARMS:
            return value
    planning = float(road_context.get("planning_probability", 0.5) or 0.5)
    recent = float(road_context.get("recent_probability", 0.5) or 0.5)
    if abs(planning - 0.5) > 1e-9:
        return "B" if planning >= 0.5 else "P"
    if abs(recent - 0.5) > 1e-9:
        return "B" if recent >= 0.5 else "P"
    bp = [value for value in history if value in ARMS]
    return bp[-1] if bp else "B"


def predict_bandit(history: Iterable[Any], *, road_context: Optional[Mapping[str, Any]] = None,
                    venue: str = "", room: str = "", user_id: str = "",
                    run_seed: Optional[int] = None) -> Dict[str, Any]:
    raw_history = _clean_history(history)
    road = dict(road_context or {})
    vector = build_context_vector(raw_history, road_context=road)
    context = np.asarray(vector, dtype=np.float64)
    with _LOCK:
        state = _read_state_unlocked()
        metrics = {arm: _arm_metrics(state, arm, context) for arm in ARMS}
    score_b, score_p = metrics["B"]["score"], metrics["P"]["score"]
    if abs(score_b - score_p) <= 1e-12:
        direction, tie_break = _fallback_direction(raw_history, road), True
    else:
        direction, tie_break = ("B" if score_b > score_p else "P"), False
    probabilities = _softmax_two(score_b, score_p)
    margin = abs(score_b - score_p)
    total_updates = int(state.get("total_updates", 0) or 0)
    maturity = 1.0 - math.exp(-total_updates / 80.0)
    quality = min(0.95, 0.34 + 0.36 * maturity + 0.25 * math.tanh(margin))
    if sum(value in ARMS for value in raw_history) < 8:
        quality = min(quality, 0.45)
    direction_edge = abs(probabilities["B"] - probabilities["P"])
    consistency = min(1.0, 0.50 + 0.50 * math.tanh(margin * 1.5))
    selected = metrics[direction]
    return {
        "ok": True,
        "engine": "CONTEXTUAL_MULTI_ARMED_BANDIT_LINUCB",
        "model_version": MODEL_VERSION,
        "model_core": "contextual_multi_armed_bandit_linu_cb",
        "mode": "screen_contextual_bandit",
        "probabilities": dict(probabilities),
        "banker_rate": round(probabilities["B"] * 100.0, 2),
        "player_rate": round(probabilities["P"] * 100.0, 2),
        "tie_rate": 0.0,
        "recommend": direction,
        "recommend_text": "莊" if direction == "B" else "閒",
        "action": direction,
        "action_text": "莊" if direction == "B" else "閒",
        "internal_recommend": direction,
        "internal_action": direction,
        "signal_allowed": True,
        "signal_status_text": "cMAB 下一局方向評估",
        "signal_reason": "LinUCB 依目前牌路上下文、歷史回饋與探索上界選擇莊／閒 Arm",
        "selected_arm": direction,
        "next_round_direction": direction,
        "next_round_direction_text": "莊" if direction == "B" else "閒",
        "direction_source": "contextual_bandit_linu_cb",
        "direction_edge": float(direction_edge),
        "direction_edge_percent": round(direction_edge * 100.0, 4),
        "quality_score": float(quality),
        "confidence_label": "較高" if quality >= 0.72 else "中等" if quality >= 0.50 else "偏低",
        "model_consistency": float(consistency),
        "uncertainty": float(selected["uncertainty"]),
        "bandit_context": list(vector),
        "context_vector": list(vector),
        "context_feature_names": list(FEATURE_NAMES),
        "bandit_scores": {arm: {key: int(value) if key == "updates" else round(float(value), 10)
                                  for key, value in metrics[arm].items()} for arm in ARMS},
        "bandit_state": {
            "total_updates": total_updates,
            "arm_updates": {arm: int(metrics[arm]["updates"]) for arm in ARMS},
            "alpha": CMAB_ALPHA,
            "l2": CMAB_L2,
            "state_file": str(CMAB_STATE_FILE),
            "cold_start_tie_break": tie_break,
        },
        "calibration": {
            "active": total_updates > 0,
            "scope": "cmab_online_reward",
            "sample_count": total_updates,
            "reason": "cMAB 直接使用每局 reward 更新，不再使用舊機率校準器",
        },
        "adaptive_ensemble": {
            "active": False,
            "effective_share": 0.0,
            "sample_count": total_updates,
            "reason": "已由 LinUCB 線上更新取代舊自適應 Stacking",
        },
        "venue": str(venue or ""), "room": str(room or ""), "user_id": str(user_id or ""),
        "run_seed": run_seed, "input_required": False,
        "disclaimer": "莊／閒百分比是 cMAB 方向分數正規化結果，不是真實開出機率；模型只依已公布結果調整。",
    }


def update_bandit(*, context: Sequence[float], selected_arm: str, reward: Optional[float],
                   event_id: str = "", actual_outcome: str = "") -> Dict[str, Any]:
    arm = str(selected_arm or "").upper().strip()
    if arm not in ARMS:
        raise ValueError("selected_arm must be B or P")
    if reward is None:
        return {"updated": False, "reason": "tie_or_skipped_reward", "selected_arm": arm,
                "actual_outcome": str(actual_outcome or "").upper()}
    x = np.asarray(list(context), dtype=np.float64)
    if x.shape != (CONTEXT_DIM,):
        raise ValueError(f"context must contain {CONTEXT_DIM} values, got {x.shape}")
    x = np.clip(x, -1.0, 1.0)
    reward_value = max(0.0, min(1.0, float(reward)))
    event_key = str(event_id or "").strip()
    with _LOCK:
        state = _read_state_unlocked()
        applied = list(state.get("applied_event_ids") or [])
        if event_key and event_key in applied:
            return {"updated": False, "reason": "duplicate_event", "event_id": event_key,
                    "total_updates": int(state.get("total_updates", 0) or 0)}
        arm_state = dict(state["arms"][arm])
        A = np.asarray(arm_state["A"], dtype=np.float64)
        b = np.asarray(arm_state["b"], dtype=np.float64)
        A += np.outer(x, x)
        b += reward_value * x
        arm_state["A"], arm_state["b"] = A.tolist(), b.tolist()
        arm_state["updates"] = int(arm_state.get("updates", 0) or 0) + 1
        arm_state["reward_sum"] = float(arm_state.get("reward_sum", 0.0) or 0.0) + reward_value
        state["arms"][arm] = arm_state
        state["total_updates"] = int(state.get("total_updates", 0) or 0) + 1
        if event_key:
            applied.append(event_key)
            state["applied_event_ids"] = applied[-CMAB_MAX_EVENT_IDS:]
        _write_state_unlocked(state)
    return {"updated": True, "event_id": event_key, "selected_arm": arm,
            "actual_outcome": str(actual_outcome or "").upper(), "reward": reward_value,
            "arm_updates": int(arm_state["updates"]), "total_updates": int(state["total_updates"])}


def get_bandit_summary() -> Dict[str, Any]:
    with _LOCK:
        state = _read_state_unlocked()
    return {
        "version": state.get("version", MODEL_VERSION),
        "context_dim": CONTEXT_DIM,
        "feature_names": list(FEATURE_NAMES),
        "total_updates": int(state.get("total_updates", 0) or 0),
        "arms": {arm: {"updates": int(state["arms"][arm].get("updates", 0) or 0),
                       "reward_sum": float(state["arms"][arm].get("reward_sum", 0.0) or 0.0)}
                 for arm in ARMS},
        "state_file": str(CMAB_STATE_FILE),
    }


__all__ = ["ARMS", "CONTEXT_DIM", "FEATURE_NAMES", "MODEL_VERSION",
           "build_context_vector", "get_bandit_summary", "predict_bandit", "update_bandit"]
