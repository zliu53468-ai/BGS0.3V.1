"""BGS CUSUM-LinUCB：可偵測牌路相變的上下文多臂老虎機。

這個模組只處理 B/P 兩個 Arm。每次使用者回報下一局 B/P 後，模型會以
預測時保存的 Context 更新選中的 Arm；CUSUM 會對「實際 reward - 預期
reward」的殘差做雙向累積。當殘差持續偏正或偏負並越過閾值時，所有 Arm
的 A/b 矩陣立刻回到 ridge identity / zero，避免舊路型綁架新牌路。

輸出的 B/P 數字是模型方向分數，不是未來開出機率或獲利保證。
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
import json
import math
import os
import time

import numpy as np


ARMS = ("B", "P")
CONTEXT_DIM = 24
MODEL_VERSION = "CUSUM-LINUCB-V1.0-ROAD-CONTEXT"

# 所有模型參數固定在程式內，避免 Render 舊環境變數不小心讓同一張牌路
# 使用不同設定。Context 已縮放到約 [-1, 1]，因此 ridge=1.0 足以在少樣本
# 時提供可逆的先驗協方差。
RIDGE = 1.0
RIDGE_JITTER = 1e-6
FORGETTING_FACTOR = 0.998
BASE_ALPHA = 0.16
MAX_EXPLORATION_BONUS = 0.35
SCORE_TEMPERATURE = 0.18

# Reward 是 0/1 Bernoulli，單局殘差本來就很大。v=0.18 會濾掉一般隨機
# 波動；h=3.00 代表冷啟動的 50% 模型約需十次同向殘差才重置，但若模型
# 已高度相信某一方向（例如預期 reward=0.80）而連續失誤，約四次就會告警。
# 這比靜態 LinUCB 反應快，同時不會因一兩手單跳或短暫連中反覆重置；應以
# 留出資料調參，不能依短期命中率追高或追低。
CUSUM_DRIFT_V = 0.18
CUSUM_THRESHOLD_H = 3.00
RESET_COOLDOWN_HANDS = 5
POST_RESET_MIN_UPDATES = 8
MAX_EVENT_IDS = 1000

FEATURE_NAMES = (
    "bias",
    "history_maturity",
    "global_banker_balance",
    "recent5_banker_balance",
    "recent10_banker_balance",
    "recent20_banker_balance",
    "last_outcome_direction",
    "previous_outcome_direction",
    "current_streak_direction",
    "current_streak_strength",
    "recent8_switch_rate",
    "recent4_alternation_strength",
    "observed_tie_rate",
    "road_planning_balance",
    "road_recent_balance",
    "road_planning_reliability",
    "road_recent_reliability",
    "full_road_balance",
    "full_road_reliability",
    "expert_consensus_signed",
    "expert_consensus_strength",
    "regime_dragon",
    "regime_chop",
    "regime_mixed",
)

BASE_DIR = Path(__file__).resolve().parent
_LOCK = RLock()


def _resolve_state_file() -> Path:
    """選擇 Render 可寫入的 CUSUM state 路徑，不使用環境變數。"""
    candidates = (
        Path("/var/data/contextual_bandit_cusum_state_v1.json"),
        BASE_DIR / "data" / "contextual_bandit_cusum_state_v1.json",
        Path("/tmp/bgs_contextual_bandit_cusum_state_v1.json"),
    )
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / f".cusum_write_probe_{time.time_ns()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    # 最後回傳 /tmp，實際寫入若失敗由 _write_state 安全忽略，不阻斷 API。
    return candidates[-1]


STATE_FILE = _resolve_state_file()


def _now() -> int:
    return int(time.time())


def _clip(value: Any, lower: float, upper: float, default: float) -> float:
    try:
        return max(lower, min(upper, float(value)))
    except (TypeError, ValueError):
        return default


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _stable_softmax_banker(score_b: float, score_p: float) -> float:
    # 只需比較兩臂分數；以較低溫度保留方向差，但限制到合理顯示範圍。
    delta = (float(score_b) - float(score_p)) / SCORE_TEMPERATURE
    return _clip(_sigmoid(delta), 0.05, 0.95, 0.50)


def _normalize_history(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in values or []:
        raw = (
            item.get("outcome")
            or item.get("actual")
            or item.get("actual_outcome")
            or item.get("virtual_outcome")
            if isinstance(item, Mapping)
            else item
        )
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            result.append(value)
    return result[-500:]


def _bp_history(values: Iterable[Any]) -> list[str]:
    return [value for value in _normalize_history(values) if value in ARMS]


def _balance(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    return (sum(value == "B" for value in values) - sum(value == "P" for value in values)) / len(values)


def _tail(values: Sequence[str], size: int) -> list[str]:
    return list(values[-min(len(values), size):])


def _streak(values: Sequence[str]) -> tuple[str, int]:
    if not values:
        return "", 0
    last = str(values[-1])
    length = 1
    for value in reversed(values[:-1]):
        if value != last:
            break
        length += 1
    return last, length


def _probability_from_mapping(values: Any) -> float:
    """讀取多種 road model 常見格式並回傳 conditional P(B)。"""
    if not isinstance(values, Mapping):
        return 0.5
    for key in ("banker_probability", "planning_probability", "recent_probability"):
        if key in values:
            return _clip(values.get(key), 0.01, 0.99, 0.5)
    banker = _clip(values.get("B", 0.0), 0.0, 1.0, 0.0)
    player = _clip(values.get("P", 0.0), 0.0, 1.0, 0.0)
    if banker + player > 1e-12:
        return banker / (banker + player)
    return 0.5


def _model_metadata(road: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    models = road.get("models")
    if isinstance(models, Mapping) and isinstance(models.get(name), Mapping):
        return models[name]
    components = road.get("component_probabilities")
    if isinstance(components, Mapping) and isinstance(components.get(name), Mapping):
        return components[name]
    return {}


def _model_reliability(road: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    metadata = _model_metadata(road, name)
    return _clip(metadata.get("reliability", metadata.get("confidence", default)), 0.0, 1.0, default)


def _regime_text(road: Mapping[str, Any]) -> str:
    regime = road.get("regime")
    if isinstance(regime, Mapping):
        return " ".join(str(value).lower() for value in regime.values())
    return str(regime or "").lower()


def _coerce_context(context: Sequence[float]) -> np.ndarray:
    values = []
    for value in list(context or [])[:CONTEXT_DIM]:
        values.append(_clip(value, -4.0, 4.0, 0.0))
    if len(values) < CONTEXT_DIM:
        values.extend([0.0] * (CONTEXT_DIM - len(values)))
    return np.asarray(values, dtype=np.float64)


def build_context_vector(
    history: Iterable[Any],
    road_context: Optional[Mapping[str, Any]] = None,
) -> list[float]:
    """建立低維、縮放後的 B/P/T 牌路 Context。

    大路與下三路資訊由 screenshot/road_model 已建構的 ``road_context`` 傳入；
    本函式不讀取 OCR 圖片，也不修改任何辨識流程。
    """
    raw = _normalize_history(history)
    bp = [value for value in raw if value in ARMS]
    road = dict(road_context or {})
    last, streak = _streak(bp)
    recent8 = _tail(bp, 8)
    recent4 = _tail(bp, 4)
    switch_rate = (
        sum(left != right for left, right in zip(recent8, recent8[1:]))
        / max(1, len(recent8) - 1)
        if len(recent8) >= 2
        else 0.0
    )
    alternation_strength = (
        1.0 if len(recent4) >= 4 and all(
            left != right for left, right in zip(recent4, recent4[1:])
        ) else 0.0
    )

    planning_probability = _clip(
        road.get("planning_probability", road.get("banker_probability", 0.5)),
        0.01, 0.99, 0.5,
    )
    recent_probability = _clip(
        road.get("recent_probability", planning_probability), 0.01, 0.99, planning_probability
    )
    full_probability = _probability_from_mapping(_model_metadata(road, "full_road"))
    planning_reliability = _clip(road.get("planning_reliability", 0.0), 0.0, 1.0, 0.0)
    recent_reliability = _clip(road.get("recent_reliability", 0.0), 0.0, 1.0, 0.0)
    full_reliability = _model_reliability(road, "full_road", planning_reliability)

    expert_probabilities: list[float] = []
    for name in ("full_road", "structural_regime", "short", "mid", "long", "pattern", "analogue"):
        metadata = _model_metadata(road, name)
        if not metadata:
            continue
        if isinstance(metadata, Mapping) and not bool(metadata.get("active", metadata.get("ok", True))):
            continue
        expert_probabilities.append(_probability_from_mapping(metadata))
    if expert_probabilities:
        consensus_signed = sum((probability - 0.5) * 2.0 for probability in expert_probabilities) / len(expert_probabilities)
        consensus_strength = abs(consensus_signed)
    else:
        consensus_signed, consensus_strength = 0.0, 0.0

    observed_tie_rate = _clip(
        road.get("observed_tie_rate", sum(value == "T" for value in raw) / max(1, len(raw))),
        0.0, 0.30, 0.0,
    )
    regime = _regime_text(road)
    dragon = 1.0 if any(token in regime for token in ("dragon", "streak", "long")) else 0.0
    chop = 1.0 if any(token in regime for token in ("chop", "alternat", "jump")) else 0.0
    mixed = 1.0 if not dragon and not chop else 0.0

    vector = [
        1.0,
        min(1.0, len(bp) / 40.0),
        _balance(bp),
        _balance(_tail(bp, 5)),
        _balance(_tail(bp, 10)),
        _balance(_tail(bp, 20)),
        1.0 if last == "B" else -1.0 if last == "P" else 0.0,
        1.0 if len(bp) >= 2 and bp[-2] == "B" else -1.0 if len(bp) >= 2 and bp[-2] == "P" else 0.0,
        1.0 if last == "B" else -1.0 if last == "P" else 0.0,
        min(1.0, streak / 6.0),
        switch_rate * 2.0 - 1.0,
        alternation_strength,
        observed_tie_rate / 0.30,
        (planning_probability - 0.5) * 2.0,
        (recent_probability - 0.5) * 2.0,
        planning_reliability,
        recent_reliability,
        (full_probability - 0.5) * 2.0,
        full_reliability,
        consensus_signed,
        consensus_strength,
        dragon,
        chop,
        mixed,
    ]
    return _coerce_context(vector).tolist()


def _empty_arm() -> Dict[str, Any]:
    return {
        "A": (np.eye(CONTEXT_DIM, dtype=np.float64) * RIDGE).tolist(),
        "b": np.zeros(CONTEXT_DIM, dtype=np.float64).tolist(),
        "updates": 0,
        "weighted_updates": 0.0,
        "reward_sum": 0.0,
    }


def _empty_scope() -> Dict[str, Any]:
    return {
        "arms": {arm: _empty_arm() for arm in ARMS},
        "cusum": {
            "g_plus": 0.0,
            "g_minus": 0.0,
            "last_residual": 0.0,
            "alarm_count": 0,
            "last_reset_at": 0,
            "last_reset_reason": "",
            "cooldown_remaining": 0,
            "updates_since_reset": 0,
        },
        "event_ids": [],
        "created_at": _now(),
        "updated_at": _now(),
    }


def _empty_state() -> Dict[str, Any]:
    return {"version": MODEL_VERSION, "scopes": {}}


def _read_state() -> Dict[str, Any]:
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, Mapping) and isinstance(data.get("scopes"), Mapping):
            return {"version": str(data.get("version") or MODEL_VERSION), "scopes": dict(data.get("scopes") or {})}
    except Exception:
        pass
    return _empty_state()


def _write_state(state: Mapping[str, Any]) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        temp.replace(STATE_FILE)
    except OSError:
        # 儲存失敗只代表本次 instance 的狀態不持久，不可讓 API 因此中斷。
        return


def _scope_id(user_id: str) -> str:
    return str(user_id or "__anonymous__").strip() or "__anonymous__"


def _arm_arrays(scope: Mapping[str, Any], arm: str) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    arms = scope.get("arms") if isinstance(scope, Mapping) else {}
    raw_arm = arms.get(arm) if isinstance(arms, Mapping) else None
    arm_data = dict(raw_arm) if isinstance(raw_arm, Mapping) else _empty_arm()
    try:
        matrix = np.asarray(arm_data.get("A"), dtype=np.float64)
        if matrix.shape != (CONTEXT_DIM, CONTEXT_DIM) or not np.all(np.isfinite(matrix)):
            raise ValueError
    except Exception:
        matrix = np.eye(CONTEXT_DIM, dtype=np.float64) * RIDGE
    matrix = (matrix + matrix.T) * 0.5
    try:
        vector = np.asarray(arm_data.get("b"), dtype=np.float64).reshape(-1)
        if vector.shape != (CONTEXT_DIM,) or not np.all(np.isfinite(vector)):
            raise ValueError
    except Exception:
        vector = np.zeros(CONTEXT_DIM, dtype=np.float64)
    return matrix, vector, arm_data


def _solve(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    identity = np.eye(CONTEXT_DIM, dtype=np.float64)
    for jitter in (RIDGE_JITTER, 1e-5, 1e-4, 1e-3):
        try:
            return np.linalg.solve(matrix + identity * jitter, vector)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(matrix + identity * 1e-2) @ vector


def _arm_metrics(scope: Mapping[str, Any], arm: str, context: np.ndarray) -> Dict[str, Any]:
    matrix, vector, arm_data = _arm_arrays(scope, arm)
    theta = _solve(matrix, vector)
    expected_reward = _sigmoid(float(theta @ context))
    inverse_context = _solve(matrix, context)
    variance = max(0.0, float(context @ inverse_context))
    cusum = dict(scope.get("cusum") or {})
    cooldown = max(0, int(cusum.get("cooldown_remaining", 0) or 0))
    updates_since_reset = max(0, int(cusum.get("updates_since_reset", 0) or 0))
    # reset 後探索係數提高，但 exploration bonus 有上限，避免分數爆掉。
    alpha = BASE_ALPHA * (1.65 if cooldown > 0 else 1.0)
    if updates_since_reset < POST_RESET_MIN_UPDATES:
        alpha *= 1.20
    uncertainty = math.sqrt(variance)
    exploration = min(MAX_EXPLORATION_BONUS, alpha * uncertainty)
    return {
        "expected_reward": float(expected_reward),
        "variance": float(variance),
        "uncertainty": float(uncertainty),
        "exploration": float(exploration),
        "score": float(expected_reward + exploration),
        "alpha": float(alpha),
        "updates": int(arm_data.get("updates", 0) or 0),
        "weighted_updates": float(arm_data.get("weighted_updates", 0.0) or 0.0),
        "reward_sum": float(arm_data.get("reward_sum", 0.0) or 0.0),
    }


def _reset_scope(scope: Dict[str, Any], reason: str) -> None:
    cusum = dict(scope.get("cusum") or {})
    scope["arms"] = {arm: _empty_arm() for arm in ARMS}
    scope["cusum"] = {
        "g_plus": 0.0,
        "g_minus": 0.0,
        "last_residual": 0.0,
        "alarm_count": int(cusum.get("alarm_count", 0) or 0) + 1,
        "last_reset_at": _now(),
        "last_reset_reason": str(reason),
        "cooldown_remaining": RESET_COOLDOWN_HANDS,
        "updates_since_reset": 0,
    }


def _cusum_snapshot(scope: Mapping[str, Any]) -> Dict[str, Any]:
    cusum = dict(scope.get("cusum") or {})
    cooldown = max(0, int(cusum.get("cooldown_remaining", 0) or 0))
    updates_since_reset = max(0, int(cusum.get("updates_since_reset", 0) or 0))
    return {
        "enabled": True,
        "g_plus": float(cusum.get("g_plus", 0.0) or 0.0),
        "g_minus": float(cusum.get("g_minus", 0.0) or 0.0),
        "drift_v": CUSUM_DRIFT_V,
        "threshold_h": CUSUM_THRESHOLD_H,
        "alarm_count": int(cusum.get("alarm_count", 0) or 0),
        "last_residual": float(cusum.get("last_residual", 0.0) or 0.0),
        "last_reset_at": int(cusum.get("last_reset_at", 0) or 0),
        "last_reset_reason": str(cusum.get("last_reset_reason") or ""),
        "cooldown_remaining": cooldown,
        "updates_since_reset": updates_since_reset,
        "post_reset_exploration": bool(cooldown > 0 or updates_since_reset < POST_RESET_MIN_UPDATES),
    }


def _road_tie_probability(road: Mapping[str, Any], history: Sequence[str]) -> float:
    observed = road.get("observed_tie_rate")
    if observed is None:
        observed = sum(value == "T" for value in history) / max(1, len(history))
    return _clip(observed, 0.0, 0.25, 0.0)


class ContextualBanditEngine:
    """CUSUM-LinUCB 核心，保留既有 predict_bandit / update_bandit 對外接口。"""

    def reset_model(self, *, user_id: str = "", reason: str = "manual_reset") -> Dict[str, Any]:
        with _LOCK:
            state = _read_state()
            key = _scope_id(user_id)
            scope = dict(state["scopes"].get(key) or _empty_scope())
            _reset_scope(scope, reason)
            scope["updated_at"] = _now()
            state["scopes"][key] = scope
            _write_state(state)
            return {"reset": True, "reason": reason, "cusum": _cusum_snapshot(scope)}

    def predict(
        self,
        history: Iterable[Any],
        *,
        road_context: Optional[Mapping[str, Any]] = None,
        venue: str = "",
        room: str = "",
        user_id: str = "",
        run_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        raw_history = _normalize_history(history)
        road = dict(road_context or {})
        context = _coerce_context(build_context_vector(raw_history, road))
        with _LOCK:
            state = _read_state()
            key = _scope_id(user_id)
            scope = dict(state["scopes"].get(key) or _empty_scope())
            metrics = {arm: _arm_metrics(scope, arm, context) for arm in ARMS}
            cusum = _cusum_snapshot(scope)

        score_b, score_p = metrics["B"]["score"], metrics["P"]["score"]
        road_direction = str(road.get("direction") or "").upper().strip()
        if abs(score_b - score_p) <= 1e-12 and road_direction in ARMS:
            selected_arm = road_direction
        elif abs(score_b - score_p) <= 1e-12:
            # 在完全相同的冷啟動分數下，較少被測試的 Arm 優先，避免 B 永遠
            # 因字典順序先被挑中；仍保持決定性而不使用隨機抽樣。
            selected_arm = min(ARMS, key=lambda arm: (metrics[arm]["updates"], arm))
        else:
            selected_arm = "B" if score_b > score_p else "P"

        conditional_banker = _stable_softmax_banker(score_b, score_p)
        tie_probability = _road_tie_probability(road, raw_history)
        bp_mass = 1.0 - tie_probability
        probabilities = {
            "B": bp_mass * conditional_banker,
            "P": bp_mass * (1.0 - conditional_banker),
            "T": tie_probability,
        }
        edge = abs(2.0 * conditional_banker - 1.0)
        average_uncertainty = (metrics["B"]["uncertainty"] + metrics["P"]["uncertainty"]) / 2.0
        post_reset = bool(cusum["post_reset_exploration"])
        confidence_score = _clip(
            0.36 + 0.34 * edge + 0.12 * min(1.0, len(_bp_history(raw_history)) / 20.0)
            - 0.16 * min(1.0, average_uncertainty / 4.0)
            - (0.18 if post_reset else 0.0),
            0.05, 0.78, 0.35,
        )
        history_fingerprint = sha256("|".join(raw_history).encode("utf-8")).hexdigest()[:24]
        reset_risk = {
            "active": post_reset,
            "cooldown_remaining": int(cusum["cooldown_remaining"]),
            "updates_since_reset": int(cusum["updates_since_reset"]),
            "recommended_max_weight": (
                0.0 if int(cusum["cooldown_remaining"]) > 0
                else 0.08 if post_reset else 0.18
            ),
            "reason": (
                "CUSUM 重置後探索期：上層應限制或暫停此專家的融合權重"
                if post_reset else "CUSUM 專家處於正常融合期"
            ),
        }
        return {
            "ok": True,
            "engine": "CUSUM_LinUCB",
            "mode": "screen_cusum_linu_cb",
            "model_version": MODEL_VERSION,
            "venue": str(venue or ""),
            "room": str(room or ""),
            "user_id": str(user_id or ""),
            "run_seed": run_seed,
            "road_support": road,
            "component_probabilities": dict(road.get("component_probabilities") or {}),
            "context_vector": context.tolist(),
            "bandit_context": context.tolist(),
            "context_feature_names": list(FEATURE_NAMES),
            "selected_arm": selected_arm,
            "base_bandit_direction": selected_arm,
            "base_bandit_direction_text": "莊" if selected_arm == "B" else "閒",
            "action": selected_arm,
            "action_text": "莊" if selected_arm == "B" else "閒",
            "recommend": selected_arm,
            "recommend_text": "莊" if selected_arm == "B" else "閒",
            "internal_action": selected_arm,
            "internal_recommend": selected_arm,
            "next_round_direction": selected_arm,
            "next_round_direction_text": "莊" if selected_arm == "B" else "閒",
            "probabilities": probabilities,
            "pre_braking_probabilities": dict(probabilities),
            "banker_rate": round(probabilities["B"] * 100.0, 2),
            "player_rate": round(probabilities["P"] * 100.0, 2),
            "tie_rate": round(probabilities["T"] * 100.0, 2),
            "direction_edge": float(edge),
            "direction_edge_percent": round(edge * 100.0, 4),
            "confidence": float(confidence_score),
            "confidence_score": float(confidence_score),
            "quality_score": float(confidence_score),
            "confidence_label": "偏低" if confidence_score < 0.50 else "中等" if confidence_score < 0.68 else "較高",
            "bandit_scores": metrics,
            "cusum": cusum,
            "reset_risk": reset_risk,
            "post_reset_exploration": post_reset,
            "signal_allowed": True,
            "signal_status_code": "CUSUM_LINUCB_ACTIVE",
            "signal_status_text": "CUSUM-LinUCB 已產生受控輔助方向",
            "signal_reason": "CUSUM-LinUCB 方向僅供 Adaptive Ensemble 受限融合，不可直接越權接管。",
            "internal_signal_reason": "CUSUM-LinUCB 方向僅供 Adaptive Ensemble 受限融合。",
            "prediction_fingerprint": history_fingerprint,
            "timeline_alignment": {"history_count": len(raw_history), "bp_count": len(_bp_history(raw_history))},
            "contextual_bandit_enabled": True,
            "contextual_bandit_update_enabled": True,
            "uncertainty": float(average_uncertainty),
            "variance": float((metrics["B"]["variance"] + metrics["P"]["variance"]) / 2.0),
            "disclaimer": "方向分數僅供牌路研究與測試，不保證未來開出結果。",
        }

    def update(
        self,
        *,
        context: Sequence[float],
        selected_arm: str,
        reward: Optional[float],
        event_id: str = "",
        actual_outcome: str = "",
        update_weight: float = 1.0,
        user_id: str = "",
        prediction_probabilities: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        arm = str(selected_arm or "").upper().strip()
        actual = str(actual_outcome or "").upper().strip()
        if arm not in ARMS:
            return {"updated": False, "reason": "invalid_arm", "cusum": {}}
        if reward is None or actual == "T":
            return {"updated": False, "reason": "tie_not_used_for_bp_arms", "actual_outcome": actual, "cusum": {}}
        bounded_reward = _clip(reward, 0.0, 1.0, 0.0)
        weight = _clip(update_weight, 0.10, 1.0, 1.0)
        vector = _coerce_context(context)
        event = str(event_id or "").strip()
        with _LOCK:
            state = _read_state()
            key = _scope_id(user_id)
            scope = dict(state["scopes"].get(key) or _empty_scope())
            seen_events = [str(value) for value in list(scope.get("event_ids") or [])]
            if event and event in seen_events:
                return {"updated": False, "reason": "duplicate_event", "event_id": event, "cusum": _cusum_snapshot(scope)}

            metrics_before = _arm_metrics(scope, arm, vector)
            expected_reward = float(metrics_before["expected_reward"])
            if isinstance(prediction_probabilities, Mapping):
                # 使用預測當時 arm 的機率作殘差基準，可避免 update 後重新
                # 估值而產生時間洩漏；沒有提供時才回退 LinUCB 當時 estimate。
                expected_reward = _clip(
                    prediction_probabilities.get(arm, expected_reward), 0.02, 0.98, expected_reward
                )
            residual = bounded_reward - expected_reward
            cusum = dict(scope.get("cusum") or {})
            g_plus = max(0.0, float(cusum.get("g_plus", 0.0) or 0.0) + residual - CUSUM_DRIFT_V)
            g_minus = max(0.0, float(cusum.get("g_minus", 0.0) or 0.0) - residual - CUSUM_DRIFT_V)
            # 浮點累積的 0.4 + 0.4 + ... 可能得到 1.599999999999；以極小
            # 容差維持「達到 h 即告警」的數學契約。
            threshold = CUSUM_THRESHOLD_H - 1e-12
            alarm = g_plus >= threshold or g_minus >= threshold
            if alarm:
                direction = "positive" if g_plus >= g_minus else "negative"
                _reset_scope(scope, f"cusum_{direction}_residual_alarm")
                # 硬重置先完全丟棄舊 regime 的 A/b；再把「觸發警報的最新
                # 已知結果」當成新 regime 的第一筆暖機觀察。這不是保留舊
                # 歷史，而是避免相變發生時連最新資訊也白白丟失。
                matrix = (
                    np.eye(CONTEXT_DIM, dtype=np.float64) * RIDGE
                    + weight * np.outer(vector, vector)
                )
                vector_b = weight * bounded_reward * vector
                arms = dict(scope.get("arms") or {})
                fresh_arm = _empty_arm()
                fresh_arm.update({
                    "A": matrix.tolist(),
                    "b": vector_b.tolist(),
                    "updates": 1,
                    "weighted_updates": float(weight),
                    "reward_sum": float(bounded_reward),
                })
                arms[arm] = fresh_arm
                scope["arms"] = arms
                reset_cusum = dict(scope.get("cusum") or {})
                scope["cusum"] = {
                    **reset_cusum,
                    "last_residual": residual,
                    "updates_since_reset": 1,
                }
                scope["event_ids"] = (seen_events + [event])[-MAX_EVENT_IDS:] if event else seen_events[-MAX_EVENT_IDS:]
                scope["updated_at"] = _now()
                state["scopes"][key] = scope
                _write_state(state)
                return {
                    "updated": True,
                    "reset_triggered": True,
                    "reset_reason": str(scope["cusum"]["last_reset_reason"]),
                    "expected_reward": expected_reward,
                    "reward": bounded_reward,
                    "residual": residual,
                    "update_weight": weight,
                    "actual_outcome": actual,
                    "cusum": _cusum_snapshot(scope),
                    "confidence_score": 0.20,
                    "warm_start_applied": True,
                }

            matrix, vector_b, arm_data = _arm_arrays(scope, arm)
            # Discounted ridge regression：CUSUM 未報警時溫和遺忘，真正相變
            # 則走上方硬重置；兩者組合比單一固定 forgetting factor 穩定。
            matrix = (
                FORGETTING_FACTOR * matrix
                + weight * np.outer(vector, vector)
                + (1.0 - FORGETTING_FACTOR) * np.eye(CONTEXT_DIM, dtype=np.float64) * RIDGE
            )
            vector_b = FORGETTING_FACTOR * vector_b + weight * bounded_reward * vector
            arm_data.update({
                "A": matrix.tolist(),
                "b": vector_b.tolist(),
                "updates": int(arm_data.get("updates", 0) or 0) + 1,
                "weighted_updates": float(arm_data.get("weighted_updates", 0.0) or 0.0) + weight,
                "reward_sum": float(arm_data.get("reward_sum", 0.0) or 0.0) + bounded_reward,
            })
            arms = dict(scope.get("arms") or {})
            arms[arm] = arm_data
            scope["arms"] = arms
            cooldown = max(0, int(cusum.get("cooldown_remaining", 0) or 0) - 1)
            scope["cusum"] = {
                **cusum,
                "g_plus": g_plus,
                "g_minus": g_minus,
                "last_residual": residual,
                "cooldown_remaining": cooldown,
                "updates_since_reset": int(cusum.get("updates_since_reset", 0) or 0) + 1,
            }
            scope["event_ids"] = (seen_events + [event])[-MAX_EVENT_IDS:] if event else seen_events[-MAX_EVENT_IDS:]
            scope["updated_at"] = _now()
            state["scopes"][key] = scope
            _write_state(state)
            return {
                "updated": True,
                "reset_triggered": False,
                "expected_reward": expected_reward,
                "reward": bounded_reward,
                "residual": residual,
                "update_weight": weight,
                "actual_outcome": actual,
                "cusum": _cusum_snapshot(scope),
                "confidence_score": 0.45 if cooldown > 0 else 0.60,
            }

    def summary(self, user_id: str = "") -> Dict[str, Any]:
        with _LOCK:
            state = _read_state()
            scope = dict(state["scopes"].get(_scope_id(user_id)) or _empty_scope())
            zero_context = np.zeros(CONTEXT_DIM, dtype=np.float64)
            arms = {arm: _arm_metrics(scope, arm, zero_context) for arm in ARMS}
            return {
                "model_version": MODEL_VERSION,
                "context_dim": CONTEXT_DIM,
                "feature_names": list(FEATURE_NAMES),
                "arms": arms,
                "cusum": _cusum_snapshot(scope),
                "state_file": str(STATE_FILE),
            }


_DEFAULT_ENGINE = ContextualBanditEngine()


def predict_bandit(
    history: Iterable[Any],
    *,
    road_context: Optional[Mapping[str, Any]] = None,
    venue: str = "",
    room: str = "",
    user_id: str = "",
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    return _DEFAULT_ENGINE.predict(
        history,
        road_context=road_context,
        venue=venue,
        room=room,
        user_id=user_id,
        run_seed=run_seed,
    )


def update_bandit(
    *,
    context: Sequence[float],
    selected_arm: str,
    reward: Optional[float],
    event_id: str = "",
    actual_outcome: str = "",
    update_weight: float = 1.0,
    user_id: str = "",
    prediction_probabilities: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return _DEFAULT_ENGINE.update(
        context=context,
        selected_arm=selected_arm,
        reward=reward,
        event_id=event_id,
        actual_outcome=actual_outcome,
        update_weight=update_weight,
        user_id=user_id,
        prediction_probabilities=prediction_probabilities,
    )


def get_bandit_summary(user_id: str = "") -> Dict[str, Any]:
    return _DEFAULT_ENGINE.summary(user_id=user_id)


def update_decision_strategy(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    """舊版 import 相容函式；策略更新統一由 update_bandit 處理。"""
    return {"updated": False, "reason": "use_update_bandit_for_cusum_linucb"}


__all__ = [
    "ARMS",
    "CONTEXT_DIM",
    "FEATURE_NAMES",
    "MODEL_VERSION",
    "ContextualBanditEngine",
    "build_context_vector",
    "get_bandit_summary",
    "predict_bandit",
    "update_bandit",
    "update_decision_strategy",
]
