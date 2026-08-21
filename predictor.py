"""BGS cMAB 統一預測入口。

正式圖片／真人桌預測使用完整 B/P/T 歷史、牌路上下文與 LinUCB cMAB。
adaptive_ensemble 負責統計混沌硬熔斷；predictor 只在後台執行
三局影子回測，達成連中 2 局且模型方差安全後才解除 No Bet。
不恢復已移除的粒子、超幾何、蒙地卡羅、Stacking 或 DeepSeek。
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from adaptive_ensemble import adapt_prediction
from contextual_bandit import predict_bandit
from particle_filter_points import counts_from_shoe, deal_ordered_hand

DB_HOLDOUT: Dict[str, Any] = {
    "status": "removed",
    "replacement": "CMAB-LINUCB-V1",
    "note": "舊粒子／有限牌組驗證層已從主要預測流程移除",
}

SHADOW_REQUIRED_CONSECUTIVE_HITS = 2
SHADOW_MAX_STREAMS = 2048


def _normalize_outcome_history(values: Iterable[Any]) -> List[str]:
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


def _fallback_direction(prediction: Mapping[str, Any]) -> str:
    for key in (
        "internal_recommend",
        "selected_arm",
        "recommend",
        "base_bandit_direction",
    ):
        direction = str(prediction.get(key) or "").upper().strip()
        if direction in {"B", "P"}:
            return direction
    banker = float(prediction.get("banker_rate", 0.0) or 0.0)
    player = float(prediction.get("player_rate", 0.0) or 0.0)
    return "B" if banker >= player else "P"


def _short_term_trend_prior(
    buffer: List[str],
    fallback_direction: str,
) -> Dict[str, Any]:
    """只依最近 3 個 B/P 建立微觀順勢先驗；T 不改變 B/P 走勢。"""
    fallback = fallback_direction if fallback_direction in {"B", "P"} else "B"
    if len(buffer) >= 2 and buffer[-1] == buffer[-2]:
        direction = buffer[-1]
        strategy = "follow_last_two_streak"
        strength = 0.60
        evidence = buffer[-2:]
    elif (
        len(buffer) >= 3
        and buffer[-3] == buffer[-1]
        and buffer[-2] != buffer[-1]
    ):
        direction = "P" if buffer[-1] == "B" else "B"
        strategy = "continue_three_step_alternation"
        strength = 0.57
        evidence = buffer[-3:]
    elif len(buffer) >= 3:
        banker_count = sum(value == "B" for value in buffer)
        direction = "B" if banker_count >= 2 else "P"
        strategy = "recent_three_majority"
        strength = 0.55
        evidence = buffer[-3:]
    else:
        direction = fallback
        strategy = "insufficient_micro_history_fallback"
        strength = 0.52
        evidence = buffer[-3:]

    opposite = "P" if direction == "B" else "B"
    return {
        "direction": direction,
        "direction_text": "莊" if direction == "B" else "閒",
        "strategy": strategy,
        "strength": float(strength),
        "evidence": list(evidence),
        "short_term_buffer": list(buffer[-3:]),
        "probabilities": {
            direction: float(strength),
            opposite: float(1.0 - strength),
            "T": 0.0,
        },
    }


class ShadowBacktestController:
    """No Bet 期間的三局影子回測與雙條件解鎖狀態機。"""

    def __init__(self) -> None:
        self.shadow_buffer: List[str] = []
        self._states: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def stream_key(
        *,
        user_id: str,
        venue: str,
        room: str,
        shoe_id: str,
    ) -> str:
        raw = "|".join((
            str(user_id or "__anonymous__"),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "").strip(),
        ))
        return sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _new_state(history: List[str]) -> Dict[str, Any]:
        return {
            "hard_brake_latched": False,
            "pending_direction": "",
            "consecutive_hits": 0,
            "total_shadow_bets": 0,
            "shadow_hits": 0,
            "shadow_misses": 0,
            "ties_skipped": 0,
            "last_history_length": len(history),
            "last_history_hash": sha256(
                "".join(history).encode("utf-8")
            ).hexdigest()[:24],
        }

    @staticmethod
    def _variance_status(prediction: Mapping[str, Any]) -> tuple[float, float, bool]:
        braking = dict(prediction.get("uncertainty_braking") or {})
        variance = float(
            prediction.get(
                "variance",
                braking.get("action_space_variance", 0.0),
            )
            or 0.0
        )
        threshold = float(
            prediction.get(
                "variance_threshold",
                braking.get("threshold_variance", float("inf")),
            )
            or float("inf")
        )
        variance_safe = bool(
            braking.get("variance_safe", variance <= threshold)
        )
        return variance, threshold, variance_safe

    @staticmethod
    def _force_no_bet(
        result: Dict[str, Any],
        *,
        reason: str,
    ) -> None:
        result.setdefault(
            "pre_hard_brake_probabilities",
            dict(result.get("probabilities") or {}),
        )
        result["probabilities"] = {"B": 0.5, "P": 0.5, "T": 0.0}
        result["banker_rate"] = 50.0
        result["player_rate"] = 50.0
        result["tie_rate"] = 0.0
        result["recommend"] = "O"
        result["recommend_text"] = "觀望"
        result["action"] = "O"
        result["action_text"] = "觀望／絕對不下注"
        result["internal_recommend"] = "O"
        result["internal_action"] = "O"
        result["next_round_direction"] = "O"
        result["next_round_direction_text"] = "觀望"
        result["signal_allowed"] = False
        result["signal_status_code"] = "HARD_BRAKE_NO_BET"
        result["signal_status_text"] = "硬熔斷鎖定：觀望／絕對不下注"
        result["signal_reason"] = reason
        result["internal_signal_reason"] = reason
        result["direction_source"] = "shadow_backtest_hard_brake"
        result["ensemble_confidence"] = 0.0
        result["confidence"] = 0.0
        result["quality_score"] = 0.0
        result["confidence_label"] = "零信心／硬熔斷"
        result["bet_multiplier"] = 0.0
        result["hard_brake_active"] = True

    def apply(
        self,
        history: List[str],
        prediction: Mapping[str, Any],
        *,
        stream_key: str = "__default__",
    ) -> Dict[str, Any]:
        result = dict(prediction or {})
        local_buffer = [value for value in history if value in {"B", "P"}][-3:]
        history_hash = sha256("".join(history).encode("utf-8")).hexdigest()[:24]
        current_extreme = bool(
            result.get("is_extreme_unseen")
            or result.get("hard_brake_active")
        )
        variance, variance_threshold, variance_safe = self._variance_status(result)

        with self._lock:
            self.shadow_buffer = list(local_buffer)
            state = self._states.get(stream_key)
            if state is None:
                state = self._new_state(history)
                self._states[stream_key] = state
            self._states.move_to_end(stream_key)
            while len(self._states) > SHADOW_MAX_STREAMS:
                self._states.popitem(last=False)

            previous_length = int(state.get("last_history_length", len(history)))
            previous_hash = str(state.get("last_history_hash") or "")
            pending_direction = str(state.get("pending_direction") or "")
            shadow_result = "PENDING"
            resolved_actual = ""

            history_replaced = bool(
                len(history) < previous_length
                or (
                    len(history) == previous_length
                    and previous_hash
                    and previous_hash != history_hash
                )
            )
            if history_replaced:
                state = self._new_state(history)
                self._states[stream_key] = state
                pending_direction = ""
                shadow_result = "STREAM_RESET"
            elif len(history) > previous_length and pending_direction in {"B", "P"}:
                new_outcomes = history[previous_length:]
                resolved_actual = next(
                    (value for value in new_outcomes if value in {"B", "P"}),
                    "",
                )
                if resolved_actual:
                    state["total_shadow_bets"] = int(
                        state.get("total_shadow_bets", 0) or 0
                    ) + 1
                    if pending_direction == resolved_actual:
                        state["consecutive_hits"] = int(
                            state.get("consecutive_hits", 0) or 0
                        ) + 1
                        state["shadow_hits"] = int(
                            state.get("shadow_hits", 0) or 0
                        ) + 1
                        shadow_result = "HIT"
                    else:
                        state["consecutive_hits"] = 0
                        state["shadow_misses"] = int(
                            state.get("shadow_misses", 0) or 0
                        ) + 1
                        shadow_result = "MISS"
                    state["pending_direction"] = ""
                    pending_direction = ""
                elif any(value == "T" for value in new_outcomes):
                    state["ties_skipped"] = int(
                        state.get("ties_skipped", 0) or 0
                    ) + 1
                    shadow_result = "TIE_SKIPPED"

            if current_extreme:
                state["hard_brake_latched"] = True

            consecutive_hits = int(state.get("consecutive_hits", 0) or 0)
            release_allowed = bool(
                state.get("hard_brake_latched")
                and not current_extreme
                and variance_safe
                and consecutive_hits >= SHADOW_REQUIRED_CONSECUTIVE_HITS
            )
            released_this_round = False
            if release_allowed:
                state["hard_brake_latched"] = False
                state["pending_direction"] = ""
                pending_direction = ""
                released_this_round = True

            hard_brake_latched = bool(state.get("hard_brake_latched"))
            shadow_prior: Dict[str, Any] = {}
            if hard_brake_latched:
                if pending_direction not in {"B", "P"}:
                    shadow_prior = _short_term_trend_prior(
                        local_buffer,
                        _fallback_direction(result),
                    )
                    pending_direction = str(shadow_prior["direction"])
                    state["pending_direction"] = pending_direction
                else:
                    shadow_prior = _short_term_trend_prior(
                        local_buffer,
                        pending_direction,
                    )

            state["last_history_length"] = len(history)
            state["last_history_hash"] = history_hash
            self._states[stream_key] = state

            shadow_payload = {
                "active": bool(hard_brake_latched),
                "hard_brake_latched": bool(hard_brake_latched),
                "released_this_round": bool(released_this_round),
                "shadow_buffer": list(local_buffer),
                "pending_direction": pending_direction,
                "pending_direction_text": (
                    "莊" if pending_direction == "B"
                    else "閒" if pending_direction == "P"
                    else ""
                ),
                "strategy": str(shadow_prior.get("strategy") or ""),
                "last_shadow_result": shadow_result,
                "last_resolved_actual": resolved_actual,
                "consecutive_hits": int(state.get("consecutive_hits", 0) or 0),
                "required_consecutive_hits": SHADOW_REQUIRED_CONSECUTIVE_HITS,
                "total_shadow_bets": int(state.get("total_shadow_bets", 0) or 0),
                "shadow_hits": int(state.get("shadow_hits", 0) or 0),
                "shadow_misses": int(state.get("shadow_misses", 0) or 0),
                "ties_skipped": int(state.get("ties_skipped", 0) or 0),
                "variance": float(variance),
                "variance_threshold": float(variance_threshold),
                "variance_safe": bool(variance_safe),
                "model_is_extreme_unseen": bool(current_extreme),
                "release_allowed": bool(release_allowed),
            }

        result["shadow_backtest"] = shadow_payload
        result["shadow_buffer"] = list(local_buffer)
        if hard_brake_latched:
            self._force_no_bet(
                result,
                reason=(
                    "統計混沌熔斷仍鎖定；只有影子回測連中 2 局且"
                    "模型方差降回安全門檻，才允許恢復正式方向。"
                ),
            )
            adaptive = dict(result.get("adaptive_ensemble") or {})
            adaptive.update({
                "active": True,
                "circuit_breaker_active": True,
                "hard_brake_active": True,
                "mode": "shadow_backtest_latched_hard_brake",
                "overall_confidence": 0.0,
                "final_action": "O",
                "bet_multiplier": 0.0,
                "shadow_backtest_required": True,
                "shadow_consecutive_hits": shadow_payload["consecutive_hits"],
                "variance_safe": bool(variance_safe),
            })
            result["adaptive_ensemble"] = adaptive
        elif released_this_round:
            result["hard_brake_released"] = True
            result["signal_reason"] = (
                "影子回測已連中 2 局且模型方差回到安全門檻，解除硬熔斷。"
            )
        return result


_SHADOW_CONTROLLER = ShadowBacktestController()
# 保留前一版類別／單例名稱，避免外部整合在部署過渡期 import 失敗；
# 行為已統一改為 No Bet 影子回測，不會恢復舊的實盤短線接管。
ShortTermTakeoverController = ShadowBacktestController
_SHORT_TERM_CONTROLLER = _SHADOW_CONTROLLER


def predict(history: Union[str, Iterable[Any], None] = None, venue: str = "", room: str = "",
            shoe_id: str = "", user_id: str = "", run_seed: Optional[int] = None,
            shoe_context: Optional[Mapping[str, Any]] = None,
            road_context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """統一預測 API；保留舊參數名稱以相容 app.py。"""
    if history is None:
        history_values: List[Any] = []
    elif isinstance(history, str):
        history_values = [part for part in history.replace("|", ",").split(",") if part.strip()]
    else:
        history_values = list(history)
    cleaned = _normalize_outcome_history(history_values)
    result = predict_bandit(
        cleaned,
        road_context=dict(road_context or {}),
        venue=venue,
        room=room,
        user_id=user_id,
        run_seed=run_seed,
    )
    # 全局閉環：統計風險訊號 -> 集成硬熔斷 -> 三局影子回測解鎖。
    result = adapt_prediction(
        result,
        venue=str(venue or ""),
        room=str(room or ""),
    )
    result = _SHADOW_CONTROLLER.apply(
        cleaned,
        result,
        stream_key=_SHADOW_CONTROLLER.stream_key(
            user_id=str(user_id or ""),
            venue=str(venue or ""),
            room=str(room or ""),
            shoe_id=str(shoe_id or ""),
        ),
    )
    result.update({
        "shoe_id": str(shoe_id or ""),
        "composition_quality": "not_applicable_cmab",
        "remaining_counts_source": "not_used",
        "shoe_context_ignored": bool(shoe_context),
        "road_quality_ok": bool(dict(road_context or {}).get(
            "quality_ok", dict(road_context or {}).get("recognition_quality_ok", True)
        )),
        "input_required": False,
    })
    return result


def run_virtual_round(session: Mapping[str, Any], run_seed: Optional[int] = None) -> Dict[str, Any]:
    """保留舊虛擬牌靴介面，但方向同樣由 cMAB 產生。"""
    hidden_shoe = [int(card) for card in list(session.get("virtual_shoe") or [])]
    if len(hidden_shoe) < 6:
        raise ValueError("虛擬牌靴不足，請重新建立牌靴。")
    outcome_history = _normalize_outcome_history(list(session.get("round_history") or []))
    seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    prediction = predict(
        history=outcome_history,
        venue=str(session.get("venue") or ""),
        room=str(session.get("room") or ""),
        shoe_id=str(session.get("shoe_id") or ""),
        user_id=str(session.get("user_id") or ""),
        run_seed=seed,
        road_context=None,
    )
    hand, remaining_shoe = deal_ordered_hand(hidden_shoe)
    hand_data = hand.as_dict()
    predicted_side = str(
        prediction.get("action")
        or prediction.get("recommend")
        or ""
    ).upper()
    actual = str(hand.outcome or "").upper()
    verdict = (
        "OBSERVE"
        if predicted_side == "O"
        else "TIE_SKIPPED"
        if actual == "T"
        else "HIT"
        if predicted_side == actual
        else "MISS"
    )
    prediction.update({
        "ok": True,
        "mode": "virtual_shoe_cmab_compatibility",
        "model_version": "CMAB-LINUCB-V1-VIRTUAL-COMPAT",
        "virtual_hand": hand_data,
        "virtual_outcome": actual,
        "virtual_outcome_text": hand_data["outcome_text"],
        "verdict": verdict,
        "verdict_text": {
            "HIT": "命中",
            "MISS": "未命中",
            "TIE_SKIPPED": "和局不計",
            "OBSERVE": "觀望／不計勝負",
        }[verdict],
        "cards_consumed": int(hand.cards_used),
        "remaining_cards_after": len(remaining_shoe),
        "remaining_counts_after": counts_from_shoe(remaining_shoe),
        "round_number": int(session.get("hand_number", 0) or 0) + 1,
        "warmup_rounds": int(session.get("warmup_rounds", 0) or 0),
        "bandit_learning_applied": False,
        "disclaimer": "虛擬相容模式方向由 cMAB 產生；虛擬結果不回寫正式 cMAB。",
    })
    return {"prediction": prediction, "hand": hand_data, "remaining_shoe": remaining_shoe}


def parse_point_observation(value: Any) -> None:
    return None


__all__ = [
    "DB_HOLDOUT",
    "ShadowBacktestController",
    "ShortTermTakeoverController",
    "parse_point_observation",
    "predict",
    "run_virtual_round",
]
