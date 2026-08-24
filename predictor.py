"""BGS cMAB 統一預測入口。

正式圖片／真人桌預測使用完整 B/P/T 歷史、牌路上下文與 LinUCB cMAB。
本檔保留既有 API 與虛擬相容工具，但正式方向不再交給 ensemble、
驗證選模或影子控制器覆寫；畫面輸出直接採用 contextual_bandit.py
的原始 B/P Arm 選擇結果。
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from adaptive_ensemble import adapt_prediction
from contextual_bandit import predict_bandit, select_decision_strategy
from particle_filter_points import counts_from_shoe, deal_ordered_hand
from shoe_composition import analyze_shoe_composition, validate_remaining_counts
from validated_decision_layer import apply_strategy_decision, apply_validated_decision

DB_HOLDOUT: Dict[str, Any] = {
    "status": "removed",
    "replacement": "CMAB-LINUCB-V1",
    "note": "舊粒子／有限牌組驗證層已從主要預測流程移除",
}

SHADOW_REQUIRED_CONSECUTIVE_HITS = 2
SHADOW_MAX_STREAMS = 2048

# 測試期間固定使用 cMAB 原始 B/P Arm，避免「沒有精確牌組／EV」時被
# 資金風控層改寫成 O。此模式只用於紀錄方向命中率；下注欄位一律維持 0，
# 不能把牌路方向分數當成已通過 EV 的正式下注訊號。
FORCE_BANDIT_DIRECTION_FOR_TESTING = True


def _bandit_learning_scope(
    *,
    user_id: str,
    venue: str,
    room: str,
) -> str:
    """建立可跨鞋泛化、但不跨使用者／場館／桌污染的 cMAB 範圍。

    回傳不可逆摘要，避免把外部 user id 直接寫進模型狀態檔。
    鞋號仍用於預測去重與影子狀態隔離，但不再重置長期學習矩陣；
    否則每次換鞋都會永久停留在牌路冷啟動與近期順勢退化。
    """
    raw = "|".join((
        str(user_id or "__anonymous__"),
        str(venue or "").upper().strip(),
        str(room or "").strip(),
    ))
    return "__cmab_scope__:" + sha256(
        raw.encode("utf-8")
    ).hexdigest()[:32]


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


def _apply_forced_test_direction(result: Mapping[str, Any]) -> Dict[str, Any]:
    """讓測試面板每局固定呈現 cMAB 原始 B/P，並鎖住所有下注欄位。

    ``apply_strategy_decision`` 仍會先執行，以保留既有資料結構與策略
    稽核資訊；但測試模式不讓它的 No Bet 動作覆寫方向。實際回報結果時，
    performance_tracker 仍可用這個 B/P 方向更新原本的 Contextual Bandit。
    """
    output = dict(result or {})
    direction = str(output.get("bandit_diagnostic_direction") or "").upper()
    if direction not in {"B", "P"}:
        direction = _fallback_direction(output)
    direction_text = "莊" if direction == "B" else "閒"
    reason = "測試模式：固定採用 Contextual Bandit 原始 B/P 方向；不代表 EV 或下注建議。"
    output.update({
        "action": direction,
        "recommend": direction,
        "internal_action": direction,
        "internal_recommend": direction,
        "next_round_direction": direction,
        "action_text": direction_text,
        "recommend_text": direction_text,
        "next_round_direction_text": direction_text,
        "direction_source": "contextual_bandit_raw_test_mode",
        "signal_allowed": False,
        "risk_gate_open": False,
        "test_mode": True,
        "test_mode_name": "forced_contextual_bandit_direction",
        "signal_status_code": "TEST_DIRECTION_ONLY",
        "signal_status_text": "測試模式：已輸出 cMAB 原始方向，未啟用下注。",
        "signal_reason": reason,
        "internal_signal_reason": reason,
        "recommended_bet_percentage": 0.0,
        "bet_percentage": 0.0,
        "kelly_fraction": 0.0,
        "kelly_percentage_applied": 0.0,
        "suggested_bet_amount": 0.0,
        "bet_level_text": "測試模式／不配置",
    })
    return output


def _trusted_physical_signal(
    shoe_context: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """只接受 10 維精確剩餘點數計數作正式 EV 輸入。

    ``observed_cards``、OCR 的剩餘總張數或 B/P/T 路單都無法證明每種點數
    各剩多少張，因此不會被誤升格為可信算牌來源。虛擬牌靴或使用者明確
    輸入並通過驗證的 ``remaining_counts`` 才能開啟這條資金決策管線。
    """
    context = dict(shoe_context or {})
    raw_counts = context.get("remaining_counts")
    if not isinstance(raw_counts, (list, tuple)) or len(raw_counts) != 10:
        signal = analyze_shoe_composition(None)
        signal["trusted_exact_counts"] = False
        signal["trust_reason"] = "missing_exact_remaining_counts"
        return signal
    try:
        decks = int(context.get("decks", 8) or 8)
        counts = validate_remaining_counts(raw_counts, decks=decks)
    except (TypeError, ValueError) as exc:
        signal = analyze_shoe_composition(None)
        signal.update({
            "trusted_exact_counts": False,
            "trust_reason": "invalid_exact_remaining_counts",
            "reason": str(exc),
        })
        return signal
    trusted_context = {
        "remaining_counts": list(counts),
        "decks": decks,
        "source": str(context.get("source") or "user_exact_remaining_counts"),
    }
    signal = analyze_shoe_composition(trusted_context)
    signal["trusted_exact_counts"] = bool(signal.get("available"))
    signal["trust_reason"] = "validated_10_value_remaining_counts"
    return signal


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
        current_probabilities = dict(result.get("probabilities") or {})
        tie_probability = max(
            0.0,
            min(0.30, float(current_probabilities.get("T", 0.0) or 0.0)),
        )
        neutral_bp = (1.0 - tie_probability) * 0.5
        result["probabilities"] = {
            "B": neutral_bp,
            "P": neutral_bp,
            "T": tie_probability,
        }
        result["banker_rate"] = round(neutral_bp * 100.0, 2)
        result["player_rate"] = round(neutral_bp * 100.0, 2)
        result["tie_rate"] = round(tie_probability * 100.0, 2)
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

            current_prefix_hash = sha256(
                "".join(history[:previous_length]).encode("utf-8")
            ).hexdigest()[:24]
            history_replaced = bool(
                len(history) < previous_length
                or (
                    len(history) >= previous_length
                    and previous_hash
                    and previous_hash != current_prefix_hash
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
    """統一預測 API：物理 EV 決策，牌路 cMAB 選擇決策策略。"""
    if history is None:
        history_values: List[Any] = []
    elif isinstance(history, str):
        history_values = [part for part in history.replace("|", ",").split(",") if part.strip()]
    else:
        history_values = list(history)
    cleaned = _normalize_outcome_history(history_values)
    bandit_learning_user_id = _bandit_learning_scope(
        user_id=str(user_id or ""),
        venue=str(venue or ""),
        room=str(room or ""),
    )
    result = predict_bandit(
        cleaned,
        road_context=dict(road_context or {}),
        venue=venue,
        room=room,
        user_id=bandit_learning_user_id,
        run_seed=run_seed,
    )
    # B/P cMAB 原始選擇只保留作可稽核的牌路診斷。正式下注的資格、方向
    # 與基礎 Kelly 由可信 exact remaining counts 的不放回 EV 決定。
    result["bandit_diagnostic_direction"] = _fallback_direction(result)
    result["bandit_diagnostic_selected_arm"] = str(result.get("selected_arm") or "")
    result["road_context"] = dict(road_context or {})
    physical_signal = _trusted_physical_signal(shoe_context)
    strategy_selection = select_decision_strategy(
        cleaned,
        road_context=dict(road_context or {}),
        physical_signal=physical_signal,
        venue=venue,
        room=room,
        user_id=bandit_learning_user_id,
    )
    result["physical_signal"] = physical_signal
    result["decision_strategy_bandit"] = strategy_selection
    result = apply_strategy_decision(
        result,
        strategy_selection=strategy_selection,
        bankroll=float(dict(shoe_context or {}).get("bankroll", 0.0) or 0.0),
    )
    if FORCE_BANDIT_DIRECTION_FOR_TESTING:
        result = _apply_forced_test_direction(result)
        result["decision_pipeline"] = "contextual_bandit_raw_direction_test_only"
        result["direction_overwrite_disabled"] = False
    else:
        result["decision_pipeline"] = "trusted_exact_ev -> strategy_linucb -> capped_kelly"
        result["direction_overwrite_disabled"] = True
    model_fingerprint = str(
        result.get("prediction_fingerprint") or ""
    ).strip()
    result["model_prediction_fingerprint"] = model_fingerprint
    result["prediction_fingerprint"] = sha256(
        "|".join((
            model_fingerprint,
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "__unspecified_shoe__").strip(),
            sha256(
                ",".join(str(value) for value in list(
                    physical_signal.get("remaining_counts") or []
                )).encode("utf-8")
            ).hexdigest()[:12],
        )).encode("utf-8")
    ).hexdigest()[:24]
    result.update({
        "shoe_id": str(shoe_id or ""),
        "bandit_learning_user_id": bandit_learning_user_id,
        "bandit_scope_mode": "user_venue_room_long_term",
        "bandit_shoe_isolated": False,
        "shoe_event_isolated": True,
        "composition_quality": (
            "trusted_exact_remaining_counts"
            if physical_signal.get("trusted_exact_counts")
            else "unavailable_or_untrusted"
        ),
        "remaining_counts_source": str(physical_signal.get("source") or "not_used"),
        "shoe_context_ignored": False,
        "road_quality_ok": bool(dict(road_context or {}).get(
            "quality_ok", dict(road_context or {}).get("recognition_quality_ok", True)
        )),
        "input_required": not bool(physical_signal.get("trusted_exact_counts")),
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
        shoe_context={
            "remaining_counts": counts_from_shoe(hidden_shoe),
            "decks": int(session.get("decks", 8) or 8),
            "source": "virtual_exact_remaining_counts",
            "bankroll": float(session.get("bankroll", 0.0) or 0.0),
        },
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
