"""BGS cMAB 統一預測入口。

正式圖片／真人桌預測使用完整 B/P/T 歷史、牌路上下文與 LinUCB cMAB。
adaptive_ensemble 僅負責 OOD 權重熔斷；不恢復已移除的粒子、
超幾何、蒙地卡羅、Stacking 或 DeepSeek。
"""
from __future__ import annotations

from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import os
import secrets

from adaptive_ensemble import adapt_prediction
from contextual_bandit import predict_bandit
from particle_filter_points import counts_from_shoe, deal_ordered_hand

DB_HOLDOUT: Dict[str, Any] = {
    "status": "removed",
    "replacement": "CMAB-LINUCB-V1",
    "note": "舊粒子／有限牌組驗證層已從主要預測流程移除",
}

SHORT_TERM_SAFE_CONFIDENCE = max(
    0.05,
    min(
        0.95,
        float(os.getenv("SHORT_TERM_SAFE_CONFIDENCE", "0.35") or "0.35"),
    ),
)
SHORT_TERM_TAKEOVER_BET_MULTIPLIER = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv("SHORT_TERM_TAKEOVER_BET_MULTIPLIER", "0.35")
            or "0.35"
        ),
    ),
)


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


class ShortTermTakeoverController:
    """集成信心崩落時的三局 Buffer 接管器。"""

    def __init__(self) -> None:
        self.short_term_buffer: List[str] = []
        self._lock = RLock()

    def apply(
        self,
        history: List[str],
        prediction: Mapping[str, Any],
    ) -> Dict[str, Any]:
        result = dict(prediction or {})
        local_buffer = [value for value in history if value in {"B", "P"}][-3:]
        with self._lock:
            # 僅作最新呼叫的可觀測快取；實際判斷使用 local_buffer，
            # 不會把不同 UID 的牌路混在一起。
            self.short_term_buffer = list(local_buffer)

        adaptive = dict(result.get("adaptive_ensemble") or {})
        confidence = float(
            result.get(
                "ensemble_confidence",
                adaptive.get("overall_confidence", result.get("quality_score", 0.0)),
            )
            or 0.0
        )
        circuit_breaker_active = bool(adaptive.get("circuit_breaker_active"))
        takeover_required = bool(
            circuit_breaker_active
            and confidence < SHORT_TERM_SAFE_CONFIDENCE
        )
        if not takeover_required:
            result["short_term_takeover"] = {
                "active": False,
                "short_term_buffer": list(local_buffer),
                "ensemble_confidence": float(confidence),
                "safe_threshold": float(SHORT_TERM_SAFE_CONFIDENCE),
                "reason": (
                    "未觸發 cMAB 熔斷"
                    if not circuit_breaker_active
                    else "熔斷後整體信心仍高於安全閾值"
                ),
            }
            return result

        prior = _short_term_trend_prior(
            local_buffer,
            _fallback_direction(result),
        )
        direction = str(prior["direction"])
        probabilities = dict(prior["probabilities"])
        original_predictor_signal = dict(result.get("predictor_signal") or {})
        original_probabilities = dict(result.get("probabilities") or {})

        result.update({
            "ensemble_probabilities_before_takeover": original_probabilities,
            "bandit_risk_signal": original_predictor_signal,
            "probabilities": probabilities,
            "banker_rate": round(probabilities["B"] * 100.0, 2),
            "player_rate": round(probabilities["P"] * 100.0, 2),
            "tie_rate": 0.0,
            "recommend": direction,
            "recommend_text": "莊" if direction == "B" else "閒",
            "action": direction,
            "action_text": "莊" if direction == "B" else "閒",
            "internal_recommend": direction,
            "internal_action": direction,
            "selected_arm": direction,
            "next_round_direction": direction,
            "next_round_direction_text": "莊" if direction == "B" else "閒",
            "signal_allowed": True,
            "signal_status_code": "SHORT_TERM_META_TREND_TAKEOVER",
            "signal_status_text": "極端未知區間：三局微觀順勢接管",
            "signal_reason": (
                "cMAB 因極端未知特徵遭動態熔斷，整體信心低於安全閾值；"
                f"改由最近 3 局策略 {prior['strategy']} 輸出短週期方向。"
            ),
            "internal_signal_reason": (
                "全局聯動防禦已啟動：熔斷後由三局微觀趨勢接管"
            ),
            "direction_source": "predictor_short_term_meta_takeover",
            "ensemble_confidence_before_takeover": float(confidence),
            "ensemble_confidence": float(prior["strength"]),
            "short_term_confidence": float(prior["strength"]),
            "bet_multiplier": float(SHORT_TERM_TAKEOVER_BET_MULTIPLIER),
            "confidence_label": "短週期接管",
            "predictor_signal": {
                "code": "SHORT_TERM_META_TREND_TAKEOVER",
                "is_extreme_unseen": bool(result.get("is_extreme_unseen")),
                "variance": float(result.get("variance", 0.0) or 0.0),
                "meta_learning_takeover": True,
                "short_term_buffer": list(local_buffer),
                "strategy": str(prior["strategy"]),
                "direction": direction,
                "confidence": float(prior["strength"]),
                "bet_multiplier": float(SHORT_TERM_TAKEOVER_BET_MULTIPLIER),
            },
            "meta_learning_takeover": {
                "active": True,
                **prior,
                "trigger": "ensemble_confidence_below_safe_threshold",
                "ensemble_confidence_before_takeover": float(confidence),
                "safe_threshold": float(SHORT_TERM_SAFE_CONFIDENCE),
            },
            "short_term_takeover": {
                "active": True,
                **prior,
                "ensemble_confidence_before_takeover": float(confidence),
                "safe_threshold": float(SHORT_TERM_SAFE_CONFIDENCE),
                "bet_multiplier": float(SHORT_TERM_TAKEOVER_BET_MULTIPLIER),
            },
        })
        adaptive["short_term_takeover_applied"] = True
        adaptive["final_direction_source"] = "predictor_short_term_meta_takeover"
        result["adaptive_ensemble"] = adaptive
        return result


_SHORT_TERM_CONTROLLER = ShortTermTakeoverController()


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
    # 全局閉環：cMAB 風險訊號 -> 集成熔斷 -> 低信心三局接管。
    result = adapt_prediction(
        result,
        venue=str(venue or ""),
        room=str(room or ""),
    )
    result = _SHORT_TERM_CONTROLLER.apply(cleaned, result)
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
    predicted_side = str(prediction.get("recommend") or "").upper()
    actual = str(hand.outcome or "").upper()
    verdict = "TIE_SKIPPED" if actual == "T" else "HIT" if predicted_side == actual else "MISS"
    prediction.update({
        "ok": True,
        "mode": "virtual_shoe_cmab_compatibility",
        "model_version": "CMAB-LINUCB-V1-VIRTUAL-COMPAT",
        "virtual_hand": hand_data,
        "virtual_outcome": actual,
        "virtual_outcome_text": hand_data["outcome_text"],
        "verdict": verdict,
        "verdict_text": {"HIT": "命中", "MISS": "未命中", "TIE_SKIPPED": "和局不計"}[verdict],
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


__all__ = ["DB_HOLDOUT", "parse_point_observation", "predict", "run_virtual_round"]
