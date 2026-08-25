"""BGS unified predictor: Full Road Adaptive Ensemble + CUSUM-LinUCB.

The road ensemble remains the structural primary source. CUSUM-LinUCB is a dynamic
expert that is replayed prequentially from already observed road history and can
influence the final B/P direction only according to its scheduler weight.

After a CUSUM change-point reset:
- hands 1-3: CUSUM expert weight = 0 and public action = Observe (O)
- hands 4-5: CUSUM expert re-enters with a small weight and reduced bet multiplier
- afterwards: its weight grows with confidence, capped below the road ensemble

Returned percentages are model direction scores, not guaranteed outcome probabilities.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from adaptive_ensemble import adapt_prediction
from contextual_bandit import MODEL_VERSION as CUSUM_MODEL_VERSION, predict_bandit
from road_model import build_road_context

ADAPTIVE_MODEL_VARIANT = "V35.0_ROAD_ADAPTIVE_PLUS_CUSUM_LINUCB"
MAX_CUSUM_ENSEMBLE_WEIGHT = 0.45

DB_HOLDOUT: Dict[str, Any] = {
    "status": "removed",
    "replacement": "FULL_ROAD_ADAPTIVE_PLUS_CUSUM_LINUCB",
    "note": "正式方向由 Full Road Adaptive 與 CUSUM-LinUCB 動態融合；不使用舊算牌/粒子層。",
}


def _normalize_outcome_history(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = (
                item.get("outcome")
                or item.get("actual")
                or item.get("actual_outcome")
                or item.get("virtual_outcome")
            )
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            result.append(value)
    return result[-2000:]


def _bandit_learning_scope(*, user_id: str, venue: str, room: str) -> str:
    raw = "|".join((
        str(user_id or "__anonymous__"),
        str(venue or "").upper().strip(),
        str(room or "").strip(),
    ))
    return "__cusum_scope__:" + sha256(raw.encode("utf-8")).hexdigest()[:32]


def _road_seed_prediction(road: Mapping[str, Any], history: Iterable[Any]) -> Dict[str, Any]:
    try:
        banker = float(road.get("banker_probability", 0.5) or 0.5)
    except (TypeError, ValueError):
        banker = 0.5
    try:
        player = float(road.get("player_probability", 1.0 - banker) or 0.0)
    except (TypeError, ValueError):
        player = 1.0 - banker
    bp_total = banker + player
    if bp_total <= 1e-12:
        banker, player = 0.5, 0.5
    else:
        banker, player = banker / bp_total, player / bp_total

    try:
        observed_tie_rate = float(road.get("observed_tie_rate", 0.0) or 0.0)
    except (TypeError, ValueError):
        observed_tie_rate = 0.0
    tie = max(0.0, min(0.30, observed_tie_rate))
    bp_mass = 1.0 - tie
    banker_probability = bp_mass * banker
    player_probability = bp_mass * player

    direction = str(road.get("direction") or "").upper().strip()
    if direction not in {"B", "P"}:
        direction = "B" if banker >= player else "P"

    normalized_history = _normalize_outcome_history(history)
    history_fingerprint = sha256(
        "|".join(normalized_history).encode("utf-8")
    ).hexdigest()[:24]
    return {
        "model_version": "FULL_ROAD_ADAPTIVE_V35.0",
        "model_variant": ADAPTIVE_MODEL_VARIANT,
        "prediction_fingerprint": history_fingerprint,
        "probabilities": {
            "B": float(banker_probability),
            "P": float(player_probability),
            "T": float(tie),
        },
        "raw_probabilities": {
            "B": float(banker_probability),
            "P": float(player_probability),
            "T": float(tie),
        },
        "banker_rate": round(banker_probability * 100.0, 2),
        "player_rate": round(player_probability * 100.0, 2),
        "tie_rate": round(tie * 100.0, 2),
        "action": direction,
        "recommend": direction,
        "internal_action": direction,
        "internal_recommend": direction,
        "next_round_direction": direction,
    }


def _normalize_probabilities(values: Any) -> Dict[str, float]:
    if not isinstance(values, Mapping):
        return {"B": 0.455, "P": 0.455, "T": 0.09}
    raw = {
        key: max(0.0, float(values.get(key, 0.0) or 0.0))
        for key in ("B", "P", "T")
    }
    total = sum(raw.values())
    if total <= 1e-12:
        return {"B": 0.455, "P": 0.455, "T": 0.09}
    return {key: raw[key] / total for key in raw}


def _conditional_banker(values: Any) -> float:
    probabilities = _normalize_probabilities(values)
    bp_total = probabilities["B"] + probabilities["P"]
    if bp_total <= 1e-12:
        return 0.5
    return float(probabilities["B"] / bp_total)


def _safe_confidence(result: Mapping[str, Any], default: float = 0.5) -> float:
    for key in ("ensemble_confidence", "confidence", "quality_score"):
        try:
            value = float(result.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            return max(0.0, min(1.0, value))
    return max(0.0, min(1.0, default))


def _fuse_adaptive_and_cusum(
    adaptive_prediction: Mapping[str, Any],
    bandit_prediction: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fuse Adaptive road model and CUSUM expert with post-reset scheduling."""
    result = dict(adaptive_prediction or {})
    bandit = dict(bandit_prediction or {})
    risk = dict(bandit.get("risk_control") or {})

    adaptive_probs = _normalize_probabilities(
        result.get("adaptive_only_probabilities")
        if isinstance(result.get("adaptive_only_probabilities"), Mapping)
        else result.get("probabilities")
    )
    bandit_probs = _normalize_probabilities(bandit.get("probabilities"))
    adaptive_banker = _conditional_banker(adaptive_probs)
    bandit_banker = _conditional_banker(bandit_probs)

    try:
        requested_weight = float(
            bandit.get(
                "ensemble_weight_suggestion",
                risk.get("ensemble_weight_suggestion", 0.0),
            ) or 0.0
        )
    except (TypeError, ValueError):
        requested_weight = 0.0
    bandit_weight = max(0.0, min(MAX_CUSUM_ENSEMBLE_WEIGHT, requested_weight))

    force_observe = bool(bandit.get("force_observe") or risk.get("force_observe"))
    vacuum_active = bool(
        bandit.get("post_reset_vacuum_active")
        or risk.get("post_reset_vacuum_active")
    )
    if force_observe:
        bandit_weight = 0.0

    road_weight = 1.0 - bandit_weight
    conditional_banker = (
        road_weight * adaptive_banker + bandit_weight * bandit_banker
    )
    conditional_banker = max(1e-6, min(1.0 - 1e-6, conditional_banker))

    tie_probability = max(0.0, min(0.30, adaptive_probs["T"]))
    bp_mass = 1.0 - tie_probability
    banker_probability = bp_mass * conditional_banker
    player_probability = bp_mass * (1.0 - conditional_banker)

    adaptive_direction = str(
        result.get("adaptive_only_direction")
        or result.get("action")
        or result.get("recommend")
        or ""
    ).upper().strip()
    if adaptive_direction not in {"B", "P"}:
        adaptive_direction = "B" if adaptive_banker >= 0.5 else "P"

    bandit_direction = str(
        bandit.get("selected_arm")
        or bandit.get("next_round_direction")
        or bandit.get("recommend")
        or ""
    ).upper().strip()
    if bandit_direction not in {"B", "P"}:
        bandit_direction = "B" if bandit_banker >= 0.5 else "P"

    if abs(conditional_banker - 0.5) <= 1e-12:
        final_direction = adaptive_direction
    else:
        final_direction = "B" if conditional_banker > 0.5 else "P"

    adaptive_confidence = _safe_confidence(result, 0.5)
    try:
        bandit_confidence = float(
            bandit.get("confidence_score", risk.get("confidence_score", 0.0)) or 0.0
        )
    except (TypeError, ValueError):
        bandit_confidence = 0.0
    bandit_confidence = max(0.0, min(1.0, bandit_confidence))
    confidence = road_weight * adaptive_confidence + bandit_weight * bandit_confidence
    if vacuum_active:
        confidence = min(confidence, 0.50 if not force_observe else 0.25)

    try:
        risk_bet_multiplier = float(
            bandit.get("bet_multiplier", risk.get("bet_multiplier", 1.0)) or 0.0
        )
    except (TypeError, ValueError):
        risk_bet_multiplier = 1.0
    risk_bet_multiplier = max(0.0, min(1.0, risk_bet_multiplier))
    bet_multiplier = 0.0 if force_observe else risk_bet_multiplier

    result.update({
        "model_version": f"FULL-ROAD-ADAPTIVE+CUSUM-LINUCB::{CUSUM_MODEL_VERSION}",
        "model_variant": ADAPTIVE_MODEL_VARIANT,
        "decision_pipeline": "full_road_adaptive_then_cusum_linucb_dynamic_fusion",
        "probabilities": {
            "B": float(banker_probability),
            "P": float(player_probability),
            "T": float(tie_probability),
        },
        "banker_rate": round(banker_probability * 100.0, 2),
        "player_rate": round(player_probability * 100.0, 2),
        "tie_rate": round(tie_probability * 100.0, 2),
        "adaptive_only_direction": adaptive_direction,
        "bandit_only_direction": bandit_direction,
        "contextual_bandit_enabled": True,
        "contextual_bandit_update_enabled": True,
        "cusum_linucb_enabled": True,
        "ucb_influenced_final_direction": bool(bandit_weight > 0.0),
        "post_reset_vacuum_active": vacuum_active,
        "force_observe": force_observe,
        "ensemble_confidence": float(confidence),
        "confidence": float(confidence),
        "quality_score": float(confidence),
        "confidence_label": (
            "重置探索期" if vacuum_active else
            "較高" if confidence >= 0.72 else
            "中等" if confidence >= 0.55 else "偏低"
        ),
        "bet_multiplier": float(bet_multiplier),
        "cusum_bandit": bandit,
        "ensemble_scheduler": {
            "road_weight": float(road_weight),
            "cusum_linucb_weight": float(bandit_weight),
            "requested_cusum_weight": float(requested_weight),
            "max_cusum_weight": float(MAX_CUSUM_ENSEMBLE_WEIGHT),
            "post_reset_vacuum_active": vacuum_active,
            "force_observe": force_observe,
            "observations_since_reset": int(
                risk.get(
                    "observations_since_reset",
                    bandit.get("observations_since_reset", 0),
                ) or 0
            ),
            "reason": (
                "CUSUM 剛重置：前 3 手 Bandit 權重歸零並強制觀望"
                if force_observe else
                "CUSUM 重置後第 4-5 手：Bandit 小權重逐步恢復"
                if vacuum_active else
                "正常區間：依 CUSUM-LinUCB confidence 動態分配權重"
            ),
        },
    })

    adaptive = dict(result.get("adaptive_ensemble") or {})
    adaptive.update({
        "active": True,
        "mode": "adaptive_road_plus_cusum_linucb",
        "contextual_bandit_enabled": True,
        "cusum_linucb_enabled": True,
        "cusum_linucb_weight": float(bandit_weight),
        "road_weight": float(road_weight),
        "post_reset_vacuum_active": vacuum_active,
        "overall_confidence": float(confidence),
        "bet_multiplier": float(bet_multiplier),
    })
    result["adaptive_ensemble"] = adaptive

    final_text = "莊" if final_direction == "B" else "閒"
    if force_observe:
        result.update({
            "recommend": "O",
            "recommend_text": "觀望",
            "action": "O",
            "action_text": "觀望／CUSUM 重置探索期",
            "internal_recommend": "O",
            "internal_action": "O",
            "next_round_direction": final_direction,
            "next_round_direction_text": final_text,
            "shadow_direction": final_direction,
            "signal_allowed": False,
            "signal_status_code": "CUSUM_POST_RESET_VACUUM_OBSERVE",
            "signal_status_text": "CUSUM 重置後真空探索期：暫停正式下注",
            "signal_reason": "變點已觸發硬重置；前 3 個新 regime 樣本僅探索，不讓 Bandit 影響正式下注。",
            "internal_signal_reason": "CUSUM_POST_RESET_VACUUM",
            "direction_source": "cusum_post_reset_vacuum_observe",
            "hard_brake_active": True,
            "is_extreme_unseen": True,
        })
        adaptive["hard_brake_active"] = True
        adaptive["circuit_breaker_active"] = True
        adaptive["final_action"] = "O"
    else:
        result.update({
            "recommend": final_direction,
            "recommend_text": final_text,
            "action": final_direction,
            "action_text": final_text,
            "internal_recommend": final_direction,
            "internal_action": final_direction,
            "next_round_direction": final_direction,
            "next_round_direction_text": final_text,
            "signal_allowed": True,
            "signal_status_code": "ADAPTIVE_CUSUM_LINUCB_FUSION",
            "signal_status_text": "Full Road Adaptive + CUSUM-LinUCB 動態融合",
            "signal_reason": f"Adaptive 權重 {road_weight:.3f}；CUSUM-LinUCB 權重 {bandit_weight:.3f}。",
            "internal_signal_reason": "ADAPTIVE_CUSUM_DYNAMIC_WEIGHT",
            "direction_source": "adaptive_cusum_linucb_fusion",
            "hard_brake_active": False,
            "is_extreme_unseen": False,
        })
        adaptive["hard_brake_active"] = False
        adaptive["circuit_breaker_active"] = False
        adaptive["final_action"] = final_direction
    result["adaptive_ensemble"] = adaptive

    result["direction_edge"] = float(abs(2.0 * conditional_banker - 1.0))
    result["direction_edge_percent"] = round(result["direction_edge"] * 100.0, 4)
    return result


class ShadowBacktestController:
    """Compatibility wrapper retained for callers from earlier predictor versions."""
    def __init__(self) -> None:
        self.shadow_buffer: List[str] = []

    @staticmethod
    def stream_key(*, user_id: str, venue: str, room: str, shoe_id: str) -> str:
        raw = "|".join((
            str(user_id or "__anonymous__"),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "").strip(),
        ))
        return sha256(raw.encode("utf-8")).hexdigest()[:24]

    def apply(
        self,
        history: List[str],
        prediction: Mapping[str, Any],
        *,
        stream_key: str = "__default__",
    ) -> Dict[str, Any]:
        del stream_key
        self.shadow_buffer = [value for value in history if value in {"B", "P"}][-3:]
        result = dict(prediction or {})
        result.setdefault("shadow_buffer", list(self.shadow_buffer))
        return result


ShortTermTakeoverController = ShadowBacktestController
_SHADOW_CONTROLLER = ShadowBacktestController()
_SHORT_TERM_CONTROLLER = _SHADOW_CONTROLLER


def predict(
    history: Union[str, Iterable[Any], None] = None,
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
    run_seed: Optional[int] = None,
    shoe_context: Optional[Mapping[str, Any]] = None,
    road_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Unified API compatible with app.py and screenshot_predictor.py."""
    if history is None:
        history_values: List[Any] = []
    elif isinstance(history, str):
        history_values = [
            part for part in history.replace("|", ",").split(",") if part.strip()
        ]
    else:
        history_values = list(history)
    cleaned = _normalize_outcome_history(history_values)

    road = dict(road_context or {})
    if not isinstance(road.get("models"), Mapping):
        road = build_road_context(cleaned, seed=run_seed)

    road_seed = _road_seed_prediction(road, cleaned)
    road_seed["road_support"] = dict(road)
    road_seed["component_probabilities"] = dict(
        road.get("component_probabilities") or {}
    )
    road_seed["decision_pipeline"] = "full_road_pattern_to_adaptive_ensemble"
    road_seed["model_variant"] = ADAPTIVE_MODEL_VARIANT
    adaptive = adapt_prediction(
        road_seed,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )

    bandit_scope = _bandit_learning_scope(
        user_id=user_id,
        venue=venue,
        room=room,
    )
    bandit = predict_bandit(
        cleaned,
        road_context=road,
        venue=venue,
        room=room,
        user_id=bandit_scope,
        run_seed=run_seed,
    )
    result = _fuse_adaptive_and_cusum(adaptive, bandit)

    model_fingerprint = str(result.get("prediction_fingerprint") or "").strip()
    bandit_fingerprint = str(bandit.get("prediction_fingerprint") or "").strip()
    result["model_prediction_fingerprint"] = model_fingerprint
    result["prediction_fingerprint"] = sha256(
        "|".join((
            model_fingerprint,
            bandit_fingerprint,
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "__unspecified_shoe__").strip(),
        )).encode("utf-8")
    ).hexdigest()[:24]
    result.update({
        "shoe_id": str(shoe_id or ""),
        "bandit_learning_user_id": bandit_scope,
        "bandit_scope_mode": "user_venue_room_hashed",
        "bandit_shoe_isolated": False,
        "shoe_event_isolated": True,
        "composition_quality": "not_applicable_road_cusum",
        "remaining_counts_source": "not_used",
        "shoe_context_ignored": bool(shoe_context),
        "road_quality_ok": bool(
            road.get("quality_ok", road.get("recognition_quality_ok", True))
        ),
        "road_support": dict(road),
        "component_probabilities": dict(
            road.get("component_probabilities") or {}
        ),
        "input_required": False,
        "probability_semantics": "direction_score_not_guaranteed_outcome_probability",
    })
    return result


def run_virtual_round(
    session: Mapping[str, Any],
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Retain virtual-shoe compatibility API used by app.py."""
    from particle_filter_points import counts_from_shoe, deal_ordered_hand

    hidden_shoe = [int(card) for card in list(session.get("virtual_shoe") or [])]
    if len(hidden_shoe) < 6:
        raise ValueError("虛擬牌靴不足，請重新建立牌靴。")
    outcome_history = _normalize_outcome_history(
        list(session.get("round_history") or [])
    )
    seed = int(
        run_seed if run_seed is not None else secrets.randbits(32)
    ) & 0xFFFFFFFF
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
        prediction.get("action") or prediction.get("recommend") or ""
    ).upper()
    actual = str(hand.outcome or "").upper()
    verdict = (
        "OBSERVE" if predicted_side == "O" else
        "TIE_SKIPPED" if actual == "T" else
        "HIT" if predicted_side == actual else "MISS"
    )
    prediction.update({
        "ok": True,
        "mode": "virtual_shoe_adaptive_cusum_compatibility",
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
        "bandit_learning_applied": True,
        "disclaimer": "虛擬相容模式方向使用 Full Road Adaptive + CUSUM-LinUCB。",
    })
    return {
        "prediction": prediction,
        "hand": hand_data,
        "remaining_shoe": remaining_shoe,
    }


def parse_point_observation(value: Any) -> None:
    del value
    return None


__all__ = [
    "DB_HOLDOUT",
    "ShadowBacktestController",
    "ShortTermTakeoverController",
    "parse_point_observation",
    "predict",
    "run_virtual_round",
]
