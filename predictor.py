"""BGS predictor: image road parsing -> three-way Markov -> B/P + mandatory sizing.

The formal direction is selected directly from the three-way Markov B/P posterior.
T is retained in the Markov state/probability model and affects entropy/risk diagnostics.
Road analysis remains available for diagnostics, but it does not override the Markov
direction. Every formal B/P prediction carries a 5%-30% bankroll sizing ratio.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from markov_model import MODEL_VERSION, update_and_predict_engine
from money_management import MoneyManagementModel, MAX_BET_RATIO
from road_model import ROAD_FEATURE_NAMES, build_road_context


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


def _road_diagnostic(road: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        banker = max(0.0, min(1.0, float(road.get("banker_probability", 0.5) or 0.5)))
    except (TypeError, ValueError):
        banker = 0.5
    try:
        player = max(0.0, min(1.0, float(road.get("player_probability", 1.0 - banker) or 0.0)))
    except (TypeError, ValueError):
        player = 1.0 - banker
    total = banker + player
    if total <= 1e-12:
        banker, player = 0.5, 0.5
    else:
        banker, player = banker / total, player / total
    try:
        confidence = max(0.0, min(1.0, float(road.get("confidence_score", 0.0) or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "direction": "B" if banker >= player else "P",
        "banker_probability": float(banker),
        "player_probability": float(player),
        "confidence": float(confidence),
        "decision_weight": 0.0,
        "diagnostic_only": True,
    }


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
    del user_id, run_seed
    if history is None:
        history_values: List[Any] = []
    elif isinstance(history, str):
        history_values = [
            part for part in history.replace("|", ",").split(",") if part.strip()
        ]
    else:
        history_values = list(history)
    cleaned = _normalize_outcome_history(history_values)

    supplied_road = dict(road_context or {})
    if (
        isinstance(supplied_road.get("road_features"), list)
        and len(supplied_road.get("road_features") or []) == len(ROAD_FEATURE_NAMES)
    ):
        road = supplied_road
    else:
        road = build_road_context(
            cleaned,
            grid_cells=list(supplied_road.get("grid_cells") or []),
            initial_image_count=int(supplied_road.get("initial_image_count", 0) or 0),
            manual_count=int(supplied_road.get("manual_count", 0) or 0),
        )
        if supplied_road:
            road["scan_metadata"] = supplied_road

    markov = update_and_predict_engine(cleaned)
    probabilities = dict(markov["probabilities"])
    direction = str(markov["direction"])
    confidence = float(markov["final_weight"])
    text = "莊" if direction == "B" else "閒"

    context = dict(shoe_context or {})
    bankroll = max(0.0, float(context.get("bankroll", 0.0) or 0.0))
    money = MoneyManagementModel().allocate(
        direction=direction,
        probabilities=probabilities,
        final_weight=confidence,
        bankroll=bankroll,
    )

    road_predict = _road_diagnostic(road)
    fingerprint = sha256(
        "|".join((
            "".join(cleaned),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "").strip(),
            MODEL_VERSION,
        )).encode("utf-8")
    ).hexdigest()[:24]

    p_b = float(probabilities["B"])
    p_p = float(probabilities["P"])
    p_t = float(probabilities["T"])
    bp_edge = abs(p_b - p_p)

    return {
        "ok": True,
        "engine": "THREEWAY_MARKOV_SHOE_DEPTH",
        "model_version": MODEL_VERSION,
        "model_variant": "THREEWAY_STATE_DECAY_BAYES_ENTROPY_SHOE_DEPTH_ALWAYS_BET",
        "model_core": "threeway_markov_primary",
        "decision_pipeline": "image_scan_to_threeway_state_to_markov_to_entropy_shoe_depth_to_mandatory_5_30_sizing",
        "prediction_fingerprint": fingerprint,
        "probabilities": {"B": p_b, "P": p_p, "T": p_t},
        "raw_direction_probabilities": {"B": p_b, "P": p_p},
        "banker_rate": round(p_b * 100.0, 2),
        "player_rate": round(p_p * 100.0, 2),
        "tie_rate": round(p_t * 100.0, 2),
        "recommend": direction,
        "recommend_text": text,
        "action": direction,
        "action_text": text,
        "internal_recommend": direction,
        "internal_action": direction,
        "next_round_direction": direction,
        "next_round_direction_text": text,
        "direction": direction,
        "direction_text": text,
        "signal_allowed": True,
        "risk_gate_open": True,
        "mandatory_bet": True,
        "signal_status_code": "MARKOV_DIRECTION_MANDATORY_BET",
        "signal_status_text": "Markov 方向成立，每局依 5%-30% 動態比例配置",
        "signal_reason": (
            f"state={markov['state_key'] or 'START'}；"
            f"H={markov['entropy_bits']:.4f} bits；"
            f"shoe_progress={markov['shoe_progress']:.3f}；"
            f"final_weight={confidence:.3f}；"
            f"sizing={money['reason']}。"
        ),
        "direction_source": "threeway_markov_primary",
        "confidence": confidence,
        "ensemble_confidence": confidence,
        "quality_score": confidence,
        "confidence_label": (
            "較高" if confidence >= 0.40 else
            "中等" if confidence >= 0.22 else "偏低"
        ),
        "entropy_bits": float(markov["entropy_bits"]),
        "entropy_base_weight": float(markov["base_weight"]),
        "shoe_progress": float(markov["shoe_progress"]),
        "shoe_depth_estimate": dict(markov["shoe_depth"]),
        "tie_risk_active": bool(markov["tie_risk_active"]),
        "direction_edge": float(bp_edge),
        "direction_edge_percent": round(bp_edge * 100.0, 4),
        "markov_predict": {
            "direction": direction,
            "probabilities": {"B": p_b, "P": p_p, "T": p_t},
            "state": dict(markov["state"]),
            "state_key": str(markov["state_key"]),
            "transition_counts": dict(markov["transition_counts"]),
            "effective_support": float(markov["effective_support"]),
            "entropy_bits": float(markov["entropy_bits"]),
            "base_weight": float(markov["base_weight"]),
            "final_weight": confidence,
            "decay": float(markov["decay"]),
            "prior": dict(markov["prior"]),
            "prior_strength": float(markov["prior_strength"]),
        },
        "markov_state": {
            "state_key": str(markov["state_key"]),
            "direction_context": str(markov["state"].get("direction_context") or ""),
            "density": str(markov["state"].get("density") or "Medium"),
            "tie_trigger": str(markov["state"].get("tie_trigger") or "NoTie"),
            "sample_count": int(markov["sample_count"]),
            "effective_support": float(markov["effective_support"]),
            "state_count": int(markov["state_count"]),
        },
        "road_predict": road_predict,
        "road_support": road,
        "road_fusion": {
            "applied": False,
            "mode": "diagnostic_only",
            "reason": "正式方向完全由三元 Markov B/P 後驗概率決定，Road 不覆寫方向。",
        },
        "component_probabilities": {
            "markov": {"B": p_b, "P": p_p, "T": p_t},
            "road_diagnostic": {
                "B": float(road_predict["banker_probability"]),
                "P": float(road_predict["player_probability"]),
                "T": 0.0,
            },
        },
        "money_management": dict(money),
        "kelly_fraction": float(money["kelly_fraction"]),
        "pre_tie_adjusted_ratio": float(money["pre_tie_adjusted_ratio"]),
        "adjusted_ratio": float(money["adjusted_ratio"]),
        "final_bet_ratio": float(money["final_bet_ratio"]),
        "bet_percentage": float(money["bet_percentage"]),
        "suggested_bet_amount": float(money["bet_amount"]),
        "bet_multiplier": (
            min(1.0, float(money["final_bet_ratio"]) / MAX_BET_RATIO)
            if MAX_BET_RATIO > 0.0 else 0.0
        ),
        "context_vector": list(road.get("road_features") or []),
        "bandit_context": [],
        "context_feature_names": list(ROAD_FEATURE_NAMES),
        "context_dim": len(list(road.get("road_features") or [])),
        "contextual_bandit_enabled": False,
        "contextual_bandit_update_enabled": False,
        "cusum_linucb_enabled": False,
        "force_observe": False,
        "hard_brake_active": False,
        "post_reset_vacuum_active": False,
        "input_required": False,
        "venue": str(venue or ""),
        "room": str(room or ""),
        "shoe_id": str(shoe_id or ""),
        "probability_semantics": "bayesian_threeway_model_probability_not_guaranteed_outcome",
    }


def run_virtual_round(
    session: Mapping[str, Any],
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Compatibility API for the existing virtual-shoe endpoint."""
    from particle_filter_points import counts_from_shoe, deal_ordered_hand

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
        run_seed=seed,
        shoe_context={"bankroll": float(session.get("bankroll", 0.0) or 0.0)},
    )
    hand, remaining_shoe = deal_ordered_hand(hidden_shoe)
    hand_data = hand.as_dict()
    predicted_side = str(prediction.get("action") or "").upper()
    actual = str(hand.outcome or "").upper()
    verdict = (
        "TIE_SKIPPED" if actual == "T" else
        "HIT" if predicted_side == actual else "MISS"
    )
    prediction.update({
        "ok": True,
        "mode": "virtual_shoe_threeway_markov_compatibility",
        "virtual_hand": hand_data,
        "virtual_outcome": actual,
        "virtual_outcome_text": hand_data["outcome_text"],
        "verdict": verdict,
        "verdict_text": {
            "HIT": "命中",
            "MISS": "未命中",
            "TIE_SKIPPED": "和局不計",
        }[verdict],
        "cards_consumed": int(hand.cards_used),
        "remaining_cards_after": len(remaining_shoe),
        "remaining_counts_after": counts_from_shoe(remaining_shoe),
        "round_number": int(session.get("hand_number", 0) or 0) + 1,
        "bandit_learning_applied": False,
        "disclaimer": "虛擬相容模式方向使用 Three-way Markov + Shoe Depth。",
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
    "parse_point_observation",
    "predict",
    "run_virtual_round",
]
