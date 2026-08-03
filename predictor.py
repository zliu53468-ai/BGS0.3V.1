"""BGS V10.9 統一機率預測入口。

虛擬牌靴有真實剩餘牌組時，有限牌組可作主要訊號；圖片／真人桌只有歷史
B/P/T 與估計組成時，會把機率收縮回 8 副牌先驗、壓低路型品質並提高觀望門檻。
線上校準器仍只做機率校準與觀望閘門，不把集成方向強行翻成相反方向。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union
import os
import secrets

from adaptive_ensemble import adapt_prediction
from online_calibrator import calibrate_prediction
from particle_filter_points import (
    DB_HOLDOUT,
    EngineSettings,
    VirtualShoeParticleEngine,
    counts_from_shoe,
    deal_ordered_hand,
    fresh_counts,
)


BASELINE = {"B": 0.458597, "P": 0.446247, "T": 0.095156}
RELIABLE_COMPOSITION_LABELS = {"observed", "actual", "known", "session_actual", "virtual_shoe"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


LEGACY_ADAPTIVE_AFTER_STACKING = os.getenv("LEGACY_ADAPTIVE_AFTER_STACKING", "0").strip() == "1"
PREDICT_SCREEN_MIN_RECOGNIZED = _env_int("PREDICT_SCREEN_MIN_RECOGNIZED", 8, 1, 90)
PREDICT_ESTIMATED_PRIOR_SHRINK = _env_float(
    "PREDICT_ESTIMATED_PRIOR_SHRINK", 0.55, 0.0, 0.98
)
PREDICT_BAD_SCAN_PRIOR_SHRINK = _env_float(
    "PREDICT_BAD_SCAN_PRIOR_SHRINK", 0.90, 0.0, 1.0
)
PREDICT_ESTIMATED_MIN_EDGE = _env_float(
    "PREDICT_ESTIMATED_MIN_EDGE", 0.025, 0.0, 0.20
)
PREDICT_ESTIMATED_MIN_STABILITY = _env_float(
    "PREDICT_ESTIMATED_MIN_STABILITY", 0.68, 0.50, 0.99
)
PREDICT_ESTIMATED_MIN_AGREEMENT = _env_float(
    "PREDICT_ESTIMATED_MIN_AGREEMENT", 0.62, 0.0, 1.0
)
PREDICT_SCREEN_ROAD_QUALITY_SCALE = _env_float(
    "PREDICT_SCREEN_ROAD_QUALITY_SCALE", 0.35, 0.0, 1.0
)
PREDICT_ESTIMATED_MAX_QUALITY = _env_float(
    "PREDICT_ESTIMATED_MAX_QUALITY", 0.58, 0.0, 1.0
)
PREDICT_BAD_SCAN_MAX_QUALITY = _env_float(
    "PREDICT_BAD_SCAN_MAX_QUALITY", 0.25, 0.0, 1.0
)
PREDICT_ALLOW_TIE_IN_ESTIMATED = os.getenv(
    "PREDICT_ALLOW_TIE_IN_ESTIMATED", "0"
).strip() == "1"

_ENGINE = VirtualShoeParticleEngine(
    EngineSettings(
        decks=_env_int("PF_DECKS", 8, 1, 16),
        particles=_env_int("PF_PARTICLES", 500, 64, 4000),
        replicas=_env_int("PF_REPLICAS", 5, 3, 11),
        simulations_per_replica=_env_int(
            "PF_PREDICT_SIMULATIONS_PER_REPLICA", 1200, 200, 20_000
        ),
        particle_draws_per_particle=_env_int("PF_DRAWS_PER_PARTICLE", 2, 1, 12),
    )
)


def _normalize_outcome_history(values: Iterable[Any]) -> List[str]:
    history: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("virtual_outcome")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            history.append(value)
    return history


def _normalize_path_history(values: Iterable[Any]) -> List[str]:
    history: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("draw_path") or item.get("path")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"N", "P", "B", "D"}:
            history.append(value)
    return history


def _normalize_probabilities(values: Mapping[str, Any]) -> Dict[str, float]:
    data = {key: max(1e-12, float(values.get(key, 0.0) or 0.0)) for key in ("B", "P", "T")}
    total = sum(data.values()) or 1.0
    return {key: data[key] / total for key in data}


def _result_probabilities(result: Mapping[str, Any]) -> Dict[str, float]:
    if isinstance(result.get("probabilities"), Mapping):
        return _normalize_probabilities(dict(result["probabilities"]))
    return _normalize_probabilities(
        {
            "B": float(result.get("banker_rate", 0.0) or 0.0) / 100.0,
            "P": float(result.get("player_rate", 0.0) or 0.0) / 100.0,
            "T": float(result.get("tie_rate", 0.0) or 0.0) / 100.0,
        }
    )


def _set_probabilities(result: Dict[str, Any], probabilities: Mapping[str, Any]) -> None:
    normalized = _normalize_probabilities(probabilities)
    result["probabilities"] = dict(normalized)
    result["banker_rate"] = round(normalized["B"] * 100.0, 2)
    result["player_rate"] = round(normalized["P"] * 100.0, 2)
    result["tie_rate"] = round(normalized["T"] * 100.0, 2)


def _prediction_label(prediction: Mapping[str, Any]) -> str:
    quality = float(prediction.get("quality_score", 0.0) or 0.0)
    if quality >= 0.72:
        return "較高"
    if quality >= 0.50:
        return "中等"
    return "偏低"


def _flatten_road_context(road_context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    source = dict(road_context or {})
    nested = source.get("road")
    if isinstance(nested, Mapping):
        merged = dict(nested)
        for key, value in source.items():
            if key != "road" and key not in merged:
                merged[key] = value
        return merged
    return source


def _road_quality(road_context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    road = _flatten_road_context(road_context)
    present = bool(
        road
        and any(
            key in road
            for key in (
                "quality_ok",
                "recognition_quality_ok",
                "recognized_count",
                "reconstructed_all",
                "sequence",
                "all_grid_cells",
            )
        )
    )
    quality_ok = bool(
        road.get("quality_ok", road.get("recognition_quality_ok", True))
    )
    reconstructed_all = bool(road.get("reconstructed_all", quality_ok))
    sequence = [str(value).upper() for value in list(road.get("sequence") or []) if str(value).upper() in {"B", "P"}]
    recognized_count = int(road.get("recognized_count", len(sequence)) or len(sequence))
    uncertain = int(road.get("uncertain_count", road.get("unknown_candidates", 0)) or 0)
    uncertain_ratio = float(
        road.get("unknown_ratio", uncertain / max(1, recognized_count + uncertain)) or 0.0
    )
    bad = bool(
        present
        and (
            not quality_ok
            or not reconstructed_all
            or recognized_count < PREDICT_SCREEN_MIN_RECOGNIZED
        )
    )
    return {
        "present": present,
        "quality_ok": quality_ok,
        "reconstructed_all": reconstructed_all,
        "recognized_count": recognized_count,
        "uncertain_count": uncertain,
        "uncertain_ratio": uncertain_ratio,
        "bad": bad,
        "fallback_reason": str(road.get("fallback_reason") or ""),
        "input_type": str(road.get("input_type") or ""),
    }


def _prepare_engine_road_context(
    road_context: Optional[Mapping[str, Any]],
    *,
    composition_quality: str,
    road_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    road = _flatten_road_context(road_context)
    if not road:
        return {}
    road["composition_quality"] = composition_quality
    road["input_mode"] = "screenshot_or_live_table"
    if bool(road_quality.get("bad")):
        road["road_available"] = False
        road["confidence_score"] = 0.0
        road["planning_reliability"] = 0.0
        road["recent_reliability"] = 0.0
    else:
        for key in ("confidence_score", "planning_reliability", "recent_reliability"):
            try:
                original = float(road.get(key, 1.0) or 0.0)
            except Exception:
                original = 0.0
            road[key] = min(original, PREDICT_SCREEN_ROAD_QUALITY_SCALE)
    return road


def _apply_prior_shrink(
    result: Mapping[str, Any],
    *,
    strength: float,
    reason: str,
) -> Dict[str, Any]:
    output = dict(result or {})
    before = _result_probabilities(output)
    shrink = max(0.0, min(1.0, float(strength)))
    after = {
        key: before[key] * (1.0 - shrink) + BASELINE[key] * shrink
        for key in BASELINE
    }
    output["pre_mode_guard_probabilities"] = dict(before)
    _set_probabilities(output, after)
    output["prior_shrink"] = {
        "active": shrink > 0.0,
        "strength": round(shrink, 6),
        "reason": reason,
        "baseline": dict(BASELINE),
        "before": dict(before),
        "after": dict(_result_probabilities(output)),
    }
    return output


def _nested_metric(result: Mapping[str, Any], key: str) -> Optional[float]:
    candidates: List[Any] = [result.get(key)]
    posterior = result.get("posterior")
    if isinstance(posterior, Mapping):
        candidates.append(posterior.get(key))
    for container_key in ("stacking", "ensemble", "group_ensemble"):
        container = result.get(container_key)
        if isinstance(container, Mapping):
            candidates.append(container.get(key))
            nested = container.get("posterior")
            if isinstance(nested, Mapping):
                candidates.append(nested.get(key))
    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _force_observe(result: Dict[str, Any], reason: str) -> None:
    result["action"] = "O"
    result["action_text"] = "觀望"
    result["signal_allowed"] = False
    result["signal_status_text"] = "等待更明確訊號"
    existing = str(result.get("signal_reason") or "")
    result["signal_reason"] = ((existing + "；") if existing else "") + reason


def _apply_mode_action_guard(
    prediction: Mapping[str, Any],
    *,
    reliable_composition: bool,
    composition_quality: str,
    road_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    result = dict(prediction or {})
    probabilities = _result_probabilities(result)
    bp_total = max(1e-12, probabilities["B"] + probabilities["P"])
    edge = abs(probabilities["B"] - probabilities["P"]) / bp_total
    stability = _nested_metric(result, "direction_stability")
    agreement = _nested_metric(result, "weighted_agreement")
    reasons: List[str] = []

    if bool(road_quality.get("bad")):
        reasons.append("圖片牌路品質未通過對齊、顏色或完整反推檢查")
    if not reliable_composition:
        if edge < PREDICT_ESTIMATED_MIN_EDGE:
            reasons.append(
                f"估計組成模式方向 edge {edge:.4f} 低於 {PREDICT_ESTIMATED_MIN_EDGE:.4f}"
            )
        if stability is not None and stability < PREDICT_ESTIMATED_MIN_STABILITY:
            reasons.append(
                f"方向穩定度 {stability:.4f} 低於 {PREDICT_ESTIMATED_MIN_STABILITY:.4f}"
            )
        if agreement is not None and agreement < PREDICT_ESTIMATED_MIN_AGREEMENT:
            reasons.append(
                f"群組方向一致度 {agreement:.4f} 低於 {PREDICT_ESTIMATED_MIN_AGREEMENT:.4f}"
            )
        if (
            str(result.get("action") or "O").upper() == "T"
            or str(result.get("recommend") or "").upper() == "T"
        ) and not PREDICT_ALLOW_TIE_IN_ESTIMATED:
            reasons.append("估計組成模式預設不開放和局下注訊號")

    if reasons:
        _force_observe(result, "；".join(reasons))

    current_quality = float(result.get("quality_score", 0.0) or 0.0)
    if bool(road_quality.get("bad")):
        result["quality_score"] = min(current_quality, PREDICT_BAD_SCAN_MAX_QUALITY)
    elif not reliable_composition:
        result["quality_score"] = min(current_quality, PREDICT_ESTIMATED_MAX_QUALITY)

    result["mode_guard"] = {
        "composition_quality": composition_quality,
        "reliable_composition": reliable_composition,
        "road_quality": dict(road_quality),
        "direction_edge": float(edge),
        "minimum_edge": PREDICT_ESTIMATED_MIN_EDGE if not reliable_composition else None,
        "direction_stability": stability,
        "minimum_stability": PREDICT_ESTIMATED_MIN_STABILITY if not reliable_composition else None,
        "weighted_agreement": agreement,
        "minimum_agreement": PREDICT_ESTIMATED_MIN_AGREEMENT if not reliable_composition else None,
        "forced_observe": bool(reasons),
        "reasons": reasons,
    }
    return result


def _post_process(
    prediction: Mapping[str, Any],
    *,
    venue: str = "",
    room: str = "",
    reliable_composition: bool = False,
    composition_quality: str = "estimated",
    road_quality: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    scan = dict(road_quality or {})
    if bool(scan.get("bad")):
        result = _apply_prior_shrink(
            prediction,
            strength=PREDICT_BAD_SCAN_PRIOR_SHRINK,
            reason="圖片辨識品質不合格，中心機率大幅收縮回標準 8 副牌先驗",
        )
    elif not reliable_composition:
        result = _apply_prior_shrink(
            prediction,
            strength=PREDICT_ESTIMATED_PRIOR_SHRINK,
            reason="沒有真實剩餘牌組，有限牌組與路型偏移只保留弱訊號",
        )
    else:
        result = dict(prediction or {})
        result["prior_shrink"] = {
            "active": False,
            "strength": 0.0,
            "reason": "有真實或高品質剩餘牌組成，保留有限牌組中心機率",
            "baseline": dict(BASELINE),
        }

    if LEGACY_ADAPTIVE_AFTER_STACKING:
        result = adapt_prediction(result, venue=venue, room=room)
    else:
        result["adaptive_ensemble"] = {
            "active": False,
            "reason": "主引擎已完成受限制五群組 Stacking，避免舊自適應層二次改寫中心機率",
            "effective_share": 0.0,
            "legacy_override_available": True,
        }
    result = calibrate_prediction(result, venue=venue, room=room)
    return _apply_mode_action_guard(
        result,
        reliable_composition=reliable_composition,
        composition_quality=composition_quality,
        road_quality=scan,
    )


def run_virtual_round(
    session: Mapping[str, Any],
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """虛擬牌靴：先以真實剩餘組成預測，再揭露程式內部下一手。"""
    hidden_shoe = [int(card) for card in list(session.get("virtual_shoe") or [])]
    if len(hidden_shoe) < 6:
        raise ValueError("虛擬牌靴不足，請重新建立牌靴。")

    remaining_counts = session.get("remaining_counts")
    if not isinstance(remaining_counts, list) or len(remaining_counts) != 10:
        remaining_counts = counts_from_shoe(hidden_shoe)

    round_history = list(session.get("round_history") or [])
    outcome_history = _normalize_outcome_history(round_history)
    path_history = _normalize_path_history(round_history)
    seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF

    prediction = _ENGINE.analyze(
        remaining_counts=remaining_counts,
        history=outcome_history,
        draw_path_history=path_history,
        seed=seed,
        road_context=None,
    )
    prediction = _post_process(
        prediction,
        venue=str(session.get("venue") or ""),
        room=str(session.get("room") or ""),
        reliable_composition=True,
        composition_quality="session_actual",
        road_quality={"present": False, "bad": False},
    )

    hand, remaining_shoe = deal_ordered_hand(hidden_shoe)
    hand_data = hand.as_dict()
    predicted_side = str(prediction.get("recommend") or "").upper()
    action = str(prediction.get("action") or "O").upper()
    actual = hand.outcome
    if action == "O":
        verdict = "OBSERVE"
    elif actual == "T" and predicted_side in {"B", "P"}:
        verdict = "TIE_SKIPPED"
    elif predicted_side == actual:
        verdict = "HIT"
    else:
        verdict = "MISS"

    prediction.update(
        {
            "model_version": "V10.9-RULE-AWARE-VIRTUAL-SHOE",
            "mode": "virtual_shoe_actual_composition",
            "composition_quality": "session_actual",
            "input_required": False,
            "confidence_label": _prediction_label(prediction),
            "virtual_hand": hand_data,
            "virtual_outcome": actual,
            "virtual_outcome_text": hand_data["outcome_text"],
            "verdict": verdict,
            "verdict_text": {
                "HIT": "命中",
                "MISS": "未命中",
                "TIE_SKIPPED": "和局不計",
                "OBSERVE": "觀望不計",
            }[verdict],
            "cards_consumed": hand.cards_used,
            "remaining_cards_after": len(remaining_shoe),
            "remaining_counts_after": counts_from_shoe(remaining_shoe),
            "shoe_id": str(session.get("shoe_id") or ""),
            "venue": str(session.get("venue") or ""),
            "room": str(session.get("room") or ""),
            "round_number": int(session.get("hand_number", 0) or 0) + 1,
            "warmup_rounds": int(session.get("warmup_rounds", 0) or 0),
            "disclaimer": "此模式使用程式內部真實剩餘牌組估計下一局機率；莊注含 5% 抽水，任何方向仍不代表正期望或獲利保證。",
        }
    )
    return {"prediction": prediction, "hand": hand_data, "remaining_shoe": remaining_shoe}


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
    """機率 API；圖片模式應傳入掃描輸出的 sequence/raw_outcomes/road_context。"""
    context = dict(shoe_context or {})
    supplied_counts = context.get("remaining_counts")
    counts_are_valid = bool(
        isinstance(supplied_counts, Sequence)
        and not isinstance(supplied_counts, (str, bytes))
        and len(supplied_counts) == 10
    )
    if counts_are_valid:
        counts = [int(value) for value in supplied_counts]
    else:
        counts = fresh_counts(_ENGINE.settings.decks)

    composition_quality = str(
        context.get("composition_quality")
        or context.get("remaining_counts_source")
        or _flatten_road_context(road_context).get("composition_quality")
        or "estimated"
    ).lower().strip()
    reliable_composition = bool(
        counts_are_valid and composition_quality in RELIABLE_COMPOSITION_LABELS
    )
    if not reliable_composition:
        composition_quality = "estimated"

    if history is None:
        history_values: List[Any] = []
    elif isinstance(history, str):
        history_values = [part for part in history.replace("|", ",").split(",") if part.strip()]
    else:
        history_values = list(history)

    scan_quality = _road_quality(road_context)
    engine_road_context = _prepare_engine_road_context(
        road_context,
        composition_quality=composition_quality,
        road_quality=scan_quality,
    )
    prediction = _ENGINE.analyze(
        remaining_counts=counts,
        history=_normalize_outcome_history(history_values),
        draw_path_history=_normalize_path_history(history_values),
        seed=run_seed,
        road_context=engine_road_context,
    )
    prediction = _post_process(
        prediction,
        venue=venue,
        room=room,
        reliable_composition=reliable_composition,
        composition_quality=composition_quality,
        road_quality=scan_quality,
    )
    screenshot_mode = bool(scan_quality.get("present"))
    prediction.update(
        {
            "venue": venue,
            "room": room,
            "shoe_id": shoe_id,
            "user_id": user_id,
            "input_required": False,
            "mode": (
                "screenshot_live_table_conservative"
                if screenshot_mode
                else "estimated_composition_conservative"
            ),
            "model_version": (
                "V10.9-RULE-AWARE-SCREENSHOT-CONSERVATIVE"
                if screenshot_mode
                else "V10.9-RULE-AWARE-ESTIMATED-CONSERVATIVE"
            ),
            "composition_quality": composition_quality,
            "remaining_counts_source": "provided_reliable" if reliable_composition else "fresh_prior_or_estimate",
            "road_quality_ok": bool(scan_quality.get("quality_ok", True)) if screenshot_mode else None,
            "confidence_label": _prediction_label(prediction),
            "disclaimer": (
                "圖片／真人桌模式沒有真實剩餘牌序；大路僅是歷史排列，系統會收縮回標準先驗並在 edge、穩定度或辨識品質不足時觀望，不暗示可跟路穩定獲利。"
                if screenshot_mode
                else "未提供可驗證的真實剩餘牌組，機率以標準先驗為中心並採保守觀望；歷史路型不具額外因果力。"
            ),
        }
    )
    return prediction


def parse_point_observation(value: Any) -> Optional[Dict[str, Any]]:
    """點數逐局輸入維持停用。"""
    return None


__all__ = ["DB_HOLDOUT", "parse_point_observation", "predict", "run_virtual_round"]
