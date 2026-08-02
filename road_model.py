"""BGS 百家樂牌路輔助與主模型融合層。

功能：
- 平滑近期加權頻率
- 一階與二階轉移
- Beta 後驗蒙地卡羅不確定性估計
- 以低權重、受門檻限制的方式融合主模型

牌路層只使用已辨識或使用者回報的 B/P 結果。它不取得外部牌序，
也不取代有限牌組超幾何主模型；資料不足或模型分歧時會自動維持主模型結果。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import os
import secrets

import numpy as np


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


ROAD_MIN_SAMPLES = _env_int("ROAD_MIN_SAMPLES", 6, 2, 100)
ROAD_HISTORY_LIMIT = _env_int("ROAD_HISTORY_LIMIT", 120, 10, 500)
ROAD_SIMULATIONS = _env_int("ROAD_SIMULATIONS", 5000, 500, 100_000)
ROAD_MIN_EDGE = _env_float("ROAD_MIN_EDGE", 0.055, 0.0, 0.30)
ROAD_MAX_UNCERTAINTY = _env_float("ROAD_MAX_UNCERTAINTY", 0.10, 0.01, 0.40)
ROAD_RECENCY_DECAY = _env_float("ROAD_RECENCY_DECAY", 0.93, 0.50, 0.999)

ROAD_FUSION_ENABLED = os.getenv("ROAD_FUSION_ENABLED", "1").strip() == "1"
ROAD_FUSION_WEIGHT = _env_float("ROAD_FUSION_WEIGHT", 0.08, 0.0, 0.20)
ROAD_FUSION_MIN_SAMPLES = _env_int("ROAD_FUSION_MIN_SAMPLES", 10, 4, 100)
ROAD_FUSION_MIN_CONFIDENCE = _env_float(
    "ROAD_FUSION_MIN_CONFIDENCE", 0.45, 0.0, 1.0
)
ROAD_FUSION_MAX_UNCERTAINTY = _env_float(
    "ROAD_FUSION_MAX_UNCERTAINTY", 0.16, 0.01, 0.50
)
ROAD_FUSION_MIN_EDGE = _env_float("ROAD_FUSION_MIN_EDGE", 0.012, 0.0, 0.20)
ROAD_FUSION_MAX_MAIN_VALIDATION_GAP = _env_float(
    "ROAD_FUSION_MAX_MAIN_VALIDATION_GAP", 0.035, 0.001, 0.20
)


def normalize_road_sequence(values: Iterable[Any]) -> List[str]:
    sequence: List[str] = []
    for item in values:
        value = str(item or "").upper().strip()
        if value in {"B", "P"}:
            sequence.append(value)
    return sequence[-ROAD_HISTORY_LIMIT:]


def _beta_counts_for_context(
    sequence: Sequence[str],
    order: int,
) -> Tuple[float, float, int]:
    """回傳指定尾端 context 後，下一局 B/P 的平滑計數。"""
    if len(sequence) <= order:
        return 2.0, 2.0, 0
    context = tuple(sequence[-order:])
    banker = 2.0
    player = 2.0
    support = 0
    for index in range(order, len(sequence)):
        if tuple(sequence[index - order : index]) != context:
            continue
        support += 1
        if sequence[index] == "B":
            banker += 1.0
        else:
            player += 1.0
    return banker, player, support


def _recent_probability(sequence: Sequence[str]) -> float:
    banker = 3.0
    player = 3.0
    for reverse_index, outcome in enumerate(reversed(sequence)):
        weight = ROAD_RECENCY_DECAY ** reverse_index
        if outcome == "B":
            banker += weight
        else:
            player += weight
    return banker / max(1e-12, banker + player)


def calculate_road_probabilities(
    values: Iterable[Any],
    seed: int | None = None,
) -> Dict[str, Any]:
    sequence = normalize_road_sequence(values)
    length = len(sequence)
    run_seed = int(seed if seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    rng = np.random.default_rng(run_seed)

    recent_b = _recent_probability(sequence)
    first_b, first_p, first_support = _beta_counts_for_context(sequence, 1)
    second_b, second_p, second_support = _beta_counts_for_context(sequence, 2)

    first_probability = first_b / (first_b + first_p)
    second_probability = second_b / (second_b + second_p)

    second_weight = min(0.42, second_support / 18.0)
    first_weight = min(0.33, first_support / 24.0)
    recent_weight = max(0.25, 1.0 - first_weight - second_weight)
    total_weight = recent_weight + first_weight + second_weight
    recent_weight /= total_weight
    first_weight /= total_weight
    second_weight /= total_weight

    center_b = (
        recent_b * recent_weight
        + first_probability * first_weight
        + second_probability * second_weight
    )

    first_samples = rng.beta(first_b, first_p, ROAD_SIMULATIONS)
    second_samples = rng.beta(second_b, second_p, ROAD_SIMULATIONS)
    recent_alpha = 3.0 + sum(
        (ROAD_RECENCY_DECAY ** index)
        for index, outcome in enumerate(reversed(sequence))
        if outcome == "B"
    )
    recent_beta = 3.0 + sum(
        (ROAD_RECENCY_DECAY ** index)
        for index, outcome in enumerate(reversed(sequence))
        if outcome == "P"
    )
    recent_samples = rng.beta(recent_alpha, recent_beta, ROAD_SIMULATIONS)
    samples = (
        recent_samples * recent_weight
        + first_samples * first_weight
        + second_samples * second_weight
    )

    banker = float(np.mean(samples)) if length else 0.5
    player = 1.0 - banker
    uncertainty = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.5
    direction = "B" if banker >= player else "P"
    edge = abs(banker - player)
    signal_allowed = bool(
        length >= ROAD_MIN_SAMPLES
        and edge >= ROAD_MIN_EDGE
        and uncertainty <= ROAD_MAX_UNCERTAINTY
    )
    action = direction if signal_allowed else "O"

    confidence_score = max(
        0.0,
        min(
            1.0,
            0.45 * min(1.0, length / 30.0)
            + 0.35 * min(1.0, edge / max(ROAD_MIN_EDGE, 1e-9))
            + 0.20 * (1.0 - min(1.0, uncertainty / ROAD_MAX_UNCERTAINTY)),
        ),
    )
    confidence_label = (
        "較高" if confidence_score >= 0.72 else "中等" if confidence_score >= 0.50 else "偏低"
    )
    if signal_allowed:
        signal_reason = "牌路樣本、方向差距與不確定性均達輔助訊號門檻"
        signal_status_text = "牌路輔助訊號已建立"
    else:
        reasons: List[str] = []
        if length < ROAD_MIN_SAMPLES:
            reasons.append("牌路樣本仍在累積")
        if edge < ROAD_MIN_EDGE:
            reasons.append("牌路方向差距不足")
        if uncertainty > ROAD_MAX_UNCERTAINTY:
            reasons.append("牌路不確定性偏高")
        signal_reason = "、".join(reasons) or "牌路資料尚未形成明確輔助訊號"
        signal_status_text = "牌路輔助資料建立中"

    return {
        "ok": bool(sequence),
        "engine": "ROAD_BAYES_MARKOV_MONTE_CARLO_V2",
        "run_seed": run_seed,
        "sequence": sequence,
        "sample_count": length,
        "banker_probability": banker,
        "player_probability": player,
        "banker_rate": round(banker * 100.0, 2),
        "player_rate": round(player * 100.0, 2),
        "direction": direction,
        "direction_text": "莊" if direction == "B" else "閒",
        "action": action,
        "action_text": "莊" if action == "B" else "閒" if action == "P" else "觀望",
        "signal_allowed": signal_allowed,
        "signal_status_text": signal_status_text,
        "signal_reason": signal_reason,
        "edge": round(edge, 6),
        "edge_percent": round(edge * 100.0, 4),
        "uncertainty": round(uncertainty, 6),
        "confidence_score": round(confidence_score, 6),
        "confidence_label": confidence_label,
        "weights": {
            "recent": round(recent_weight, 6),
            "first_order": round(first_weight, 6),
            "second_order": round(second_weight, 6),
        },
        "supports": {
            "first_order": first_support,
            "second_order": second_support,
        },
        "center_probability_before_simulation": round(center_b, 6),
        "data_scope": "recognized_banker_player_sequence",
    }


def _probability(value: Any, fallback: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except Exception:
        return fallback


def fuse_road_with_main_prediction(
    main_prediction: Mapping[str, Any],
    road_analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    """將牌路分析以低權重融合到主模型，並保留完整診斷資料。"""
    result = dict(main_prediction or {})
    road = dict(road_analysis or {})
    result["road_support"] = road

    main_b = _probability(result.get("banker_rate")) / 100.0
    main_p = _probability(result.get("player_rate")) / 100.0
    main_t = _probability(result.get("tie_rate")) / 100.0
    total = main_b + main_p + main_t
    if total <= 0:
        result["road_fusion"] = {"applied": False, "reason": "主模型機率資料不足"}
        return result
    main_b, main_p, main_t = main_b / total, main_p / total, main_t / total

    road_samples = int(road.get("sample_count", 0) or 0)
    road_confidence = _probability(road.get("confidence_score"))
    road_uncertainty = _probability(road.get("uncertainty"), 1.0)
    road_b = _probability(road.get("banker_probability"), 0.5)
    road_b = max(0.0, min(1.0, road_b))
    road_direction = str(road.get("direction") or ("B" if road_b >= 0.5 else "P"))
    main_direction = str(result.get("recommend") or ("B" if main_b >= main_p else "P"))
    main_validation_gap = _probability(result.get("validation_gap"), 0.0)

    eligible = bool(
        ROAD_FUSION_ENABLED
        and road.get("ok")
        and road_samples >= ROAD_FUSION_MIN_SAMPLES
        and road_confidence >= ROAD_FUSION_MIN_CONFIDENCE
        and road_uncertainty <= ROAD_FUSION_MAX_UNCERTAINTY
        and main_validation_gap <= ROAD_FUSION_MAX_MAIN_VALIDATION_GAP
    )

    support_factor = min(1.0, road_samples / 30.0)
    confidence_factor = min(1.0, road_confidence / 0.72) if road_confidence > 0 else 0.0
    effective_weight = ROAD_FUSION_WEIGHT * support_factor * confidence_factor if eligible else 0.0

    main_bp_total = max(1e-12, main_b + main_p)
    main_b_no_tie = main_b / main_bp_total
    fused_b_no_tie = (
        main_b_no_tie * (1.0 - effective_weight) + road_b * effective_weight
    )
    fused_b = (1.0 - main_t) * fused_b_no_tie
    fused_p = (1.0 - main_t) * (1.0 - fused_b_no_tie)
    fused_t = main_t

    direction = "B" if fused_b >= fused_p else "P"
    direction_edge = abs(fused_b_no_tie - (1.0 - fused_b_no_tie))
    aligned = road_direction == main_direction
    main_quality = _probability(result.get("quality_score"), 0.0)
    main_signal_allowed = bool(result.get("signal_allowed"))
    road_signal_allowed = bool(road.get("signal_allowed"))

    fusion_signal_allowed = bool(
        direction_edge >= ROAD_FUSION_MIN_EDGE
        and (
            main_signal_allowed
            or (
                eligible
                and aligned
                and road_signal_allowed
                and main_quality >= 0.45
            )
        )
    )
    action = direction if fusion_signal_allowed else "O"

    blended_quality = main_quality
    if eligible:
        blended_quality = max(
            0.0,
            min(1.0, main_quality * (1.0 - effective_weight) + road_confidence * effective_weight),
        )
    confidence_label = (
        "較高" if blended_quality >= 0.72 else "中等" if blended_quality >= 0.50 else "偏低"
    )

    main_consistency = _probability(result.get("model_consistency"), 0.0)
    direction_alignment = 1.0 if aligned else 0.0
    model_consistency = max(
        0.0,
        min(
            1.0,
            main_consistency * (1.0 - effective_weight)
            + direction_alignment * effective_weight,
        ),
    )

    if fusion_signal_allowed:
        if eligible and aligned:
            signal_reason = "主模型與牌路輔助方向一致，且方向差距與模型品質通過正式訊號門檻"
        else:
            signal_reason = str(result.get("signal_reason") or "主模型正式方向訊號已開放")
        signal_status_text = "方向訊號已開放"
    else:
        if eligible and not aligned:
            signal_reason = "主模型與牌路輔助方向分歧，系統維持風險控管"
        elif not eligible:
            signal_reason = str(result.get("signal_reason") or "牌路樣本或品質尚未達融合條件")
        else:
            signal_reason = "融合後方向差距尚未達正式訊號門檻"
        signal_status_text = "等待更明確訊號"

    result.update(
        {
            "banker_rate": round(fused_b * 100.0, 2),
            "player_rate": round(fused_p * 100.0, 2),
            "tie_rate": round(fused_t * 100.0, 2),
            "probabilities": {"B": fused_b, "P": fused_p, "T": fused_t},
            "no_tie_probabilities": {"B": fused_b_no_tie, "P": 1.0 - fused_b_no_tie},
            "recommend": direction,
            "recommend_text": "莊" if direction == "B" else "閒",
            "action": action,
            "action_text": "莊" if action == "B" else "閒" if action == "P" else "觀望",
            "direction_edge": round(direction_edge, 8),
            "direction_edge_percent": round(direction_edge * 100.0, 4),
            "quality_score": round(blended_quality, 6),
            "confidence_label": confidence_label,
            "model_consistency": round(model_consistency, 6),
            "signal_allowed": fusion_signal_allowed,
            "signal_status_text": signal_status_text,
            "signal_reason": signal_reason,
            "direction_source": "main_plus_road" if effective_weight > 0 else "main_model",
            "road_fusion": {
                "applied": effective_weight > 0,
                "eligible": eligible,
                "aligned": aligned,
                "requested_weight": ROAD_FUSION_WEIGHT,
                "effective_weight": round(effective_weight, 6),
                "sample_count": road_samples,
                "road_confidence": round(road_confidence, 6),
                "road_uncertainty": round(road_uncertainty, 6),
                "main_validation_gap": round(main_validation_gap, 6),
            },
        }
    )
    return result


__all__ = [
    "calculate_road_probabilities",
    "fuse_road_with_main_prediction",
    "normalize_road_sequence",
]
