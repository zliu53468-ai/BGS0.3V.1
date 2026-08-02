"""BGS 百家樂牌路先行分析模組。

處理順序：
1. 先把已辨識或使用者回報的 B/P 結果正規化。
2. 使用近期衰減、一階轉移、二階轉移與 Beta 後驗蒙地卡羅建立牌路 context。
3. 將牌路 context 傳入有限牌組主引擎，由主引擎統一決定下一局方向與正式訊號。

牌路資料只描述已發生結果，不取得外部真人桌的隱藏牌序，也不保證下一局結果。
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

# 牌路 context 是否可進入主引擎，以及最多可使用多少權重。
ROAD_FUSION_ENABLED = os.getenv("ROAD_FUSION_ENABLED", "1").strip() == "1"
ROAD_FUSION_WEIGHT = _env_float("ROAD_FUSION_WEIGHT", 0.08, 0.0, 0.20)
ROAD_FUSION_MIN_SAMPLES = _env_int("ROAD_FUSION_MIN_SAMPLES", 10, 4, 100)
ROAD_FUSION_MIN_CONFIDENCE = _env_float(
    "ROAD_FUSION_MIN_CONFIDENCE", 0.45, 0.0, 1.0
)
ROAD_FUSION_MAX_UNCERTAINTY = _env_float(
    "ROAD_FUSION_MAX_UNCERTAINTY", 0.16, 0.01, 0.50
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
        if tuple(sequence[index - order:index]) != context:
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
    """先分析牌路，回傳可直接傳給主引擎的 road context。"""
    sequence = normalize_road_sequence(values)
    length = len(sequence)
    run_seed = int(seed if seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    rng = np.random.default_rng(run_seed)

    recent_b = _recent_probability(sequence)
    first_b, first_p, first_support = _beta_counts_for_context(sequence, 1)
    second_b, second_p, second_support = _beta_counts_for_context(sequence, 2)

    first_probability = first_b / (first_b + first_p)
    second_probability = second_b / (second_b + second_p)

    # 支援度越高，轉移模型權重越大；資料不足時回到近期平滑頻率。
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
        ROAD_RECENCY_DECAY ** index
        for index, outcome in enumerate(reversed(sequence))
        if outcome == "B"
    )
    recent_beta = 3.0 + sum(
        ROAD_RECENCY_DECAY ** index
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
        "較高" if confidence_score >= 0.72
        else "中等" if confidence_score >= 0.50
        else "偏低"
    )

    eligible_for_core = bool(
        ROAD_FUSION_ENABLED
        and length >= ROAD_FUSION_MIN_SAMPLES
        and confidence_score >= ROAD_FUSION_MIN_CONFIDENCE
        and uncertainty <= ROAD_FUSION_MAX_UNCERTAINTY
    )
    support_factor = min(1.0, length / 30.0)
    confidence_factor = min(1.0, confidence_score / 0.72) if confidence_score > 0 else 0.0
    edge_factor = min(1.0, edge / max(ROAD_MIN_EDGE, 1e-9))
    suggested_core_weight = (
        ROAD_FUSION_WEIGHT
        * support_factor
        * confidence_factor
        * (0.65 + 0.35 * edge_factor)
        if eligible_for_core
        else 0.0
    )

    if signal_allowed:
        signal_reason = "牌路樣本、方向差距與不確定性均達輔助訊號門檻"
        signal_status_text = "牌路方向已建立"
    else:
        reasons: List[str] = []
        if length < ROAD_MIN_SAMPLES:
            reasons.append("牌路樣本仍在累積")
        if edge < ROAD_MIN_EDGE:
            reasons.append("牌路方向差距不足")
        if uncertainty > ROAD_MAX_UNCERTAINTY:
            reasons.append("牌路不確定性偏高")
        signal_reason = "、".join(reasons) or "牌路資料尚未形成明確方向"
        signal_status_text = "牌路資料建立中"

    return {
        "ok": bool(sequence),
        "engine": "ROAD_BAYES_MARKOV_MONTE_CARLO_V3",
        "pipeline_stage": "road_first",
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
        "eligible_for_core": eligible_for_core,
        "suggested_core_weight": round(suggested_core_weight, 8),
        "max_core_weight": ROAD_FUSION_WEIGHT,
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


def build_road_context(
    values: Iterable[Any],
    seed: int | None = None,
) -> Dict[str, Any]:
    """語意化別名：明確表示這份結果會先送入主引擎。"""
    return calculate_road_probabilities(values, seed=seed)


def _probability(value: Any, fallback: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except Exception:
        return fallback


def fuse_road_with_main_prediction(
    main_prediction: Mapping[str, Any],
    road_analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    """相容舊版 app。

    V9.5 的正確流程已在主引擎內整合 road context；若結果已含
    ``road_integration.processed_inside_core``，此函式只補相容欄位，不再二次融合。
    舊版 predictor 尚未整合時，才執行保守的後置低權重融合。
    """
    result = dict(main_prediction or {})
    road = dict(road_analysis or {})
    result["road_support"] = road
    internal = dict(result.get("road_integration") or {})
    if internal.get("processed_inside_core"):
        result["road_fusion"] = dict(internal)
        return result

    main_b = _probability(result.get("banker_rate")) / 100.0
    main_p = _probability(result.get("player_rate")) / 100.0
    main_t = _probability(result.get("tie_rate")) / 100.0
    total = main_b + main_p + main_t
    if total <= 0:
        result["road_fusion"] = {"applied": False, "reason": "主模型機率資料不足"}
        return result
    main_b, main_p, main_t = main_b / total, main_p / total, main_t / total

    effective_weight = min(
        ROAD_FUSION_WEIGHT,
        _probability(road.get("suggested_core_weight"), 0.0),
    )
    eligible = bool(road.get("eligible_for_core"))
    if not eligible:
        effective_weight = 0.0
    road_b = min(1.0, _probability(road.get("banker_probability"), 0.5))
    main_bp_total = max(1e-12, main_b + main_p)
    main_b_no_tie = main_b / main_bp_total
    fused_b_no_tie = main_b_no_tie * (1.0 - effective_weight) + road_b * effective_weight
    fused_b = (1.0 - main_t) * fused_b_no_tie
    fused_p = (1.0 - main_t) * (1.0 - fused_b_no_tie)
    direction = "B" if fused_b >= fused_p else "P"

    result.update({
        "banker_rate": round(fused_b * 100.0, 2),
        "player_rate": round(fused_p * 100.0, 2),
        "tie_rate": round(main_t * 100.0, 2),
        "probabilities": {"B": fused_b, "P": fused_p, "T": main_t},
        "recommend": direction,
        "recommend_text": "莊" if direction == "B" else "閒",
        "direction_source": "legacy_post_fusion" if effective_weight > 0 else "main_model",
        "road_fusion": {
            "applied": effective_weight > 0,
            "eligible": eligible,
            "effective_weight": round(effective_weight, 8),
            "processed_inside_core": False,
        },
    })
    return result


__all__ = [
    "build_road_context",
    "calculate_road_probabilities",
    "fuse_road_with_main_prediction",
    "normalize_road_sequence",
]
