"""百家樂路紙序列的輕量模擬模型。

此模型只根據已辨識的 B/P 結果估算「路紙傾向」，不會改變虛擬牌靴的
剩餘牌數，也不代表能取得外部真人桌的牌序。模型融合：
- 平滑後的近期加權頻率
- 一階轉移（上一局 -> 下一局）
- 二階轉移（最近兩局 -> 下一局）
- Beta 後驗蒙地卡羅抽樣，用來估計不確定性
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple
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

    # 後驗蒙地卡羅：每次抽取可能的轉移機率，再融合近期頻率。
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

    return {
        "ok": bool(sequence),
        "engine": "ROAD_BAYES_MARKOV_MONTE_CARLO",
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
        "edge": round(edge, 6),
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
        "disclaimer": "此結果只反映已辨識路紙的統計傾向，不代表可預知外部牌局。",
    }


__all__ = [
    "calculate_road_probabilities",
    "normalize_road_sequence",
]
