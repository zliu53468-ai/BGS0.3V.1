"""BGS V10.8 全歷史機率擬合模型。

本模組不只看最近幾局，而是使用整副已知 B/P/T 歷史建立：
- 全局與分段出現比例
- 累積機率曲線漂移
- 早／中／後段差異
- 歷史前綴中與目前機率狀態相似的下一局結果

所有歷史相似狀態都採 walk-forward：只用某一時點以前的資料描述該狀態，
再用該時點的下一局作為標籤，避免偷看未來資料。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import math
import os

import numpy as np


OUTCOMES = ("B", "P", "T")
BASELINE = np.asarray([0.458597, 0.446247, 0.095156], dtype=np.float64)


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


GLOBAL_MIN_PREFIX = _env_int("GLOBAL_PROB_MIN_PREFIX", 10, 6, 40)
GLOBAL_NEIGHBORS = _env_int("GLOBAL_PROB_NEIGHBORS", 32, 8, 120)
GLOBAL_PRIOR_STRENGTH = _env_float("GLOBAL_PROB_PRIOR_STRENGTH", 34.0, 8.0, 200.0)
GLOBAL_SIMILARITY_SCALE = _env_float("GLOBAL_PROB_SIMILARITY_SCALE", 3.6, 1.0, 12.0)


def _normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = np.maximum(array, 1e-12)
    total = float(array.sum())
    return array / total if total > 0 else BASELINE.copy()


def _clean(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("virtual_outcome")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in OUTCOMES:
            result.append(value)
    return result[-2000:]


def _probability_vector(values: Sequence[str], prior_strength: float = 18.0) -> np.ndarray:
    counts = BASELINE * prior_strength
    for value in values:
        counts[OUTCOMES.index(value)] += 1.0
    return _normalize(counts)


def _bp_rate(values: Sequence[str]) -> float:
    bp = [value for value in values if value in {"B", "P"}]
    if not bp:
        return float(BASELINE[0] / (BASELINE[0] + BASELINE[1]))
    return (bp.count("B") + 3.0) / (len(bp) + 6.0)


def _segment(values: Sequence[str], start: float, end: float) -> List[str]:
    length = len(values)
    left = max(0, min(length, int(math.floor(length * start))))
    right = max(left, min(length, int(math.ceil(length * end))))
    return list(values[left:right])


def _block_volatility(values: Sequence[str], block_size: int = 8) -> float:
    bp = [value for value in values if value in {"B", "P"}]
    if len(bp) < block_size * 2:
        return 0.0
    rates: List[float] = []
    for start in range(0, len(bp), block_size):
        block = bp[start:start + block_size]
        if len(block) >= max(4, block_size // 2):
            rates.append(_bp_rate(block))
    return float(np.std(rates, ddof=1)) if len(rates) >= 2 else 0.0


def _cumulative_curve_slope(values: Sequence[str]) -> float:
    bp = [value for value in values if value in {"B", "P"}]
    if len(bp) < 8:
        return 0.0
    cumulative: List[float] = []
    banker = 0
    for index, value in enumerate(bp, 1):
        banker += int(value == "B")
        cumulative.append(banker / index)
    x = np.linspace(-1.0, 1.0, len(cumulative))
    y = np.asarray(cumulative, dtype=np.float64)
    x_var = float(np.sum(np.square(x)))
    return float(np.sum(x * (y - y.mean())) / max(1e-12, x_var))


def _state(values: Sequence[str]) -> Dict[str, Any]:
    raw = list(values)
    length = len(raw)
    full = _probability_vector(raw, GLOBAL_PRIOR_STRENGTH)
    early = _probability_vector(_segment(raw, 0.0, 1.0 / 3.0), 14.0)
    middle = _probability_vector(_segment(raw, 1.0 / 3.0, 2.0 / 3.0), 14.0)
    late = _probability_vector(_segment(raw, 2.0 / 3.0, 1.0), 14.0)
    recent8 = _probability_vector(raw[-8:], 14.0)
    recent18 = _probability_vector(raw[-18:], 18.0)
    recent36 = _probability_vector(raw[-36:], 24.0)

    global_bp = _bp_rate(raw)
    early_bp = _bp_rate(_segment(raw, 0.0, 1.0 / 3.0))
    middle_bp = _bp_rate(_segment(raw, 1.0 / 3.0, 2.0 / 3.0))
    late_bp = _bp_rate(_segment(raw, 2.0 / 3.0, 1.0))
    recent8_bp = _bp_rate(raw[-8:])
    recent18_bp = _bp_rate(raw[-18:])
    recent36_bp = _bp_rate(raw[-36:])

    vector = np.asarray([
        min(1.0, length / 80.0),
        global_bp,
        early_bp,
        middle_bp,
        late_bp,
        recent8_bp,
        recent18_bp,
        recent36_bp,
        max(-0.35, min(0.35, late_bp - early_bp)),
        max(-0.35, min(0.35, recent18_bp - global_bp)),
        min(0.35, _block_volatility(raw)),
        max(-0.08, min(0.08, _cumulative_curve_slope(raw))),
        float(full[2]),
        float(late[2]),
        float(recent18[2]),
    ], dtype=np.float64)

    return {
        "vector": vector,
        "full": full,
        "early": early,
        "middle": middle,
        "late": late,
        "recent8": recent8,
        "recent18": recent18,
        "recent36": recent36,
        "length": length,
        "global_bp": global_bp,
        "early_bp": early_bp,
        "middle_bp": middle_bp,
        "late_bp": late_bp,
        "recent8_bp": recent8_bp,
        "recent18_bp": recent18_bp,
        "recent36_bp": recent36_bp,
        "volatility": _block_volatility(raw),
        "curve_slope": _cumulative_curve_slope(raw),
    }


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    weights = np.asarray([
        0.55, 1.20, 0.75, 0.85, 1.00,
        0.65, 0.85, 0.90, 0.85, 1.05,
        0.70, 0.55, 0.55, 0.45, 0.50,
    ], dtype=np.float64)
    diff = left - right
    return float(math.sqrt(np.sum(weights * np.square(diff)) / np.sum(weights)))


def analyze_global_probability(values: Iterable[Any]) -> Dict[str, Any]:
    history = _clean(values)
    length = len(history)
    current = _state(history)

    candidates: List[Tuple[float, int, str]] = []
    for index in range(GLOBAL_MIN_PREFIX, length):
        prefix = history[:index]
        past = _state(prefix)
        distance = _distance(current["vector"], past["vector"])
        similarity = math.exp(-GLOBAL_SIMILARITY_SCALE * distance)
        # 全歷史都可參與，只給較新的狀態非常輕微的加成，避免又退化成近期模型。
        position_factor = 0.90 + 0.10 * (index / max(1, length - 1))
        candidates.append((similarity * position_factor, index, history[index]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    neighbors = candidates[:GLOBAL_NEIGHBORS]

    neighbor_counts = BASELINE * 12.0
    similarity_total = 0.0
    strong_support = 0
    for similarity, _, actual in neighbors:
        similarity_total += similarity
        if similarity >= 0.45:
            strong_support += 1
        neighbor_counts[OUTCOMES.index(actual)] += similarity
    neighbor_probability = _normalize(neighbor_counts)

    full_fit = current["full"]
    segmented_fit = _normalize(
        current["early"] * 0.22
        + current["middle"] * 0.28
        + current["late"] * 0.34
        + current["recent36"] * 0.16
    )

    mean_similarity = similarity_total / max(1, len(neighbors))
    maturity = min(1.0, length / 54.0)
    support_score = min(1.0, strong_support / 14.0)
    neighbor_share = min(0.52, maturity * (0.18 + 0.34 * support_score * mean_similarity))
    segmented_share = min(0.30, 0.12 + 0.18 * maturity)
    baseline_share = max(0.18, 1.0 - neighbor_share - segmented_share)

    fitted = _normalize(
        neighbor_probability * neighbor_share
        + segmented_fit * segmented_share
        + BASELINE * baseline_share
    )

    reliability = min(
        0.82,
        maturity
        * (
            0.34
            + 0.34 * support_score
            + 0.22 * mean_similarity
            + 0.10 * (1.0 - min(1.0, current["volatility"] / 0.20))
        ),
    )

    direction = "B" if fitted[0] >= fitted[1] else "P"
    bp_total = max(1e-12, float(fitted[0] + fitted[1]))
    bp_edge = abs(float(fitted[0] - fitted[1])) / bp_total

    return {
        "ok": length >= GLOBAL_MIN_PREFIX,
        "engine": "GLOBAL_FULL_HISTORY_PROBABILITY_FIT_V10_8",
        "sample_count": length,
        "full_history_used_count": length,
        "probabilities": {key: float(fitted[index]) for index, key in enumerate(OUTCOMES)},
        "banker_probability": float(fitted[0]),
        "player_probability": float(fitted[1]),
        "tie_probability": float(fitted[2]),
        "direction": direction,
        "reliability": float(reliability),
        "support": strong_support,
        "candidate_count": len(candidates),
        "mean_neighbor_similarity": float(mean_similarity),
        "bp_edge": float(bp_edge),
        "fit_components": {
            "baseline_share": float(baseline_share),
            "segmented_share": float(segmented_share),
            "neighbor_share": float(neighbor_share),
            "full_empirical": {key: float(full_fit[index]) for index, key in enumerate(OUTCOMES)},
            "segmented_fit": {key: float(segmented_fit[index]) for index, key in enumerate(OUTCOMES)},
            "neighbor_fit": {key: float(neighbor_probability[index]) for index, key in enumerate(OUTCOMES)},
        },
        "segments": {
            "early": {key: float(current["early"][index]) for index, key in enumerate(OUTCOMES)},
            "middle": {key: float(current["middle"][index]) for index, key in enumerate(OUTCOMES)},
            "late": {key: float(current["late"][index]) for index, key in enumerate(OUTCOMES)},
            "recent8": {key: float(current["recent8"][index]) for index, key in enumerate(OUTCOMES)},
            "recent18": {key: float(current["recent18"][index]) for index, key in enumerate(OUTCOMES)},
            "recent36": {key: float(current["recent36"][index]) for index, key in enumerate(OUTCOMES)},
        },
        "diagnostics": {
            "global_bp_rate": float(current["global_bp"]),
            "early_bp_rate": float(current["early_bp"]),
            "middle_bp_rate": float(current["middle_bp"]),
            "late_bp_rate": float(current["late_bp"]),
            "recent8_bp_rate": float(current["recent8_bp"]),
            "recent18_bp_rate": float(current["recent18_bp"]),
            "recent36_bp_rate": float(current["recent36_bp"]),
            "block_volatility": float(current["volatility"]),
            "cumulative_curve_slope": float(current["curve_slope"]),
            "walk_forward_only": True,
        },
    }


__all__ = ["analyze_global_probability"]
