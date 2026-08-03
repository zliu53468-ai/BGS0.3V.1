"""BGS V10.8 全盤牌路規劃模型。

完整重建六列大路，使用整副歷史的欄高、龍長分布、分段交替率、
齊腳率與大眼仔／小路／曱甴路結構。下一局不是由「長龍必續／單跳必反」
這類硬規則決定，而是用 walk-forward 的歷史結構相似狀態與龍長尾段類比擬合。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import math
import os

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


ROAD_PLAN_MIN_PREFIX = _env_int("ROAD_PLAN_MIN_PREFIX", 10, 6, 40)
ROAD_PLAN_NEIGHBORS = _env_int("ROAD_PLAN_NEIGHBORS", 32, 8, 120)
ROAD_PLAN_LOOKBACK = _env_int("ROAD_PLAN_LOOKBACK", 500, 36, 2000)
ROAD_PLAN_SIMILARITY_SCALE = _env_float("ROAD_PLAN_SIMILARITY_SCALE", 4.0, 1.0, 12.0)


def _clean(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("virtual_outcome")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P"}:
            result.append(value)
    return result[-ROAD_PLAN_LOOKBACK:]


def build_big_road(sequence: Sequence[str]) -> Dict[str, Any]:
    seq = [str(value).upper() for value in sequence if str(value).upper() in {"B", "P"}]
    cells: List[Dict[str, Any]] = []
    occupied: set[Tuple[int, int]] = set()
    if not seq:
        return {"cells": [], "columns": [], "column_heights": []}

    column = row = start_column = 0
    previous = seq[0]
    cells.append({"index": 0, "outcome": previous, "column": column, "row": row})
    occupied.add((column, row))

    for index, outcome in enumerate(seq[1:], 1):
        if outcome == previous:
            below = (column, row + 1)
            if row < 5 and below not in occupied:
                row += 1
            else:
                column += 1
                while (column, row) in occupied:
                    column += 1
        else:
            start_column += 1
            while (start_column, 0) in occupied:
                start_column += 1
            column, row = start_column, 0
        occupied.add((column, row))
        cells.append({"index": index, "outcome": outcome, "column": column, "row": row})
        previous = outcome

    maximum_column = max(cell["column"] for cell in cells)
    columns: List[List[Dict[str, Any]]] = []
    for column_index in range(maximum_column + 1):
        cells_in_column = sorted(
            (cell for cell in cells if cell["column"] == column_index),
            key=lambda cell: cell["row"],
        )
        columns.append(cells_in_column)
    return {
        "cells": cells,
        "columns": columns,
        "column_heights": [len(items) for items in columns],
    }


def _runs(sequence: Sequence[str]) -> List[Tuple[str, int]]:
    result: List[Tuple[str, int]] = []
    for value in sequence:
        if result and result[-1][0] == value:
            result[-1] = (value, result[-1][1] + 1)
        else:
            result.append((value, 1))
    return result


def _change_rate(values: Sequence[str]) -> float:
    if len(values) < 2:
        return 0.5
    return sum(a != b for a, b in zip(values, values[1:])) / (len(values) - 1)


def _derived_road(heights: Sequence[int], offset: int) -> List[str]:
    result: List[str] = []
    for index in range(offset + 1, len(heights)):
        current = int(heights[index])
        reference = int(heights[index - offset])
        previous_reference = int(heights[index - offset - 1])
        delta_current = current - reference
        delta_previous = reference - previous_reference
        regular = (
            current == reference
            or delta_current == delta_previous
            or abs(current - reference) <= 1
        )
        result.append("R" if regular else "U")
    return result


def _binary_stats(values: Sequence[str]) -> Dict[str, float]:
    if len(values) < 2:
        return {"continuation": 0.5, "recent_continuation": 0.5, "balance": 0.0}
    full = sum(a == b for a, b in zip(values, values[1:])) / (len(values) - 1)
    recent = list(values[-12:])
    recent_continuation = (
        sum(a == b for a, b in zip(recent, recent[1:])) / (len(recent) - 1)
        if len(recent) >= 2
        else 0.5
    )
    balance = abs(values.count("R") / len(values) - 0.5) * 2.0
    return {
        "continuation": full,
        "recent_continuation": recent_continuation,
        "balance": balance,
    }


def _state(sequence: Sequence[str]) -> Dict[str, Any]:
    seq = list(sequence)
    length = len(seq)
    road = build_big_road(seq)
    heights = list(road["column_heights"])
    runs = _runs(seq)
    run_lengths = [length for _, length in runs]
    recent_runs = run_lengths[-10:]
    current_run = run_lengths[-1] if run_lengths else 0

    mean_run = float(np.mean(run_lengths)) if run_lengths else 0.0
    run_variance = float(np.var(run_lengths)) if run_lengths else 0.0
    recent_mean_run = float(np.mean(recent_runs)) if recent_runs else 0.0
    recent_run_variance = float(np.var(recent_runs)) if recent_runs else 0.0

    run_histogram = [
        sum(value == 1 for value in run_lengths) / max(1, len(run_lengths)),
        sum(value == 2 for value in run_lengths) / max(1, len(run_lengths)),
        sum(value == 3 for value in run_lengths) / max(1, len(run_lengths)),
        sum(value >= 4 for value in run_lengths) / max(1, len(run_lengths)),
    ]

    early = seq[: max(1, length // 3)]
    middle = seq[max(0, length // 3): max(1, 2 * length // 3)]
    late = seq[max(0, 2 * length // 3):]

    height_mean = float(np.mean(heights)) if heights else 0.0
    height_variance = float(np.var(heights)) if heights else 0.0
    equal_foot = (
        sum(a == b for a, b in zip(heights, heights[1:])) / (len(heights) - 1)
        if len(heights) >= 2
        else 0.0
    )
    recent_heights = heights[-8:]
    recent_equal_foot = (
        sum(a == b for a, b in zip(recent_heights, recent_heights[1:]))
        / (len(recent_heights) - 1)
        if len(recent_heights) >= 2
        else 0.0
    )

    big_eye = _derived_road(heights, 1)
    small_road = _derived_road(heights, 2)
    cockroach_road = _derived_road(heights, 3)
    derived_stats = {
        "big_eye": _binary_stats(big_eye),
        "small_road": _binary_stats(small_road),
        "cockroach_road": _binary_stats(cockroach_road),
    }

    padded_last_heights = ([0] * 5 + heights)[-5:]
    padded_last_runs = ([0] * 5 + run_lengths)[-5:]

    vector = np.asarray([
        min(1.0, length / 80.0),
        min(1.0, current_run / 6.0),
        _change_rate(seq),
        _change_rate(early),
        _change_rate(middle),
        _change_rate(late),
        min(1.0, mean_run / 5.0),
        min(1.0, run_variance / 8.0),
        min(1.0, recent_mean_run / 5.0),
        min(1.0, recent_run_variance / 8.0),
        *run_histogram,
        min(1.0, height_mean / 5.0),
        min(1.0, height_variance / 8.0),
        equal_foot,
        recent_equal_foot,
        *(min(1.0, value / 6.0) for value in padded_last_heights),
        *(min(1.0, value / 6.0) for value in padded_last_runs),
        derived_stats["big_eye"]["continuation"],
        derived_stats["big_eye"]["recent_continuation"],
        derived_stats["small_road"]["continuation"],
        derived_stats["small_road"]["recent_continuation"],
        derived_stats["cockroach_road"]["continuation"],
        derived_stats["cockroach_road"]["recent_continuation"],
    ], dtype=np.float64)

    return {
        "vector": vector,
        "road": road,
        "heights": heights,
        "runs": runs,
        "run_lengths": run_lengths,
        "current_run": current_run,
        "mean_run": mean_run,
        "run_variance": run_variance,
        "recent_mean_run": recent_mean_run,
        "recent_run_variance": recent_run_variance,
        "run_histogram": run_histogram,
        "alternation_rate": _change_rate(seq),
        "early_alternation_rate": _change_rate(early),
        "middle_alternation_rate": _change_rate(middle),
        "late_alternation_rate": _change_rate(late),
        "equal_foot_rate": equal_foot,
        "recent_equal_foot_rate": recent_equal_foot,
        "derived_roads": {
            "big_eye": big_eye,
            "small_road": small_road,
            "cockroach_road": cockroach_road,
        },
        "derived_stats": derived_stats,
    }


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    weights = np.ones_like(left, dtype=np.float64)
    weights[:6] = [0.45, 1.15, 0.90, 0.60, 0.70, 0.85]
    weights[6:10] = [0.75, 0.55, 0.85, 0.65]
    weights[10:14] = [0.65, 0.75, 0.60, 0.70]
    weights[14:18] = [0.70, 0.60, 0.60, 0.70]
    weights[-6:] = [0.45, 0.55, 0.40, 0.50, 0.35, 0.45]
    difference = left - right
    return float(math.sqrt(np.sum(weights * np.square(difference)) / np.sum(weights)))


def _run_suffix_probability(sequence: Sequence[str]) -> Dict[str, Any]:
    """比較完整歷史中相同龍長尾段，標籤為下一顆續龍或轉折。"""
    target_runs = [length for _, length in _runs(sequence)]
    if len(target_runs) < 2:
        return {"continuation_probability": 0.5, "support": 0, "order": 0, "reliability": 0.0}

    for order in (5, 4, 3, 2):
        if len(target_runs) < order:
            continue
        suffix = tuple(target_runs[-order:])
        continued = 3.0
        turned = 3.0
        support = 0
        for index in range(ROAD_PLAN_MIN_PREFIX, len(sequence)):
            prefix = sequence[:index]
            prefix_runs = [length for _, length in _runs(prefix)]
            if len(prefix_runs) < order:
                continue
            if tuple(prefix_runs[-order:]) != suffix:
                continue
            support += 1
            if sequence[index] == prefix[-1]:
                continued += 1.0
            else:
                turned += 1.0
        if support >= 2:
            probability = continued / (continued + turned)
            return {
                "continuation_probability": probability,
                "support": support,
                "order": order,
                "reliability": min(0.74, support / 12.0),
                "walk_forward_only": True,
            }
    return {
        "continuation_probability": 0.5,
        "support": 0,
        "order": 0,
        "reliability": 0.0,
        "walk_forward_only": True,
    }


def analyze_road_planning(
    values: Iterable[Any],
    *,
    grid_cells: Sequence[Mapping[str, Any]] | None = None,
    initial_image_count: int = 0,
    manual_count: int = 0,
) -> Dict[str, Any]:
    sequence = _clean(values)
    sample_count = len(sequence)
    current = _state(sequence)
    last_side = sequence[-1] if sequence else ""

    candidates: List[Tuple[float, int, bool, str]] = []
    for index in range(ROAD_PLAN_MIN_PREFIX, sample_count):
        prefix = sequence[:index]
        past = _state(prefix)
        distance = _distance(current["vector"], past["vector"])
        similarity = math.exp(-ROAD_PLAN_SIMILARITY_SCALE * distance)
        actual_next = sequence[index]
        continued = actual_next == prefix[-1]
        candidates.append((similarity, index, continued, actual_next))

    candidates.sort(key=lambda item: item[0], reverse=True)
    neighbors = candidates[:ROAD_PLAN_NEIGHBORS]

    continue_mass = 3.0
    turn_mass = 3.0
    similarity_total = 0.0
    strong_support = 0
    for similarity, _, continued, _ in neighbors:
        similarity_total += similarity
        if similarity >= 0.42:
            strong_support += 1
        if continued:
            continue_mass += similarity
        else:
            turn_mass += similarity
    knn_continuation = continue_mass / (continue_mass + turn_mass)

    suffix = _run_suffix_probability(sequence)
    suffix_reliability = float(suffix.get("reliability", 0.0) or 0.0)
    suffix_share = min(0.28, 0.08 + 0.20 * suffix_reliability)
    knn_share = 1.0 - suffix_share
    continuation_probability = (
        knn_continuation * knn_share
        + float(suffix.get("continuation_probability", 0.5) or 0.5) * suffix_share
    )

    if last_side == "B":
        banker_probability = continuation_probability
    elif last_side == "P":
        banker_probability = 1.0 - continuation_probability
    else:
        banker_probability = 0.5

    mean_similarity = similarity_total / max(1, len(neighbors))
    maturity = min(1.0, sample_count / 54.0)
    support_score = min(1.0, strong_support / 14.0)
    derived_depth = min(
        1.0,
        (
            len(current["derived_roads"]["big_eye"])
            + len(current["derived_roads"]["small_road"])
            + len(current["derived_roads"]["cockroach_road"])
        ) / 36.0,
    )
    reliability = min(
        0.84,
        maturity * (
            0.28
            + 0.30 * support_score
            + 0.24 * mean_similarity
            + 0.10 * suffix_reliability
            + 0.08 * derived_depth
        ),
    )

    direction = "B" if banker_probability >= 0.5 else "P"
    return {
        "ok": sample_count >= ROAD_PLAN_MIN_PREFIX,
        "active": sample_count >= ROAD_PLAN_MIN_PREFIX and strong_support >= 2,
        "engine": "FULL_ROAD_PLANNING_WALK_FORWARD_V10_8",
        "sample_count": sample_count,
        "full_history_used_count": sample_count,
        "support": strong_support,
        "candidate_count": len(candidates),
        "banker_probability": float(banker_probability),
        "player_probability": float(1.0 - banker_probability),
        "direction": direction,
        "reliability": float(reliability),
        "edge": abs(float(banker_probability) - 0.5) * 2.0,
        "continuation_probability": float(continuation_probability),
        "knn_continuation_probability": float(knn_continuation),
        "run_suffix_model": suffix,
        "mean_neighbor_similarity": float(mean_similarity),
        "last_side": last_side,
        "current_run": int(current["current_run"]),
        "column_heights": current["heights"],
        "run_lengths": current["run_lengths"],
        "alternation_rate": float(current["alternation_rate"]),
        "early_alternation_rate": float(current["early_alternation_rate"]),
        "middle_alternation_rate": float(current["middle_alternation_rate"]),
        "late_alternation_rate": float(current["late_alternation_rate"]),
        "equal_foot_rate": float(current["equal_foot_rate"]),
        "recent_equal_foot_rate": float(current["recent_equal_foot_rate"]),
        "run_variance": float(current["run_variance"]),
        "run_histogram": [float(value) for value in current["run_histogram"]],
        "derived_roads": current["derived_roads"],
        "derived_stats": current["derived_stats"],
        "geometry": current["road"],
        "feature_vector": [round(float(value), 6) for value in current["vector"]],
        "nearest_states": [
            {
                "history_index": index,
                "similarity": round(similarity, 6),
                "behavior": "continue" if continued else "turn",
                "actual_next": actual_next,
            }
            for similarity, index, continued, actual_next in neighbors[:16]
        ],
        "initial_image_count": max(0, int(initial_image_count or 0)),
        "manual_count": max(0, int(manual_count or 0)),
        "grid_cell_count": len(list(grid_cells or [])),
        "hard_pattern_rules": False,
        "walk_forward_only": True,
    }


__all__ = ["analyze_road_planning", "build_big_road"]
