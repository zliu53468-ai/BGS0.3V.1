"""BGS V10.7 全盤牌路結構相似狀態模型。

本模組重建六列大路，抽取龍長、交替率、欄高、齊腳與三種衍生路特徵。
它不再把「長龍一定續、單跳一定反」寫成固定答案；改為從同一副牌已發生
歷史中尋找結構最相近的過去狀態，統計其下一局是延續或轉折。

限制：牌路只描述已發生結果，不取得真人桌隱藏牌序，也不保證下一局。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import math
import os


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


FULL_ROAD_NEIGHBORS = _env_int("FULL_ROAD_NEIGHBORS", 24, 6, 80)
FULL_ROAD_LOOKBACK = _env_int("FULL_ROAD_LOOKBACK", 180, 36, 600)
FULL_ROAD_MIN_PREFIX = _env_int("FULL_ROAD_MIN_PREFIX", 10, 6, 40)


def _clean(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        raw = (
            item.get("outcome") or item.get("actual")
            if isinstance(item, Mapping)
            else item
        )
        value = str(raw or "").upper().strip()
        if value in {"B", "P"}:
            result.append(value)
    return result


def build_big_road(sequence: Sequence[str]) -> Dict[str, Any]:
    """依百家樂大路落點規則重建格位。"""
    seq = [str(v).upper() for v in sequence if str(v).upper() in {"B", "P"}]
    cells: List[Dict[str, Any]] = []
    occupied: set[Tuple[int, int]] = set()
    if not seq:
        return {"cells": [], "columns": [], "column_heights": []}

    col = row = start_col = 0
    previous = seq[0]
    cells.append({"index": 0, "outcome": previous, "column": col, "row": row})
    occupied.add((col, row))

    for index, outcome in enumerate(seq[1:], 1):
        if outcome == previous:
            below = (col, row + 1)
            if row < 5 and below not in occupied:
                row += 1
            else:
                col += 1
                while (col, row) in occupied:
                    col += 1
        else:
            start_col += 1
            while (start_col, 0) in occupied:
                start_col += 1
            col, row = start_col, 0
        occupied.add((col, row))
        cells.append(
            {"index": index, "outcome": outcome, "column": col, "row": row}
        )
        previous = outcome

    max_col = max(cell["column"] for cell in cells)
    columns: List[List[Dict[str, Any]]] = []
    for column_index in range(max_col + 1):
        column = sorted(
            (cell for cell in cells if cell["column"] == column_index),
            key=lambda cell: cell["row"],
        )
        columns.append(column)
    return {
        "cells": cells,
        "columns": columns,
        "column_heights": [len(column) for column in columns],
    }


def _runs(sequence: Sequence[str]) -> List[Tuple[str, int]]:
    result: List[Tuple[str, int]] = []
    for value in sequence:
        if result and result[-1][0] == value:
            result[-1] = (value, result[-1][1] + 1)
        else:
            result.append((value, 1))
    return result


def _rate_change(sequence: Sequence[str], size: int) -> float:
    recent = list(sequence[-size:])
    if len(recent) < 2:
        return 0.5
    return sum(a != b for a, b in zip(recent, recent[1:])) / (
        len(recent) - 1
    )


def _derived_road(heights: Sequence[int], offset: int) -> List[str]:
    result: List[str] = []
    for index in range(offset + 1, len(heights)):
        current = int(heights[index])
        reference = int(heights[index - offset])
        previous_reference = int(heights[index - offset - 1])
        delta_a = current - reference
        delta_b = reference - previous_reference
        regular = (
            current == reference
            or delta_a == delta_b
            or abs(current - reference) <= 1
        )
        result.append("R" if regular else "U")
    return result


def _binary_stats(values: Sequence[str]) -> Dict[str, float]:
    recent = list(values[-12:])
    if len(recent) < 2:
        return {
            "continuation": 0.5,
            "alternation": 0.5,
            "stability": 0.0,
        }
    continuation = sum(
        a == b for a, b in zip(recent, recent[1:])
    ) / (len(recent) - 1)
    alternation = 1.0 - continuation
    balance = abs(recent.count(recent[-1]) / len(recent) - 0.5) * 2.0
    stability = min(
        1.0,
        0.55 * max(continuation, alternation) + 0.45 * balance,
    )
    return {
        "continuation": continuation,
        "alternation": alternation,
        "stability": stability,
    }


def _structural_state(sequence: Sequence[str]) -> Dict[str, Any]:
    seq = list(sequence)
    road = build_big_road(seq)
    heights = list(road["column_heights"])
    runs = _runs(seq)
    run_lengths = [length for _, length in runs]
    recent_runs = run_lengths[-10:]
    current_run = recent_runs[-1] if recent_runs else 0
    mean_run = sum(recent_runs) / max(1, len(recent_runs))
    variance = sum(
        (length - mean_run) ** 2 for length in recent_runs
    ) / max(1, len(recent_runs))
    pair_rate = sum(length == 2 for length in recent_runs) / max(
        1, len(recent_runs)
    )
    equal_foot_rate = (
        sum(a == b for a, b in zip(heights[-10:], heights[-9:]))
        / max(1, len(heights[-10:]) - 1)
        if len(heights) >= 2
        else 0.0
    )

    big_eye = _derived_road(heights, 1)
    small = _derived_road(heights, 2)
    cockroach = _derived_road(heights, 3)
    derived_stats = {
        "big_eye": _binary_stats(big_eye),
        "small_road": _binary_stats(small),
        "cockroach_road": _binary_stats(cockroach),
    }

    vector = (
        min(1.0, current_run / 6.0),
        _rate_change(seq, 8),
        _rate_change(seq, 18),
        _rate_change(seq, 36),
        min(1.0, mean_run / 5.0),
        min(1.0, variance / 8.0),
        pair_rate,
        equal_foot_rate,
        derived_stats["big_eye"]["continuation"],
        derived_stats["small_road"]["continuation"],
        derived_stats["cockroach_road"]["continuation"],
    )
    return {
        "vector": vector,
        "road": road,
        "heights": heights,
        "runs": runs,
        "run_lengths": run_lengths,
        "current_run": current_run,
        "mean_run": mean_run,
        "variance": variance,
        "pair_rate": pair_rate,
        "equal_foot_rate": equal_foot_rate,
        "alternation_rate": _rate_change(seq, 24),
        "derived_roads": {
            "big_eye": big_eye,
            "small_road": small,
            "cockroach_road": cockroach,
        },
        "derived_stats": derived_stats,
    }


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    feature_weights = (
        1.20, 1.00, 0.90, 0.70, 0.85, 0.60,
        0.85, 0.65, 0.55, 0.50, 0.45,
    )
    total = 0.0
    for index, (a, b) in enumerate(zip(left, right)):
        total += feature_weights[index] * (float(a) - float(b)) ** 2
    return math.sqrt(total / sum(feature_weights))


def analyze_full_road_pattern(
    values: Iterable[Any],
    *,
    grid_cells: Sequence[Mapping[str, Any]] | None = None,
    initial_image_count: int = 0,
    manual_count: int = 0,
) -> Dict[str, Any]:
    sequence = _clean(values)
    sample_count = len(sequence)
    current = _structural_state(sequence)
    last_side = sequence[-1] if sequence else ""

    candidates: List[Tuple[float, int, bool, str]] = []
    start = max(FULL_ROAD_MIN_PREFIX, sample_count - FULL_ROAD_LOOKBACK)
    for index in range(start, sample_count):
        prefix = sequence[:index]
        if len(prefix) < FULL_ROAD_MIN_PREFIX:
            continue
        past = _structural_state(prefix)
        distance = _distance(current["vector"], past["vector"])
        similarity = math.exp(-4.2 * distance)
        previous = prefix[-1]
        actual_next = sequence[index]
        continued = actual_next == previous
        candidates.append((similarity, index, continued, actual_next))

    candidates.sort(key=lambda item: item[0], reverse=True)
    neighbors = candidates[:FULL_ROAD_NEIGHBORS]
    continue_mass = 2.5
    turn_mass = 2.5
    similarity_total = 0.0
    for similarity, _, continued, _ in neighbors:
        similarity_total += similarity
        if continued:
            continue_mass += similarity
        else:
            turn_mass += similarity

    continuation_probability = continue_mass / (continue_mass + turn_mass)
    banker_probability = (
        continuation_probability
        if last_side == "B"
        else 1.0 - continuation_probability
        if last_side == "P"
        else 0.5
    )
    direction = "B" if banker_probability >= 0.5 else "P"

    neighbor_support = len(
        [item for item in neighbors if item[0] >= 0.30]
    )
    mean_similarity = (
        similarity_total / len(neighbors) if neighbors else 0.0
    )
    maturity = min(1.0, sample_count / 48.0)
    support_score = min(1.0, neighbor_support / 12.0)
    reliability = min(
        1.0,
        maturity * (0.45 * support_score + 0.55 * mean_similarity),
    )
    active = sample_count >= 12 and neighbor_support >= 3

    return {
        "active": active,
        "support": neighbor_support,
        "historical_candidate_count": len(candidates),
        "banker_probability": banker_probability if sequence else 0.5,
        "direction": direction,
        "reliability": reliability if active else 0.0,
        "edge": abs(banker_probability - 0.5) * 2.0,
        "continuation_probability": continuation_probability,
        "continuation_score": (continuation_probability - 0.5) * 2.0,
        "last_side": last_side,
        "current_run": current["current_run"],
        "column_heights": current["heights"],
        "run_lengths": current["run_lengths"],
        "alternation_rate": current["alternation_rate"],
        "pair_rate": current["pair_rate"],
        "equal_foot_rate": current["equal_foot_rate"],
        "run_variance": current["variance"],
        "mean_neighbor_similarity": mean_similarity,
        "derived_roads": current["derived_roads"],
        "derived_stats": current["derived_stats"],
        "votes": [
            {
                "model": "historical_structural_neighbor",
                "history_index": index,
                "similarity": round(similarity, 6),
                "behavior": "continue" if continued else "turn",
                "actual_next": actual_next,
            }
            for similarity, index, continued, actual_next in neighbors[:12]
        ],
        "geometry": current["road"],
        "feature_vector": [round(float(v), 6) for v in current["vector"]],
        "engine": "FULL_ROAD_STRUCTURAL_KNN_V10_7",
        "full_history_used_count": sample_count,
        "initial_image_count": max(0, int(initial_image_count or 0)),
        "manual_count": max(0, int(manual_count or 0)),
        "grid_cell_count": len(list(grid_cells or [])),
        "lookback_mode": "walk_forward_historical_structural_neighbors",
        "hard_pattern_rules": False,
    }


__all__ = ["analyze_full_road_pattern", "build_big_road"]
