"""BGS 全盤牌路結構模型 V10.2。

本模組把完整 B/P 序列重建為六列大路，抽取全局欄高、龍長、轉折、
齊腳、長短欄交替與三種衍生結構路（大眼仔／小路／曱甴路）特徵。
輸出的是延續／轉折傾向與 B/P 機率，供 road_model.py 作為獨立專家。

限制：牌路只描述已發生結果，無法取得真人桌隱藏牌序，也不保證下一局。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import math


def _clean(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual")
        else:
            raw = item
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
        cells.append({"index": index, "outcome": outcome, "column": col, "row": row})
        previous = outcome

    max_col = max(cell["column"] for cell in cells)
    columns: List[List[Dict[str, Any]]] = []
    for c in range(max_col + 1):
        column = sorted((x for x in cells if x["column"] == c), key=lambda x: x["row"])
        columns.append(column)
    heights = [len(column) for column in columns]
    return {"cells": cells, "columns": columns, "column_heights": heights}


def _runs(sequence: Sequence[str]) -> List[Tuple[str, int]]:
    result: List[Tuple[str, int]] = []
    for value in sequence:
        if result and result[-1][0] == value:
            result[-1] = (value, result[-1][1] + 1)
        else:
            result.append((value, 1))
    return result


def _derived_road(heights: Sequence[int], offset: int) -> List[str]:
    """以欄高結構生成衍生路紅/藍序列；R=規整，U=不規整。"""
    result: List[str] = []
    for index in range(offset + 1, len(heights)):
        current = int(heights[index])
        reference = int(heights[index - offset])
        previous_reference = int(heights[index - offset - 1])
        # 同高或相鄰兩欄高度變化一致視為規整紅；否則藍。
        delta_a = current - reference
        delta_b = reference - previous_reference
        regular = current == reference or (delta_a == delta_b) or abs(current - reference) <= 1
        result.append("R" if regular else "U")
    return result


def _binary_pattern_score(values: Sequence[str]) -> Dict[str, float]:
    if len(values) < 2:
        return {"continuation": 0.5, "stability": 0.0, "alternation": 0.0}
    recent = list(values[-12:])
    same = sum(a == b for a, b in zip(recent, recent[1:]))
    transitions = max(1, len(recent) - 1)
    continuation = same / transitions
    alternation = 1.0 - continuation
    balance = abs(recent.count(recent[-1]) / len(recent) - 0.5) * 2.0
    stability = min(1.0, 0.55 * max(continuation, alternation) + 0.45 * balance)
    return {"continuation": continuation, "stability": stability, "alternation": alternation}


def analyze_full_road_pattern(values: Iterable[Any]) -> Dict[str, Any]:
    sequence = _clean(values)
    road = build_big_road(sequence)
    heights = road["column_heights"]
    runs = _runs(sequence)
    sample_count = len(sequence)
    last_side = sequence[-1] if sequence else ""
    current_run = runs[-1][1] if runs else 0
    run_lengths = [length for _, length in runs]

    big_eye = _derived_road(heights, 1)
    small = _derived_road(heights, 2)
    cockroach = _derived_road(heights, 3)
    derived = {
        "big_eye": _binary_pattern_score(big_eye),
        "small_road": _binary_pattern_score(small),
        "cockroach_road": _binary_pattern_score(cockroach),
    }

    # 全盤欄高規律：比較最近欄與全局歷史中相似欄高後的下一欄。
    analogue_continue = 0
    analogue_turn = 0
    support = 0
    context = tuple(run_lengths[-3:]) if len(run_lengths) >= 3 else tuple(run_lengths)
    if context:
        order = len(context)
        for i in range(order, len(run_lengths)):
            if tuple(run_lengths[i-order:i]) != context:
                continue
            support += 1
            previous_side = runs[i-1][0]
            next_side = runs[i][0]
            if next_side == previous_side:
                analogue_continue += 1
            else:
                analogue_turn += 1

    recent_runs = run_lengths[-8:]
    mean_run = sum(recent_runs) / max(1, len(recent_runs))
    variance = sum((x - mean_run) ** 2 for x in recent_runs) / max(1, len(recent_runs))
    alternation_rate = (
        sum(a != b for a, b in zip(sequence[-24:], sequence[-23:])) / max(1, len(sequence[-24:]) - 1)
        if sample_count >= 2 else 0.0
    )
    pair_rate = sum(length == 2 for length in recent_runs) / max(1, len(recent_runs))
    equal_foot_rate = (
        sum(a == b for a, b in zip(heights[-10:], heights[-9:])) / max(1, len(heights[-10:]) - 1)
        if len(heights) >= 2 else 0.0
    )

    # 多種結構專家投票：延續(+1)／轉折(-1)。
    votes: List[Tuple[str, float, float]] = []
    if current_run >= 4:
        votes.append(("current_streak", 1.0, min(1.0, 0.45 + current_run * 0.08)))
    elif alternation_rate >= 0.72:
        votes.append(("alternation", -1.0, min(1.0, alternation_rate)))
    elif pair_rate >= 0.55:
        # 雙跳：本欄第一顆偏延續，第二顆偏轉折。
        votes.append(("double_pattern", 1.0 if current_run == 1 else -1.0, 0.65 + 0.25 * pair_rate))
    else:
        votes.append(("run_balance", 1.0 if current_run < max(2.0, mean_run) else -1.0, 0.35))

    if support >= 2:
        total = analogue_continue + analogue_turn
        analogue_score = (analogue_continue - analogue_turn) / max(1, total)
        votes.append(("global_run_analogue", 1.0 if analogue_score >= 0 else -1.0, min(0.85, 0.35 + support / 12.0)))

    if len(heights) >= 4:
        last_delta = heights[-1] - heights[-2]
        prior_delta = heights[-2] - heights[-3]
        votes.append(("column_height_rhythm", 1.0 if last_delta == prior_delta else -1.0, 0.45 + 0.25 * equal_foot_rate))

    for name, stats in derived.items():
        seq = {"big_eye": big_eye, "small_road": small, "cockroach_road": cockroach}[name]
        if len(seq) >= 3:
            # 衍生路最近延續表示結構規律延續，交替表示結構可能轉折。
            sign = 1.0 if stats["continuation"] >= stats["alternation"] else -1.0
            votes.append((name, sign, 0.25 + 0.45 * stats["stability"]))

    weighted_sum = sum(sign * weight for _, sign, weight in votes)
    total_weight = sum(weight for _, _, weight in votes) or 1.0
    continuation_score = weighted_sum / total_weight
    continuation_probability = max(0.08, min(0.92, 0.5 + 0.36 * continuation_score))
    predicted = last_side if continuation_probability >= 0.5 else ("P" if last_side == "B" else "B")
    banker_probability = continuation_probability if last_side == "B" else 1.0 - continuation_probability
    reliability = min(1.0, (sample_count / 48.0) * (0.45 + 0.55 * abs(continuation_score)))
    active = sample_count >= 12 and len(runs) >= 4

    return {
        "active": active,
        "support": sample_count,
        "banker_probability": banker_probability if sequence else 0.5,
        "direction": predicted if predicted in {"B", "P"} else "B",
        "reliability": reliability if active else 0.0,
        "edge": abs(banker_probability - 0.5) * 2.0 if sequence else 0.0,
        "continuation_probability": continuation_probability,
        "continuation_score": continuation_score,
        "last_side": last_side,
        "current_run": current_run,
        "column_heights": heights,
        "run_lengths": run_lengths,
        "alternation_rate": alternation_rate,
        "pair_rate": pair_rate,
        "equal_foot_rate": equal_foot_rate,
        "run_variance": variance,
        "derived_roads": {
            "big_eye": big_eye,
            "small_road": small,
            "cockroach_road": cockroach,
        },
        "derived_stats": derived,
        "votes": [
            {"model": name, "behavior": "continue" if sign > 0 else "turn", "weight": round(weight, 6)}
            for name, sign, weight in votes
        ],
        "geometry": road,
    }


__all__ = ["analyze_full_road_pattern", "build_big_road"]
