"""BBB V18 + V19 parity Contextual LinUCB core for BGS0.3V.1.

Formal prediction parity target:
- zliu53468-ai/BBB app256forward.js (V18)
- zliu53468-ai/BBB app256continuation.js (V19)

Invariants:
- 256D context = 128D shoe/progression + 128D road/6x15 structure.
- B/P two-arm frozen Contextual LinUCB.
- ALPHA=0.5, RIDGE=1.0, confidence bounded to 42%-58%.
- No bootstrap, walk-forward, replay, previous-prediction settlement, A/b update, or decay.
- Only last_selected / selection_streak are persisted per user/venue/room/shoe scope.
- OCR, screenshot parsing, LINE/LIFF UI, public predictor fields and money management live outside this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import os
import time

import numpy as np

from shoe_constants import SHOE_DECKS

ARMS = ("P", "B")
SHOE_CONTEXT_DIM = 128
ROAD_CONTEXT_DIM = 128
CONTEXT_DIM = 256
BIG_ROAD_ROWS = 6
BIG_ROAD_COLS = 15
LINUCB_ALPHA = 0.5
LINUCB_RIDGE = 1.0
LINUCB_UPDATE_WEIGHT = 0.0
LINUCB_FORGETTING = 1.0
LINUCB_ARM_ALPHA_MAX_SCALE = 1.0
LINUCB_SCORE_TIE_EPSILON = 1e-9
LINUCB_SCORE_TEMPERATURE = 0.42
ROAD_PRIOR_SCORE_WEIGHT = 0.0
ROAD_PRIOR_PROBABILITY_SPAN = 0.0
LINUCB_PROBABILITY_CORRECTION_SPAN = 0.0
PROBABILITY_MIN = 0.42
PROBABILITY_MAX = 0.58
TOTAL_CARDS = 416.0
ESTIMATED_CARDS_PER_ROUND = 4.9
STATE_VERSION = "LINUCB-2ARM-FROZEN-256D-BBB-V18-V19-6X15-PARITY"
_LOCK = RLock()


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _signed(value: Any) -> float:
    return _clip(value, -1.0, 1.0)


def _normalize_history(history: Iterable[Any] | str | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = history.replace("|", "").replace(",", "").replace(" ", "").upper()
        if compact and all(char in {"B", "P", "T"} for char in compact):
            return list(compact)[-2000:]
        values: Iterable[Any] = [part for part in history.replace("|", ",").split(",") if part.strip()]
    else:
        values = deepcopy(list(history))
    out: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            out.append(value)
    return out[-2000:]


def _bp(sequence: Sequence[str]) -> list[str]:
    return [x for x in sequence if x in {"B", "P"}]


def _side_sign(side: str) -> float:
    return 1.0 if side == "B" else -1.0 if side == "P" else 0.0


def _runs(sequence: Sequence[str]) -> list[tuple[str, int]]:
    values = _bp(sequence)
    if not values:
        return []
    out: list[tuple[str, int]] = []
    side, n = values[0], 1
    for value in values[1:]:
        if value == side:
            n += 1
        else:
            out.append((side, n))
            side, n = value, 1
    out.append((side, n))
    return out


def _banker_ratio(sequence: Sequence[str], window: int) -> float:
    values = _bp(sequence)[-max(1, int(window)):]
    return float(sum(x == "B" for x in values) / len(values)) if values else 0.5


def _turn_rate(sequence: Sequence[str], window: int) -> float:
    values = _bp(sequence)[-max(2, int(window)):]
    if len(values) < 2:
        return 0.5
    turns = sum(values[i] != values[i - 1] for i in range(1, len(values)))
    return float(turns / (len(values) - 1))


def _tie_ratio(sequence: Sequence[str], window: int = 0) -> float:
    values = list(sequence[-window:]) if window else list(sequence)
    return float(sum(x == "T" for x in values) / len(values)) if values else 0.0


def _binary_entropy(sequence: Sequence[str], window: int = 12) -> float:
    values = _bp(sequence)[-window:]
    if not values:
        return 1.0
    p = sum(x == "B" for x in values) / len(values)
    q = 1.0 - p
    e = 0.0
    if p:
        e -= p * math.log2(p)
    if q:
        e -= q * math.log2(q)
    return _clip(e)


def _outcome_entropy(sequence: Sequence[str], window: int = 12) -> float:
    values = list(sequence[-window:])
    if not values:
        return 1.0
    e = 0.0
    for outcome in ("B", "P", "T"):
        p = sum(x == outcome for x in values) / len(values)
        if p:
            e -= p * math.log2(p)
    return _clip(e / math.log2(3.0))


def _balance(sequence: Sequence[str], window: int) -> float:
    return _clip(1.0 - abs(_banker_ratio(sequence, window) - 0.5) * 2.0)


def _same_tail(sequence: Sequence[str], window: int) -> float:
    values = _bp(sequence)[-window:]
    if len(values) < window:
        return 0.5
    return 1.0 if all(x == values[0] for x in values) else 0.0


def _alternating_tail(sequence: Sequence[str], window: int) -> float:
    values = _bp(sequence)[-window:]
    if len(values) < window:
        return 0.5
    return 1.0 if all(values[i] != values[i - 1] for i in range(1, len(values))) else 0.0


def _run_volatility(sequence: Sequence[str], window: int = 6) -> float:
    heights = [x[1] for x in _runs(sequence)[-window:]]
    if len(heights) < 2:
        return 0.25
    delta = sum(abs(heights[i] - heights[i - 1]) for i in range(1, len(heights)))
    return _clip(delta / (len(heights) - 1) / 3.0)


def _run_trend(sequence: Sequence[str], window: int = 5) -> float:
    heights = [x[1] for x in _runs(sequence)[-window:]]
    if len(heights) < 2:
        return 0.5
    return _clip(0.5 + ((heights[-1] - heights[0]) / (len(heights) - 1)) / 6.0)


def _run_stats(sequence: Sequence[str], window: int) -> dict[str, float]:
    heights = [x[1] for x in _runs(sequence)[-window:]]
    if not heights:
        return {"avg": 0.0, "max": 0.0, "std": 0.0}
    mean = sum(heights) / len(heights)
    variance = sum((x - mean) ** 2 for x in heights) / len(heights)
    return {"avg": _clip(mean / 8.0), "max": _clip(max(heights) / 12.0), "std": _clip(math.sqrt(variance) / 6.0)}


def _cell_key(row: int, col: int) -> tuple[int, int]:
    return row, col


def _build_big_road(sequence: Sequence[str]) -> dict[str, Any]:
    occupied: dict[tuple[int, int], dict[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    streaks: list[dict[str, Any]] = []
    current_side = ""
    current_row = 0
    current_col = -1
    current_origin_col = -1
    streak_index = -1
    pending_ties = 0

    def place(side: str, row: int, col: int, origin_col: int, move: str) -> dict[str, Any]:
        cell = {
            "side": side, "row": row, "col": col, "originCol": origin_col,
            "streakIndex": streak_index, "move": move, "ties": 0, "index": len(cells),
        }
        occupied[_cell_key(row, col)] = cell
        cells.append(cell)
        streaks[streak_index]["cells"].append(cell)
        return cell

    for outcome in sequence:
        if outcome == "T":
            if cells:
                cells[-1]["ties"] += 1
            else:
                pending_ties += 1
            continue
        if outcome not in {"B", "P"}:
            continue
        if not current_side:
            current_side = outcome
            current_row = 0
            current_col = 0
            current_origin_col = 0
            streak_index = 0
            streaks.append({"side": outcome, "originCol": 0, "cells": []})
            first = place(outcome, 0, 0, 0, "start")
            first["ties"] += pending_ties
            pending_ties = 0
            continue
        if outcome != current_side:
            current_side = outcome
            current_row = 0
            current_col = current_origin_col + 1
            while _cell_key(0, current_col) in occupied:
                current_col += 1
            current_origin_col = current_col
            streak_index += 1
            streaks.append({"side": outcome, "originCol": current_origin_col, "cells": []})
            place(outcome, 0, current_col, current_origin_col, "new-column")
            continue
        below_row = current_row + 1
        if below_row < BIG_ROAD_ROWS and _cell_key(below_row, current_col) not in occupied:
            current_row = below_row
            place(outcome, current_row, current_col, current_origin_col, "down")
        else:
            next_col = current_col + 1
            while _cell_key(current_row, next_col) in occupied:
                next_col += 1
            current_col = next_col
            place(outcome, current_row, current_col, current_origin_col, "tail-bottom" if below_row >= BIG_ROAD_ROWS else "tail-collision")

    for streak in streaks:
        rows = [c["row"] for c in streak["cells"]]
        streak["logicalLength"] = len(streak["cells"])
        streak["verticalHeight"] = max(rows) + 1 if rows else 0
        streak["tailLength"] = sum(c["col"] > streak["originCol"] for c in streak["cells"])
        streak["endRow"] = streak["cells"][-1]["row"] if streak["cells"] else 0
        streak["endCol"] = streak["cells"][-1]["col"] if streak["cells"] else streak["originCol"]
        streak["hasCollisionTail"] = any(c["move"] == "tail-collision" for c in streak["cells"])
        streak["hasBottomTail"] = any(c["move"] == "tail-bottom" for c in streak["cells"])

    max_col = max((c["col"] for c in cells), default=0)
    view_start_col = max(0, max_col - BIG_ROAD_COLS + 1)
    view_end_col = view_start_col + BIG_ROAD_COLS - 1
    visible_cells = [c for c in cells if view_start_col <= c["col"] <= view_end_col]
    grid = [[None for _ in range(BIG_ROAD_COLS)] for _ in range(BIG_ROAD_ROWS)]
    for c in visible_cells:
        vc = c["col"] - view_start_col
        if 0 <= vc < BIG_ROAD_COLS:
            grid[c["row"]][vc] = c["side"]
    return {
        "rows": BIG_ROAD_ROWS, "cols": BIG_ROAD_COLS, "cells": cells, "streaks": streaks,
        "occupied": occupied, "maxCol": max_col, "viewStartCol": view_start_col, "viewEndCol": view_end_col,
        "visibleCells": visible_cells, "grid": grid, "currentCell": cells[-1] if cells else None,
        "currentStreak": streaks[-1] if streaks else None,
    }


def _full_completed_streaks(road: Mapping[str, Any]) -> list[dict[str, Any]]:
    streaks = list(road.get("streaks") or [])
    return streaks[:-1] if len(streaks) > 1 else []


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    a = sorted(float(x) for x in values)
    m = len(a) // 2
    return a[m] if len(a) % 2 else (a[m - 1] + a[m]) / 2.0


def _mode_value(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"value": 0.0, "support": 0.0}
    counts: dict[float, int] = {}
    for x in values:
        fx = float(x)
        counts[fx] = counts.get(fx, 0) + 1
    med = _median(values)
    best = float(values[0])
    best_count = 0
    for value, count in counts.items():
        if count > best_count or (count == best_count and abs(value - med) < abs(best - med)):
            best, best_count = value, count
    return {"value": best, "support": best_count / len(values)}


def _summarize(values: Sequence[float], std_scale: float = 6.0) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "med": 0.0, "mode": 0.0, "recent": 0.0, "std": 0.0, "target": 0.0, "reliability": 0.0}
    vals = [float(x) for x in values]
    mean = sum(vals) / len(vals)
    med = _median(vals)
    mv = _mode_value(vals)
    recent_vals = vals[-4:]
    recent = sum(recent_vals) / len(recent_vals)
    variance = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = math.sqrt(variance)
    target = 0.42 * mv["value"] + 0.28 * med + 0.30 * recent
    reliability = _clip(len(vals) / 5.0) * _clip(0.45 + 0.55 * mv["support"]) * _clip(1.0 - std / std_scale)
    return {"count": len(vals), "mean": mean, "med": med, "mode": mv["value"], "recent": recent, "std": std, "target": target, "reliability": reliability}


def _side_road_stats(completed: Sequence[Mapping[str, Any]], side: str) -> dict[str, float]:
    items = [x for x in completed if x.get("side") == side][-12:]
    if not items:
        return {"count": 0, "verticalTarget": 0.0, "logicalTarget": 0.0, "tailTarget": 0.0, "reliability": 0.0}
    v = _summarize([x["verticalHeight"] for x in items], 5.0)
    l = _summarize([x["logicalLength"] for x in items], 8.0)
    t = _summarize([x["tailLength"] for x in items], 6.0)
    return {
        "count": len(items), "verticalTarget": v["target"], "logicalTarget": l["target"], "tailTarget": t["target"],
        "verticalReliability": v["reliability"], "logicalReliability": l["reliability"], "tailReliability": t["reliability"],
        "reliability": _clip(0.52 * v["reliability"] + 0.40 * l["reliability"] + 0.08 * t["reliability"]),
        "verticalStd": v["std"], "logicalStd": l["std"],
    }


def _stage_survival(completed: Sequence[Mapping[str, Any]], side: str, current_logical: int) -> dict[str, float]:
    items = [x for x in completed if x.get("side") == side][-16:]
    reached_w = continued_w = ended_w = 0.0
    for i, streak in enumerate(items):
        if int(streak.get("logicalLength", 0)) < current_logical:
            continue
        recency = 0.94 ** (len(items) - 1 - i)
        reached_w += recency
        if int(streak.get("logicalLength", 0)) > current_logical:
            continued_w += recency
        else:
            ended_w += recency
    prior = 1.15
    denom = reached_w + 2.0 * prior
    return {"cont": _clip((continued_w + prior) / denom), "turn": _clip((ended_w + prior) / denom), "support": _clip(reached_w / 4.0), "reached": reached_w}


def _pair_road_stats(completed: Sequence[Mapping[str, Any]], from_side: str) -> dict[str, float]:
    to_side = "P" if from_side == "B" else "B"
    pairs: list[tuple[float, float, float, float]] = []
    for i in range(len(completed) - 1):
        if completed[i].get("side") == from_side and completed[i + 1].get("side") == to_side:
            pairs.append((completed[i]["verticalHeight"], completed[i]["logicalLength"], completed[i + 1]["verticalHeight"], completed[i + 1]["logicalLength"]))
    use = pairs[-10:]
    if not use:
        return {"count": 0, "fromVertical": 0.0, "fromLogical": 0.0, "toVertical": 0.0, "toLogical": 0.0, "reliability": 0.0}
    fv = _summarize([x[0] for x in use], 5.0)
    fl = _summarize([x[1] for x in use], 8.0)
    tv = _summarize([x[2] for x in use], 5.0)
    tl = _summarize([x[3] for x in use], 8.0)
    return {"count": len(use), "fromVertical": fv["target"], "fromLogical": fl["target"], "toVertical": tv["target"], "toLogical": tl["target"], "reliability": _clip(len(use) / 6.0) * _clip(0.55 + 0.25 * fv["reliability"] + 0.20 * tv["reliability"])}


def _closeness(a: float, b: float, span: float = 3.0) -> float:
    if not b:
        return 0.5
    return _clip(1.0 - abs(a - b) / max(span, b * 0.65 + 1.0))


def _contextual_stage_stats(completed: Sequence[Mapping[str, Any]], current_index: int, current_side: str, current_logical: int) -> dict[str, float]:
    current_prev1 = completed[current_index - 1] if current_index > 0 and current_index - 1 < len(completed) else None
    current_prev2 = completed[current_index - 2] if current_index > 1 and current_index - 2 < len(completed) else None
    if not current_prev1:
        return {"cont": 0.5, "turn": 0.5, "support": 0.0}
    total = cont = turn = 0.0
    for i, streak in enumerate(completed):
        if streak.get("side") != current_side or int(streak.get("logicalLength", 0)) < current_logical or i < 1:
            continue
        p1 = completed[i - 1]
        p2 = completed[i - 2] if i > 1 else None
        sim = 0.72 * _closeness(float(p1["logicalLength"]), float(current_prev1["logicalLength"]), 3.0)
        if current_prev2 and p2:
            sim += 0.28 * _closeness(float(p2["logicalLength"]), float(current_prev2["logicalLength"]), 3.5)
        else:
            sim += 0.14
        recency = 0.96 ** (len(completed) - 1 - i)
        weight = _clip(sim) ** 2 * recency
        total += weight
        if int(streak.get("logicalLength", 0)) > current_logical:
            cont += weight
        else:
            turn += weight
    prior = 0.75
    denom = total + 2.0 * prior
    return {"cont": _clip((cont + prior) / denom), "turn": _clip((turn + prior) / denom), "support": _clip(total / 3.0)}


def _big_road_geometry(road: Mapping[str, Any]) -> dict[str, float]:
    cells = list(road.get("visibleCells") or [])
    if not cells:
        return {"occupancy":0.0,"bankerRatio":0.5,"topOccupancy":0.0,"bottomOccupancy":0.0,"tailRatio":0.0,"activeColumns":0.0,"avgFill":0.0,"maxFill":0.0,"fillStd":0.0,"profileRegularity":0.5,"currentRow":0.0,"currentViewCol":0.0,"currentInTail":0.0,"currentAtBottom":0.0,"blockedBelow":0.0}
    counts = [0] * BIG_ROAD_COLS
    profiles = [0] * BIG_ROAD_COLS
    top = bottom = tail = banker = 0
    view_start = int(road.get("viewStartCol", 0))
    for c in cells:
        vc = int(c["col"]) - view_start
        if not (0 <= vc < BIG_ROAD_COLS):
            continue
        counts[vc] += 1
        profiles[vc] |= 1 << int(c["row"])
        top += int(c["row"] == 0)
        bottom += int(c["row"] == BIG_ROAD_ROWS - 1)
        tail += int(c["col"] > c["originCol"])
        banker += int(c["side"] == "B")
    active = [x for x in counts if x > 0]
    avg = sum(active) / len(active) if active else 0.0
    mx = max(active) if active else 0.0
    variance = sum((x - avg) ** 2 for x in active) / len(active) if active else 0.0
    used = [x for x in profiles if x != 0]
    sim = 0.0
    n = 0
    for i in range(1, len(used)):
        xor = used[i] ^ used[i - 1]
        bits = sum((xor >> b) & 1 for b in range(BIG_ROAD_ROWS))
        sim += 1.0 - bits / BIG_ROAD_ROWS
        n += 1
    current = road.get("currentCell")
    occupied = road.get("occupied") or {}
    blocked = bool(current and (int(current["row"]) >= BIG_ROAD_ROWS - 1 or _cell_key(int(current["row"]) + 1, int(current["col"])) in occupied))
    return {
        "occupancy": _clip(len(cells) / (BIG_ROAD_ROWS * BIG_ROAD_COLS)), "bankerRatio": banker / len(cells),
        "topOccupancy": _clip(top / BIG_ROAD_COLS), "bottomOccupancy": _clip(bottom / BIG_ROAD_COLS), "tailRatio": tail / len(cells),
        "activeColumns": _clip(len(active) / BIG_ROAD_COLS), "avgFill": _clip(avg / BIG_ROAD_ROWS), "maxFill": _clip(mx / BIG_ROAD_ROWS),
        "fillStd": _clip(math.sqrt(variance) / BIG_ROAD_ROWS), "profileRegularity": _clip(sim / n) if n else 0.5,
        "currentRow": float(current["row"]) if current else 0.0, "currentViewCol": float(current["col"] - view_start) if current else 0.0,
        "currentInTail": float(bool(current and current["col"] > current["originCol"])), "currentAtBottom": float(bool(current and current["row"] == BIG_ROAD_ROWS - 1)),
        "blockedBelow": float(blocked),
    }


def _candidate_move_info(base_road: Mapping[str, Any], candidate: str, sequence: Sequence[str]) -> dict[str, Any]:
    after = _build_big_road([*sequence, candidate])
    added = after["cells"][-1] if len(after["cells"]) > len(base_road.get("cells") or []) else None
    if not added:
        return {"after": after, "row": 0, "col": 0, "viewCol": 0, "down": 0.0, "right": 0.0, "newColumn": 0.0}
    return {"after": after, "row": added["row"], "col": added["col"], "viewCol": added["col"] - after["viewStartCol"], "down": float(added["move"] == "down"), "right": float(added["move"] in {"tail-bottom", "tail-collision"}), "newColumn": float(added["move"] == "new-column")}


def _grid_candidate_quality(base_road: Mapping[str, Any], after_road: Mapping[str, Any]) -> float:
    before = _big_road_geometry(base_road)
    after = _big_road_geometry(after_road)
    regularity_gain = _clip(0.5 + (after["profileRegularity"] - before["profileRegularity"]) * 1.4)
    smooth = _clip(1.0 - after["fillStd"])
    return _clip(0.64 * regularity_gain + 0.36 * smooth)


def _derived_mark(heights: Sequence[int], column: int, row: int, new_column: bool, offset: int) -> str:
    if new_column:
        if column < offset + 1:
            return ""
        return "R" if heights[column - 1] == heights[column - 1 - offset] else "U"
    if column < offset:
        return ""
    ref = heights[column - offset]
    return "R" if (ref >= row) == (ref >= row - 1) else "U"


def _build_derived_roads(sequence: Sequence[str]) -> dict[str, list[str]]:
    road = _build_big_road(sequence)
    streaks = road["streaks"]
    out = {"big_eye": [], "small_road": [], "cockroach_road": []}
    offsets = {"big_eye": 1, "small_road": 2, "cockroach_road": 3}
    heights: list[int] = []
    for i, streak in enumerate(streaks):
        heights.append(int(streak["verticalHeight"]))
        for name, off in offsets.items():
            mark = _derived_mark(heights, i, int(streak["verticalHeight"]), True, off)
            if mark:
                out[name].append(mark)
        for row in range(2, int(streak["verticalHeight"]) + 1):
            for name, off in offsets.items():
                mark = _derived_mark(heights, i, row, False, off)
                if mark:
                    out[name].append(mark)
    return out


def _regularity(values: Sequence[str], window: int = 8) -> tuple[float, int]:
    marks = [x for x in list(values[-window:]) if x in {"R", "U"}]
    return (sum(x == "R" for x in marks) / len(marks), len(marks)) if marks else (0.5, 0)


def _derived_info(sequence: Sequence[str], window: int = 8) -> dict[str, float]:
    roads = _build_derived_roads(sequence)
    be, bn = _regularity(roads["big_eye"], window)
    sm, sn = _regularity(roads["small_road"], window)
    cr, cn = _regularity(roads["cockroach_road"], window)
    mean = (be + sm + cr) / 3.0
    return {"be": be, "sm": sm, "cr": cr, "bn": bn, "sn": sn, "cn": cn, "consensus": _clip(1.0 - (abs(be - mean) + abs(sm - mean) + abs(cr - mean)) / 1.5), "support": _clip((bn + sn + cn) / (window * 3.0))}


def _branch_future_quality(sequence: Sequence[str], first: str) -> float:
    road1 = _build_big_road([*sequence, first])
    current = road1.get("currentStreak")
    if not current:
        return 0.5
    completed = _full_completed_streaks(road1)
    own = _side_road_stats(completed, first)
    stage = _stage_survival(completed, first, int(current["logicalLength"]))
    context = _contextual_stage_stats(completed, len(completed), first, int(current["logicalLength"]))
    target_fit = _closeness(current["logicalLength"], own.get("logicalTarget", 0.0), 3.0) if own.get("count") else 0.5
    second_same = _build_big_road([*sequence, first, first])
    same_streak = second_same.get("currentStreak")
    same_next_fit = _closeness(same_streak["logicalLength"], own.get("logicalTarget", 0.0), 3.0) if own.get("count") and same_streak else 0.5
    pair = _pair_road_stats(completed, first)
    turn_fit = _clip(0.55 * _closeness(current["logicalLength"], pair.get("fromLogical", 0.0), 3.0) + 0.45 * pair.get("reliability", 0.0)) if pair.get("count") else 0.5
    q_continue = _clip(0.42 * stage["cont"] + 0.25 * context["cont"] + 0.23 * same_next_fit + 0.10 * target_fit)
    q_turn = _clip(0.42 * stage["turn"] + 0.25 * context["turn"] + 0.23 * turn_fit + 0.10 * target_fit)
    return _clip(0.62 * max(q_continue, q_turn) + 0.38 * ((q_continue + q_turn) / 2.0))


def _big_road_candidates(sequence: Sequence[str]) -> dict[str, Any]:
    road = _build_big_road(sequence)
    current = road.get("currentStreak")
    if not current:
        return {"B":0.5,"P":0.5,"support":0.0,"currentSide":"","road":road,"geometry":_big_road_geometry(road),"forwardDirectional":0.0,"survivalDirectional":0.0,"contextDirectional":0.0,"twoStepDirectional":0.0}
    completed = _full_completed_streaks(road)
    current_side = str(current["side"])
    opposite = "P" if current_side == "B" else "B"
    sign = _side_sign(current_side)
    own = _side_road_stats(completed, current_side)
    opp = _side_road_stats(completed, opposite)
    stage = _stage_survival(completed, current_side, int(current["logicalLength"]))
    context = _contextual_stage_stats(completed, len(completed), current_side, int(current["logicalLength"]))
    pair = _pair_road_stats(completed, current_side)
    cont_info = _candidate_move_info(road, current_side, sequence)
    rev_info = _candidate_move_info(road, opposite, sequence)
    cont_streak = cont_info["after"].get("currentStreak")
    next_logical = int(cont_streak["logicalLength"]) if cont_streak else int(current["logicalLength"])
    target_fit_now = _closeness(current["logicalLength"], own.get("logicalTarget", 0.0), 3.0) if own.get("count") else 0.5
    target_fit_next = _closeness(next_logical, own.get("logicalTarget", 0.0), 3.0) if own.get("count") else 0.5
    pair_fit = _clip(0.58 * _closeness(current["logicalLength"], pair.get("fromLogical", 0.0), 3.0) + 0.42 * _closeness(current["verticalHeight"], pair.get("fromVertical", 0.0), 2.2)) if pair.get("count") else 0.5
    continue_grid = _grid_candidate_quality(road, cont_info["after"])
    reverse_grid = _grid_candidate_quality(road, rev_info["after"])
    cdi = _derived_info([*sequence, current_side], 8)
    rdi = _derived_info([*sequence, opposite], 8)
    continue_derived = _clip(0.6 * cdi["consensus"] + 0.4 * cdi["support"])
    reverse_derived = _clip(0.6 * rdi["consensus"] + 0.4 * rdi["support"])
    continue_score = _clip(0.31*stage["cont"] + 0.22*context["cont"] + 0.19*target_fit_next + 0.10*own.get("reliability",0.0) + 0.08*continue_derived + 0.06*continue_grid + 0.04*(1.0-target_fit_now))
    reverse_score = _clip(0.31*stage["turn"] + 0.22*context["turn"] + 0.19*target_fit_now + 0.14*pair_fit + 0.06*pair.get("reliability",0.0) + 0.05*reverse_derived + 0.03*reverse_grid)
    scores = {"B": continue_score, "P": reverse_score} if current_side == "B" else {"P": continue_score, "B": reverse_score}
    two_b = _branch_future_quality(sequence, "B")
    two_p = _branch_future_quality(sequence, "P")
    two_step_directional = _signed(two_b - two_p)
    survival_directional = _signed(sign * (stage["cont"] - stage["turn"]))
    context_directional = _signed(sign * (context["cont"] - context["turn"]))
    forward_directional = _signed(0.48*(scores["B"]-scores["P"]) + 0.24*survival_directional + 0.18*context_directional + 0.10*two_step_directional)
    support = _clip(0.30*stage["support"] + 0.24*context["support"] + 0.18*own.get("reliability",0.0) + 0.12*pair.get("reliability",0.0) + 0.16*min(1.0, len(completed)/8.0))
    return {
        **scores, "support": support, "currentSide": current_side, "road": road, "geometry": _big_road_geometry(road),
        "stageCont": stage["cont"], "stageTurn": stage["turn"], "stageSupport": stage["support"],
        "contextCont": context["cont"], "contextTurn": context["turn"], "contextSupport": context["support"],
        "ownTarget": own.get("logicalTarget",0.0), "oppositeTarget": opp.get("logicalTarget",0.0),
        "pairReliability": pair.get("reliability",0.0), "pairFit": pair_fit,
        "continueGrid": continue_grid, "reverseGrid": reverse_grid, "continueMove": cont_info, "reverseMove": rev_info,
        "targetFitNow": target_fit_now, "targetFitNext": target_fit_next,
        "forwardDirectional": forward_directional, "survivalDirectional": survival_directional, "contextDirectional": context_directional,
        "twoStepDirectional": two_step_directional, "twoB": two_b, "twoP": two_p,
    }


def _build_shoe_vector(sequence: Sequence[str]) -> tuple[list[float], list[str]]:
    used = min(TOTAL_CARDS, len(sequence) * ESTIMATED_CARDS_PER_ROUND)
    remaining = max(0.0, TOTAL_CARDS - used)
    rr = _clip(remaining / TOTAL_CARDS)
    pen = _clip(1.0 - rr)
    maturity = _clip(len(sequence) / 70.0)
    hands = _clip(len(sequence) / (TOTAL_CARDS / ESTIMATED_CARDS_PER_ROUND))
    v: list[float] = [rr,pen,rr,maturity,*([1.0]*10),0.0,0.0,_clip(1-pen/.35),_clip(1-abs(pen-.5)/.35),_clip((pen-.55)/.35),hands,rr,_clip(math.log1p(len(sequence))/math.log1p(TOTAL_CARDS/ESTIMATED_CARDS_PER_ROUND)),_tie_ratio(sequence),_tie_ratio(sequence,8),_tie_ratio(sequence,16),_balance(sequence,max(1,len(_bp(sequence)))),_binary_entropy(sequence,12),_outcome_entropy(sequence,12),_outcome_entropy(sequence,24),_clip(len(sequence)/32.0),1.0,_clip(math.sqrt(len(sequence))/math.sqrt(TOTAL_CARDS/ESTIMATED_CARDS_PER_ROUND))]
    names = [
        "remaining_cards_ratio","penetration_ratio","remaining_cards_ratio_repeat","shoe_maturity_ratio",
        *[f"neutral_rank_{i}" for i in range(10)],"physical_edge_proxy","shoe_information_reliability",
        "shoe_phase_early","shoe_phase_middle","shoe_phase_late","estimated_hands_played_norm","remaining_decks_ratio","hands_elapsed_log_norm",
        "tie_ratio_all","tie_ratio_recent8","tie_ratio_recent16","bp_balance_strength","bp_entropy_recent12","outcome_entropy_recent12","outcome_entropy_recent24","sample_support_norm","composition_missing_indicator","shoe_progression_confidence",
    ]
    for w in (4,6,12,24,32): v.append(_tie_ratio(sequence,w)); names.append(f"tie_ratio_recent{w}")
    for w in (4,6,8,16,24,32): v.append(_binary_entropy(sequence,w)); names.append(f"bp_entropy_recent{w}")
    for w in (6,8,16,32): v.append(_outcome_entropy(sequence,w)); names.append(f"outcome_entropy_recent{w}")
    for w in (6,8,16,24,32): v.append(_balance(sequence,w)); names.append(f"bp_balance_recent{w}")
    extra = [pen*pen,math.sqrt(pen),rr*rr,math.sqrt(rr),_clip(1-pen/.18),_clip(1-abs(pen-.3)/.22),_clip(1-abs(pen-.62)/.24),_clip((pen-.72)/.22),_clip(len(sequence)/8.0),_clip(len(sequence)/16.0),_clip(len(sequence)/24.0),_clip(len(sequence)/48.0)]
    extra_names = ["penetration_squared","penetration_sqrt","remaining_squared","remaining_sqrt","shoe_phase_very_early","shoe_phase_early_mid","shoe_phase_mid_late","shoe_phase_very_late","sample_support_8","sample_support_16","sample_support_24","sample_support_48"]
    v.extend(extra); names.extend(extra_names)
    for w in (2,3,5,7,10,14,20,28,40,48,56,64): v.append(_tie_ratio(sequence,w)); names.append(f"tie_ratio_recent{w}")
    for w in (2,3,5,7,10,14,20,28,40,48,56,64): v.append(_binary_entropy(sequence,w)); names.append(f"bp_entropy_recent{w}")
    for w in (2,3,5,7,10,14,20,28,40,48,56,64): v.append(_outcome_entropy(sequence,w)); names.append(f"outcome_entropy_recent{w}")
    for w in (2,3,5,7,10,14,20,28,40,48,56,64): v.append(_balance(sequence,w)); names.append(f"bp_balance_recent{w}")
    v.extend([pen**3,rr**3,math.sqrt(math.sqrt(pen)),math.sqrt(math.sqrt(rr)),_clip(1-abs(pen-.125)/.125),_clip(1-abs(pen-.375)/.125),_clip(1-abs(pen-.625)/.125),_clip(1-abs(pen-.875)/.125),_clip(len(sequence)/4.0),_clip(len(sequence)/12.0),_clip(len(sequence)/20.0),_clip(len(sequence)/32.0),_clip(.5+(_tie_ratio(sequence,8)-_tie_ratio(sequence,32))/2),_clip(.5+(_binary_entropy(sequence,8)-_binary_entropy(sequence,32))/2),_clip(.5+(_balance(sequence,8)-_balance(sequence,32))/2),_clip(math.log1p(len(sequence))/math.log1p(128.0))])
    names.extend(["penetration_cubic","remaining_cubic","penetration_quarter_root","remaining_quarter_root","shoe_phase_q1","shoe_phase_q2","shoe_phase_q3","shoe_phase_q4","sample_support_4","sample_support_12","sample_support_20","sample_support_32","tie_short_long_delta","entropy_short_long_delta","balance_short_long_delta","maturity_log_norm"])
    if len(v) != SHOE_CONTEXT_DIM or len(names) != SHOE_CONTEXT_DIM:
        raise RuntimeError(f"shoe mismatch {len(v)}/{len(names)}")
    return v, names


def _build_road_vector(sequence: Sequence[str]) -> tuple[list[float], list[str], dict[str, Any]]:
    rs = _runs(sequence)
    current = rs[-1] if rs else ("",0)
    def prior(i: int) -> tuple[str,int]: return rs[-1-i] if len(rs)>i else ("",0)
    d8 = _derived_info(sequence,8)
    s4,s8,s12 = _run_stats(sequence,4),_run_stats(sequence,8),_run_stats(sequence,12)
    cand = _big_road_candidates(sequence)
    g = cand["geometry"]
    hs = _clip(1-_run_volatility(sequence,6)*.55-abs(_turn_rate(sequence,8)-_turn_rate(sequence,24))*.45)
    road: list[float] = []
    names: list[str] = []
    def push(name: str, value: Any) -> None:
        names.append(name); road.append(float(value) if math.isfinite(float(value)) else 0.0)
    push("current_side_banker",_side_sign(current[0])); push("current_run_norm",_clip(current[1]/12.0))
    for i in range(1,7): push(f"previous_run_{i}",_clip(prior(i)[1]/12.0))
    push("stage_cont_probability",cand.get("stageCont",.5)); push("stage_turn_probability",cand.get("stageTurn",.5)); push("context_cont_probability",cand.get("contextCont",.5)); push("context_turn_probability",cand.get("contextTurn",.5)); push("structure_stability",hs); push("run_volatility",_run_volatility(sequence,6)); push("run_trend",_run_trend(sequence,5))
    push("derived_big_eye",d8["be"]);push("derived_small",d8["sm"]);push("derived_cockroach",d8["cr"]);push("derived_consensus",d8["consensus"]);push("derived_support",d8["support"]);push("same2",_same_tail(sequence,2));push("same3",_same_tail(sequence,3));push("alt4",_alternating_tail(sequence,4));push("alt6",_alternating_tail(sequence,6))
    for w in (2,4,6,8,10,12,16,20,24,32,48,64): push(f"banker_bias_{w}",(_banker_ratio(sequence,w)-.5)*2)
    for w in (2,4,6,8,10,12,16,20,24,32,48,64): push(f"turn_rate_{w}",_turn_rate(sequence,w))
    for w in (4,8,16,24,32):
        d=_derived_info(sequence,w);push(f"be_{w}",d["be"]);push(f"sm_{w}",d["sm"]);push(f"cr_{w}",d["cr"])
    push("run_avg4",s4["avg"]);push("run_avg8",s8["avg"]);push("run_avg12",s12["avg"]);push("run_max8",s8["max"]);push("run_max12",s12["max"]);push("run_std8",s8["std"]);push("run_std12",s12["std"]);push("run_trend8",_run_trend(sequence,8));push("run_trend12",_run_trend(sequence,12))
    for w in (3,5,8,10): push(f"alternating_{w}",_alternating_tail(sequence,w))
    for w in (4,5,6,8,10): push(f"same_{w}",_same_tail(sequence,w))
    push("banker_delta_4_16",_banker_ratio(sequence,4)-_banker_ratio(sequence,16));push("banker_delta_8_32",_banker_ratio(sequence,8)-_banker_ratio(sequence,32));push("banker_delta_16_64",_banker_ratio(sequence,16)-_banker_ratio(sequence,64));push("turn_delta_4_16",_turn_rate(sequence,4)-_turn_rate(sequence,16));push("turn_delta_8_32",_turn_rate(sequence,8)-_turn_rate(sequence,32));push("turn_delta_16_64",_turn_rate(sequence,16)-_turn_rate(sequence,64))
    push("bigroad_candidate_B",cand.get("B",.5)-.5);push("bigroad_candidate_P",cand.get("P",.5)-.5);push("bigroad_candidate_gap_B",cand.get("B",.5)-cand.get("P",.5));push("bigroad_support",cand.get("support",0));push("target_fit_now",cand.get("targetFitNow",.5));push("target_fit_next",cand.get("targetFitNext",.5));push("pair_reliability",cand.get("pairReliability",0));push("pair_fit",cand.get("pairFit",.5))
    push("forward_directional",cand.get("forwardDirectional",0));push("survival_directional",cand.get("survivalDirectional",0));push("context_directional",cand.get("contextDirectional",0));push("two_step_directional",cand.get("twoStepDirectional",0));push("two_step_B",cand.get("twoB",.5)-.5);push("two_step_P",cand.get("twoP",.5)-.5)
    push("grid_current_row",_clip(g["currentRow"]/(BIG_ROAD_ROWS-1)));push("grid_current_col",_clip(g["currentViewCol"]/(BIG_ROAD_COLS-1)));push("grid_visible_occupancy",g["occupancy"]);push("grid_visible_banker_ratio",(g["bankerRatio"]-.5)*2);push("grid_top_occupancy",g["topOccupancy"]);push("grid_bottom_occupancy",g["bottomOccupancy"]);push("grid_tail_ratio",g["tailRatio"]);push("grid_current_in_tail",g["currentInTail"]);push("grid_current_at_bottom",g["currentAtBottom"]);push("grid_blocked_below",g["blockedBelow"]);push("grid_continue_moves_down",cand.get("continueMove",{}).get("down",0));push("grid_continue_moves_right",cand.get("continueMove",{}).get("right",0));push("grid_continue_row",_clip(cand.get("continueMove",{}).get("row",0)/(BIG_ROAD_ROWS-1)));push("grid_continue_col",_clip(cand.get("continueMove",{}).get("viewCol",0)/(BIG_ROAD_COLS-1)));push("grid_reverse_col",_clip(cand.get("reverseMove",{}).get("viewCol",0)/(BIG_ROAD_COLS-1)));push("grid_active_columns",g["activeColumns"]);push("grid_avg_column_fill",g["avgFill"]);push("grid_max_column_fill",g["maxFill"]);push("grid_column_fill_std",g["fillStd"]);push("grid_profile_regularity",g["profileRegularity"])
    while len(road)<ROAD_CONTEXT_DIM: push(f"reserved_{len(road)}",0.0)
    if len(road)!=ROAD_CONTEXT_DIM: raise RuntimeError(f"road mismatch {len(road)}")
    return road,names,cand


def _context256(sequence: Sequence[str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    shoe,shoe_names=_build_shoe_vector(sequence)
    road,road_names,cand=_build_road_vector(sequence)
    vector=np.nan_to_num(np.asarray(shoe+road,dtype=np.float64),nan=0.0,posinf=2.0,neginf=-1.0)
    if vector.size!=CONTEXT_DIM: raise RuntimeError(f"context mismatch {vector.size}")
    road_obj=cand["road"]
    metadata={
        "raw_round_count":len(sequence),"bp_round_count":len(_bp(sequence)),"tie_count":sum(x=="T" for x in sequence),
        "context_layout":"128_shoe_plus_128_road_256d","context_compatibility":"bbb_app256forward_v18_plus_continuation_v19",
        "formal_direction_source":"contextual_linucb","single_brain":True,"external_direction_votes_enabled":False,
        "big_road_rows":BIG_ROAD_ROWS,"big_road_cols":BIG_ROAD_COLS,"big_road_view_start_col":road_obj["viewStartCol"],"big_road_view_end_col":road_obj["viewEndCol"],"big_road_max_col":road_obj["maxCol"],
        "big_road_grid":road_obj["grid"],"road_candidates":_serializable_candidates(cand),
        "shoe_feature_values":[float(x) for x in shoe],"road_feature_values":[float(x) for x in road],
        "exact_card_input_ignored_for_web_panel_compatibility":True,"rank_ratio_source":"neutral_fallback_web_panel",
    }
    return vector,shoe_names+road_names,metadata


def _serializable_candidates(cand: Mapping[str, Any]) -> dict[str, Any]:
    keys=("B","P","support","currentSide","stageCont","stageTurn","stageSupport","contextCont","contextTurn","contextSupport","ownTarget","oppositeTarget","pairReliability","pairFit","continueGrid","reverseGrid","targetFitNow","targetFitNext","forwardDirectional","survivalDirectional","contextDirectional","twoStepDirectional","twoB","twoP")
    return {k:cand.get(k) for k in keys}


def _transition_rate(values: Sequence[str]) -> float:
    if len(values)<2: return 0.5
    return sum(values[i]!=values[i-1] for i in range(1,len(values)))/(len(values)-1)


def _persistence_shift(sequence: Sequence[str]) -> dict[str,float]:
    a=_bp(sequence)
    if len(a)<7: return {"value":0.5,"support":_clip(len(a)/10.0)}
    recent=a[-5:]; previous=a[max(0,len(a)-13):max(0,len(a)-5)]
    recent_turn=_transition_rate(recent); previous_turn=_transition_rate(previous)
    acceleration=(1-recent_turn)-(1-previous_turn)
    return {"value":_clip(0.5+acceleration*0.95),"support":_clip(min(len(recent)-1,len(previous)-1)/5.0),"recentTurn":recent_turn,"previousTurn":previous_turn}


def _empirical_exhaustion(completed: Sequence[Mapping[str,Any]],side:str,current_length:int)->dict[str,float]:
    items=[s for s in completed if s.get("side")==side][-16:]
    if not items: return {"value":0.5,"support":0.0,"tailShare":0.5,"continueShare":0.5}
    weight=ended=exceeded=0.0
    for i,s in enumerate(items):
        recency=0.95**(len(items)-1-i);weight+=recency
        if int(s.get("logicalLength",0))<=current_length: ended+=recency
        if int(s.get("logicalLength",0))>current_length: exceeded+=recency
    prior=.8;denom=weight+2*prior
    tail=(ended+prior)/denom;cont=(exceeded+prior)/denom
    return {"value":_clip(tail),"continueShare":_clip(cont),"support":_clip(weight/5),"tailShare":_clip(tail)}


def _continuation_signals(sequence:Sequence[str],base_prediction:Mapping[str,Any])->dict[str,Any]:
    rs=_runs(sequence);current=rs[-1] if rs else ("",0);current_side,current_length=current;sign=_side_sign(current_side)
    cand=base_prediction.get("candidates") or _big_road_candidates(sequence)
    road=_build_big_road(sequence);completed=_full_completed_streaks(road)
    stage_now=_stage_survival(completed,current_side,current_length);stage_next=_stage_survival(completed,current_side,current_length+1)
    shift=_persistence_shift(sequence);exhaustion=_empirical_exhaustion(completed,current_side,current_length)
    same_future=_branch_future_quality(sequence,current_side) if current_side else .5;opposite="P" if current_side=="B" else "B"
    reverse_future=_branch_future_quality(sequence,opposite) if current_side else .5
    branch_current_adv=_clip(.5+(same_future-reverse_future)*1.55);branch_reverse_adv=_clip(.5+(reverse_future-same_future)*1.55)
    two_step_current=_clip(.5+sign*float(cand.get("twoStepDirectional",0))*0.5);forward_current=_clip(.5+sign*float(cand.get("forwardDirectional",0))*0.5)
    context_cont=float(cand.get("contextCont",.5));context_turn=float(cand.get("contextTurn",.5))
    early_gate=_clip((4-current_length)/3.0)
    start_raw=_clip(.25*stage_now["cont"]+.20*context_cont+.18*branch_current_adv+.15*shift["value"]+.12*two_step_current+.10*forward_current)
    start_support=_clip(.30*stage_now["support"]+.22*float(cand.get("contextSupport",0))+.18*shift["support"]+.15*float(cand.get("support",0))+.15*_clip(len(completed)/8.0))
    start_signal=_clip(early_gate*start_support*_clip((start_raw-.50)/.30))
    survival_drop=_clip(.5+(stage_now["cont"]-stage_next["cont"])*1.8);maturity_gate=_clip((current_length-1)/3.0)
    own_target=float(cand.get("ownTarget",0) or 0);overshoot=_clip((current_length-own_target+.25)/max(2.0,own_target*.75)) if own_target else 0.0
    break_raw=_clip(.25*stage_now["turn"]+.19*context_turn+.18*survival_drop+.15*exhaustion["value"]+.13*branch_reverse_adv+.10*overshoot)
    break_support=_clip(.32*stage_now["support"]+.22*float(cand.get("contextSupport",0))+.20*exhaustion["support"]+.14*float(cand.get("support",0))+.12*_clip(len(completed)/8.0))
    break_signal=_clip(maturity_gate*break_support*_clip((break_raw-.50)/.30))
    net=_signed(start_signal-break_signal)
    return {"currentSide":current_side,"currentLength":current_length,"startSignal":start_signal,"breakSignal":break_signal,"startRaw":start_raw,"breakRaw":break_raw,"startSupport":start_support,"breakSupport":break_support,"stageNow":stage_now,"stageNext":stage_next,"survivalDrop":survival_drop,"shift":shift,"exhaustion":exhaustion,"sameFuture":same_future,"reverseFuture":reverse_future,"net":net,"directional":_signed(sign*net)}


def _frozen_prior(feature_names:Sequence[str])->tuple[dict[str,np.ndarray],dict[str,np.ndarray]]:
    A={arm:np.full(CONTEXT_DIM,LINUCB_RIDGE,dtype=np.float64) for arm in ARMS};b={arm:np.zeros(CONTEXT_DIM,dtype=np.float64) for arm in ARMS}
    index={name:i for i,name in enumerate(feature_names)}
    def set_directional(name:str,weight:float,precision:float=1.0)->None:
        if name not in index:return
        i=index[name];A["B"][i]=precision;A["P"][i]=precision;b["B"][i]=weight;b["P"][i]=-weight
    set_directional("banker_bias_8",.042,1.20);set_directional("banker_bias_16",.034,1.24);set_directional("banker_bias_32",.025,1.30)
    set_directional("banker_delta_4_16",.032,1.18);set_directional("banker_delta_8_32",.026,1.22);set_directional("banker_delta_16_64",.018,1.28)
    set_directional("bigroad_candidate_gap_B",.27,.98);set_directional("forward_directional",.36,.92);set_directional("survival_directional",.23,.98);set_directional("context_directional",.21,1.00);set_directional("two_step_directional",.18,1.02)
    set_directional("grid_visible_banker_ratio",.008,1.42);set_directional("current_side_banker",.004,1.48)
    return A,b


def _score_arm(arm:str,x:np.ndarray,A:Mapping[str,np.ndarray],b:Mapping[str,np.ndarray])->dict[str,float]:
    diag=np.maximum(1e-9,A[arm]);theta=b[arm]/diag
    mean=float(np.dot(x,theta));uncertainty=float(math.sqrt(max(0.0,float(np.sum((x*x)/diag)))))
    return {"mean":mean,"uncertainty":uncertainty,"score":mean+LINUCB_ALPHA*uncertainty,"effective_alpha":LINUCB_ALPHA,"raw_n":0.0,"effective_n":0.0}


def _deterministic_tie(sequence:Sequence[str])->str:
    token="FROZEN256_BIGROAD_6X15_FORWARD_V18|"+"".join(sequence);h=0
    for ch in token:h=(h*31+ord(ch))&0xFFFFFFFF
    return "B" if h%2 else "P"


def _base_choose(sequence:Sequence[str],vector:np.ndarray,feature_names:Sequence[str],metadata:Mapping[str,Any])->dict[str,Any]:
    A,b=_frozen_prior(feature_names);scores={arm:_score_arm(arm,vector,A,b) for arm in ARMS};gap=float(scores["B"]["score"]-scores["P"]["score"])
    direction=_deterministic_tie(sequence) if abs(gap)<=LINUCB_SCORE_TIE_EPSILON else ("B" if gap>0 else "P")
    raw_pb=1/(1+math.exp(-max(-8,min(8,gap/LINUCB_SCORE_TEMPERATURE))));p_b=_clip(raw_pb,PROBABILITY_MIN,PROBABILITY_MAX);p_p=1-p_b
    cand=_big_road_candidates(sequence);support=float(cand.get("support",0));forward=float(cand.get("forwardDirectional",0));two=float(cand.get("twoStepDirectional",0))
    regime="混合"
    if support>=.25:
        current_cont=bool(cand.get("currentSide") and direction==cand.get("currentSide"))
        if abs(float(cand.get("B",.5))-float(cand.get("P",.5)))>=.045 or abs(forward)>=.20: regime="大路延續" if current_cont else "大路反轉"
        else: regime="大路觀察"
    strength=_clip(.40+.32*support+.14*abs(forward)+.08*abs(two)+min(.16,abs(gap)*.66))
    return {"direction":direction,"gap":gap,"scores":scores,"probabilities":{"B":p_b,"P":p_p,"T":0.0},"confidence":p_b if direction=="B" else p_p,"regime":regime,"strength":strength,"candidates":cand,"metadata":dict(metadata)}


def _enhanced_choose(sequence:Sequence[str],vector:np.ndarray,feature_names:Sequence[str],metadata:Mapping[str,Any])->dict[str,Any]:
    base=_base_choose(sequence,vector,feature_names,metadata);sig=_continuation_signals(sequence,base);adjustment=_signed(sig["directional"])*.16;gap=base["gap"]+adjustment
    direction=base["direction"] if abs(gap)<=1e-9 else ("B" if gap>0 else "P")
    raw_pb=1/(1+math.exp(-max(-8,min(8,gap/LINUCB_SCORE_TEMPERATURE))));p_b=_clip(raw_pb,PROBABILITY_MIN,PROBABILITY_MAX);p_p=1-p_b
    label="前瞻平衡"
    if sig["startSignal"]>=.28 and sig["startSignal"]>sig["breakSignal"]+.08:label="延續前兆"
    elif sig["breakSignal"]>=.28 and sig["breakSignal"]>sig["startSignal"]+.08:label="延續衰竭"
    strength=_clip(base["strength"]*.78+.12*max(sig["startSupport"],sig["breakSupport"])+.10*max(sig["startSignal"],sig["breakSignal"]))
    return {**base,"direction":direction,"gap":gap,"probabilities":{"B":p_b,"P":p_p,"T":0.0},"confidence":p_b if direction=="B" else p_p,"regime":label,"strength":strength,"continuationSignals":sig,"v19":{"version":"V19_CONTINUATION_START_BREAK","baseGap":base["gap"],"adjustment":adjustment,"startSignal":sig["startSignal"],"breakSignal":sig["breakSignal"]}}


@dataclass(frozen=True)
class ContextSnapshot:
    vector: np.ndarray
    feature_names: tuple[str,...]
    metadata: dict[str,Any]


class ContextGenerator:
    def build(self, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None = None) -> ContextSnapshot:
        del shoe_context
        raw=_normalize_history(deepcopy(history));vector,names,metadata=_context256(raw)
        return ContextSnapshot(vector=vector,feature_names=tuple(names),metadata=metadata)


def _state_path()->Path:
    candidates=[];configured=str(os.getenv("LINUCB_STATE_FILE","") or "").strip()
    if configured:candidates.append(Path(configured).expanduser())
    candidates.extend([Path("/var/data/contextual_linucb_state.json"),Path(__file__).resolve().parent/"data"/"contextual_linucb_state.json",Path("/tmp/contextual_linucb_state.json")])
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True,exist_ok=True);probe=candidate.parent/f".linucb_write_{time.time_ns()}";probe.write_text("ok",encoding="utf-8");probe.unlink(missing_ok=True);return candidate
        except OSError:continue
    return Path("/tmp/contextual_linucb_state.json")

STATE_FILE=_state_path()


def _new_scope()->dict[str,Any]:
    now=int(time.time());return {"last_selected":"","selection_streak":0,"pending":{},"updates":0,"direct_predict_only":True,"no_bootstrap_on_start":True,"no_feedback_update":True,"no_ab_update":True,"no_decay":True,"created_at":now,"updated_at":now}


def _read_state()->dict[str,Any]:
    try:
        payload=json.loads(STATE_FILE.read_text(encoding="utf-8"));
        if not isinstance(payload,dict):raise ValueError
    except Exception:payload={}
    if payload.get("version")!=STATE_VERSION or payload.get("dim")!=CONTEXT_DIM:payload={}
    return {"version":STATE_VERSION,"dim":CONTEXT_DIM,"alpha":LINUCB_ALPHA,"ridge":LINUCB_RIDGE,"forgetting":1.0,"scopes":payload.get("scopes") if isinstance(payload.get("scopes"),dict) else {}}


def _write_state(payload:Mapping[str,Any])->None:
    temporary=STATE_FILE.with_suffix(STATE_FILE.suffix+".tmp");temporary.write_text(json.dumps(dict(payload),ensure_ascii=False),encoding="utf-8");temporary.replace(STATE_FILE)


def make_scope_key(*,user_id:str="",venue:str="",room:str="",shoe_id:str="")->str:
    raw="|".join((str(user_id or "").strip(),str(venue or "").upper().strip(),str(room or "").strip(),str(shoe_id or "").strip()));return sha256((raw or "GLOBAL").encode()).hexdigest()[:24]


def _history_fingerprint(history:Sequence[str])->str:return sha256("".join(history).encode()).hexdigest()[:24]


try:
    _, _FEATURE_NAMES, _ = _context256([])
except Exception:
    _FEATURE_NAMES = [f"feature_{i}" for i in range(CONTEXT_DIM)]
CONTEXT_FEATURE_NAMES = tuple(_FEATURE_NAMES)


class ContextualLinUCB:
    def __init__(self,alpha:float=LINUCB_ALPHA):
        self.alpha=LINUCB_ALPHA;self.generator=ContextGenerator()

    @staticmethod
    def _remember_selection(scope:dict[str,Any],direction:str)->int:
        previous=str(scope.get("last_selected") or "").upper().strip();streak=int(scope.get("selection_streak",0) or 0)+1 if previous==direction else 1
        scope.update({"last_selected":direction,"selection_streak":streak,"updated_at":int(time.time())});return streak

    def predict(self,*,history:Iterable[Any]|str|None,shoe_context:Mapping[str,Any]|None,scope_key:str)->dict[str,Any]:
        raw_history=_normalize_history(deepcopy(history));snapshot=self.generator.build(raw_history,deepcopy(dict(shoe_context or {})));x=np.nan_to_num(snapshot.vector.copy(),nan=0.0,posinf=2.0,neginf=-1.0)
        chosen=_enhanced_choose(raw_history,x,snapshot.feature_names,snapshot.metadata);direction=chosen["direction"];probabilities=chosen["probabilities"];confidence=chosen["confidence"]
        fingerprint=_history_fingerprint(raw_history)
        with _LOCK:
            root=_read_state();scope=deepcopy(dict(root["scopes"].get(scope_key) or _new_scope()));streak=self._remember_selection(scope,direction)
            scope.update({"pending":{},"frozen_direct_mode":True,"direct_predict_only":True,"no_bootstrap_on_start":True,"no_feedback_update":True,"no_ab_update":True,"no_decay":True});root["scopes"][scope_key]=scope;_write_state(root)
        bootstrap={"applied":False,"reason":"bbb_v19_parity_no_bootstrap","bootstrap_rounds":0,"source_rounds":len(raw_history)}
        feedback={"updated":False,"reason":"bbb_v19_parity_no_feedback_update","diagnostic_only":False,"formal_model":"contextual_linucb","a_b_frozen_without_bootstrap":True,"no_settlement":True,"no_decay":True}
        metadata=deepcopy(snapshot.metadata);metadata.update({"selection_streak":streak,"linucb_direction_weight":1.0,"prediction_mode":"bbb_v18_v19_frozen_256d_6x15_parity","automatic_feedback_update_enabled":False,"a_b_frozen_without_bootstrap":True,"no_bootstrap_on_start":True,"no_replay":True,"no_decay":True,"continuation_signals":deepcopy(chosen["continuationSignals"]),"v19":deepcopy(chosen["v19"]),"regime":chosen["regime"],"structure_strength":chosen["strength"]})
        scores=deepcopy(chosen["scores"])
        return {
            "model":"contextual_linucb_single_brain","version":STATE_VERSION,"legacy_state_version":STATE_VERSION,"direction":direction,"selected_arm":direction,"arm_index":1 if direction=="B" else 0,
            "probabilities":probabilities,"selected_win_probability":confidence,"confidence":confidence,"context_vector":[float(v) for v in snapshot.vector],"model_context_vector":[float(v) for v in x],
            "context_feature_names":list(snapshot.feature_names),"context_dim":CONTEXT_DIM,"context_metadata":metadata,
            "road_prior":{"diagnostic_only":True,"direction_weight":0.0,"banker_probability":0.5,"player_probability":0.5},"road_prior_probability":{"B":0.5,"P":0.5},"road_forecaster":{"available":False,"diagnostic_only":True,"formal_direction_weight":0.0},
            "features_used":dict(zip(snapshot.feature_names,[float(v) for v in snapshot.vector])),"effective_support":0.0,"uncertainty":scores[direction]["uncertainty"],"linucb_probability_correction":0.0,"linucb_direction_weight":1.0,"learning_reliability":0.0,
            "scores":scores,"score_gap":chosen["gap"],"base_score_gap":chosen["v19"]["baseGap"],"v19_adjustment":chosen["v19"]["adjustment"],"score_semantics":"bbb_v18_frozen_linucb_gap_plus_v19_continuation_forewarning","alpha":LINUCB_ALPHA,"ridge":LINUCB_RIDGE,
            "forgetting":1.0,"feedback_update":feedback,"bootstrap_update":bootstrap,"panel_bootstrap_applied":False,"scope_key":scope_key,"arms":list(ARMS),"selection_reason":"bbb_v18_v19_frozen_argmax","selection_streak":streak,
            "effective_arm_samples":{"B":0.0,"P":0.0},"history_round_count":len(raw_history),"bp_history_round_count":len(_bp(raw_history)),"history_fingerprint":fingerprint,"short_shoe_target_rounds":"50-70",
            "formal_context_source":"bbb_v18_v19_256d_128shoe_128road_6x15_frozen_context","formal_direction_source":"contextual_linucb","road_context_direction_weight":0.0,"card_composition_direction_weight":0.0,
            "probability_semantics":"bounded_logistic_mapping_of_bbb_v19_adjusted_score_gap","cold_start_uses_road_prior":False,"shoe_context_used_for_formal_direction":False,"shoe_context_used_as_features":False,"history_estimated_shoe_features_used":True,
            "shoe_context_independent_vote":False,"external_road_vote_enabled":False,"anti_echo_external_penalty":False,"panel_compatible":True,"frozen_direct_mode":True,"direct_predict_only":True,"no_bootstrap_on_start":True,
            "automatic_feedback_update_enabled":False,"no_replay":True,"no_previous_settlement":True,"no_ab_update":True,"no_decay":True,"regime":chosen["regime"],"structure_strength":chosen["strength"],"continuation_signals":deepcopy(chosen["continuationSignals"]),"v19":deepcopy(chosen["v19"]),
            "big_road":{"rows":BIG_ROAD_ROWS,"cols":BIG_ROAD_COLS,"viewStartCol":chosen["candidates"]["road"]["viewStartCol"],"viewEndCol":chosen["candidates"]["road"]["viewEndCol"],"maxCol":chosen["candidates"]["road"]["maxCol"],"grid":chosen["candidates"]["road"]["grid"]},
            "anti_lock":{"enabled":False,"method":"none_external_feedback_only","tie_is_non_directional":True,"old_state_reused":False},
        }

    def update(self,*,scope_key:str,action:str,context_vector:Sequence[float],actual_outcome:str,clear_pending:bool=True)->dict[str,Any]:
        del scope_key,action,context_vector,actual_outcome,clear_pending
        return {"updated":False,"reason":"bbb_v19_parity_frozen_no_ab_update","explicit_update_only":True,"a_b_frozen":True,"decay_applied":False}


_DEFAULT_BANDIT=ContextualLinUCB()

def predict_bandit(*,history:Iterable[Any]|str|None,shoe_context:Mapping[str,Any]|None,scope_key:str)->dict[str,Any]:
    return _DEFAULT_BANDIT.predict(history=deepcopy(history),shoe_context=deepcopy(dict(shoe_context or {})),scope_key=str(scope_key or ""))

def update_bandit(*,scope_key:str,action:str,context_vector:Sequence[float],actual_outcome:str,clear_pending:bool=True)->dict[str,Any]:
    return _DEFAULT_BANDIT.update(scope_key=str(scope_key or ""),action=action,context_vector=deepcopy(list(context_vector)),actual_outcome=actual_outcome,clear_pending=clear_pending)

__all__=["ARMS","CONTEXT_DIM","CONTEXT_FEATURE_NAMES","ContextGenerator","ContextualLinUCB","ESTIMATED_CARDS_PER_ROUND","SHOE_DECKS","LINUCB_ALPHA","LINUCB_ARM_ALPHA_MAX_SCALE","LINUCB_FORGETTING","LINUCB_RIDGE","LINUCB_SCORE_TIE_EPSILON","LINUCB_UPDATE_WEIGHT","PROBABILITY_MIN","PROBABILITY_MAX","ROAD_PRIOR_PROBABILITY_SPAN","ROAD_PRIOR_SCORE_WEIGHT","STATE_VERSION","make_scope_key","predict_bandit","update_bandit"]