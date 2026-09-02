"""Frozen-direct 256D Single-Brain Contextual LinUCB core.

Parity target: zliu53468-ai/BBB app256.js.
- 256D context = 128D shoe/progression + 128D road/structure.
- B/P two-arm LinUCB.
- No bootstrap, walk-forward, replay, previous-prediction settlement, A/b update, or decay.
- Only selection metadata (last_selected / selection_streak) changes after predict(), matching the web panel.

OCR, screenshot parsing, public predictor fields, LINE/LIFF UI and money management live outside this module.
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

from shoe_constants import AVERAGE_CARDS_PER_HAND, SHOE_DECKS

ARMS = ("P", "B")
SHOE_CONTEXT_DIM = 128
ROAD_CONTEXT_DIM = 128
CONTEXT_DIM = 256
LINUCB_ALPHA = max(0.0, float(os.getenv("LINUCB_ALPHA", "0.5") or "0.5"))
LINUCB_RIDGE = max(1e-6, float(os.getenv("LINUCB_RIDGE", "1.0") or "1.0"))
LINUCB_UPDATE_WEIGHT = 0.0
LINUCB_FORGETTING = 1.0
LINUCB_ARM_ALPHA_MAX_SCALE = max(1.0, min(2.5, float(os.getenv("LINUCB_ARM_ALPHA_MAX_SCALE", "1.60") or "1.60")))
LINUCB_SCORE_TIE_EPSILON = max(1e-12, float(os.getenv("LINUCB_SCORE_TIE_EPSILON", "0.000001") or "0.000001"))
LINUCB_SCORE_TEMPERATURE = max(0.25, min(10.0, float(os.getenv("LINUCB_SCORE_TEMPERATURE", "2.0") or "2.0")))
ROAD_PRIOR_SCORE_WEIGHT = 0.0
ROAD_PRIOR_PROBABILITY_SPAN = 0.0
LINUCB_PROBABILITY_CORRECTION_SPAN = 0.0
PROBABILITY_MIN = 0.42
PROBABILITY_MAX = 0.58
STATE_VERSION = "LINUCB-2ARM-SINGLE-BRAIN-128SHOE-128ROAD-256D-BBB-WEB-PARITY-V11"
ESTIMATED_CARDS_PER_ROUND = AVERAGE_CARDS_PER_HAND
_LOCK = RLock()


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


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


def _balance_strength(sequence: Sequence[str], window: int) -> float:
    return _clip(1.0 - abs(_banker_ratio(sequence, window) - 0.5) * 2.0)


def _turn_rate(sequence: Sequence[str], window: int) -> float:
    values = _bp(sequence)[-max(2, int(window)):]
    if len(values) < 2:
        return 0.5
    turns = sum(values[i] != values[i - 1] for i in range(1, len(values)))
    return float(turns / (len(values) - 1))


def _tie_ratio(sequence: Sequence[str], window: int = 0) -> float:
    values = list(sequence[-window:]) if window else list(sequence)
    return float(sum(x == "T" for x in values) / len(values)) if values else 0.0


def _entropy(sequence: Sequence[str], window: int = 12) -> float:
    values = list(sequence[-window:])
    if not values:
        return 1.0
    e = 0.0
    for outcome in ("B", "P", "T"):
        p = sum(x == outcome for x in values) / len(values)
        if p > 0:
            e -= p * math.log2(p)
    return _clip(e / math.log2(3.0))


def _binary_entropy(sequence: Sequence[str], window: int = 12) -> float:
    values = _bp(sequence)[-window:]
    if not values:
        return 1.0
    p = sum(x == "B" for x in values) / len(values)
    q = 1.0 - p
    e = 0.0
    if p > 0:
        e -= p * math.log2(p)
    if q > 0:
        e -= q * math.log2(q)
    return _clip(e)


def _derived_mark(heights: list[int], column: int, row: int, new_column: bool, offset: int) -> str:
    if new_column:
        if column < offset + 1:
            return ""
        return "R" if heights[column - 1] == heights[column - 1 - offset] else "U"
    if column < offset:
        return ""
    ref_height = heights[column - offset]
    return "R" if (ref_height >= row) == (ref_height >= row - 1) else "U"


def _build_derived_roads(sequence: Sequence[str]) -> dict[str, list[str]]:
    values = _bp(sequence)
    sides: list[str] = []
    heights: list[int] = []
    out = {"big_eye": [], "small_road": [], "cockroach_road": []}
    offsets = {"big_eye": 1, "small_road": 2, "cockroach_road": 3}
    for side in values:
        new_column = not sides or side != sides[-1]
        if new_column:
            sides.append(side)
            heights.append(1)
        else:
            heights[-1] += 1
        column = len(heights) - 1
        row = heights[column]
        for name, offset in offsets.items():
            mark = _derived_mark(heights, column, row, new_column, offset)
            if mark:
                out[name].append(mark)
    return out


def _regularity(values: Sequence[str], window: int = 8) -> tuple[float, int]:
    marks = [x for x in list(values[-window:]) if x in {"R", "U"}]
    return (float(sum(x == "R" for x in marks) / len(marks)), len(marks)) if marks else (0.5, 0)


def _length_bucket(n: int) -> str:
    n = max(1, int(n))
    return str(n) if n <= 5 else "6+"


def _hazard_contexts(side: str, current: int, previous: Sequence[int]) -> list[tuple[str, str]]:
    previous_height = previous[-1] if previous else 0
    deltas = ["UP" if previous[i] > previous[i - 1] else "DOWN" if previous[i] < previous[i - 1] else "EQUAL" for i in range(1, len(previous))]
    d1 = deltas[-1] if deltas else "NA"
    d2 = deltas[-2] if len(deltas) > 1 else "NA"
    cur = _length_bucket(current)
    prev = _length_bucket(previous_height) if previous_height else "0"
    return [
        ("full", f"HZF|side={side or 'NA'}|cur={cur}|prev={prev}|d1={d1}|d2={d2}"),
        ("structure", f"HZS|cur={cur}|prev={prev}|d1={d1}|d2={d2}"),
        ("shape", f"HZP|cur={cur}|prev={prev}|d1={d1}"),
        ("length", f"HZL|cur={cur}"),
        ("global", "HZG|GLOBAL"),
    ]


def _hazard_table(run_values: Sequence[tuple[str, int]]) -> dict[str, dict[str, float]]:
    completed = list(run_values[:-1])
    heights = [x[1] for x in completed]
    table: dict[str, dict[str, float]] = {}
    for index, (side, final_length) in enumerate(completed):
        previous = heights[:index]
        for at in range(1, max(1, final_length) + 1):
            event = "CONTINUE" if at < final_length else "TURN"
            for _, key in _hazard_contexts(side, at, previous):
                table.setdefault(key, {"CONTINUE": 0.0, "TURN": 0.0})[event] += 1.0
    return table


def _hazard_posterior(counts: Mapping[str, Any]) -> dict[str, float]:
    continued = float(counts.get("CONTINUE", 0.0) or 0.0)
    turned = float(counts.get("TURN", 0.0) or 0.0)
    denominator = continued + turned + 6.0
    return {"CONTINUE": (continued + 3.0) / denominator, "TURN": (turned + 3.0) / denominator}


def _hazard(sequence: Sequence[str]) -> float:
    run_values = _runs(sequence)
    if not run_values:
        return 0.5
    side, current = run_values[-1]
    heights = [x[1] for x in run_values[:-1]]
    table = _hazard_table(run_values)
    contexts = _hazard_contexts(side, current, heights)
    probabilities = {"CONTINUE": 0.5, "TURN": 0.5}
    penalty, found = 1.0, False
    for index, (_, key) in enumerate(contexts):
        counts = table.get(key, {"CONTINUE": 0.0, "TURN": 0.0})
        if counts["CONTINUE"] + counts["TURN"] >= 4:
            probabilities = _hazard_posterior(counts)
            found = True
            break
        if index < len(contexts) - 1:
            penalty *= 0.75
    if not found:
        counts = table.get("HZG|GLOBAL", {"CONTINUE": 0.0, "TURN": 0.0})
        if counts["CONTINUE"] + counts["TURN"] > 0:
            probabilities = _hazard_posterior(counts)
        else:
            penalty = 0.0
    cont = (1.0 - penalty) * 0.5 + penalty * probabilities["CONTINUE"]
    return _clip(1.0 - cont)


def _run_volatility(sequence: Sequence[str], window: int = 6) -> float:
    heights = [x[1] for x in _runs(sequence)[-window:]]
    if len(heights) < 2:
        return 0.25
    delta = sum(abs(heights[i] - heights[i - 1]) for i in range(1, len(heights)))
    return _clip(delta / (len(heights) - 1) / 3.0)


def _run_height_trend(sequence: Sequence[str], window: int = 5) -> float:
    heights = [x[1] for x in _runs(sequence)[-window:]]
    if len(heights) < 2:
        return 0.5
    slope = (heights[-1] - heights[0]) / (len(heights) - 1)
    return _clip(0.5 + slope / 6.0)


def _hsmm_stable(sequence: Sequence[str]) -> float:
    alternation = _turn_rate(sequence, 10)
    run_values = _runs(sequence)
    current = run_values[-1][1] if run_values else 0
    r = _clip(current / 6.0)
    e = _entropy(sequence, 12)
    v = _run_volatility(sequence)
    persistent = math.exp(-((alternation-.25)/.24)**2-((r-.70)/.28)**2-((e-.62)/.24)**2-((v-.26)/.24)**2)
    alternating = math.exp(-((alternation-.84)/.18)**2-((r-.18)/.20)**2-((e-.70)/.23)**2-((v-.30)/.24)**2)
    transition = math.exp(-((alternation-.52)/.28)**2-((r-.34)/.26)**2-((e-.82)/.18)**2-((v-.72)/.23)**2)
    noise = math.exp(-((alternation-.55)/.30)**2-((r-.27)/.24)**2-((e-.94)/.11)**2-((v-.55)/.28)**2)
    weights = (0.25*persistent, 0.25*alternating, 0.20*transition, 0.30*noise)
    total = sum(weights) or 1.0
    return _clip((weights[0] + weights[1]) / total)


def _alternating_tail(sequence: Sequence[str], window: int) -> float:
    values = _bp(sequence)[-window:]
    if len(values) < window:
        return 0.5
    return 1.0 if all(values[i] != values[i - 1] for i in range(1, len(values))) else 0.0


def _same_tail(sequence: Sequence[str], window: int) -> float:
    values = _bp(sequence)[-window:]
    if len(values) < window:
        return 0.5
    return 1.0 if all(x == values[0] for x in values) else 0.0


def _run_stats(sequence: Sequence[str], window: int) -> dict[str, float]:
    values = [x[1] for x in _runs(sequence)[-window:]]
    mean = sum(values) / len(values) if values else 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values) if values else 0.0
    return {
        "avg": _clip(mean / 8.0),
        "max": _clip((max(values) if values else 0.0) / 12.0),
        "std": _clip(math.sqrt(variance) / 6.0),
    }


_SHOE_NAMES: list[str] = []
_ROAD_NAMES: list[str] = []
def _sn(name: str) -> None: _SHOE_NAMES.append(name)
def _rn(name: str) -> None: _ROAD_NAMES.append(name)

for _name in (
    "remaining_cards_ratio","penetration_ratio","estimated_hands_remaining_norm","shoe_maturity_ratio",
    "rank_A_relative_ratio","rank_2_relative_ratio","rank_3_relative_ratio","rank_4_relative_ratio",
    "rank_5_relative_ratio","rank_6_relative_ratio","rank_7_relative_ratio","rank_8_relative_ratio",
    "rank_9_relative_ratio","rank_10JQK_relative_ratio","physical_edge_proxy","shoe_information_reliability",
    "shoe_phase_early","shoe_phase_middle","shoe_phase_late","estimated_hands_played_norm",
    "remaining_decks_ratio","hands_elapsed_log_norm","tie_ratio_all","tie_ratio_recent8",
    "tie_ratio_recent16","bp_balance_strength","bp_entropy_recent12","outcome_entropy_recent12",
    "outcome_entropy_recent24","sample_support_norm","composition_missing_indicator","shoe_progression_confidence",
): _sn(_name)
for _w in (4,6,12,24,32): _sn(f"tie_ratio_recent{_w}")
for _w in (4,6,8,16,24,32): _sn(f"bp_entropy_recent{_w}")
for _w in (6,8,16,32): _sn(f"outcome_entropy_recent{_w}")
for _w in (6,8,16,24,32): _sn(f"bp_balance_recent{_w}")
for _name in (
    "penetration_squared","penetration_sqrt","remaining_squared","remaining_sqrt",
    "shoe_phase_very_early","shoe_phase_early_mid","shoe_phase_mid_late","shoe_phase_very_late",
    "sample_support_8","sample_support_16","sample_support_24","sample_support_48",
): _sn(_name)
for _w in (2,3,5,7,10,14,20,28,40,48,56,64): _sn(f"tie_ratio_recent{_w}")
for _w in (2,3,5,7,10,14,20,28,40,48,56,64): _sn(f"bp_entropy_recent{_w}")
for _w in (2,3,5,7,10,14,20,28,40,48,56,64): _sn(f"outcome_entropy_recent{_w}")
for _w in (2,3,5,7,10,14,20,28,40,48,56,64): _sn(f"bp_balance_recent{_w}")
for _name in (
    "penetration_cubic","remaining_cubic","penetration_quarter_root","remaining_quarter_root",
    "shoe_phase_q1","shoe_phase_q2","shoe_phase_q3","shoe_phase_q4",
    "sample_support_4","sample_support_12","sample_support_20","sample_support_32",
    "tie_short_long_delta","entropy_short_long_delta","balance_short_long_delta","maturity_log_norm",
): _sn(_name)

for _name in (
    "current_side_banker_binary","current_run_length_norm","previous_run_length_norm","previous2_run_length_norm",
    "recent5_banker_ratio","recent8_banker_ratio","recent12_banker_ratio","recent5_turn_rate",
    "recent8_turn_rate","recent12_turn_rate","run_length_hazard_rate","hsmm_stable_probability",
    "big_eye_regularity","small_road_regularity","cockroach_road_regularity","derived_road_consensus",
    "current_side_player_binary","previous3_run_length_norm","recent3_banker_ratio","recent20_banker_ratio",
    "recent3_turn_rate","recent20_turn_rate","run_continue_probability","recent8_outcome_entropy",
    "recent20_outcome_entropy","recent6_run_volatility","recent5_run_height_trend","big_eye_support_norm",
    "small_road_support_norm","cockroach_road_support_norm","last2_same_side","last3_same_side",
): _rn(_name)
for _w in (2,4,6,10,16,24,32,48): _rn(f"recent{_w}_banker_ratio")
for _w in (2,4,6,10,16,24,32,48): _rn(f"recent{_w}_turn_rate")
for _name in (
    "big_eye_regularity_w4","small_road_regularity_w4","cockroach_road_regularity_w4",
    "big_eye_regularity_w16","small_road_regularity_w16","cockroach_road_regularity_w16",
    "previous4_run_length_norm","avg_run_last4_norm","avg_run_last8_norm","max_run_last8_norm",
    "run_std_last8_norm","run_delta_last_norm","alternating_last4","alternating_last6","last4_same_side","last5_same_side",
): _rn(_name)
for _w in (7,9,11,14,18,22,28,36,40,56,64,72): _rn(f"recent{_w}_banker_ratio")
for _w in (7,9,11,14,18,22,28,36,40,56,64,72): _rn(f"recent{_w}_turn_rate")
for _w in (6,12,24,32):
    for _kind in ("big_eye","small_road","cockroach_road"):
        _rn(f"{_kind}_regularity_w{_w}")
for _name in (
    "previous5_run_length_norm","previous6_run_length_norm","avg_run_last6_norm","avg_run_last12_norm",
    "max_run_last12_norm","run_std_last12_norm","run_height_trend_w8","run_height_trend_w12",
    "alternating_last3","alternating_last5","alternating_last8","alternating_last10",
    "last6_same_side","last7_same_side","last8_same_side","last10_same_side",
    "banker_ratio_delta_4_16","banker_ratio_delta_8_32","banker_ratio_delta_16_64",
    "turn_rate_delta_4_16","turn_rate_delta_8_32","turn_rate_delta_16_64",
    "hazard_squared","continue_squared","hsmm_stable_squared","derived_consensus_squared",
    "derived_support_mean","derived_regularity_mean",
): _rn(_name)

if len(_SHOE_NAMES) != SHOE_CONTEXT_DIM or len(_ROAD_NAMES) != ROAD_CONTEXT_DIM:
    raise RuntimeError(f"256D feature-name mismatch: {len(_SHOE_NAMES)}/{len(_ROAD_NAMES)}")
CONTEXT_FEATURE_NAMES = tuple(_SHOE_NAMES + _ROAD_NAMES)


def _context256(sequence: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    total_cards = float(52 * SHOE_DECKS)
    used = min(total_cards, len(sequence) * float(AVERAGE_CARDS_PER_HAND))
    remaining = max(0.0, total_cards - used)
    rr = _clip(remaining / total_cards)
    penetration = _clip(1.0 - rr)
    maturity = _clip(len(sequence) / 70.0)
    capacity = total_cards / float(AVERAGE_CARDS_PER_HAND)
    hands_played = _clip(len(sequence) / capacity)
    all_banker = _banker_ratio(sequence, max(1, len(_bp(sequence))))

    shoe = [
        rr, penetration, rr, maturity,
        *([1.0] * 10), 0.0, 0.0,
        _clip(1.0 - penetration / .35), _clip(1.0 - abs(penetration - .5) / .35), _clip((penetration - .55) / .35), hands_played,
        rr, _clip(math.log1p(len(sequence)) / math.log1p(capacity)),
        _tie_ratio(sequence), _tie_ratio(sequence, 8), _tie_ratio(sequence, 16), _clip(1.0 - abs(all_banker - .5) * 2.0),
        _binary_entropy(sequence, 12), _entropy(sequence, 12), _entropy(sequence, 24),
        _clip(len(sequence) / 32.0), 1.0, _clip(math.sqrt(len(sequence)) / math.sqrt(capacity)),
    ]
    for w in (4,6,12,24,32): shoe.append(_tie_ratio(sequence, w))
    for w in (4,6,8,16,24,32): shoe.append(_binary_entropy(sequence, w))
    for w in (6,8,16,32): shoe.append(_entropy(sequence, w))
    for w in (6,8,16,24,32): shoe.append(_balance_strength(sequence, w))
    shoe.extend([
        penetration**2, math.sqrt(penetration), rr**2, math.sqrt(rr),
        _clip(1.0 - penetration/.18), _clip(1.0 - abs(penetration-.3)/.22),
        _clip(1.0 - abs(penetration-.62)/.24), _clip((penetration-.72)/.22),
        _clip(len(sequence)/8.0), _clip(len(sequence)/16.0), _clip(len(sequence)/24.0), _clip(len(sequence)/48.0),
    ])
    for w in (2,3,5,7,10,14,20,28,40,48,56,64): shoe.append(_tie_ratio(sequence, w))
    for w in (2,3,5,7,10,14,20,28,40,48,56,64): shoe.append(_binary_entropy(sequence, w))
    for w in (2,3,5,7,10,14,20,28,40,48,56,64): shoe.append(_entropy(sequence, w))
    for w in (2,3,5,7,10,14,20,28,40,48,56,64): shoe.append(_balance_strength(sequence, w))
    shoe.extend([
        penetration**3, rr**3, math.sqrt(math.sqrt(penetration)), math.sqrt(math.sqrt(rr)),
        _clip(1.0 - abs(penetration-.125)/.125), _clip(1.0 - abs(penetration-.375)/.125),
        _clip(1.0 - abs(penetration-.625)/.125), _clip(1.0 - abs(penetration-.875)/.125),
        _clip(len(sequence)/4.0), _clip(len(sequence)/12.0), _clip(len(sequence)/20.0), _clip(len(sequence)/32.0),
        _clip(.5 + (_tie_ratio(sequence,8)-_tie_ratio(sequence,32))/2.0),
        _clip(.5 + (_binary_entropy(sequence,8)-_binary_entropy(sequence,32))/2.0),
        _clip(.5 + (_balance_strength(sequence,8)-_balance_strength(sequence,32))/2.0),
        _clip(math.log1p(len(sequence))/math.log1p(128.0)),
    ])

    run_values = _runs(sequence)
    def prior_run(offset: int) -> tuple[str, int]:
        return run_values[-1-offset] if len(run_values) > offset else ("", 0)
    cur, p1, p2, p3, p4, p5, p6 = (prior_run(i) for i in range(7))
    side_b = 1.0 if cur[0] == "B" else 0.0 if cur[0] == "P" else 0.5
    side_p = 1.0 if cur[0] == "P" else 0.0 if cur[0] == "B" else 0.5
    derived = _build_derived_roads(sequence)
    def R(window: int) -> dict[str, float]:
        be, bn = _regularity(derived["big_eye"], window)
        sm, sn = _regularity(derived["small_road"], window)
        cr, cn = _regularity(derived["cockroach_road"], window)
        return {"be":be,"sm":sm,"cr":cr,"bn":bn,"sn":sn,"cn":cn}
    r4,r6,r8,r12,r16,r24,r32 = (R(w) for w in (4,6,8,12,16,24,32))
    reg_mean = (r8["be"] + r8["sm"] + r8["cr"]) / 3.0
    consensus = _clip(1.0 - (abs(r8["be"]-reg_mean)+abs(r8["sm"]-reg_mean)+abs(r8["cr"]-reg_mean))/1.5)
    hz = _hazard(sequence)
    bp_values = _bp(sequence)
    def same(n: int) -> float:
        if len(bp_values) < n:
            return 0.5
        tail = bp_values[-n:]
        return 1.0 if all(v == tail[-1] for v in tail) else 0.0
    s4,s6,s8,s12 = (_run_stats(sequence,w) for w in (4,6,8,12))
    run_delta = _clip(.5 + (cur[1]-p1[1])/12.0) if len(run_values)>1 else .5

    road = [
        side_b,_clip(cur[1]/8.0),_clip(p1[1]/8.0),_clip(p2[1]/8.0),
        _banker_ratio(sequence,5),_banker_ratio(sequence,8),_banker_ratio(sequence,12),
        _turn_rate(sequence,5),_turn_rate(sequence,8),_turn_rate(sequence,12),
        hz,_hsmm_stable(sequence),r8["be"],r8["sm"],r8["cr"],consensus,
        side_p,_clip(p3[1]/8.0),_banker_ratio(sequence,3),_banker_ratio(sequence,20),
        _turn_rate(sequence,3),_turn_rate(sequence,20),_clip(1.0-hz),
        _entropy(sequence,8),_entropy(sequence,20),_run_volatility(sequence),_run_height_trend(sequence,5),
        _clip(r8["bn"]/8.0),_clip(r8["sn"]/8.0),_clip(r8["cn"]/8.0),same(2),same(3),
    ]
    for w in (2,4,6,10,16,24,32,48): road.append(_banker_ratio(sequence,w))
    for w in (2,4,6,10,16,24,32,48): road.append(_turn_rate(sequence,w))
    road.extend([
        r4["be"],r4["sm"],r4["cr"],r16["be"],r16["sm"],r16["cr"],
        _clip(p4[1]/8.0),s4["avg"],s8["avg"],s8["max"],s8["std"],run_delta,
        _alternating_tail(sequence,4),_alternating_tail(sequence,6),same(4),same(5),
    ])
    for w in (7,9,11,14,18,22,28,36,40,56,64,72): road.append(_banker_ratio(sequence,w))
    for w in (7,9,11,14,18,22,28,36,40,56,64,72): road.append(_turn_rate(sequence,w))
    for item in (r6,r12,r24,r32): road.extend([item["be"],item["sm"],item["cr"]])
    road.extend([
        _clip(p5[1]/8.0),_clip(p6[1]/8.0),s6["avg"],s12["avg"],s12["max"],s12["std"],
        _run_height_trend(sequence,8),_run_height_trend(sequence,12),
        _alternating_tail(sequence,3),_alternating_tail(sequence,5),_alternating_tail(sequence,8),_alternating_tail(sequence,10),
        same(6),same(7),same(8),same(10),
        _clip(.5+(_banker_ratio(sequence,4)-_banker_ratio(sequence,16))/2.0),
        _clip(.5+(_banker_ratio(sequence,8)-_banker_ratio(sequence,32))/2.0),
        _clip(.5+(_banker_ratio(sequence,16)-_banker_ratio(sequence,64))/2.0),
        _clip(.5+(_turn_rate(sequence,4)-_turn_rate(sequence,16))/2.0),
        _clip(.5+(_turn_rate(sequence,8)-_turn_rate(sequence,32))/2.0),
        _clip(.5+(_turn_rate(sequence,16)-_turn_rate(sequence,64))/2.0),
        hz**2,(1.0-hz)**2,_hsmm_stable(sequence)**2,consensus**2,
        _clip((r8["bn"]+r8["sn"]+r8["cn"])/24.0),_clip((r8["be"]+r8["sm"]+r8["cr"])/3.0),
    ])
    if len(shoe) != 128 or len(road) != 128:
        raise RuntimeError(f"256D context mismatch: {len(shoe)}/{len(road)}")
    vector = np.nan_to_num(np.asarray(shoe + road, dtype=np.float64), nan=0.0, posinf=2.0, neginf=-1.0)
    metadata = {
        "raw_round_count": len(sequence), "bp_round_count": len(_bp(sequence)), "tie_count": sum(x == "T" for x in sequence),
        "remaining_cards": remaining, "remaining_ratio": rr, "penetration_ratio": penetration,
        "shoe_maturity_ratio": maturity, "shoe_decks": SHOE_DECKS,
        "context_layout": "128_shoe_plus_128_road_256d",
        "context_compatibility": "bbb_app256_frozen_direct",
        "formal_direction_source": "contextual_linucb", "single_brain": True,
        "external_direction_votes_enabled": False, "anti_echo_external_penalty": False,
        "shoe_feature_values": [float(x) for x in shoe], "road_feature_values": [float(x) for x in road],
        "exact_card_input_ignored_for_web_panel_compatibility": True,
        "rank_ratio_source": "neutral_fallback_web_panel",
        "physical_edge_proxy": 0.0, "shoe_information_reliability": 0.0,
        "hazard_rate": hz, "hsmm_stable_probability": _hsmm_stable(sequence), "derived_road_consensus": consensus,
    }
    return vector, metadata


def _model_x(vector: Sequence[float]) -> np.ndarray:
    return np.nan_to_num(np.asarray(vector, dtype=np.float64).reshape(CONTEXT_DIM), nan=0.0, posinf=2.0, neginf=-1.0)


@dataclass(frozen=True)
class ContextSnapshot:
    vector: np.ndarray
    metadata: dict[str, Any]


class ContextGenerator:
    def build(self, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None = None) -> ContextSnapshot:
        del shoe_context
        raw = _normalize_history(deepcopy(history))
        vector, metadata = _context256(raw)
        return ContextSnapshot(vector=vector, metadata=metadata)


def _state_path() -> Path:
    candidates: list[Path] = []
    configured = str(os.getenv("LINUCB_STATE_FILE", "") or "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([Path("/var/data/contextual_linucb_state.json"), Path(__file__).resolve().parent / "data" / "contextual_linucb_state.json", Path("/tmp/contextual_linucb_state.json")])
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / f".linucb_write_{time.time_ns()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return Path("/tmp/contextual_linucb_state.json")

STATE_FILE = _state_path()


def _new_arm() -> dict[str, Any]:
    return {"A": (np.eye(CONTEXT_DIM) * LINUCB_RIDGE).tolist(), "b": np.zeros(CONTEXT_DIM).tolist(), "n": 0, "effective_n": 0.0}


def _new_scope() -> dict[str, Any]:
    now = int(time.time())
    return {"arms": {arm: _new_arm() for arm in ARMS}, "pending": {}, "updates": 0, "last_selected": "", "selection_streak": 0,
            "direct_predict_only": True, "no_bootstrap_on_start": True, "no_feedback_update": True, "no_ab_update": True,
            "no_decay": True, "created_at": now, "updated_at": now}


def _read_state() -> dict[str, Any]:
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict): raise ValueError
    except Exception:
        payload = {}
    if payload.get("version") != STATE_VERSION or payload.get("dim") != CONTEXT_DIM:
        payload = {}
    return {"version": STATE_VERSION, "dim": CONTEXT_DIM, "alpha": LINUCB_ALPHA, "ridge": LINUCB_RIDGE,
            "forgetting": 1.0, "scopes": payload.get("scopes") if isinstance(payload.get("scopes"), dict) else {}}


def _write_state(payload: Mapping[str, Any]) -> None:
    temporary = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False), encoding="utf-8")
    temporary.replace(STATE_FILE)


def make_scope_key(*, user_id: str = "", venue: str = "", room: str = "", shoe_id: str = "") -> str:
    raw = "|".join((str(user_id or "").strip(), str(venue or "").upper().strip(), str(room or "").strip(), str(shoe_id or "").strip()))
    return sha256((raw or "GLOBAL").encode("utf-8")).hexdigest()[:24]


def _history_fingerprint(history: Sequence[str]) -> str:
    return sha256("".join(history).encode("utf-8")).hexdigest()[:24]


def _arm_arrays(state: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        matrix = np.asarray(state.get("A"), dtype=np.float64).reshape(CONTEXT_DIM, CONTEXT_DIM)
        vector = np.asarray(state.get("b"), dtype=np.float64).reshape(CONTEXT_DIM)
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)): raise ValueError
        return matrix, vector
    except Exception:
        return np.eye(CONTEXT_DIM) * LINUCB_RIDGE, np.zeros(CONTEXT_DIM)


class ContextualLinUCB:
    def __init__(self, alpha: float = LINUCB_ALPHA):
        self.alpha = max(0.0, float(alpha))
        self.generator = ContextGenerator()

    def _score(self, arm_state: Mapping[str, Any], x: np.ndarray, alpha_scale: float) -> dict[str, float]:
        x = _model_x(x)
        matrix, reward_vector = _arm_arrays(arm_state)
        try:
            theta = np.linalg.solve(matrix, reward_vector)
            solved_x = np.linalg.solve(matrix, x)
        except np.linalg.LinAlgError:
            matrix = matrix + np.eye(CONTEXT_DIM) * LINUCB_RIDGE
            theta = np.linalg.solve(matrix, reward_vector)
            solved_x = np.linalg.solve(matrix, x)
        mean = float(x @ theta)
        uncertainty = float(math.sqrt(max(0.0, x @ solved_x)))
        effective_alpha = self.alpha * max(0.5, min(2.5, float(alpha_scale)))
        return {"score": mean + effective_alpha * uncertainty, "mean": mean, "uncertainty": uncertainty,
                "effective_alpha": effective_alpha, "raw_n": float(arm_state.get("n",0) or 0),
                "effective_n": float(arm_state.get("effective_n", arm_state.get("n",0)) or 0.0)}

    @staticmethod
    def _tie_choice(scope: Mapping[str, Any], raw_history: Sequence[str]) -> tuple[str, str]:
        arms = dict(scope.get("arms") or {})
        banker_n = float((arms.get("B") or {}).get("effective_n",0.0) or 0.0)
        player_n = float((arms.get("P") or {}).get("effective_n",0.0) or 0.0)
        if abs(banker_n-player_n) > 1e-9:
            return ("B" if banker_n < player_n else "P"), "tie_less_sampled_arm"
        last = str(scope.get("last_selected") or "").upper().strip()
        if last in ARMS:
            return ("P" if last == "B" else "B"), "tie_opposite_previous_arm"
        token = "LOCAL_256D_128PLUS128|" + "".join(raw_history)
        panel_hash = 0
        for char in token:
            panel_hash = (panel_hash * 31 + ord(char)) & 0xFFFFFFFF
        return ("B" if panel_hash % 2 else "P"), "tie_deterministic_history_hash"

    def _choose(self, scope: Mapping[str, Any], x: np.ndarray, raw_history: Sequence[str]):
        bp_rounds = len(_bp(raw_history))
        base_scale = 1.35 if bp_rounds < 8 else 1.15 if bp_rounds < 15 else 1.0
        arms = dict(scope.get("arms") or {})
        effective = {arm:max(0.0,float((arms.get(arm) or {}).get("effective_n",0.0) or 0.0)) for arm in ARMS}
        total = sum(effective.values())
        scores: dict[str, dict[str,float]] = {}
        for arm in ARMS:
            imbalance = math.sqrt(max(1.0,total+2.0)/max(1.0,effective[arm]+1.0))
            alpha_scale = base_scale * _clip(imbalance,0.85,LINUCB_ARM_ALPHA_MAX_SCALE)
            item = self._score(arms.get(arm,{}),x,alpha_scale)
            item.update({"linucb_score":item["score"],"alpha_scale":alpha_scale,"external_score_component":0.0})
            scores[arm] = item
        gap = float(scores["B"]["score"]-scores["P"]["score"])
        if abs(gap) <= LINUCB_SCORE_TIE_EPSILON:
            direction, reason = self._tie_choice(scope,raw_history)
        else:
            direction, reason = ("B" if gap > 0 else "P"), "linucb_ucb_score_argmax"
        return scores,effective,total,direction,reason,gap

    @staticmethod
    def _remember_selection(scope: dict[str,Any], direction: str) -> int:
        previous = str(scope.get("last_selected") or "").upper().strip()
        streak = int(scope.get("selection_streak",0) or 0)+1 if previous == direction else 1
        scope.update({"last_selected":direction,"selection_streak":streak,"updated_at":int(time.time())})
        return streak

    def predict(self, *, history: Iterable[Any] | str | None, shoe_context: Mapping[str,Any] | None, scope_key: str) -> dict[str,Any]:
        raw_history = _normalize_history(deepcopy(history))
        snapshot = self.generator.build(raw_history, deepcopy(dict(shoe_context or {})))
        raw_x = snapshot.vector.copy(); x = _model_x(raw_x); fingerprint = _history_fingerprint(raw_history)
        with _LOCK:
            root = _read_state(); scope = deepcopy(dict(root["scopes"].get(scope_key) or _new_scope()))
            bootstrap = {"applied":False,"reason":"web_panel_direct_no_bootstrap","bootstrap_rounds":0,"source_rounds":len(raw_history)}
            feedback = {"updated":False,"reason":"web_panel_direct_no_feedback_update","diagnostic_only":False,"formal_model":"contextual_linucb",
                        "a_b_frozen_without_bootstrap":True,"no_settlement":True,"no_decay":True}
            scores,effective,total,direction,reason,gap = self._choose(scope,x,raw_history)
            raw_pb = 1.0/(1.0+math.exp(-max(-8.0,min(8.0,gap/LINUCB_SCORE_TEMPERATURE))))
            p_b = _clip(raw_pb,PROBABILITY_MIN,PROBABILITY_MAX); p_p = 1.0-p_b
            probabilities = {"B":p_b,"P":p_p,"T":0.0}; confidence = p_b if direction=="B" else p_p
            streak = self._remember_selection(scope,direction)
            snapshot.metadata.update({"selection_streak":streak,"linucb_direction_weight":1.0,"panel_bootstrap":deepcopy(bootstrap),
                                      "prediction_mode":"frozen_256d_128plus128_local_brain_direct","automatic_feedback_update_enabled":False,
                                      "a_b_frozen_without_bootstrap":True,"no_bootstrap_on_start":True,"no_replay":True,"no_decay":True})
            scope.update({"pending":{},"frozen_direct_mode":True,"direct_predict_only":True,"no_bootstrap_on_start":True,
                          "no_feedback_update":True,"no_ab_update":True,"no_decay":True})
            root["scopes"][scope_key] = scope; _write_state(root)
        return {
            "model":"contextual_linucb_single_brain","version":STATE_VERSION,"legacy_state_version":STATE_VERSION,
            "direction":direction,"selected_arm":direction,"arm_index":1 if direction=="B" else 0,
            "probabilities":probabilities,"selected_win_probability":confidence,"confidence":confidence,
            "context_vector":[float(v) for v in raw_x],"model_context_vector":[float(v) for v in x],
            "context_feature_names":list(CONTEXT_FEATURE_NAMES),"context_dim":CONTEXT_DIM,"context_metadata":deepcopy(snapshot.metadata),
            "road_prior":{"diagnostic_only":True,"direction_weight":0.0,"banker_probability":0.5,"player_probability":0.5},
            "road_prior_probability":{"B":0.5,"P":0.5},"road_forecaster":{"available":False,"diagnostic_only":True,"formal_direction_weight":0.0},
            "features_used":dict(zip(CONTEXT_FEATURE_NAMES,[float(v) for v in raw_x])),"effective_support":total,"uncertainty":scores[direction]["uncertainty"],
            "linucb_probability_correction":0.0,"linucb_direction_weight":1.0,"learning_reliability":_clip(total/10.0),
            "scores":scores,"score_gap":gap,"score_semantics":"contextual_linucb_ucb_scores_only","alpha":self.alpha,"ridge":LINUCB_RIDGE,
            "forgetting":1.0,"feedback_update":feedback,"bootstrap_update":deepcopy(bootstrap),"panel_bootstrap_applied":False,
            "scope_key":scope_key,"arms":list(ARMS),"selection_reason":reason,"selection_streak":streak,
            "effective_arm_samples":effective,"history_round_count":len(raw_history),"bp_history_round_count":len(_bp(raw_history)),
            "history_fingerprint":fingerprint,"short_shoe_target_rounds":"50-70",
            "formal_context_source":"single_brain_256d_128shoe_128road_panel_frozen_direct_context","formal_direction_source":"contextual_linucb",
            "road_context_direction_weight":0.0,"card_composition_direction_weight":0.0,
            "probability_semantics":"bounded_logistic_mapping_of_linucb_ucb_score_gap","cold_start_uses_road_prior":False,
            "shoe_context_used_for_formal_direction":False,"shoe_context_used_as_features":False,"history_estimated_shoe_features_used":True,
            "shoe_context_independent_vote":False,"external_road_vote_enabled":False,"anti_echo_external_penalty":False,
            "panel_compatible":True,"frozen_direct_mode":True,"direct_predict_only":True,"no_bootstrap_on_start":True,
            "automatic_feedback_update_enabled":False,"no_replay":True,"no_previous_settlement":True,"no_ab_update":True,"no_decay":True,
            "anti_lock":{"enabled":False,"method":"none_external_feedback_only","tie_is_non_directional":True,"old_state_reused":False},
        }

    def update(self, *, scope_key: str, action: str, context_vector: Sequence[float], actual_outcome: str, clear_pending: bool = True) -> dict[str,Any]:
        del scope_key, action, context_vector, actual_outcome, clear_pending
        return {"updated":False,"reason":"frozen_256d_web_parity_no_ab_update","explicit_update_only":True,"a_b_frozen":True,"decay_applied":False}


_DEFAULT_BANDIT = ContextualLinUCB()

def predict_bandit(*, history: Iterable[Any] | str | None, shoe_context: Mapping[str,Any] | None, scope_key: str) -> dict[str,Any]:
    return _DEFAULT_BANDIT.predict(history=deepcopy(history),shoe_context=deepcopy(dict(shoe_context or {})),scope_key=str(scope_key or ""))

def update_bandit(*, scope_key: str, action: str, context_vector: Sequence[float], actual_outcome: str, clear_pending: bool = True) -> dict[str,Any]:
    return _DEFAULT_BANDIT.update(scope_key=str(scope_key or ""),action=action,context_vector=deepcopy(list(context_vector)),actual_outcome=actual_outcome,clear_pending=clear_pending)

__all__ = [
    "ARMS","CONTEXT_DIM","CONTEXT_FEATURE_NAMES","ContextGenerator","ContextualLinUCB","ESTIMATED_CARDS_PER_ROUND","SHOE_DECKS",
    "LINUCB_ALPHA","LINUCB_ARM_ALPHA_MAX_SCALE","LINUCB_FORGETTING","LINUCB_RIDGE","LINUCB_SCORE_TIE_EPSILON","LINUCB_UPDATE_WEIGHT",
    "PROBABILITY_MIN","PROBABILITY_MAX","ROAD_PRIOR_PROBABILITY_SPAN","ROAD_PRIOR_SCORE_WEIGHT","STATE_VERSION","make_scope_key","predict_bandit","update_bandit",
]
