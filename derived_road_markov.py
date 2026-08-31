"""Human-style probabilistic model for baccarat derived roads.

Big Eye Boy, Small Road and Cockroach Pig are deterministic transformations of
Big Road. Their red/blue marks are structural marks, not Banker/Player sides.
This module models the way experienced table players read those roads:

* recent red/blue continuation over 4/6/10 marks;
* single-jump, double-jump, colour-dragon and repeating run rhythm;
* same/switch Pattern Replay;
* direct R/U N-grams with recency decay;
* pattern-break probability;
* cross-road agreement and synchronized structural breaks;
* standard Ask-Road scoring for hypothetical next Banker vs Player.

The final derived-road reliability is capped because all three roads originate
from the same Big-Road history. It is intended as an auxiliary signal only.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence
import math

DERIVED_SYMBOLS = ("R", "U")
DERIVED_MAX_ORDER = 4
DERIVED_SUPPORT_THRESHOLD = 3
DERIVED_BACKOFF_ALPHA = 0.78
DERIVED_PRIOR_STRENGTH = 3.0
DERIVED_RECENCY_DECAY = 0.965
MAX_DERIVED_ROAD_RELIABILITY = 0.18
MIN_FORMAL_ACTIVE_ROADS = 2

ROAD_WEIGHTS = {
    "big_eye": 1.00,
    "small_road": 0.85,
    "cockroach_road": 0.70,
}
RECENT_WINDOW_WEIGHTS = {4: 0.45, 6: 0.35, 10: 0.20}
REPLAY_LENGTHS = (3, 4, 5, 6)
COMPONENT_WEIGHTS = {
    "recent": 0.26,
    "pattern_replay": 0.26,
    "ngram": 0.26,
    "run_rhythm": 0.22,
}


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _logit(probability: float) -> float:
    p = _clip(probability, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    x = max(-20.0, min(20.0, float(value)))
    return 1.0 / (1.0 + math.exp(-x))


def _clean(values: Sequence[Any]) -> list[str]:
    return [
        str(value).upper().strip()
        for value in values
        if str(value).upper().strip() in DERIVED_SYMBOLS
    ][-300:]


def _runs(values: Sequence[str]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))
    return runs


def _relation_signature(values: Sequence[str]) -> tuple[int, ...]:
    return tuple(1 if a == b else 0 for a, b in zip(values, values[1:]))


def _support_reliability(support: float, half_saturation: float) -> float:
    support = max(0.0, float(support))
    return _clip(support / (support + max(1e-9, float(half_saturation))))


def _probability_to_red(last_mark: str, p_same: float) -> float:
    p_same = _clip(p_same)
    return p_same if last_mark == "R" else 1.0 - p_same


def _weighted_same_probability(
    observations: Sequence[tuple[bool, float]],
    prior_strength: float,
) -> tuple[float, float]:
    same = prior_strength * 0.5
    switch = prior_strength * 0.5
    support = 0.0
    for is_same, raw_weight in observations:
        weight = max(0.0, float(raw_weight))
        support += weight
        if is_same:
            same += weight
        else:
            switch += weight
    total = same + switch
    return (same / total if total > 1e-12 else 0.5), support


def _recent_component(sequence: Sequence[str]) -> Dict[str, Any]:
    if len(sequence) < 2:
        return {"p_r": 0.5, "p_u": 0.5, "reliability": 0.0, "windows": {}}
    last = sequence[-1]
    numerator = 0.0
    denominator = 0.0
    details: Dict[str, Any] = {}
    for size, base_weight in RECENT_WINDOW_WEIGHTS.items():
        values = list(sequence[-size:])
        transitions = list(zip(values, values[1:]))
        support = len(transitions)
        same = sum(a == b for a, b in transitions)
        p_same = (same + 1.75) / (support + 3.5) if support else 0.5
        reliability = _support_reliability(support, max(2.5, size * 0.45))
        p_r = _probability_to_red(last, p_same)
        effective = base_weight * reliability
        numerator += effective * (p_r - 0.5)
        denominator += effective
        details[str(size)] = {
            "p_same": float(p_same),
            "p_switch": float(1.0 - p_same),
            "p_r": float(p_r),
            "p_u": float(1.0 - p_r),
            "support": int(support),
            "reliability": float(reliability),
            "base_weight": float(base_weight),
        }
    edge = numerator / denominator if denominator > 1e-12 else 0.0
    p_r = _clip(0.5 + edge, 0.34, 0.66)
    return {
        "p_r": float(p_r),
        "p_u": float(1.0 - p_r),
        "reliability": float(_clip(denominator)),
        "windows": details,
        "semantics": "recent_derived_colour_same_switch_4_6_10",
    }


def _pattern_replay_component(sequence: Sequence[str]) -> Dict[str, Any]:
    n = len(sequence)
    if n < 5:
        return {"p_r": 0.5, "p_u": 0.5, "reliability": 0.0, "matches": {}}
    last = sequence[-1]
    numerator = 0.0
    denominator = 0.0
    details: Dict[str, Any] = {}
    for length in REPLAY_LENGTHS:
        if n < length + 1:
            continue
        target = _relation_signature(sequence[-length:])
        observations: list[tuple[bool, float]] = []
        for start in range(0, n - length):
            prior = sequence[start : start + length]
            if _relation_signature(prior) != target:
                continue
            next_mark = sequence[start + length]
            age = (n - 1) - (start + length)
            observations.append(
                (next_mark == prior[-1], DERIVED_RECENCY_DECAY ** max(0, age))
            )
        p_same, support = _weighted_same_probability(observations, 3.0)
        reliability = _support_reliability(support, 3.5 + 0.45 * length)
        p_r = _probability_to_red(last, p_same)
        specificity = 0.60 + 0.40 * length / max(REPLAY_LENGTHS)
        effective = specificity * reliability
        numerator += effective * (p_r - 0.5)
        denominator += effective
        details[str(length)] = {
            "signature": "".join("S" if item else "X" for item in target),
            "p_same": float(p_same),
            "p_switch": float(1.0 - p_same),
            "p_r": float(p_r),
            "support": float(support),
            "reliability": float(reliability),
        }
    edge = numerator / denominator if denominator > 1e-12 else 0.0
    p_r = _clip(0.5 + edge, 0.34, 0.66)
    return {
        "p_r": float(p_r),
        "p_u": float(1.0 - p_r),
        "reliability": float(_clip(denominator / 2.5)),
        "matches": details,
        "semantics": "human_style_same_switch_pattern_replay",
    }


def _ngram_component(sequence: Sequence[str]) -> Dict[str, Any]:
    n = len(sequence)
    if n < 3:
        return {"p_r": 0.5, "p_u": 0.5, "reliability": 0.0, "orders": {}}
    details: Dict[str, Any] = {}
    numerator = 0.0
    denominator = 0.0
    for order in range(1, DERIVED_MAX_ORDER + 1):
        if n <= order:
            continue
        context = tuple(sequence[-order:])
        counts = {"R": 0.0, "U": 0.0}
        support = 0.0
        last_start = n - order - 1
        for start in range(0, n - order):
            if tuple(sequence[start : start + order]) != context:
                continue
            target = sequence[start + order]
            age = max(0, last_start - start)
            weight = DERIVED_RECENCY_DECAY ** age
            counts[target] += weight
            support += weight
        prior_each = DERIVED_PRIOR_STRENGTH * 0.5
        total = support + DERIVED_PRIOR_STRENGTH
        p_r = (counts["R"] + prior_each) / total if total > 0 else 0.5
        reliability = _support_reliability(support, 3.0 + order)
        specificity = 0.70 + 0.30 * order / DERIVED_MAX_ORDER
        effective = specificity * reliability
        numerator += effective * (p_r - 0.5)
        denominator += effective
        details[str(order)] = {
            "context": "".join(context),
            "counts": counts,
            "support": float(support),
            "p_r": float(p_r),
            "p_u": float(1.0 - p_r),
            "reliability": float(reliability),
        }
    edge = numerator / denominator if denominator > 1e-12 else 0.0
    p_r = _clip(0.5 + edge, 0.34, 0.66)
    return {
        "p_r": float(p_r),
        "p_u": float(1.0 - p_r),
        "reliability": float(_clip(denominator / 2.4)),
        "orders": details,
        "semantics": "direct_RU_ngram_orders_1_to_4_with_recency_decay",
    }


def _run_survival(sequence: Sequence[str], current_length: int) -> tuple[float, int]:
    runs = _runs(sequence)
    completed = [length for _, length in runs[:-1]]
    eligible = [length for length in completed if length >= current_length]
    survived = sum(length > current_length for length in eligible)
    p_continue = (survived + 2.0) / (len(eligible) + 4.0)
    return p_continue, len(eligible)


def _run_rhythm_component(sequence: Sequence[str]) -> Dict[str, Any]:
    if not sequence:
        return {
            "p_r": 0.5,
            "p_u": 0.5,
            "reliability": 0.0,
            "pattern": "COLD_START",
            "pattern_break_probability": 0.5,
        }
    runs = _runs(sequence)
    last_mark, current_run = runs[-1]
    completed_lengths = [length for _, length in runs[:-1]]
    recent_lengths = [length for _, length in runs[-6:]]
    desired_same: bool | None = None
    pattern = "GENERIC"
    base_reliability = 0.0
    target_run_length: int | None = None

    if len(recent_lengths) >= 4 and all(length == 1 for length in recent_lengths[-4:]):
        pattern = "SINGLE_JUMP"
        desired_same = False
        base_reliability = 0.72
    elif len(completed_lengths) >= 2 and all(
        length == 2 for length in completed_lengths[-2:]
    ) and current_run <= 2:
        pattern = "DOUBLE_JUMP"
        target_run_length = 2
        desired_same = current_run < 2
        base_reliability = 0.66
    elif len(completed_lengths) >= 4:
        a, b, c, d = completed_lengths[-4:]
        if a == c and b == d and a != b and 1 <= a <= 4 and 1 <= b <= 4:
            pattern = f"RUN_RHYTHM_{a}_{b}"
            target_run_length = a
            desired_same = current_run < target_run_length
            base_reliability = 0.56
    if desired_same is None and current_run >= 3:
        pattern = "COLOR_DRAGON"
        p_continue, support = _run_survival(sequence, current_run)
        desired_same = p_continue >= 0.5
        base_reliability = 0.45 + 0.20 * _support_reliability(support, 4.0)
    else:
        p_continue, support = _run_survival(sequence, current_run)

    support_rel = _support_reliability(support, 4.0)
    empirical_same = p_continue
    if desired_same is None:
        p_same = empirical_same
        reliability = 0.30 * support_rel
    else:
        rule_same = 0.68 if desired_same else 0.32
        empirical_weight = 0.45 * support_rel
        p_same = (1.0 - empirical_weight) * rule_same + empirical_weight * empirical_same
        reliability = _clip(base_reliability * (0.50 + 0.50 * support_rel))

    p_r = _probability_to_red(last_mark, p_same)
    if desired_same is None:
        break_probability = 0.5
    else:
        expected_probability = p_same if desired_same else 1.0 - p_same
        break_probability = 1.0 - expected_probability

    return {
        "p_r": float(_clip(p_r, 0.32, 0.68)),
        "p_u": float(1.0 - _clip(p_r, 0.32, 0.68)),
        "p_same": float(p_same),
        "p_switch": float(1.0 - p_same),
        "reliability": float(reliability),
        "pattern": pattern,
        "desired_relation": (
            "SAME" if desired_same is True else "SWITCH" if desired_same is False else "EMPIRICAL"
        ),
        "current_mark": last_mark,
        "current_run_length": int(current_run),
        "target_run_length": target_run_length,
        "run_lengths": recent_lengths,
        "run_survival_probability": float(empirical_same),
        "run_survival_support": int(support),
        "pattern_break_probability": float(_clip(break_probability)),
        "semantics": "human_derived_run_rhythm_single_double_dragon_and_break",
    }


def predict_next_derived_mark(sequence: Sequence[Any]) -> Dict[str, Any]:
    """Predict the next derived-road red/blue mark using human-style features."""
    values = _clean(sequence)
    components = {
        "recent": _recent_component(values),
        "pattern_replay": _pattern_replay_component(values),
        "ngram": _ngram_component(values),
        "run_rhythm": _run_rhythm_component(values),
    }

    numerator = 0.0
    denominator = 0.0
    used: Dict[str, Any] = {}
    for name, base_weight in COMPONENT_WEIGHTS.items():
        item = components[name]
        reliability = _clip(item.get("reliability", 0.0))
        effective = base_weight * reliability
        p_r = _clip(item.get("p_r", 0.5), 0.05, 0.95)
        numerator += effective * _logit(p_r)
        denominator += effective
        used[name] = {
            "base_weight": float(base_weight),
            "reliability": float(reliability),
            "effective_weight": float(effective),
            "p_r": float(p_r),
            "p_u": float(1.0 - p_r),
        }

    raw_logit = numerator / denominator if denominator > 1e-12 else 0.0
    maturity = _clip(len(values) / 12.0)
    final_logit = raw_logit * (0.30 + 0.70 * maturity)
    p_r = _clip(_sigmoid(final_logit), 0.32, 0.68)
    confidence = _clip(denominator * maturity)
    rhythm = components["run_rhythm"]

    return {
        "model_id": "HUMAN-DERIVED-ROAD-V2",
        "probabilities": {"R": float(p_r), "U": float(1.0 - p_r)},
        "direction": "R" if p_r >= 0.5 else "U",
        "confidence": float(confidence),
        "sample_count": len(values),
        "maturity": float(maturity),
        "pattern": str(rhythm.get("pattern") or "GENERIC"),
        "pattern_break_probability": float(rhythm.get("pattern_break_probability", 0.5) or 0.5),
        "current_run_length": int(rhythm.get("current_run_length", 0) or 0),
        "components": components,
        "component_weights": used,
        "semantics": "human_style_derived_colour_probability_not_banker_player_probability",
    }


def score_ask_road_scenarios(
    derived_roads: Mapping[str, Sequence[Any]],
    scenario_marks: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    """Score hypothetical next Banker/Player using human-style Ask-Road logic."""
    models = {
        name: predict_next_derived_mark(list(derived_roads.get(name) or []))
        for name in ROAD_WEIGHTS
    }

    weighted_logs = {"B": 0.0, "P": 0.0}
    weight_sums = {"B": 0.0, "P": 0.0}
    scenario_details: Dict[str, Any] = {"B": {}, "P": {}}
    preference_numerator = 0.0
    preference_denominator = 0.0
    active_roads: set[str] = set()

    for name, road_weight in ROAD_WEIGHTS.items():
        model = models[name]
        confidence = float(model.get("confidence", 0.0) or 0.0)
        mark_b = str((scenario_marks.get("B") or {}).get(name) or "").upper()
        mark_p = str((scenario_marks.get("P") or {}).get(name) or "").upper()
        if (
            mark_b not in DERIVED_SYMBOLS
            or mark_p not in DERIVED_SYMBOLS
            or confidence <= 0.02
        ):
            for side, mark in (("B", mark_b), ("P", mark_p)):
                scenario_details[side][name] = {
                    "mark": mark,
                    "active": False,
                    "reason": "road_not_mature_or_no_standard_ask_mark",
                }
            continue

        active_roads.add(name)
        prob_b = max(1e-6, float(model["probabilities"].get(mark_b, 0.5) or 0.5))
        prob_p = max(1e-6, float(model["probabilities"].get(mark_p, 0.5) or 0.5))
        effective = road_weight * confidence
        weighted_logs["B"] += effective * math.log(prob_b)
        weighted_logs["P"] += effective * math.log(prob_p)
        weight_sums["B"] += effective
        weight_sums["P"] += effective

        preference_edge = prob_b - prob_p
        preference_numerator += effective * preference_edge
        preference_denominator += effective * abs(preference_edge)

        scenario_details["B"][name] = {
            "mark": mark_b,
            "active": True,
            "mark_probability": float(prob_b),
            "model_confidence": confidence,
            "effective_weight": float(effective),
        }
        scenario_details["P"][name] = {
            "mark": mark_p,
            "active": True,
            "mark_probability": float(prob_p),
            "model_confidence": confidence,
            "effective_weight": float(effective),
        }

    log_scores = {"B": math.log(0.5), "P": math.log(0.5)}
    for side in ("B", "P"):
        if weight_sums[side] > 1e-12:
            log_scores[side] = weighted_logs[side] / weight_sums[side]

    max_log = max(log_scores.values())
    exp_b = math.exp(log_scores["B"] - max_log)
    exp_p = math.exp(log_scores["P"] - max_log)
    total = exp_b + exp_p
    likelihood = {
        "B": exp_b / total if total > 1e-12 else 0.5,
        "P": exp_p / total if total > 1e-12 else 0.5,
    }

    active_list = sorted(active_roads)
    active_fraction = len(active_list) / 3.0
    if active_list:
        total_road_weight = sum(ROAD_WEIGHTS[name] for name in active_list)
        mean_confidence = sum(
            ROAD_WEIGHTS[name] * float(models[name]["confidence"])
            for name in active_list
        ) / max(1e-12, total_road_weight)
        breaks = [float(models[name].get("pattern_break_probability", 0.5) or 0.5) for name in active_list]
        mean_break = sum(breaks) / len(breaks)
        break_dispersion = max(breaks) - min(breaks) if len(breaks) >= 2 else 1.0
        synchronized_break = _clip(mean_break * (1.0 - break_dispersion))
    else:
        mean_confidence = 0.0
        synchronized_break = 0.0

    cross_road_agreement = (
        abs(preference_numerator) / preference_denominator
        if preference_denominator > 1e-12
        else 0.0
    )
    separation = abs(likelihood["B"] - likelihood["P"])
    base_reliability = (
        MAX_DERIVED_ROAD_RELIABILITY
        * mean_confidence
        * active_fraction
        * (0.45 + 0.55 * _clip(separation / 0.20))
        * (0.65 + 0.35 * _clip(cross_road_agreement))
        * (0.85 + 0.15 * synchronized_break)
    )
    reliability = (
        min(MAX_DERIVED_ROAD_RELIABILITY, base_reliability)
        if len(active_list) >= MIN_FORMAL_ACTIVE_ROADS
        else 0.0
    )

    return {
        "model_id": "HUMAN-DERIVED-ASK-ROAD-V2",
        "likelihood": likelihood,
        "reliability": float(reliability),
        "raw_reliability": float(min(MAX_DERIVED_ROAD_RELIABILITY, base_reliability)),
        "max_reliability": float(MAX_DERIVED_ROAD_RELIABILITY),
        "active_roads": active_list,
        "active_road_count": len(active_list),
        "minimum_formal_active_roads": int(MIN_FORMAL_ACTIVE_ROADS),
        "cross_road_agreement": float(_clip(cross_road_agreement)),
        "synchronized_break": float(synchronized_break),
        "scenario_marks": {
            side: dict(scenario_marks.get(side) or {}) for side in ("B", "P")
        },
        "scenario_details": scenario_details,
        "models": models,
        "log_scores": log_scores,
        "human_patterns": {name: str(models[name].get("pattern") or "GENERIC") for name in ROAD_WEIGHTS},
        "semantics": (
            "human_style_three_derived_roads_ask_road_likelihood_with_capped_shared_history_reliability"
        ),
    }


__all__ = [
    "DERIVED_MAX_ORDER",
    "DERIVED_SUPPORT_THRESHOLD",
    "DERIVED_BACKOFF_ALPHA",
    "MAX_DERIVED_ROAD_RELIABILITY",
    "MIN_FORMAL_ACTIVE_ROADS",
    "predict_next_derived_mark",
    "score_ask_road_scenarios",
]
