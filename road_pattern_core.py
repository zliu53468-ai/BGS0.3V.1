"""Road-primary baccarat B/P forecasting with probabilistic turning analysis.

Formal B/P direction is derived only from the observed Big-Road B/P sequence.
Shoe composition, cut-card position, LSTM, LinUCB, HSMM and hazard models have
zero formal direction authority.

V2 keeps the inspectable road-pattern stack and adds a probabilistic turning
layer designed for short 50-70 hand shoes:

1. Multi-window SAME/SWITCH behaviour over 6/10/16/24 resolved hands.
2. Orientation-invariant historical Pattern Replay.
3. Orientation-invariant relation N-grams, orders 2-5.
4. Pattern survival for single-jump, double-jump and dragon-like runs.
5. Probabilistic turning analysis:
   - Bayesian run survival,
   - multi-scale disagreement,
   - pattern-break probability,
   - light change-point probability.

When change pressure rises, the system shifts weight toward 6/10-hand evidence
and the turning layer while reducing stale 16/24-hand and replay influence.
Final component fusion is reliability-weighted log-odds rather than a raw
probability average. There is no forced follow-last and no forced alternation.

This is an inspectable sequence model, not evidence that baccarat is
predictable or profitable.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence
import math

MODEL_ID = "ROAD-PATTERN-PROBABILITY-V2"
VERSION = "ROAD-PATTERN-PROBABILITY-50-70-V2"
OUTCOMES = ("B", "P")

STABLE_WINDOW_WEIGHTS = {6: 0.20, 10: 0.25, 16: 0.27, 24: 0.28}
TURN_WINDOW_WEIGHTS = {6: 0.40, 10: 0.30, 16: 0.20, 24: 0.10}
WINDOW_WEIGHTS = dict(STABLE_WINDOW_WEIGHTS)

STABLE_COMPONENT_WEIGHTS = {
    "multi_window": 0.25,
    "pattern_replay": 0.30,
    "ngram": 0.25,
    "pattern_survival": 0.15,
    "probabilistic_turning": 0.05,
}
TURN_COMPONENT_WEIGHTS = {
    "multi_window": 0.30,
    "pattern_replay": 0.20,
    "ngram": 0.20,
    "pattern_survival": 0.10,
    "probabilistic_turning": 0.20,
}
COMPONENT_WEIGHTS = dict(STABLE_COMPONENT_WEIGHTS)

NGRAM_ORDERS = (2, 3, 4, 5)
REPLAY_LENGTHS = (3, 4, 5, 6, 8)
RECENCY_DECAY = 0.965
MAX_DIRECTION_EDGE = 0.13
MIN_COMPONENT_RELIABILITY = 0.02
RUN_PRIOR_ALPHA = 2.5
RUN_PRIOR_BETA = 2.5


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _sigmoid(value: float) -> float:
    value = max(-20.0, min(20.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _logit(probability: float) -> float:
    p = _clip(probability, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _blend_weight_maps(
    stable: Mapping[int | str, float],
    turning: Mapping[int | str, float],
    pressure: float,
) -> dict[int | str, float]:
    t = _clip(pressure)
    keys = set(stable) | set(turning)
    blended = {
        key: (1.0 - t) * float(stable.get(key, 0.0))
        + t * float(turning.get(key, 0.0))
        for key in keys
    }
    total = sum(max(0.0, value) for value in blended.values())
    if total <= 1e-12:
        return {key: 0.0 for key in blended}
    return {key: max(0.0, value) / total for key, value in blended.items()}


def normalize_bp(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = "".join(
            ch for ch in history.upper() if not ch.isspace() and ch not in ",|"
        )
        if compact and all(ch in {"B", "P", "T"} for ch in compact):
            return [ch for ch in compact if ch in OUTCOMES][-500:]
        raw_items: Iterable[Any] = [
            part for part in history.replace("|", ",").split(",") if part.strip()
        ]
    else:
        raw_items = history
    result: list[str] = []
    for item in raw_items:
        if isinstance(item, Mapping):
            raw = (
                item.get("outcome")
                or item.get("actual")
                or item.get("actual_outcome")
                or item.get("virtual_outcome")
            )
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in OUTCOMES:
            result.append(value)
    return result[-500:]


def _runs(sequence: Sequence[str]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for side in sequence:
        if runs and runs[-1][0] == side:
            runs[-1] = (side, runs[-1][1] + 1)
        else:
            runs.append((side, 1))
    return runs


def _relation_signature(values: Sequence[str]) -> tuple[int, ...]:
    """Orientation-invariant transition signature: 1=same, 0=switch."""
    return tuple(1 if left == right else 0 for left, right in zip(values, values[1:]))


def _same_probability_to_b(last_side: str, p_same: float) -> float:
    p_same = _clip(p_same)
    return p_same if last_side == "B" else 1.0 - p_same


def _support_reliability(support: float, *, half_saturation: float) -> float:
    support = max(0.0, float(support))
    return _clip(support / (support + max(1e-9, float(half_saturation))))


def _weighted_same_probability(
    observations: Sequence[tuple[bool, float]],
    *,
    prior_strength: float = 3.0,
) -> tuple[float, float]:
    same = 0.5 * max(0.0, prior_strength)
    switch = 0.5 * max(0.0, prior_strength)
    support = 0.0
    for is_same, weight in observations:
        w = max(0.0, float(weight))
        support += w
        if is_same:
            same += w
        else:
            switch += w
    total = same + switch
    return (same / total if total > 1e-12 else 0.5), support


def _window_same_stats(sequence: Sequence[str]) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    for size in STABLE_WINDOW_WEIGHTS:
        values = list(sequence[-size:])
        transitions = list(zip(values, values[1:]))
        support = len(transitions)
        same_count = sum(left == right for left, right in transitions)
        p_same = (same_count + 2.0) / (support + 4.0) if support else 0.5
        reliability = _support_reliability(
            support, half_saturation=max(3.0, size * 0.45)
        )
        stats[size] = {
            "p_same": float(p_same),
            "p_switch": float(1.0 - p_same),
            "support": int(support),
            "reliability": float(reliability),
        }
    return stats


def _multi_window_component(
    sequence: Sequence[str],
    *,
    turn_pressure: float = 0.0,
    stats: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(sequence) < 2:
        return {
            "p_b": 0.5,
            "p_p": 0.5,
            "reliability": 0.0,
            "support": 0.0,
            "windows": {},
            "dynamic_window_weights": dict(STABLE_WINDOW_WEIGHTS),
        }
    last = sequence[-1]
    window_stats = dict(stats or _window_same_stats(sequence))
    dynamic = _blend_weight_maps(
        STABLE_WINDOW_WEIGHTS, TURN_WINDOW_WEIGHTS, turn_pressure
    )
    numerator = 0.0
    denominator = 0.0
    total_support = 0.0
    diagnostics: dict[str, Any] = {}
    for size in STABLE_WINDOW_WEIGHTS:
        item = dict(window_stats.get(size) or {})
        p_same = _clip(item.get("p_same", 0.5))
        reliability = _clip(item.get("reliability", 0.0))
        support = int(item.get("support", 0) or 0)
        p_b = _same_probability_to_b(last, p_same)
        base_weight = float(dynamic.get(size, 0.0))
        weight = base_weight * reliability
        numerator += weight * (p_b - 0.5)
        denominator += weight
        total_support += support * base_weight
        diagnostics[str(size)] = {
            **item,
            "p_b": float(p_b),
            "p_p": float(1.0 - p_b),
            "base_weight": float(base_weight),
        }
    edge = numerator / denominator if denominator > 1e-12 else 0.0
    p_b = _clip(0.5 + edge, 0.38, 0.62)
    reliability = _clip(denominator)
    return {
        "p_b": float(p_b),
        "p_p": float(1.0 - p_b),
        "reliability": float(reliability),
        "support": float(total_support),
        "windows": diagnostics,
        "turn_pressure": float(_clip(turn_pressure)),
        "dynamic_window_weights": {str(k): float(v) for k, v in dynamic.items()},
        "semantics": "dynamic_6_10_16_24_same_switch_rates_orientation_invariant",
    }


def _pattern_replay_component(sequence: Sequence[str]) -> dict[str, Any]:
    n = len(sequence)
    if n < 4:
        return {
            "p_b": 0.5,
            "p_p": 0.5,
            "reliability": 0.0,
            "support": 0.0,
            "matches": {},
        }
    last = sequence[-1]
    numerator = 0.0
    denominator = 0.0
    total_support = 0.0
    details: dict[str, Any] = {}
    for length in REPLAY_LENGTHS:
        if n < length + 1:
            continue
        target_signature = _relation_signature(sequence[-length:])
        observations: list[tuple[bool, float]] = []
        for start in range(0, n - length):
            window = sequence[start : start + length]
            if _relation_signature(window) != target_signature:
                continue
            next_side = sequence[start + length]
            is_same = next_side == window[-1]
            age = (n - 1) - (start + length)
            observations.append((is_same, RECENCY_DECAY ** max(0, age)))
        p_same, support = _weighted_same_probability(observations, prior_strength=3.0)
        reliability = _support_reliability(
            support, half_saturation=3.5 + 0.4 * length
        )
        p_b = _same_probability_to_b(last, p_same)
        specificity = 0.55 + 0.45 * (length / max(REPLAY_LENGTHS))
        weight = specificity * reliability
        numerator += weight * (p_b - 0.5)
        denominator += weight
        total_support += support
        details[str(length)] = {
            "signature": "".join("S" if value else "X" for value in target_signature),
            "p_same": float(p_same),
            "p_switch": float(1.0 - p_same),
            "p_b": float(p_b),
            "support": float(support),
            "reliability": float(reliability),
        }
    edge = numerator / denominator if denominator > 1e-12 else 0.0
    p_b = _clip(0.5 + edge, 0.37, 0.63)
    reliability = _clip(
        denominator / max(1.0, len(REPLAY_LENGTHS) * 0.65)
    )
    return {
        "p_b": float(p_b),
        "p_p": float(1.0 - p_b),
        "reliability": float(reliability),
        "support": float(total_support),
        "matches": details,
        "semantics": "historical_same_switch_signature_replay_not_raw_BP_chasing",
    }


def _ngram_component(sequence: Sequence[str]) -> dict[str, Any]:
    n = len(sequence)
    if n < 3:
        return {
            "p_b": 0.5,
            "p_p": 0.5,
            "reliability": 0.0,
            "support": 0.0,
            "orders": {},
        }
    numerator = 0.0
    denominator = 0.0
    total_support = 0.0
    details: dict[str, Any] = {}
    for order in NGRAM_ORDERS:
        if n <= order - 1:
            continue
        current_context = sequence[-order:]
        current_rel = _relation_signature(current_context)
        observations: list[tuple[bool, float]] = []
        for start in range(0, n - order):
            prior = sequence[start : start + order]
            if _relation_signature(prior) != current_rel:
                continue
            next_side = sequence[start + order]
            is_same = next_side == prior[-1]
            age = (n - 1) - (start + order)
            observations.append((is_same, RECENCY_DECAY ** max(0, age)))
        p_same, support = _weighted_same_probability(observations, prior_strength=3.5)
        reliability = _support_reliability(support, half_saturation=4.0 + order)
        p_b = _same_probability_to_b(sequence[-1], p_same)
        order_weight = (
            0.70 + 0.30 * order / max(NGRAM_ORDERS)
        ) * reliability
        numerator += order_weight * (p_b - 0.5)
        denominator += order_weight
        total_support += support
        details[str(order)] = {
            "relation_context": "".join("S" if value else "X" for value in current_rel),
            "p_same": float(p_same),
            "p_switch": float(1.0 - p_same),
            "p_b": float(p_b),
            "support": float(support),
            "reliability": float(reliability),
        }
    edge = numerator / denominator if denominator > 1e-12 else 0.0
    p_b = _clip(0.5 + edge, 0.38, 0.62)
    reliability = _clip(
        denominator / max(1.0, len(NGRAM_ORDERS) * 0.75)
    )
    return {
        "p_b": float(p_b),
        "p_p": float(1.0 - p_b),
        "reliability": float(reliability),
        "support": float(total_support),
        "orders": details,
        "semantics": "orientation_invariant_relation_ngram_orders_2_to_5",
    }


def _completed_run_lengths(sequence: Sequence[str]) -> list[int]:
    runs = _runs(sequence)
    return [length for _, length in runs[:-1]] if len(runs) > 1 else []


def _bayesian_run_survival(
    sequence: Sequence[str], current_length: int
) -> dict[str, Any]:
    lengths = _completed_run_lengths(sequence)
    eligible = [length for length in lengths if length >= current_length]
    survived = sum(length > current_length for length in eligible)
    failed = len(eligible) - survived
    alpha = RUN_PRIOR_ALPHA + survived
    beta = RUN_PRIOR_BETA + failed
    total = alpha + beta
    mean = alpha / total if total > 0 else 0.5
    variance = (
        alpha * beta / (total * total * (total + 1.0))
        if total > 0
        else 0.05
    )
    sd = math.sqrt(max(0.0, variance))
    reliability = _support_reliability(len(eligible), half_saturation=4.0)
    return {
        "continue_probability": float(mean),
        "turn_probability": float(1.0 - mean),
        "support": int(len(eligible)),
        "survived": int(survived),
        "failed": int(failed),
        "posterior_alpha": float(alpha),
        "posterior_beta": float(beta),
        "posterior_sd": float(sd),
        "credible_low_approx": float(_clip(mean - 1.64 * sd)),
        "credible_high_approx": float(_clip(mean + 1.64 * sd)),
        "reliability": float(reliability),
        "semantics": "beta_binomial_run_survival_shrunk_to_neutral_on_low_support",
    }


def _pattern_survival_component(sequence: Sequence[str]) -> dict[str, Any]:
    if not sequence:
        return {
            "p_b": 0.5,
            "p_p": 0.5,
            "reliability": 0.0,
            "support": 0,
            "pattern": "COLD_START",
            "survival_probability": 0.5,
        }
    runs = _runs(sequence)
    last_side, current_run = runs[-1]
    recent_lengths = [length for _, length in runs[-5:]]
    pattern = "GENERIC"
    desired_same: bool | None = None
    base_strength = 0.0

    if len(recent_lengths) >= 4 and all(
        length == 1 for length in recent_lengths[-4:]
    ):
        pattern = "SINGLE_JUMP"
        desired_same = False
        base_strength = 0.70
    elif len(recent_lengths) >= 4 and all(
        length == 2 for length in recent_lengths[-4:-1]
    ):
        pattern = "DOUBLE_JUMP"
        desired_same = current_run < 2
        base_strength = 0.62
    elif current_run >= 3:
        pattern = "DRAGON"
        desired_same = True
        base_strength = 0.52

    bayes_run = _bayesian_run_survival(sequence, current_run)
    empirical_same = float(bayes_run["continue_probability"])
    support = int(bayes_run["support"])
    support_reliability = float(bayes_run["reliability"])

    if desired_same is None:
        p_same = empirical_same
        reliability = 0.35 * support_reliability
    else:
        rule_same = 0.62 if desired_same else 0.38
        empirical_weight = 0.55 * support_reliability
        p_same = (
            (1.0 - empirical_weight) * rule_same
            + empirical_weight * empirical_same
        )
        reliability = _clip(
            base_strength * (0.45 + 0.55 * support_reliability)
        )

    p_b = _same_probability_to_b(last_side, p_same)
    desired_probability = (
        p_same
        if desired_same is True
        else 1.0 - p_same
        if desired_same is False
        else 0.5
    )
    pattern_break_probability = (
        1.0 - desired_probability if desired_same is not None else 0.5
    )
    return {
        "p_b": float(_clip(p_b, 0.38, 0.62)),
        "p_p": float(1.0 - _clip(p_b, 0.38, 0.62)),
        "reliability": float(reliability),
        "support": int(support),
        "pattern": pattern,
        "current_run_length": int(current_run),
        "desired_relation": (
            "SAME"
            if desired_same is True
            else "SWITCH"
            if desired_same is False
            else "EMPIRICAL"
        ),
        "survival_probability": float(p_same),
        "empirical_run_survival": float(empirical_same),
        "pattern_break_probability": float(pattern_break_probability),
        "bayesian_run_survival": bayes_run,
        "semantics": "weak_pattern_prior_blended_with_bayesian_in_shoe_run_survival",
    }


def _weighted_window_rate(
    stats: Mapping[int, Mapping[str, Any]], sizes: Sequence[int]
) -> tuple[float, float]:
    numerator = 0.0
    denominator = 0.0
    for size in sizes:
        item = stats.get(size) or {}
        reliability = _clip(item.get("reliability", 0.0))
        p_same = _clip(item.get("p_same", 0.5))
        numerator += reliability * p_same
        denominator += reliability
    return (
        numerator / denominator if denominator > 1e-12 else 0.5,
        _clip(denominator / max(1.0, float(len(sizes)))),
    )


def _relation_change_probability(sequence: Sequence[str]) -> dict[str, Any]:
    relations = list(_relation_signature(sequence))
    if len(relations) < 5:
        return {
            "probability": 0.0,
            "short_same_rate": 0.5,
            "baseline_same_rate": 0.5,
            "absolute_delta": 0.0,
            "z_like": 0.0,
            "support_maturity": 0.0,
        }

    short = relations[-min(6, len(relations)) :]
    baseline = relations[: -len(short)]
    if len(baseline) > 18:
        baseline = baseline[-18:]

    short_same = (sum(short) + 2.0) / (len(short) + 4.0)
    if baseline:
        base_same = (sum(baseline) + 2.0) / (len(baseline) + 4.0)
    else:
        base_same = 0.5

    delta = abs(short_same - base_same)
    pooled = _clip((short_same + base_same) / 2.0, 0.05, 0.95)
    se = math.sqrt(
        pooled
        * (1.0 - pooled)
        * (1.0 / max(1, len(short)) + 1.0 / max(1, len(baseline)))
    )
    z_like = delta / max(0.08, se)
    delta_score = _sigmoid((delta - 0.16) * 12.0)
    z_score = _sigmoid((z_like - 1.35) * 2.0)
    support_maturity = _clip(len(relations) / 18.0)
    probability = _clip(
        (0.55 * delta_score + 0.45 * z_score) * support_maturity
    )
    return {
        "probability": float(probability),
        "short_same_rate": float(short_same),
        "baseline_same_rate": float(base_same),
        "absolute_delta": float(delta),
        "z_like": float(z_like),
        "delta_score": float(delta_score),
        "z_score": float(z_score),
        "support_maturity": float(support_maturity),
        "short_support": len(short),
        "baseline_support": len(baseline),
        "semantics": "light_change_point_on_recent_vs_prior_same_switch_rate",
    }


def _probabilistic_turning_component(
    sequence: Sequence[str],
    *,
    window_stats: Mapping[int, Mapping[str, Any]],
    survival: Mapping[str, Any],
) -> dict[str, Any]:
    if not sequence:
        return {
            "p_b": 0.5,
            "p_p": 0.5,
            "p_same": 0.5,
            "continue_probability": 0.5,
            "turn_probability": 0.5,
            "change_probability": 0.0,
            "turning_pressure": 0.0,
            "reliability": 0.0,
            "support": 0,
            "regime": "COLD_START",
        }

    runs = _runs(sequence)
    current_run = runs[-1][1]
    bayes_run = _bayesian_run_survival(sequence, current_run)
    short_same, short_rel = _weighted_window_rate(window_stats, (6, 10))
    long_same, long_rel = _weighted_window_rate(window_stats, (16, 24))
    disagreement_raw = abs(short_same - long_same)
    disagreement = _clip(disagreement_raw / 0.22)

    change = _relation_change_probability(sequence)
    change_probability = float(change["probability"])

    survival_p_same = _clip(survival.get("survival_probability", 0.5))
    survival_rel = _clip(survival.get("reliability", 0.0))
    pattern = str(survival.get("pattern") or "GENERIC")
    pattern_break = _clip(survival.get("pattern_break_probability", 0.5))
    pattern_break_signal = (
        pattern_break * survival_rel if pattern != "GENERIC" else 0.0
    )

    run_p_same = _clip(bayes_run.get("continue_probability", 0.5))
    run_rel = _clip(bayes_run.get("reliability", 0.0))

    w_short = 0.30 + 0.45 * change_probability
    w_run = 0.35 * (0.35 + 0.65 * run_rel)
    w_survival = 0.25 * (0.35 + 0.65 * survival_rel)
    w_neutral = 0.10
    denom = w_short + w_run + w_survival + w_neutral
    p_same = (
        w_short * short_same
        + w_run * run_p_same
        + w_survival * survival_p_same
        + w_neutral * 0.5
    ) / max(1e-12, denom)

    support_maturity = _clip(max(0, len(sequence) - 2) / 18.0)
    p_same = 0.5 + (p_same - 0.5) * (0.35 + 0.65 * support_maturity)
    p_same = _clip(p_same, 0.36, 0.64)

    turning_pressure = _clip(
        0.45 * change_probability
        + 0.30 * disagreement
        + 0.25 * pattern_break_signal
    )
    reliability = _clip(
        support_maturity
        * (
            0.25
            + 0.30 * max(short_rel, long_rel)
            + 0.25 * max(run_rel, survival_rel)
            + 0.20 * turning_pressure
        )
    )

    p_b = _same_probability_to_b(sequence[-1], p_same)
    regime = (
        "TRANSITION"
        if turning_pressure >= 0.60
        else "WATCH"
        if turning_pressure >= 0.30
        else "STABLE"
    )
    return {
        "p_b": float(_clip(p_b, 0.36, 0.64)),
        "p_p": float(1.0 - _clip(p_b, 0.36, 0.64)),
        "p_same": float(p_same),
        "continue_probability": float(p_same),
        "turn_probability": float(1.0 - p_same),
        "change_probability": float(change_probability),
        "turning_pressure": float(turning_pressure),
        "reliability": float(reliability),
        "support": int(max(0, len(sequence) - 1)),
        "regime": regime,
        "current_run_length": int(current_run),
        "bayesian_run_survival": bayes_run,
        "multi_scale": {
            "short_same_probability": float(short_same),
            "long_same_probability": float(long_same),
            "short_reliability": float(short_rel),
            "long_reliability": float(long_rel),
            "absolute_disagreement": float(disagreement_raw),
            "disagreement_score": float(disagreement),
        },
        "pattern_break": {
            "pattern": pattern,
            "probability": float(pattern_break),
            "reliability": float(survival_rel),
            "pressure_contribution": float(pattern_break_signal),
        },
        "change_point": change,
        "semantics": "bayesian_turn_probability_plus_multiscale_disagreement_and_change_point",
    }


def forecast_road_pattern(
    history: str | Iterable[Any] | None,
) -> dict[str, Any]:
    sequence = normalize_bp(history)

    window_stats = _window_same_stats(sequence)
    replay = _pattern_replay_component(sequence)
    ngram = _ngram_component(sequence)
    survival = _pattern_survival_component(sequence)
    turning = _probabilistic_turning_component(
        sequence, window_stats=window_stats, survival=survival
    )
    turn_pressure = _clip(turning.get("turning_pressure", 0.0))
    multi_window = _multi_window_component(
        sequence, turn_pressure=turn_pressure, stats=window_stats
    )

    components = {
        "multi_window": multi_window,
        "pattern_replay": replay,
        "ngram": ngram,
        "pattern_survival": survival,
        "probabilistic_turning": turning,
    }
    dynamic_component_weights = _blend_weight_maps(
        STABLE_COMPONENT_WEIGHTS, TURN_COMPONENT_WEIGHTS, turn_pressure
    )

    logit_numerator = 0.0
    denominator = 0.0
    used: dict[str, Any] = {}
    for name, component in components.items():
        base_weight = float(dynamic_component_weights.get(name, 0.0))
        reliability = _clip(component.get("reliability", 0.0))
        effective = (
            base_weight * reliability
            if reliability >= MIN_COMPONENT_RELIABILITY
            else 0.0
        )
        p_b = _clip(component.get("p_b", 0.5), 0.34, 0.66)
        component_logit = _logit(p_b)
        logit_numerator += effective * component_logit
        denominator += effective
        used[name] = {
            "base_weight": float(base_weight),
            "reliability": float(reliability),
            "effective_weight": float(effective),
            "p_b": float(p_b),
            "p_p": float(1.0 - p_b),
            "logit": float(component_logit),
        }

    raw_logit = logit_numerator / denominator if denominator > 1e-12 else 0.0
    raw_probability_b = _sigmoid(raw_logit)
    raw_edge = raw_probability_b - 0.5

    maturity = _clip(len(sequence) / 20.0)
    maturity_factor = 0.35 + 0.65 * maturity
    final_logit = raw_logit * maturity_factor
    max_logit = abs(_logit(0.5 + MAX_DIRECTION_EDGE))
    final_logit = max(-max_logit, min(max_logit, final_logit))
    p_b = _clip(_sigmoid(final_logit), 0.37, 0.63)
    p_p = 1.0 - p_b
    direction = "B" if p_b >= p_p else "P"
    confidence = max(p_b, p_p)

    runs = _runs(sequence)
    current_run = runs[-1][1] if runs else 0
    return {
        "model_id": MODEL_ID,
        "version": VERSION,
        "available": True,
        "direction": direction,
        "action": direction,
        "probabilities": {"B": float(p_b), "P": float(p_p), "T": 0.0},
        "confidence": float(confidence),
        "selected_win_probability": float(confidence),
        "margin": float(abs(p_b - p_p)),
        "sequence_length": len(sequence),
        "big_road_sequence": "".join(sequence[-24:]),
        "current_run_length": int(current_run),
        "maturity": float(maturity),
        "maturity_factor": float(maturity_factor),
        "raw_edge": float(raw_edge),
        "final_edge": float(p_b - 0.5),
        "raw_logit": float(raw_logit),
        "final_logit": float(final_logit),
        "effective_weight_sum": float(denominator),
        "components": components,
        "component_weights": used,
        "dynamic_component_weights": {
            str(k): float(v) for k, v in dynamic_component_weights.items()
        },
        "dynamic_window_weights": dict(
            multi_window.get("dynamic_window_weights") or {}
        ),
        "turning_layer": turning,
        "turning_pressure": float(turn_pressure),
        "change_point_probability": float(
            turning.get("change_probability", 0.0) or 0.0
        ),
        "continue_probability": float(
            turning.get("continue_probability", 0.5) or 0.5
        ),
        "turn_probability": float(
            turning.get("turn_probability", 0.5) or 0.5
        ),
        "regime": str(turning.get("regime") or "STABLE"),
        "pattern": str(survival.get("pattern") or "GENERIC"),
        "pattern_survival_score": float(
            survival.get("survival_probability", 0.5) or 0.5
        ),
        "fusion_method": "reliability_weighted_log_odds_dynamic_turning",
        "direction_authority": "road_pattern_probability_core_only",
        "shoe_direction_weight": 0.0,
        "lstm_direction_weight": 0.0,
        "linucb_direction_weight": 0.0,
        "hazard_direction_weight": 0.0,
        "semantics": "road_only_dynamic_multiwindow_replay_ngram_survival_probabilistic_turning",
    }


__all__ = [
    "MODEL_ID",
    "VERSION",
    "WINDOW_WEIGHTS",
    "STABLE_WINDOW_WEIGHTS",
    "TURN_WINDOW_WEIGHTS",
    "COMPONENT_WEIGHTS",
    "STABLE_COMPONENT_WEIGHTS",
    "TURN_COMPONENT_WEIGHTS",
    "NGRAM_ORDERS",
    "REPLAY_LENGTHS",
    "normalize_bp",
    "forecast_road_pattern",
]
