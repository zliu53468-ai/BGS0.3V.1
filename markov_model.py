"""Support-aware variable-order three-way Markov predictor for BGS.

V3.3 keeps the stable V3.2 road/regime design and replaces hard single-order
backoff with a conservative hierarchical blend across order-1..4 contexts:
- every available order contributes according to raw/effective support,
- higher-order specificity is rewarded only modestly,
- posterior entropy slightly calibrates each order's contribution,
- cross-order direction disagreement shrinks the aggregate toward the prior,
- sparse high-order contexts can contribute but cannot abruptly take control,
- nested road coarse/full auxiliary state remains single-selected to avoid
  double counting.

This is a stochastic history model. It does not imply deterministic baccarat
patterns or guaranteed future outcomes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping
import math

from shoe_depth_estimator import ShoeDepthEstimator

MODEL_VERSION = "THREEWAY-VARIABLE-ORDER-MULTIORDER-BLEND-V3.3-QUANT-CANDIDATE"
OUTCOMES = ("B", "P", "T")
PHYSICAL_PRIOR = {"B": 0.4586, "P": 0.4462, "T": 0.0952}

MAX_ORDER = 4
SUPPORT_THRESHOLD = 4
BACKOFF_ALPHA = 0.75  # legacy compatibility diagnostic; no hard backoff in V3.3.
BAYES_PRIOR_STRENGTH = 6.0
DECAY = 0.95  # legacy name: this is retention lambda, not decay intensity.
MAX_ENTROPY = math.log2(3.0)
ENTROPY_WINDOW = 12
ENTROPY_SPIKE_THRESHOLD = 0.22
RECENT_FOCUS_WINDOW = 6

# Multi-order blend calibration. These values only control how the Markov
# hierarchy combines its own correlated order-1..4 contexts; they do not add a
# new model or a new fusion channel.
_ORDER_SPECIFICITY_STEP = 0.08
_ORDER_ENTROPY_FLOOR = 0.75
_ORDER_AGREEMENT_SHRINK_FLOOR = 0.65

_ROAD_SUPPORT_SCALE = {"road_coarse": 5.0, "road_full": 8.0}
_ROAD_ALPHA_CAP = {"road_coarse": 0.34, "road_full": 0.26}
_ROAD_FULL_MIN_RELIABILITY = 0.50


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize(values: Mapping[str, float]) -> Dict[str, float]:
    raw = {outcome: max(1e-12, float(values.get(outcome, 0.0) or 0.0)) for outcome in OUTCOMES}
    total = sum(raw.values())
    if total <= 1e-12:
        return dict(PHYSICAL_PRIOR)
    return {outcome: raw[outcome] / total for outcome in OUTCOMES}


def _clean_threeway(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in values:
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
    return result[-2000:]


def _clean_bp(sequence: Iterable[str]) -> list[str]:
    return [value for value in sequence if value in {"B", "P"}]


def _bp_runs(sequence: Iterable[str]) -> list[tuple[str, int]]:
    bp = _clean_bp(sequence)
    if not bp:
        return []
    runs: list[tuple[str, int]] = []
    side, length = bp[0], 1
    for value in bp[1:]:
        if value == side:
            length += 1
        else:
            runs.append((side, length))
            side, length = value, 1
    runs.append((side, length))
    return runs


def _run_bucket(length: int) -> str:
    value = max(0, int(length))
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    return "4+"


def _alternation_ratio(sequence: Iterable[str], window: int = 10) -> float:
    bp = _clean_bp(sequence)[-max(2, int(window)):]
    if len(bp) < 2:
        return 0.0
    changes = sum(1 for left, right in zip(bp, bp[1:]) if left != right)
    return changes / max(1, len(bp) - 1)


def _window_entropy(sequence: Iterable[str], window: int = ENTROPY_WINDOW) -> float:
    values = list(sequence)[-max(1, int(window)):]
    if not values:
        return MAX_ENTROPY
    counts = {outcome: values.count(outcome) for outcome in OUTCOMES}
    total = float(len(values))
    entropy = 0.0
    for outcome in OUTCOMES:
        p = counts[outcome] / total
        if p > 0.0:
            entropy -= p * math.log2(p)
    return float(entropy)


def _base_regime(sequence: Iterable[str]) -> str:
    bp = _clean_bp(sequence)
    runs = _bp_runs(bp)
    if len(bp) < 4 or not runs:
        return "MIXED"

    current_length = runs[-1][1]
    recent_runs = [length for _, length in runs[-5:]]
    recent_completed = [length for _, length in runs[-5:-1]]

    if current_length >= 4:
        return "DRAGON"

    alt_ratio = _alternation_ratio(bp, window=10)
    one_ratio = (
        sum(1 for length in recent_runs if length == 1) / len(recent_runs)
        if recent_runs else 0.0
    )
    if alt_ratio >= 0.72 and one_ratio >= 0.60:
        return "CHOP"

    double_evidence = recent_completed[-3:]
    if len(double_evidence) >= 2:
        near_two = sum(1 for length in double_evidence if length == 2)
        if near_two >= 2 and current_length in {1, 2}:
            return "DOUBLE_CHOP"

    return "MIXED"


def _detect_regime(sequence: Iterable[str]) -> Dict[str, Any]:
    values = list(sequence)
    bp = _clean_bp(values)
    runs = _bp_runs(bp)

    current_base = _base_regime(values)
    previous_base = _base_regime(values[:-3]) if len(values) >= 7 else "MIXED"

    entropy_current = _window_entropy(values, ENTROPY_WINDOW)
    if len(values) >= ENTROPY_WINDOW * 2:
        previous_window = values[-ENTROPY_WINDOW * 2:-ENTROPY_WINDOW]
        entropy_previous = _window_entropy(previous_window, ENTROPY_WINDOW)
    else:
        entropy_previous = entropy_current
    entropy_delta = entropy_current - entropy_previous
    entropy_spike = bool(
        len(values) >= ENTROPY_WINDOW * 2
        and entropy_delta >= ENTROPY_SPIKE_THRESHOLD
    )

    break_from_dragon = bool(
        len(runs) >= 2 and runs[-2][1] >= 4 and runs[-1][1] == 1
    )

    previous_chop = _base_regime(values[:-1]) == "CHOP" if len(values) >= 5 else False
    break_from_chop = bool(
        previous_chop and runs and runs[-1][1] >= 2
    )

    current_run_length = runs[-1][1] if runs else 0
    regime_changed = bool(
        len(values) >= 7
        and previous_base != current_base
        and (
            (current_base == "DRAGON" and current_run_length == 4)
            or (current_base == "MIXED" and previous_base != "MIXED")
            or (current_base == "DOUBLE_CHOP" and current_run_length == 1)
        )
    )

    pattern_break = bool(break_from_dragon or break_from_chop or regime_changed)
    change_point = bool(entropy_spike or pattern_break)
    transition = change_point
    regime = "TRANSITION" if transition else current_base

    recent_runs = [length for _, length in runs[-5:]]
    if regime == "TRANSITION":
        stability = 0.20
    elif regime == "MIXED":
        stability = 0.45
    elif len(recent_runs) >= 3:
        stability = 0.80
    else:
        stability = 0.60

    return {
        "regime": regime,
        "base_regime": current_base,
        "previous_regime": previous_base,
        "transition": transition,
        "change_point": change_point,
        "pattern_break": pattern_break,
        "entropy_window": ENTROPY_WINDOW,
        "entropy_current": float(entropy_current),
        "entropy_previous": float(entropy_previous),
        "entropy_delta": float(entropy_delta),
        "entropy_spike": entropy_spike,
        "entropy_spike_threshold": float(ENTROPY_SPIKE_THRESHOLD),
        "stability": float(stability),
        "alternation_ratio": float(_alternation_ratio(values, window=10)),
        "recent_run_lengths": recent_runs,
        "current_run_length": int(current_run_length),
        "previous_run_length": int(runs[-2][1]) if len(runs) >= 2 else 0,
        "break_from_dragon": break_from_dragon,
        "break_from_chop": break_from_chop,
        "regime_changed": regime_changed,
        "focus_window": int(RECENT_FOCUS_WINDOW if change_point else 0),
    }


def _adaptive_retention_lambda(
    sequence: Iterable[str],
    base_lambda: float = DECAY,
) -> float:
    """Return retention lambda used by w(age)=lambda**age.

    The user-facing "decay intensity" is delta = 1-lambda.
    Therefore a change point *raises decay intensity* by lowering lambda.
    """
    profile = _detect_regime(sequence)
    base = _clip(base_lambda, 0.70, 0.995)

    if profile["change_point"]:
        return 0.72
    if profile["regime"] == "DRAGON":
        return _clip(base + 0.025, 0.90, 0.98)
    if profile["regime"] in {"CHOP", "DOUBLE_CHOP"}:
        return _clip(base + 0.015, 0.89, 0.975)
    if profile["regime"] == "MIXED":
        return _clip(base - 0.015, 0.86, 0.96)
    return _clip(base - 0.05, 0.78, 0.94)


def _density_state(sequence: list[str]) -> Dict[str, Any]:
    recent5 = sequence[-5:]
    banker_count = recent5.count("B")
    player_count = recent5.count("P")
    delta = banker_count - player_count
    density = "High" if delta >= 2 else "Low" if delta <= -2 else "Medium"
    return {
        "recent5": recent5,
        "banker_count_recent5": banker_count,
        "player_count_recent5": player_count,
        "density_delta": delta,
        "density": density,
    }


def _road_state(sequence: list[str]) -> Dict[str, Any]:
    runs = _bp_runs(sequence)
    regime = _detect_regime(sequence)
    density = _density_state(sequence)

    current_side = runs[-1][0] if runs else ""
    current_length = runs[-1][1] if runs else 0
    previous_length = runs[-2][1] if len(runs) >= 2 else 0
    previous2_length = runs[-3][1] if len(runs) >= 3 else 0

    current_bucket = _run_bucket(current_length) if current_length else "0"
    previous_bucket = _run_bucket(previous_length) if previous_length else "0"
    previous2_bucket = _run_bucket(previous2_length) if previous2_length else "0"

    tie_trigger = "HasTie" if "T" in sequence[-3:] else "NoTie"
    coarse_key = (
        f"RC|side={current_side or 'NA'}|cur={current_bucket}|"
        f"prev={previous_bucket}|reg={regime['regime']}|tie={tie_trigger}"
    )
    full_key = (
        f"RF|side={current_side or 'NA'}|cur={current_bucket}|"
        f"prev={previous_bucket}|prev2={previous2_bucket}|"
        f"reg={regime['regime']}|density={density['density']}|tie={tie_trigger}"
    )
    return {
        "current_side": current_side,
        "current_run_length": int(current_length),
        "current_run_bucket": current_bucket,
        "previous_run_length": int(previous_length),
        "previous_run_bucket": previous_bucket,
        "previous2_run_length": int(previous2_length),
        "previous2_run_bucket": previous2_bucket,
        "coarse_key": coarse_key,
        "full_key": full_key,
        "regime": regime,
        **density,
    }


def encode_threeway_state(history: Iterable[Any]) -> Dict[str, Any]:
    sequence = _clean_threeway(history)
    density = _density_state(sequence)
    road = _road_state(sequence)
    recent3 = sequence[-3:]
    return {
        "ready": len(sequence) >= 2,
        "direction_context": "".join(sequence[-2:]) if sequence else "",
        "density": density["density"],
        "tie_trigger": "HasTie" if "T" in recent3 else "NoTie",
        "key": road["full_key"] if sequence else "",
        "recent5": density["recent5"],
        "recent3": recent3,
        "banker_count_recent5": density["banker_count_recent5"],
        "player_count_recent5": density["player_count_recent5"],
        "density_delta": density["density_delta"],
        "current_side": road["current_side"],
        "current_run_length": road["current_run_length"],
        "current_run_bucket": road["current_run_bucket"],
        "previous_run_length": road["previous_run_length"],
        "previous_run_bucket": road["previous_run_bucket"],
        "previous2_run_length": road["previous2_run_length"],
        "previous2_run_bucket": road["previous2_run_bucket"],
        "road_coarse_key": road["coarse_key"],
        "road_full_key": road["full_key"],
        "regime": road["regime"]["regime"],
        "base_regime": road["regime"]["base_regime"],
        "previous_regime": road["regime"]["previous_regime"],
        "regime_transition": road["regime"]["transition"],
        "regime_stability": road["regime"]["stability"],
        "alternation_ratio": road["regime"]["alternation_ratio"],
        "recent_run_lengths": road["regime"]["recent_run_lengths"],
        "change_point": road["regime"]["change_point"],
        "entropy_current": road["regime"]["entropy_current"],
        "entropy_delta": road["regime"]["entropy_delta"],
    }


def _context_keys(sequence: list[str]) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    for order in range(1, MAX_ORDER + 1):
        if len(sequence) >= order:
            keys[f"order_{order}"] = f"O{order}|{''.join(sequence[-order:])}"
    if sequence:
        road = _road_state(sequence)
        keys["road_coarse"] = road["coarse_key"]
        keys["road_full"] = road["full_key"]
    return keys


def _new_counts() -> Dict[str, float]:
    return {"B": 0.0, "P": 0.0, "T": 0.0}


def _build_transition_tables(
    sequence: list[str],
    *,
    retention_lambda: float,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Build decayed and raw transition counters for all 1..4 orders.

    For a transition with age a:
        effective_count = lambda ** a
    Raw support remains the literal number of observations and is retained for
    compatibility diagnostics and blend reliability calibration.
    """
    weighted: Dict[str, Dict[str, float]] = defaultdict(_new_counts)
    raw: Dict[str, Dict[str, float]] = defaultdict(_new_counts)
    n = len(sequence)

    for index in range(1, n):
        prefix = sequence[:index]
        age = max(0, n - 1 - index)
        weight = float(retention_lambda ** age)
        outcome = sequence[index]
        for key in _context_keys(prefix).values():
            weighted[key][outcome] += weight
            raw[key][outcome] += 1.0

    return (
        {key: dict(value) for key, value in weighted.items()},
        {key: dict(value) for key, value in raw.items()},
    )


def _support(counts: Mapping[str, float]) -> float:
    return sum(float(counts.get(outcome, 0.0) or 0.0) for outcome in OUTCOMES)


def _reliability(support: float, scale: float) -> float:
    value = max(0.0, float(support))
    return value / (value + max(1e-9, float(scale)))


def _posterior_probabilities(
    counts: Mapping[str, float],
    *,
    prior_strength: float,
    parent_prior: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    """Dirichlet/Bayesian smoothing.

    P(y|c) = (N(c,y) + beta * pi_y) / (N(c) + beta)
    where pi is either the physical baccarat prior or a supplied parent prior.
    """
    prior = _normalize(parent_prior or PHYSICAL_PRIOR)
    denominator = _support(counts) + prior_strength
    if denominator <= 1e-12:
        return dict(prior)
    return {
        outcome: (
            float(counts.get(outcome, 0.0) or 0.0)
            + prior_strength * prior[outcome]
        ) / denominator
        for outcome in OUTCOMES
    }


def _blend(
    base: Mapping[str, float],
    overlay: Mapping[str, float],
    alpha: float,
) -> Dict[str, float]:
    weight = _clip(alpha)
    mixed = {
        outcome: (1.0 - weight) * float(base[outcome])
        + weight * float(overlay[outcome])
        for outcome in OUTCOMES
    }
    return _normalize(mixed)


def _direction_vote(probabilities: Mapping[str, float]) -> str:
    return "B" if float(probabilities["B"]) >= float(probabilities["P"]) else "P"


def _entropy(probabilities: Mapping[str, float]) -> float:
    value = 0.0
    for outcome in OUTCOMES:
        p = max(1e-15, min(1.0, float(probabilities[outcome])))
        value -= p * math.log2(p)
    return float(value)


def _support_aware_markov(
    sequence: list[str],
    weighted_table: Mapping[str, Mapping[str, float]],
    raw_table: Mapping[str, Mapping[str, float]],
    *,
    prior_strength: float,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """Blend all available order-1..4 contexts instead of hard backoff.

    Each order receives a conservative raw weight from:
      sqrt(raw_support_reliability * effective_support_reliability)
      * modest order-specificity factor
      * modest posterior-certainty factor

    The per-order weights are normalized before averaging, so nested orders are
    not treated as independent additive votes. The resulting aggregate is then
    shrunk toward the physical prior using the strongest available support and
    cross-order B/P agreement. This keeps sparse high orders informative without
    allowing a single low-support context to abruptly replace lower-order evidence.
    """
    contexts = _context_keys(sequence)
    highest = min(MAX_ORDER, len(sequence))
    details: Dict[str, Any] = {}
    raw_order_weights: Dict[str, float] = {}

    for order in range(1, highest + 1):
        name = f"order_{order}"
        key = contexts.get(name)
        if not key:
            continue

        weighted_counts = dict(weighted_table.get(key, _new_counts()))
        raw_counts = dict(raw_table.get(key, _new_counts()))
        raw_support = _support(raw_counts)
        effective_support = _support(weighted_counts)
        posterior = _posterior_probabilities(
            weighted_counts,
            prior_strength=prior_strength,
        )

        raw_support_reliability = _reliability(raw_support, SUPPORT_THRESHOLD)
        effective_support_reliability = _reliability(
            effective_support,
            SUPPORT_THRESHOLD,
        )
        support_score = math.sqrt(
            max(0.0, raw_support_reliability * effective_support_reliability)
        )
        posterior_certainty = _clip(1.0 - _entropy(posterior) / MAX_ENTROPY)
        entropy_factor = _ORDER_ENTROPY_FLOOR + (
            1.0 - _ORDER_ENTROPY_FLOOR
        ) * posterior_certainty
        specificity_factor = 1.0 + _ORDER_SPECIFICITY_STEP * (order - 1)
        raw_weight = support_score * entropy_factor * specificity_factor

        raw_order_weights[name] = float(raw_weight)
        details[name] = {
            "key": key,
            "raw_support": float(raw_support),
            "effective_support": float(effective_support),
            "support_threshold": int(SUPPORT_THRESHOLD),
            "qualifies": bool(raw_support >= SUPPORT_THRESHOLD),
            "posterior": dict(posterior),
            "counts": weighted_counts,
            "raw_support_reliability": float(raw_support_reliability),
            "effective_support_reliability": float(effective_support_reliability),
            "support_score": float(support_score),
            "posterior_entropy_bits": float(_entropy(posterior)),
            "posterior_certainty": float(posterior_certainty),
            "entropy_factor": float(entropy_factor),
            "specificity_factor": float(specificity_factor),
            "raw_blend_weight": float(raw_weight),
        }

    weight_total = sum(raw_order_weights.values())
    if weight_total > 1e-12:
        normalized_weights = {
            name: float(weight / weight_total)
            for name, weight in raw_order_weights.items()
        }
    else:
        normalized_weights = {name: 0.0 for name in raw_order_weights}

    for name, item in details.items():
        item["blend_weight"] = float(normalized_weights.get(name, 0.0))

    if weight_total <= 1e-12:
        aggregate = dict(PHYSICAL_PRIOR)
    else:
        aggregate = _normalize({
            outcome: sum(
                normalized_weights.get(name, 0.0)
                * float(item["posterior"][outcome])
                for name, item in details.items()
            )
            for outcome in OUTCOMES
        })

    b_weight = sum(
        raw_order_weights.get(name, 0.0)
        for name, item in details.items()
        if _direction_vote(item["posterior"]) == "B"
    )
    p_weight = sum(
        raw_order_weights.get(name, 0.0)
        for name, item in details.items()
        if _direction_vote(item["posterior"]) == "P"
    )
    vote_total = b_weight + p_weight
    if vote_total <= 1e-12:
        agreement = 0.5
    else:
        agreement = max(b_weight, p_weight) / vote_total

    max_raw_support = max(
        [float(item["raw_support"]) for item in details.values()] or [0.0]
    )
    max_effective_support = max(
        [float(item["effective_support"]) for item in details.values()] or [0.0]
    )
    raw_strength = _reliability(max_raw_support, SUPPORT_THRESHOLD)
    effective_strength = _reliability(
        max_effective_support,
        SUPPORT_THRESHOLD,
    )
    support_strength = math.sqrt(max(0.0, raw_strength * effective_strength))
    agreement_factor = _ORDER_AGREEMENT_SHRINK_FLOOR + (
        1.0 - _ORDER_AGREEMENT_SHRINK_FLOOR
    ) * agreement
    hierarchical_evidence_strength = _clip(support_strength * agreement_factor)
    probability = _blend(
        PHYSICAL_PRIOR,
        aggregate,
        hierarchical_evidence_strength,
    )

    if normalized_weights:
        selected_name = max(
            normalized_weights,
            key=lambda name: (
                normalized_weights[name],
                int(name.rsplit("_", 1)[-1]),
            ),
        )
        selected_order = int(selected_name.rsplit("_", 1)[-1])
        selected_item = details[selected_name]
        selected_counts = dict(selected_item["counts"])
        selected_posterior = dict(selected_item["posterior"])
    else:
        selected_name = "physical_prior"
        selected_order = 0
        selected_counts = _new_counts()
        selected_posterior = dict(PHYSICAL_PRIOR)

    # Compatibility fields are retained for downstream diagnostics. V3.3 does
    # not execute hard N->N-1 backoff; backoff_penalty now represents the
    # conservative hierarchical evidence-strength shrinkage applied to the
    # multi-order aggregate.
    backoff_steps = 0
    backoff_penalty = float(hierarchical_evidence_strength)

    return probability, {
        "mode": "support_entropy_weighted_multi_order_blend",
        "support_threshold": int(SUPPORT_THRESHOLD),
        "backoff_alpha": float(BACKOFF_ALPHA),
        "backoff_alpha_legacy_only": True,
        "highest_available_order": int(highest),
        "selected_order": int(selected_order),
        "selected_context": selected_name,
        "backoff_steps": int(backoff_steps),
        "backoff_penalty": float(backoff_penalty),
        "contexts": details,
        "selected_counts": dict(selected_counts),
        "selected_posterior": dict(selected_posterior),
        "multi_order_agreement": float(agreement),
        "order_weights": {
            name: float(weight) for name, weight in normalized_weights.items()
        },
        "aggregate_posterior": {
            outcome: float(aggregate[outcome]) for outcome in OUTCOMES
        },
        "hierarchical_evidence_strength": float(hierarchical_evidence_strength),
        "max_raw_support": float(max_raw_support),
        "max_effective_support": float(max_effective_support),
        "agreement_factor": float(agreement_factor),
        "semantics": (
            "normalized_correlated_order_blend_then_prior_shrink_"
            "not_independent_vote_summing"
        ),
    }


def _apply_nested_road(
    sequence: list[str],
    base_probability: Mapping[str, float],
    weighted_table: Mapping[str, Mapping[str, float]],
    raw_table: Mapping[str, Mapping[str, float]],
    *,
    prior_strength: float,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    contexts = _context_keys(sequence)
    candidates: Dict[str, Dict[str, Any]] = {}

    for name in ("road_coarse", "road_full"):
        key = contexts.get(name)
        if not key:
            continue
        weighted_counts = dict(weighted_table.get(key, _new_counts()))
        raw_counts = dict(raw_table.get(key, _new_counts()))
        raw_support = _support(raw_counts)
        effective_support = _support(weighted_counts)
        posterior = _posterior_probabilities(
            weighted_counts,
            prior_strength=prior_strength,
            parent_prior=base_probability,
        )
        reliability = _reliability(effective_support, _ROAD_SUPPORT_SCALE[name])
        candidates[name] = {
            "key": key,
            "raw_support": float(raw_support),
            "effective_support": float(effective_support),
            "posterior": posterior,
            "reliability": float(reliability),
            "raw_alpha": float(_ROAD_ALPHA_CAP[name] * reliability),
            "counts": weighted_counts,
        }

    full = candidates.get("road_full")
    coarse = candidates.get("road_coarse")
    full_ready = bool(
        full
        and float(full["raw_support"]) >= SUPPORT_THRESHOLD
        and float(full["reliability"]) >= _ROAD_FULL_MIN_RELIABILITY
    )
    selected_name = (
        "road_full" if full_ready
        else "road_coarse" if coarse
        else "road_full" if full
        else ""
    )

    probability = dict(base_probability)
    selected_alpha = 0.0
    if selected_name:
        selected = candidates[selected_name]
        selected_alpha = float(selected["raw_alpha"])
        probability = _blend(probability, selected["posterior"], selected_alpha)

    details: Dict[str, Any] = {}
    for name, item in candidates.items():
        details[name] = {
            **item,
            "applied": bool(name == selected_name),
            "alpha": float(item["raw_alpha"]) if name == selected_name else 0.0,
        }

    return probability, {
        "mode": "nested_full_else_coarse",
        "selected_context": selected_name,
        "selected_alpha": float(selected_alpha),
        "full_min_reliability": float(_ROAD_FULL_MIN_RELIABILITY),
        "double_count_prevented": True,
        "contexts": details,
    }


def update_and_predict_engine(
    history: Iterable[Any],
    *,
    decay: float = DECAY,
    prior_strength: float = BAYES_PRIOR_STRENGTH,
) -> Dict[str, Any]:
    sequence = _clean_threeway(history)
    prior_strength = max(1e-9, float(prior_strength))
    shoe = ShoeDepthEstimator().estimate(sequence)
    current_state = encode_threeway_state(sequence)
    regime_profile = _detect_regime(sequence)

    retention_lambda = _adaptive_retention_lambda(sequence, float(decay))
    decay_intensity = 1.0 - retention_lambda

    # Keep the existing V3.2 change-point refocus behavior unchanged in this PR.
    if regime_profile["change_point"]:
        training_sequence = sequence[-RECENT_FOCUS_WINDOW:]
        focus_applied = True
    else:
        training_sequence = sequence
        focus_applied = False

    weighted_table, raw_table = _build_transition_tables(
        training_sequence,
        retention_lambda=retention_lambda,
    )

    markov_probability, backoff = _support_aware_markov(
        sequence,
        weighted_table,
        raw_table,
        prior_strength=prior_strength,
    )
    probabilities, road_selection = _apply_nested_road(
        sequence,
        markov_probability,
        weighted_table,
        raw_table,
        prior_strength=prior_strength,
    )

    entropy_bits = _entropy(probabilities)
    entropy_weight = _clip(1.0 - entropy_bits / MAX_ENTROPY)
    agreement = float(backoff["multi_order_agreement"])
    selected_order = int(backoff["selected_order"])
    selected_raw_support = 0.0
    selected_item = dict(backoff["contexts"].get(f"order_{selected_order}") or {})
    if selected_item:
        selected_raw_support = float(selected_item.get("raw_support", 0.0) or 0.0)
    support_strength = float(
        backoff.get(
            "hierarchical_evidence_strength",
            _reliability(selected_raw_support, SUPPORT_THRESHOLD),
        )
        or 0.0
    )
    support_strength = _clip(support_strength)
    regime_stability = float(regime_profile["stability"])
    backoff_penalty = float(backoff["backoff_penalty"])

    evidence_quality = _clip(
        0.35
        + 0.20 * agreement
        + 0.20 * support_strength
        + 0.15 * regime_stability
        + 0.10 * backoff_penalty
    )
    base_weight = _clip(entropy_weight * evidence_quality)

    # Maturity only calibrates confidence; it never creates deterministic certainty.
    maturity = max(0.15, float(shoe.shoe_progress))
    final_weight = _clip(base_weight * maturity)

    direction = "B" if probabilities["B"] >= probabilities["P"] else "P"
    selected_counts = dict(backoff["selected_counts"])
    effective_support = _support(selected_counts)

    hierarchy = {
        "mode": "support_weighted_multiorder_blend_plus_nested_road",
        # Keep the old key name for callers that already inspect this structure.
        "markov_backoff": backoff,
        "markov_multiorder_blend": backoff,
        "road_selection": road_selection,
        "dominant_context": str(backoff["selected_context"]),
        "dominant_counts": selected_counts,
        "multi_order_agreement": float(agreement),
        "support_strength": float(support_strength),
        "order_weights": dict(backoff.get("order_weights") or {}),
        "hierarchical_evidence_strength": float(
            backoff.get("hierarchical_evidence_strength", 0.0) or 0.0
        ),
        "max_context_support": float(
            max(
                [
                    float(item.get("raw_support", 0.0) or 0.0)
                    for item in backoff["contexts"].values()
                ]
                or [0.0]
            )
        ),
    }

    return {
        "model_version": MODEL_VERSION,
        "engine": "THREEWAY_VARIABLE_ORDER_MULTIORDER_BLEND_MARKOV",
        "history": sequence,
        "sample_count": len(sequence),
        "state": current_state,
        "state_key": current_state["key"],
        "decay": float(retention_lambda),
        "base_decay": float(decay),
        "adaptive_decay": float(retention_lambda),
        "retention_lambda": float(retention_lambda),
        "decay_intensity": float(decay_intensity),
        "focus_applied": bool(focus_applied),
        "focus_window": int(RECENT_FOCUS_WINDOW if focus_applied else 0),
        "training_sample_count": int(len(training_sequence)),
        "regime": str(regime_profile["regime"]),
        "regime_profile": dict(regime_profile),
        "prior": dict(PHYSICAL_PRIOR),
        "prior_strength": float(prior_strength),
        "max_order": int(MAX_ORDER),
        "support_threshold": int(SUPPORT_THRESHOLD),
        "backoff_alpha": float(BACKOFF_ALPHA),
        "selected_order": int(selected_order),
        "backoff_steps": int(backoff["backoff_steps"]),
        "backoff_penalty": float(backoff_penalty),
        "transition_counts": {
            outcome: float(selected_counts.get(outcome, 0.0) or 0.0)
            for outcome in OUTCOMES
        },
        "effective_support": float(effective_support),
        "state_count": int(len(weighted_table)),
        "probabilities": {
            outcome: float(probabilities[outcome]) for outcome in OUTCOMES
        },
        "markov_probabilities_before_road": {
            outcome: float(markov_probability[outcome]) for outcome in OUTCOMES
        },
        "direction": direction,
        "entropy_bits": float(entropy_bits),
        "max_entropy_bits": float(MAX_ENTROPY),
        "entropy_weight": float(entropy_weight),
        "base_weight": float(base_weight),
        "shoe_progress": float(shoe.shoe_progress),
        "final_weight": float(final_weight),
        "shoe_depth": shoe.as_dict(),
        "tie_risk_active": bool(probabilities["T"] > 0.15),
        "tie_risk_threshold": 0.15,
        "hierarchical_backoff": hierarchy,
        "multi_order_agreement": float(agreement),
        "support_strength": float(support_strength),
        "dominant_context": str(backoff["selected_context"]),
        "order_weights": dict(backoff.get("order_weights") or {}),
        "hierarchical_evidence_strength": float(
            backoff.get("hierarchical_evidence_strength", 0.0) or 0.0
        ),
        "confidence_semantics": (
            "entropy_support_multiorder_agreement_regime_maturity_weight_"
            "not_win_probability"
        ),
    }


def predict_markov(
    history: Iterable[Any],
    *,
    road_context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    del road_context
    return update_and_predict_engine(history)


__all__ = [
    "MODEL_VERSION",
    "OUTCOMES",
    "PHYSICAL_PRIOR",
    "DECAY",
    "BAYES_PRIOR_STRENGTH",
    "MAX_ORDER",
    "SUPPORT_THRESHOLD",
    "BACKOFF_ALPHA",
    "ENTROPY_WINDOW",
    "encode_threeway_state",
    "update_and_predict_engine",
    "predict_markov",
]
